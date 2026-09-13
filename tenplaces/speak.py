"""The robot talks back (Speechmatics text-to-speech): it says its plan and why, confirms spoken changes, says
when it stops or repairs something and what it cannot do. Routine events (skill starts, checks) stay silent.

    speaker = Speaker()                                            # plays live; records every line's sim time
    on_event = SpeakEvents(speaker, forward=rec.on_event)          # chains with DemoRecorder
    run_command(..., on_event=on_event, voice=EchoGuard(voice, speaker))
    speaker.close(); speaker.write_track("demo.wav", rec.duration_s); mux("demo.mp4", "demo.wav", "demo.mp4")

say() never blocks: synthesis and playback run on a worker thread, so the 25 Hz control loop never waits on
the network. Every phrase is cached on disk by (voice, text), so repeated lines cost nothing and a re-run
renders its audio track offline. The track is laid out on the simulated timeline the demo video uses
(one frame per 40 ms control step), not on wall-clock time: the robot pauses while the VLM plans.
"""
import hashlib
import io
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .listen import STOP
from .paths import OUT

TTS_URL = "https://preview.tts.speechmatics.com/generate/{voice}"
VOICES = ("sarah", "theo", "megan", "jack")  # UK female, UK male, US female, US male
RATE = 16000  # the API's wav_16000 output: mono int16
CACHE = OUT / "audio" / "tts_cache"
# Lines that do not depend on the plan; Speaker.prewarm() caches them before the run.
COMMON = ("Stopping.", "One moment.", "The table is set.", "Done.", "OK — no change to the plan.",
          "OK — nothing more after this.")


def _fetch(text: str, voice: str, timeout: float = 15.0, stats: dict | None = None) -> bytes:
    """One TTS request; returns the WAV bytes."""
    import requests

    from .voice import _require_key

    t = time.perf_counter()
    r = requests.post(TTS_URL.format(voice=voice), params={"output_format": "wav_16000"}, json={"text": text},
                      headers={"Authorization": f"Bearer {_require_key()}"}, timeout=timeout, stream=True)
    r.raise_for_status()
    chunks = []
    for chunk in r.iter_content(4096):
        if not chunks and stats is not None:
            stats["first_audio_ms"] = 1000 * (time.perf_counter() - t)
        chunks.append(chunk)
    if stats is not None:
        stats["total_ms"] = 1000 * (time.perf_counter() - t)
    return b"".join(chunks)


def cache_path(text: str, voice: str, cache_dir=CACHE) -> Path:
    return Path(cache_dir) / f"{voice}_{hashlib.sha1(f'{voice}\n{text}'.encode()).hexdigest()[:16]}.wav"


def synthesize(text: str, voice: str = "sarah", cache_dir=CACHE, stats: dict | None = None):
    """Text -> (int16 mono samples, sample_rate), from the disk cache when this (voice, text) was said before."""
    import soundfile as sf

    path = cache_path(text, voice, cache_dir)
    if stats is not None:
        stats["cached"] = path.exists()
    if not path.exists():
        data = _fetch(text, voice, stats=stats)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.part")  # prewarm and the speaker may race
        tmp.write_bytes(data)
        tmp.replace(path)  # never leave a half-written file that later reads as a cache hit
    samples, rate = sf.read(str(path), dtype="int16", always_2d=True)
    return samples[:, 0].copy(), rate


class Speaker:
    """Lines are spoken one at a time on a worker thread. If more than `max_pending` wait, the oldest waiting
    ones are dropped from playback (stale news); they stay in `lines`, so the video track still has them."""

    def __init__(self, voice: str = "sarah", play: bool = True, synth=synthesize, max_pending: int = 2):
        self.voice, self.play, self.synth, self.max_pending = voice, play, synth, max_pending
        self.lines = []  # {"t": sim time (or wall seconds since start), "text": ...} for every line said
        self.latest, self.error = "", None
        self.playing = []  # [(wall start, wall end or None, text)] of what went out of the speakers
        self._t0, self._pending, self._cv = time.perf_counter(), deque(), threading.Condition()
        self._interrupt, self._closed, self._busy = threading.Event(), False, False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def prewarm(self, texts=None):
        """Synthesize fixed phrases in the background now, so they later play without the ~1.1 s request."""
        def run():
            for text in texts or COMMON:
                try:
                    self.synth(text, self.voice)
                except Exception as e:
                    self.error = e

        threading.Thread(target=run, daemon=True).start()

    def say(self, text: str, sim_time: float | None = None, interrupt: bool = False):
        t = sim_time if sim_time is not None else time.perf_counter() - self._t0
        self.lines.append({"t": float(t), "text": text, "cut": interrupt})
        self.latest = text
        with self._cv:
            if interrupt:  # e.g. "Stopping." cuts off whatever is being said and everything queued
                self._pending.clear()
                self._interrupt.set()
            self._pending.append(text)
            while len(self._pending) > self.max_pending:
                self._pending.popleft()
            self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                while not self._pending and not self._closed:
                    self._cv.wait()
                if not self._pending:
                    return
                text = self._pending.popleft()
                self._busy = True
                self._interrupt.clear()
            try:
                samples, rate = self.synth(text, self.voice)
                if self.play and not self._interrupt.is_set():
                    self._play(samples, rate, text)
            except Exception as e:  # a failed line must not end the voice; kept for the caller to inspect
                self.error = e
            finally:
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()

    def _play(self, samples, rate, text):
        import sounddevice as sd

        entry = [time.perf_counter(), None, text]
        self.playing.append(entry)
        sd.play(samples, rate)
        end = time.perf_counter() + len(samples) / rate + 0.2
        while time.perf_counter() < end and not self._interrupt.is_set():
            time.sleep(0.02)
        sd.stop()
        entry[1] = time.perf_counter()

    def is_echo(self, heard: str, window_s: float = 2.0, overlap: float = 0.6) -> bool:
        """Was `heard` most likely the robot's own voice picked up by the microphone? True when most of its
        words belong to a line played within the last `window_s` seconds."""
        words = set(re.findall(r"[a-z']+", heard.lower()))
        if not words:
            return False
        now = time.perf_counter()
        for start, end, text in list(self.playing):
            if end is None or now - end < window_s:
                said = set(re.findall(r"[a-z']+", text.lower()))
                if len(words & said) >= overlap * len(words):
                    return True
        return False

    def flush(self, timeout: float = 30.0):
        """Wait until everything queued has been synthesized (and played)."""
        with self._cv:
            self._cv.wait_for(lambda: not self._pending and not self._busy, timeout=timeout)

    def close(self, timeout: float = 30.0):
        self.flush(timeout)
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._thread.join(timeout=timeout)

    def write_track(self, path, duration_s: float, start_s: float = 0.0, rate: int = RATE, gap_s: float = 0.15,
                    max_late_s: float = 3.0):
        """A mono WAV of every line at its simulated time minus `start_s` (the sim time of the video's first
        frame, DemoRecorder.t_start). A line said with interrupt=True starts on time and cuts the one before, as
        it did live; any other line that would overlap waits for the previous one, and is left out if that makes
        it more than `max_late_s` late. Returns [(start_s, text)] in video time."""
        import soundfile as sf

        track = np.zeros(int(round(duration_s * rate)), np.int16)
        placed, free_at = [], 0.0
        for line in sorted(self.lines, key=lambda x: x["t"]):
            t = max(0.0, line["t"] - start_s)
            if line.get("cut"):
                track[int(round(t * rate)):] = 0  # silence the rest of what was being said
                free_at = t
            start = max(t, free_at)
            if start - t > max_late_s or start >= duration_s:
                continue
            samples, r = self.synth(line["text"], self.voice)
            if r != rate:
                n = int(len(samples) * rate / r)
                samples = np.interp(np.arange(n) * r / rate, np.arange(len(samples)), samples).astype(np.int16)
            i = int(round(start * rate))
            chunk = samples[: len(track) - i]
            track[i: i + len(chunk)] = chunk
            placed.append((round(start, 3), line["text"]))
            free_at = start + len(samples) / rate + gap_s
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), track, rate, subtype="PCM_16")
        return placed


def mux(video_mp4, wav, out_mp4):
    """Put the audio track into the video (video stream copied, audio as AAC). out_mp4 may be video_mp4."""
    import imageio_ffmpeg

    out = Path(out_mp4)
    tmp = out.with_name(out.stem + ".mux_tmp.mp4")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video_mp4), "-i", str(wav),
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", str(tmp)]
    subprocess.run(cmd, check=True, capture_output=True)
    tmp.replace(out)
    return out


# ---- what the robot says --------------------------------------------------------------------------------------

NOUN = {"drawer": "drawer", "spoon": "spoon", "plate": "plate", "fork": "fork", "cup": "cup"}
# Spoken versions of planner.PREREQ_WHY, keyed (prerequisite, skill).
WHY = {("drawer", "spoon"): "The drawer comes first — the spoon is inside.",
       ("drawer", "fork"): "The drawer comes first — the fork is inside.",
       ("plate", "fork"): "The plate comes first — it sits on the fork's spot.",
       ("drawer", "plate"): "The drawer comes first — its tray reaches the placemat."}
ADDED = re.compile(r"added '(\w+)' before '(\w+)'")


def _join(items, last="and"):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {last} " + items[-1]


def _the(skills, last="and"):
    return "the " + _join([NOUN.get(s, s) for s in skills], last)


def _plan_sentence(steps):
    objs = [s for s in steps if s != "drawer"]
    if "drawer" in steps:
        return f"I'll open the drawer, then place {_the(objs)}." if objs else "I'll open the drawer."
    return f"I'll place {_the(objs)}."


def _why(corrections):
    """The verifier's most important correction, spoken: the first prerequisite it added."""
    for c in corrections or []:
        m = ADDED.search(c)
        if m and (m[1], m[2]) in WHY:
            return WHY[(m[1], m[2])]
    return None


def _cannot(unsupported):
    items = [u.strip().rstrip(".") for u in unsupported or [] if u and u.strip()]
    return f"Sorry, I can't {_join(items, 'or')}." if items else None


def _is_are(skills):
    return f"{_the(skills)} {'is' if len(skills) == 1 else 'are'}"


def phrase_for(event: dict, state: dict | None = None) -> str | None:
    """A short sentence for an agent event, or None to stay quiet. `state` (a dict owned by the caller, e.g.
    SpeakEvents) remembers the plan and what is done, so a spoken change can be confirmed as a difference
    ("OK — skipping the fork.") and the end can say what is missing."""
    st = state if state is not None else {}
    st.setdefault("done", [])
    kind = event.get("kind")
    text = None
    if kind == "plan":
        steps = event.get("steps", [])
        parts = [_plan_sentence(steps) if steps else None, _why(event.get("corrections")),
                 _cannot(event.get("unsupported"))]
        if not steps and not event.get("unsupported"):
            parts = ["Alright — nothing to do."]
        text = " ".join(p for p in parts if p)
        st["remaining"], st["target"] = list(steps), set(st["done"]) | set(steps)
    elif kind == "replan":
        failed = str(event.get("reason", "")).split(" ")[0]
        steps = event.get("steps", [])
        head = f"The {NOUN.get(failed, failed)} isn't in place." if failed in NOUN else "That didn't work."
        text = f"{head} New plan: {_plan_sentence(steps)[5:]}" if steps else f"{head} Nothing more I can do."
        st["remaining"], st["target"] = list(steps), set(st["done"]) | set(steps)
    elif kind == "amend":
        steps, after = event.get("steps", []), event.get("after")
        if "remaining" in st:
            before = [s for s in st["remaining"] if s != after and s not in st["done"]]
            removed = [s for s in before if s not in steps]
            added = [s for s in steps if s not in before]
            changes = ([f"skipping {_the(removed)}"] if removed else []) + ([f"adding {_the(added)}"] if added else [])
            if changes:
                text = "OK — " + " and ".join(changes) + "."
            else:
                text = "OK — no change to the plan."
        else:
            text = f"OK — next, {_the(steps)}." if steps else None
        if not steps:
            text = "OK — nothing more after this." if after else "OK — that's all."
        why = _why(event.get("corrections"))
        text = " ".join(p for p in (text, why) if p)
        st["remaining"] = list(steps)
        st["target"] = set(st["done"]) | ({after} if after else set()) | set(steps)
    elif kind == "thinking":
        text = None if STOP.search(str(event.get("heard", ""))) else "One moment."
    elif kind == "stop":
        text = "Stopping."
    elif kind == "skill_start":
        if event.get("attempt", 1) > 1:
            text = f"Trying the {NOUN.get(event['skill'], event['skill'])} again."
    elif kind == "check":
        if event.get("done") and event.get("skill") not in st["done"]:
            st["done"].append(event["skill"])
    elif kind == "regressed":
        s = event.get("skill")
        if s in st["done"]:
            st["done"].remove(s)
        text = "The drawer closed — opening it again." if s == "drawer" else f"The {NOUN.get(s, s)} moved — putting it back."
    elif kind == "gave_up":
        s = event.get("skill")
        text = f"I can't get the {NOUN.get(s, s)} back in place — leaving it."
    elif kind == "finished":
        if event.get("stopped"):
            return None  # "Stopping." was already said
        done = set(event.get("done", []))
        missing = [s for s in ("drawer", "spoon", "plate", "fork", "cup") if s in st.get("target", ()) and s not in done]
        if missing:
            text = f"Done — but {_is_are(missing)}n't in place."
        elif done - {"drawer"}:
            text = "The table is set."
        else:
            text = "Done."
    return text


class SpeakEvents:
    """on_event callback: speaks phrase_for(event) at the event's sim time, records it in the event as
    "said" (so the log, the JSON and the demo panel show it), then forwards the event.

    When the person has spoken, the robot's next line cuts off whatever it was still saying: the answer to
    them ("One moment.", "OK — skipping the fork.") matters more than the rest of a 7 s plan sentence."""

    def __init__(self, speaker: Speaker, forward=None):
        self.speaker, self.forward, self.state, self._barge_in = speaker, forward, {}, False

    def __call__(self, event: dict):
        self._barge_in |= event.get("kind") == "heard"
        text = phrase_for(event, self.state)
        if text:
            event["said"] = text
            self.speaker.say(text, sim_time=event.get("t"), interrupt=self._barge_in or event.get("kind") == "stop")
            self._barge_in = False
        if self.forward is not None:
            self.forward(event)


class EchoGuard:
    """A voice source (tenplaces.listen) that ignores the robot's own words picked up by the microphone, so
    "The plate moved — putting it back." is not heard as a request. A person talking over it still counts."""

    def __init__(self, voice, speaker: Speaker):
        self.voice, self.speaker, self.ignored = voice, speaker, []

    @property
    def partial(self):
        return getattr(self.voice, "partial", "")

    def poll(self, sim_time=None):
        out = []
        for text in self.voice.poll(sim_time):
            (self.ignored if self.speaker.is_echo(text) else out).append(text)
        return out

    def close(self):
        self.voice.close()

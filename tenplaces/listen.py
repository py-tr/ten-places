"""Things said while the robot works. The agent polls every control step; each complete sentence is either
"stop" (handled locally, no model call) or a change to the plan (sent to the VLM planner's amend()).

    voice = LiveVoice()                                      # microphone -> Speechmatics RT, one session per run
    voice = ScriptedVoice([(12.0, "Skip the fork.")])        # the same at fixed simulated times (tests, videos)

Both expose poll(sim_time) -> [sentences], .partial (the words being spoken now, for the demo panel), close().
LiveVoice also keeps what the microphone heard for the whole run, where each sentence lies in that audio
(Speechmatics' own timings) and the computer's clock against simulated time: save() writes them, and a demo video
places each sentence at the simulated moment it was said (live_placements, scripts/voice_over.py).
"""
import asyncio
import json
import re
import threading
import time
from pathlib import Path

import numpy as np

STOP = re.compile(r"\b(stop|halt|freeze)\b", re.IGNORECASE)
# A released sentence carries at least one letter or digit. Speechmatics punctuates, and the tail of a sentence
# can arrive as a final segment of its own ("."), which is not an instruction: it costs a planner call and holds
# the arms until the answer comes back.
WORD = re.compile(r"[^\W_]")


class ScriptedVoice:
    """Sentences released once the simulation reaches their time; nothing is transcribed."""

    def __init__(self, schedule):
        self.schedule = sorted(schedule)
        self.partial = ""

    def poll(self, sim_time: float):
        due = [text for t, text in self.schedule if t <= sim_time]
        self.schedule = [(t, text) for t, text in self.schedule if t > sim_time]
        return due

    def close(self):
        pass


class LiveVoice:
    """The default microphone streamed to Speechmatics RT for the whole run. A sentence is released when its
    final transcript ends in . ? or ! (Speechmatics punctuates), or after `gap_s` without new words."""

    def __init__(self, language: str = "en", gap_s: float = 1.2, device=None):
        from .voice import MicSource, _stream

        self.mic = MicSource(device=device, stop_on_silence=False)
        self.gap_s, self.partial = gap_s, ""
        self.buf, self.ready, self.last, self.lock = "", [], 0.0, threading.Lock()
        self.error = None
        self.span = None  # (start, end) in the audio of the sentence being heard
        self.utterances = []  # every sentence released: text, its audio span, the wall time it was released
        self.clock = []  # (wall time, simulated time), sampled as the agent polls
        self.mic.__enter__()
        self.t0_wall = time.time()  # the microphone's first sample, to within its ~10 ms buffer
        self.t_closed = None

        def on_final(text, start, end):
            with self.lock:
                self.buf += text
                self.last = time.perf_counter()
                if text.strip():
                    self.span = (self.span[0] if self.span else start, end)
                if re.search(r"[.?!]\s*$", self.buf):
                    self._release()

        def on_segment_partial(text):
            self.partial = " ".join((self.buf + text).split())

        def run():
            try:
                asyncio.run(_stream(self.mic, self.mic.rate, language, on_final_span=on_final,
                                    on_segment_partial=on_segment_partial))
            except Exception as e:  # surfaced by poll(), never swallowed
                self.error = e

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        print(f"[voice] listening on {self.mic.name} ({self.mic.rate} Hz)", flush=True)

    def _release(self):
        text = " ".join(self.buf.split())
        if text and WORD.search(text):
            self.ready.append(text)
            now = time.time()
            start, end = self.span or (None, None)
            self.utterances.append({"text": text, "audio_start": start, "audio_end": end, "wall": round(now, 3)})
            # The computer's clock on screen: a screen recording of the run shows the same moment in the terminal.
            print(f"[voice {time.strftime('%H:%M:%S', time.localtime(now))}.{int(now % 1 * 10)}] heard {text!r} "
                  f"(said {start:.2f}-{end:.2f} s into the recording)" if start is not None else
                  f"[voice] heard {text!r}", flush=True)
        self.buf, self.partial, self.span = "", "", None

    def poll(self, sim_time=None):
        if self.error is not None:
            raise RuntimeError(f"speech stream failed: {self.error}") from self.error
        now = time.time()
        if sim_time is not None and (not self.clock or now - self.clock[-1][0] >= 0.05):
            self.clock.append((round(now, 3), round(float(sim_time), 3)))
        with self.lock:
            if self.buf and time.perf_counter() - self.last > self.gap_s:
                self._release()
            out, self.ready = self.ready, []
        return out

    def close(self):
        self.mic.stop()
        self.thread.join(timeout=10)
        self.mic.__exit__()
        self.t_closed = time.time()

    def save(self, wav_path) -> dict:
        """After close(): what the microphone heard as a WAV, and next to it (.json) the sentences, their spans
        and the clock. The audio's length against the wall time it ran shows whether any audio went missing."""
        import soundfile as sf

        wav_path = Path(wav_path)
        audio = np.frombuffer(b"".join(self.mic.captured), np.int16)
        sf.write(str(wav_path), audio, self.mic.rate, subtype="PCM_16")
        info = {"wav": wav_path.name, "rate": self.mic.rate, "device": self.mic.name, "t0_wall": self.t0_wall,
                "audio_s": round(len(audio) / self.mic.rate, 3),
                "wall_s": round((self.t_closed or time.time()) - self.t0_wall, 3),
                "utterances": self.utterances, "clock": self.clock}
        wav_path.with_suffix(".json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        return info


def sim_time_of(wall: float, clock) -> float:
    """Simulated time at a wall-clock moment, interpolated between the agent's polls (clamped at the ends)."""
    walls, sims = zip(*clock)
    return float(np.interp(wall, walls, sims))


def live_placements(events, live: dict):
    """Where each sentence the robot heard goes in simulated time: for every "heard" event, the saved utterance
    with the same text (in order), its span in the recording, and the simulated times its speech started and
    ended (wall clock of the recording's first sample + the span, through the clock). Sentences the robot ignored
    (its own voice picked up by the microphone) have no heard event and are not placed."""
    todo = [u for u in live["utterances"] if u.get("audio_start") is not None]
    out = []
    for e in events:
        if e["kind"] != "heard":
            continue
        match = next((u for u in todo if u["text"] == e["text"]), None)
        if match is None:
            continue
        todo.remove(match)
        start, end = (sim_time_of(live["t0_wall"] + match[k], live["clock"]) for k in ("audio_start", "audio_end"))
        out.append({"text": e["text"], "heard_t": e["t"], "audio_start": match["audio_start"],
                    "audio_end": match["audio_end"], "sim_start": round(start, 3), "sim_end": round(end, 3)})
    return out

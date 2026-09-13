"""Things said while the robot works. The agent polls every control step; each complete sentence is either
"stop" (handled locally, no model call) or a change to the plan (sent to the VLM planner's amend()).

    voice = LiveVoice()                                      # microphone -> Speechmatics RT, one session per run
    voice = ScriptedVoice([(12.0, "Skip the fork.")])        # the same at fixed simulated times (tests, videos)

Both expose poll(sim_time) -> [sentences], .partial (the words being spoken now, for the demo panel), close().
"""
import asyncio
import re
import threading
import time

STOP = re.compile(r"\b(stop|halt|freeze)\b", re.IGNORECASE)


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
        self.mic.__enter__()

        def on_final(text):
            with self.lock:
                self.buf += text
                self.last = time.perf_counter()
                if re.search(r"[.?!]\s*$", self.buf):
                    self._release()

        def on_segment_partial(text):
            self.partial = " ".join((self.buf + text).split())

        def run():
            try:
                asyncio.run(_stream(self.mic, self.mic.rate, language, on_final=on_final,
                                    on_segment_partial=on_segment_partial))
            except Exception as e:  # surfaced by poll(), never swallowed
                self.error = e

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        print(f"[voice] listening on {self.mic.name} ({self.mic.rate} Hz)", flush=True)

    def _release(self):
        text = " ".join(self.buf.split())
        if text:
            self.ready.append(text)
        self.buf, self.partial = "", ""

    def poll(self, sim_time=None):
        if self.error is not None:
            raise RuntimeError(f"speech stream failed: {self.error}") from self.error
        with self.lock:
            if self.buf and time.perf_counter() - self.last > self.gap_s:
                self._release()
            out, self.ready = self.ready, []
        return out

    def close(self):
        self.mic.stop()
        self.thread.join(timeout=10)
        self.mic.__exit__()

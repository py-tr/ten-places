"""Spoken commands via Speechmatics real-time speech-to-text.

Audio is streamed to Speechmatics RT as it is captured; partial transcripts arrive while the person speaks
and the final transcript goes straight to the planner.

    text, partials, ms = transcribe_mic(on_partial=print)            # live microphone, stops on silence
    text, partials, ms = transcribe_file("out/audio/command.wav")     # recorded command (wav/flac/ogg)

ms is the time from the end of the audio (end of speech, for the mic) to the final transcript. Commands
spoken while the robot works go through tenplaces.listen.LiveVoice, which keeps one session open.
The API key comes from the SPEECHMATICS_API_KEY environment variable (never from code or the repo).
"""
import asyncio
import io
import os
import queue
import time
from pathlib import Path

import numpy as np


def _require_key() -> str:
    key = os.environ.get("SPEECHMATICS_API_KEY")
    if not key and os.name == "nt":  # set with `setx` after this process started: read the user environment
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as env:
                key = winreg.QueryValueEx(env, "SPEECHMATICS_API_KEY")[0]
        except OSError:
            key = None
    if not key:
        raise RuntimeError("Set SPEECHMATICS_API_KEY in the environment to use voice commands.")
    return key


async def _stream(source, rate: int, language: str, on_partial=None, end_time=None, on_final=None,
                  on_segment_partial=None, on_final_span=None):
    """Stream raw mono int16 PCM from source.read() to Speechmatics RT.

    on_partial(text so far, finals + current partial), on_final(new final segment), on_segment_partial(current
    partial segment only); on_final_span(segment, start_s, end_s): the same with where it lies in the audio sent;
    end_time() -> when the audio ended, for the latency figure.
    """
    from speechmatics.rt import (AsyncClient, AudioEncoding, AudioFormat, ServerMessageType, TranscriptionConfig,
                                 TranscriptResult)

    finals, partials, t_final = [], [], {}
    async with AsyncClient(api_key=_require_key()) as client:
        @client.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT)
        def _partial(message):
            text = TranscriptResult.from_message(message).metadata.transcript
            if on_segment_partial is not None:
                on_segment_partial(text)
            if text:
                partials.append(text)
                if on_partial is not None:
                    on_partial(" ".join("".join(finals + [text]).split()))

        @client.on(ServerMessageType.ADD_TRANSCRIPT)
        def _final(message):
            meta = TranscriptResult.from_message(message).metadata
            text = meta.transcript
            if text:
                finals.append(text)
                t_final["t"] = time.perf_counter()
                if on_final is not None:
                    on_final(text)
                if on_final_span is not None:
                    on_final_span(text, meta.start_time, meta.end_time)

        config = TranscriptionConfig(language=language, enable_partials=True, max_delay=1.0)
        fmt = AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=rate)
        await client.transcribe(source, transcription_config=config, audio_format=fmt)
    t_end = end_time() if end_time else None
    ms = 1000 * (t_final["t"] - t_end) if t_end and "t" in t_final else float("nan")
    return " ".join("".join(finals).split()), partials, ms  # finals carry their own spacing


def transcribe_file(path: str | Path, language: str = "en", on_partial=None):
    """A recorded command through Speechmatics RT. Returns (final_text, partials, ms_after_audio_end)."""
    import soundfile as sf

    samples, rate = sf.read(str(path), dtype="int16", always_2d=True)
    pcm = io.BytesIO(samples.mean(axis=1).astype("int16").tobytes())
    sent = {}

    class Source:  # marks when the last chunk has been handed to the client
        def read(self, n):
            chunk = pcm.read(n)
            if not chunk:
                sent.setdefault("t", time.perf_counter())
            return chunk

    return asyncio.run(_stream(Source(), rate, language, on_partial, lambda: sent.get("t")))


class MicSource:
    """Default input device as a blocking read() source. With stop_on_silence it ends after speech followed by
    `silence_s` of quiet, or after `max_s` (noise floor from the first 0.3 s); otherwise it runs until stop()."""

    def __init__(self, max_s: float = 10.0, silence_s: float = 0.8, device=None, stop_on_silence: bool = True):
        import sounddevice as sd

        info = sd.query_devices(device, kind="input")
        self.rate = int(info["default_samplerate"])
        self.name = info["name"]
        self.max_s, self.silence_s, self.stop_on_silence = max_s, silence_s, stop_on_silence
        self.q, self.buf = queue.Queue(), b""
        self.t_speech_end, self.done = None, False
        self.floor, self.heard, self.quiet_since, self.samples, self.peak = [], False, None, 0, 0.0
        # An audio interface shows its inputs as one multi-channel device ("Analogue 1 + 2"); the microphone can be
        # on any of them, so capture two and keep the louder.
        self.channels = max(1, min(2, int(info["max_input_channels"])))
        self.captured = []  # everything the microphone delivered, mono int16 (transcribe_mic(save_to=...))
        self.stream = sd.RawInputStream(samplerate=self.rate, channels=self.channels, dtype="int16", device=device,
                                        callback=lambda data, frames, t, status: self._capture(bytes(data)))

    def _capture(self, data: bytes):
        mono = self._mono(data)
        self.captured.append(mono)
        self.q.put(mono)

    def _mono(self, data: bytes) -> bytes:
        if self.channels == 1:
            return data
        x = np.frombuffer(data, dtype=np.int16).reshape(-1, self.channels)
        return np.ascontiguousarray(x[:, int(np.argmax(np.abs(x.astype(np.int32)).sum(axis=0)))]).tobytes()

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()

    def stop(self):
        self.done = True

    def _update_vad(self, chunk: bytes):
        if not self.stop_on_silence:
            return
        x = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
        self.samples += len(x)
        t = self.samples / self.rate
        if t < 0.3:
            self.floor.append(rms)
            return
        self.peak = max(self.peak, rms)
        threshold = max(4 * (np.median(self.floor) if self.floor else 0.0), 300.0)
        self.threshold = threshold
        if rms > threshold:
            self.heard, self.quiet_since = True, None
        elif self.heard:
            self.quiet_since = self.quiet_since or time.perf_counter()
            if time.perf_counter() - self.quiet_since >= self.silence_s:
                self.done, self.t_speech_end = True, self.quiet_since
        if t >= self.max_s:
            self.done, self.t_speech_end = True, self.t_speech_end or time.perf_counter()

    def read(self, n: int) -> bytes:
        while len(self.buf) < n and not self.done:
            try:
                chunk = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            self._update_vad(chunk)
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out  # b"" once done and drained: the client then ends the stream


def transcribe_mic(language: str = "en", on_partial=None, max_s: float = 10.0, silence_s: float = 0.8, device=None,
                   save_to=None):
    """Speak a command into the default microphone. Returns (final_text, partials, ms_after_speech_end).
    save_to: also write what the microphone heard as a WAV (the demo video's voice-over uses it)."""
    with MicSource(max_s, silence_s, device) as mic:
        print(f"listening on {mic.name} ({mic.rate} Hz, {mic.channels} ch) — speak, then pause", flush=True)
        result = asyncio.run(_stream(mic, mic.rate, language, on_partial, lambda: mic.t_speech_end))
        if save_to is not None:
            import soundfile as sf

            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(save_to), np.frombuffer(b"".join(mic.captured), np.int16), mic.rate, subtype="PCM_16")
        if not mic.heard:
            print(f"heard nothing above the noise floor in {mic.max_s:.0f} s (loudest {mic.peak:.0f} RMS, speech needs "
                  f"> {getattr(mic, 'threshold', 300):.0f}): check the input, its gain and Windows' default microphone",
                  flush=True)
        return result

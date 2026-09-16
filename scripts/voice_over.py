"""Give a demo video the person's side of the conversation: the command, heard before the robot moves, and
anything said while it works.

Spoken commands use the microphone recording saved during the run (<video>.command.wav, from
run_agent.py --mic --video): the audio Speechmatics transcribed. What was said while the robot worked
(run_agent.py --listen --video) comes from the whole-run recording (<video>.live.wav/.json): each sentence the
robot heard is cut at Speechmatics' own timings and placed at the simulated moment it was said, from the
computer's clock logged against simulated time (tenplaces.listen.live_placements) — not by eye. Typed commands and
scripted interjections (run_agent.py --say) are read by a text-to-speech voice different from the robot's, and the
intro labels them as typed. The intro holds the first frame with only the command on the panel while it is said;
the original video and the robot's own track (<video>.wav) follow unchanged.

    python scripts/voice_over.py                  # out/video/demo/seed<N>.mp4 -> out/video/demo/voiced/seed<N>.mp4
"""
import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.demo_video import MAIN_HW, _font, _wrap  # noqa: E402
from tenplaces.speak import RATE, mux, synthesize  # noqa: E402

FPS = 25


def resample(x: np.ndarray, rate: int) -> np.ndarray:
    if rate == RATE:
        return x
    n = int(len(x) * RATE / rate)
    return np.interp(np.arange(n) * rate / RATE, np.arange(len(x)), x).astype(np.float32)


def load_wav(path) -> np.ndarray:
    import soundfile as sf

    x, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(x.mean(axis=1), rate)


def tts(text: str, voice: str) -> np.ndarray:
    samples, rate = synthesize(text, voice)
    return resample(samples.astype(np.float32) / 32768, rate)


def trim(x: np.ndarray, pad_s: float = 0.3) -> np.ndarray:
    """Only the speech: the microphone recording starts before the person speaks and runs on after."""
    frame = int(0.02 * RATE)
    n = len(x) // frame
    if n == 0:
        return x
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    loud = np.flatnonzero(rms > max(4 * np.median(rms), 0.005))
    if not len(loud):
        return x
    a, b = loud[0] * frame - int(pad_s * RATE), (loud[-1] + 1) * frame + int(pad_s * RATE)
    return x[max(0, a): min(len(x), b)]


def fade(x: np.ndarray, s: float = 0.08) -> np.ndarray:
    """Ramp in and out: a quiet microphone recording is raised ~10x, and its room noise must not start and stop
    with a hard edge."""
    n = min(int(s * RATE), len(x) // 2)
    if n == 0:
        return x
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x = x.copy()
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def level(x: np.ndarray, peak: float = 0.8) -> np.ndarray:
    m = float(np.abs(x).max()) if len(x) else 0.0
    return x * (peak / m) if m > 0 else x


def intro_frame(first: np.ndarray, label: str, text: str, notes=(), listening_for: str | None = None) -> np.ndarray:
    """The first frame with the panel cleared down to the command: the plan appears when the video starts.
    notes: smaller lines under it (how the audio was placed, which take)."""
    img = Image.fromarray(first)
    d = ImageDraw.Draw(img)
    x0 = MAIN_HW[1]
    d.rectangle([x0, 0, img.width, img.height], fill=(18, 20, 24))
    x, y = x0 + 16, 14
    d.text((x, y), "Ten Places · two SO-101 arms · OpenVINO on Intel CPU", font=_font(16), fill=(170, 170, 180))
    y += 30
    for line in _wrap(label, 46):
        d.text((x, y), line, font=_font(16), fill=(140, 200, 255))
        y += 22
    lines = _wrap(f'"{text}"', 38)
    for i, line in enumerate(_wrap(listening_for, 38) if listening_for else lines):
        d.text((x, y + i * 26), line, font=_font(20), fill=(120, 124, 132) if listening_for else (255, 255, 255))
    y += 26 * len(lines)  # the command's space is reserved either way, so nothing below it moves when it appears
    y += 12
    for note in notes:
        for line in _wrap(note, 52):
            d.text((x, y), line, font=_font(14), fill=(170, 170, 180))
            y += 19
    return np.asarray(img)


def voice_over(stem: Path, entry: dict, out: Path, voice: str, lead_s: float = 0.5, tail_s: float = 0.8):
    run = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    mic = stem.with_suffix(".command.wav")
    if entry["mode"] == "spoken":
        if not mic.exists():
            return f"skipped: spoken, but no microphone recording ({mic.name}); re-record with run_agent.py --mic --video"
        cmd, label, shown = trim(load_wav(mic)), "Command — spoken (microphone → Speechmatics, live)", run["command"]
    else:
        cmd, label, shown = tts(entry["command"], voice), "Command — typed (read aloud by a text-to-speech voice)", entry["command"]
    cmd = fade(level(cmd))
    reader = iio.get_reader(str(stem.with_suffix(".mp4")))
    first = reader.get_data(0)
    # A spoken command belongs on the panel when Speechmatics finalised it, not before: the microphone clip plays
    # under an empty panel, and the run's own measurement (ms after the speech ended) says when the words appear.
    spoken_ms = (run.get("voice") or {}).get("ms_final_after_speech_end") if entry["mode"] == "spoken" else None
    reveal_s = (lead_s + len(cmd) / RATE - 0.3 + spoken_ms / 1000) if spoken_ms is not None else None
    n_intro = int(np.ceil(((reveal_s + tail_s) if reveal_s is not None else
                           (lead_s + len(cmd) / RATE + tail_s)) * FPS))
    robot = load_wav(stem.with_suffix(".wav"))
    # Video time 0 is the simulated time of the video's first frame (older runs: the first control step).
    t0 = run.get("video_t_start")
    if t0 is None:
        t0 = min((e["t"] for e in run["events"] if e["kind"] == "skill_start"), default=0.5)
    notes, live_json = [], stem.with_suffix(".live.json")
    if spoken_ms is not None:
        notes.append(f"The command appears when the transcript was final: {spoken_ms} ms after the speech ended")
    if live_json.exists():  # said while the robot works, live: the person's own voice where it was said
        from tenplaces.listen import live_placements

        live = json.loads(live_json.read_text(encoding="utf-8"))
        rec = load_wav(live_json.with_name(live["wav"]))
        for p in live_placements(run["events"], live):
            clip = fade(level(trim(rec[max(0, int((p["audio_start"] - 0.3) * RATE)): int((p["audio_end"] + 0.4) * RATE)])))
            # the clip's last speech (0.3 s before its end, see trim) lands at the simulated moment it ended
            i = max(0, int((p["sim_end"] - t0 + 0.3) * RATE) - len(clip))
            robot = np.concatenate([robot, np.zeros(max(0, i + len(clip) - len(robot)), np.float32)])
            robot[i: i + len(clip)] += clip
            print(f"  live: {p['text']!r} said at sim {p['sim_start']:.2f}-{p['sim_end']:.2f} s, heard at "
                  f"{p['heard_t']:.2f} s -> video {i / RATE:.2f} s", flush=True)
        notes.append("Said while working: the microphone, placed at the simulated moment it was said "
                     "(from the computer's clock logged against simulated time)")
    else:  # scripted (--say): in the TTS voice, ending just before the robot hears it
        for e in run["events"]:
            if e["kind"] == "heard" and entry["mode"] != "spoken":
                clip = level(tts(e["text"], voice))
                i = max(0, int((e["t"] - t0 - 0.2) * RATE) - len(clip))
                seg = clip[: max(0, len(robot) - i)]
                robot[i: i + len(seg)] += seg
    if entry.get("take"):
        notes.append(f"Take {entry['take']}")
    audio = np.concatenate([np.zeros(int(lead_s * RATE), np.float32), cmd])
    audio = np.concatenate([audio, np.zeros(max(0, int(n_intro / FPS * RATE) - len(audio)), np.float32), robot])
    out.mkdir(parents=True, exist_ok=True)
    video = out / f"{stem.name}.mp4"
    writer = iio.get_writer(str(video), fps=FPS, macro_block_size=8)
    frame = intro_frame(first, label, shown, notes)
    n_wait = min(n_intro, int(round(reveal_s * FPS))) if reveal_s is not None else 0
    # The transcript as it actually grew: each partial placed at the moment it arrived, counted back from the
    # final. Only a handful of distinct frames exist, so they are rendered once and reused.
    log = sorted((run.get("voice") or {}).get("partials_log") or [], key=lambda p: -p["before_final_s"])
    cache = {}
    for i in range(n_wait):
        left = reveal_s - i / FPS  # how long still to go before the final transcript
        arrived = [p["text"] for p in log if p["before_final_s"] >= left]
        key = arrived[-1] if arrived else ""
        if key not in cache:
            cache[key] = intro_frame(first, label, shown, notes, listening_for=key or "listening…")
        writer.append_data(cache[key])
    for _ in range(n_intro - n_wait):
        writer.append_data(frame)
    for f in reader:
        writer.append_data(f)
    writer.close()
    reader.close()
    import soundfile as sf

    wav = out / f"{stem.name}.wav"
    sf.write(str(wav), np.clip(audio, -1, 1), RATE, subtype="PCM_16")
    mux(video, wav, video)
    return f"{video} (intro {n_intro / FPS:.1f} s, {'microphone' if entry['mode'] == 'spoken' else voice})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo_seeds.json")
    ap.add_argument("--dir", default="out/video/demo")
    ap.add_argument("--out", default="out/video/demo/voiced")
    ap.add_argument("--voice", default="jack", help="the person's TTS voice; the robot speaks as 'sarah'")
    ap.add_argument("--seeds", type=int, nargs="*")
    args = ap.parse_args()
    for entry in json.loads(Path(args.config).read_text(encoding="utf-8"))["seeds"]:
        if args.seeds and entry["seed"] not in args.seeds:
            continue
        stem = Path(args.dir) / f"seed{entry['seed']}"
        if not stem.with_suffix(".json").exists():
            print(f"seed {entry['seed']}: no run")
            continue
        print(f"seed {entry['seed']}: {voice_over(stem, entry, Path(args.out), args.voice)}", flush=True)


if __name__ == "__main__":
    main()

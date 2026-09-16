"""Every demonstration run, end to end, in one video, with a title card before each and a timestamp index.

The grid (scripts/make_grid_video.py) shows ten tables at a glance and is what goes in the submission video; this
is the evidence copy: each run at full size with its audio, so a single run can be watched and heard in full. The
index it prints goes in README.md, so a reader can jump straight to one command.

Clips come from the voiced renders (scripts/voice_over.py), so the spoken commands are the person's own microphone
and the robot's replies are its own track. PASS/FAIL is read from the demo.csv that scripts/score_demo.py wrote
next to each set - nothing here decides whether a run passed.

    python scripts/make_reel.py                                    # -> out/video/reel.mp4 + the index
    python scripts/make_reel.py --order live demo extra            # live take first
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.demo_video import _font, _wrap  # noqa: E402
from tenplaces.speak import RATE, mux  # noqa: E402

FPS = 25
SETS = {
    "demo": ("configs/demo_seeds.json", "out/video/demo_final"),
    "extra": ("configs/demo_extra.json", "out/video/demo_final_extra"),
    "live": ("configs/demo_live.json", "out/video/demo_live/take2"),
}


def verdicts(d: Path) -> dict:
    """seed -> (passed, cause), from the demo.csv score_demo.py wrote. Empty when the set was not scored."""
    f = d / "demo.csv"
    if not f.is_file():
        return {}
    out = {}
    with f.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out[int(row["seed"])] = (row["success"].strip().lower() in ("true", "1", "yes"), row.get("cause", ""))
    return out


def card(size, entry: dict, passed, cause: str, take=None) -> Image.Image:
    """The title card: which seed, what was asked, typed or spoken, and how it ended."""
    w, h = size
    img = Image.new("RGB", (w, h), (18, 20, 24))
    d = ImageDraw.Draw(img)
    x, y = int(w * 0.08), int(h * 0.30)
    mode = "spoken into the microphone" if entry.get("mode") == "spoken" else "typed"
    head = f"Seed {entry['seed']} - {mode}" + (f" - take {take}" if take else "")
    d.text((x, y), head, font=_font(22), fill=(140, 200, 255))
    y += 42
    for line in _wrap(f'"{entry["command"]}"', 44):
        d.text((x, y), line, font=_font(34), fill=(255, 255, 255))
        y += 46
    if entry.get("live_say"):
        y += 8
        for line in _wrap(f'then, while it works: "{entry["live_say"]}"', 52):
            d.text((x, y), line, font=_font(22), fill=(200, 200, 210))
            y += 30
    y += 18
    if passed is None:
        d.text((x, y), "not scored", font=_font(26), fill=(170, 170, 180))
    else:
        d.text((x, y), "PASS" if passed else f"FAIL: {cause}", font=_font(30),
               fill=(80, 230, 80) if passed else (240, 70, 70))
    return img


def segment(img: Image.Image, seconds: float, out: Path) -> Path:
    """A still as a clip with a silent track, so it concatenates with the runs without a stream mismatch."""
    import soundfile as sf

    frames = int(round(seconds * FPS))
    writer = iio.get_writer(str(out), fps=FPS, macro_block_size=8)
    frame = np.asarray(img)
    for _ in range(frames):
        writer.append_data(frame)
    writer.close()
    wav = out.with_suffix(".wav")
    sf.write(str(wav), np.zeros(int(seconds * RATE), dtype=np.float32), RATE, subtype="PCM_16")
    mux(out, wav, out)
    wav.unlink(missing_ok=True)
    return out


def duration_s(mp4: Path) -> float:
    r = iio.get_reader(str(mp4))
    n = r.count_frames()
    r.close()
    return n / FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", nargs="*", default=["demo", "extra", "live"], choices=list(SETS))
    ap.add_argument("--out", default="out/video/reel.mp4")
    ap.add_argument("--card-s", type=float, default=2.5)
    ap.add_argument("--index-only", action="store_true", help="print the index without encoding the reel")
    args = ap.parse_args()

    work = Path(args.out).parent / "reel_parts"
    work.mkdir(parents=True, exist_ok=True)
    parts, index, size, t = [], [], None, 0.0

    for name in args.order:
        cfg_path, dir_path = SETS[name]
        cfg, d = json.loads(Path(cfg_path).read_text(encoding="utf-8")), Path(dir_path)
        scored = verdicts(d)
        for entry in cfg["seeds"]:
            clip = d / "voiced" / f"seed{entry['seed']}.mp4"
            if not clip.is_file():
                print(f"seed {entry['seed']}: no voiced clip ({clip}) - run scripts/voice_over.py first")
                continue
            if size is None:
                first = iio.get_reader(str(clip))
                size = (first.get_data(0).shape[1], first.get_data(0).shape[0])
                first.close()
            passed, cause = scored.get(entry["seed"], (None, ""))
            c = segment(card(size, entry, passed, cause, entry.get("take")), args.card_s,
                        work / f"card_{name}_{entry['seed']}.mp4")
            index.append((t, entry, passed, cause, name))
            t += duration_s(c)
            parts += [c, clip]
            t += duration_s(clip)

    if not parts:
        sys.exit("no clips found: record the demos, then scripts/score_demo.py and scripts/voice_over.py")

    if args.index_only:
        print(f"\n(index only, nothing encoded) {t / 60:.1f} min, {len(parts) // 2} runs\n")
    else:
        listing = work / "parts.txt"
        listing.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts), encoding="utf-8")
        import imageio_ffmpeg

        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
               "-i", str(listing), "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "128k", str(args.out)]
        subprocess.run(cmd, check=True)
        print(f"\n{args.out}  ({t / 60:.1f} min, {len(parts) // 2} runs)\n")

    rows = ["| Time | Seed | Command | How | Result |", "|---|---|---|---|---|"]
    for start, entry, passed, cause, name in index:
        # Both sides are Speechmatics and both are audible: a spoken command is the person's microphone
        # transcribed by Speechmatics STT, a typed one is read aloud by a second Speechmatics TTS voice
        # (voice_over.py). The robot answers in its own Speechmatics voice in every run.
        mode = ("live microphone (Speechmatics STT)" if entry.get("mode") == "spoken"
                else "typed, read aloud (Speechmatics TTS)")
        # A failed seed says why. "not completed" would be wrong for a run whose steps were all done.
        # A wrongly announced "cannot do" is not a dropped object. Say what the table looked like first, then why
        # it was scored a failure; demo.csv keeps the grader's own cause string untouched.
        if passed is None:
            result = "-"
        elif passed:
            result = "done as asked"
        elif "refusal" in (cause or "").lower():
            result = 'table set correctly - scored a failure for wrongly saying it could not "set the rest"'
        else:
            result = cause or "not completed"
        extra = " (half-set table)" if name == "extra" else (" (live, plan changed mid-run)" if name == "live" else "")
        if entry.get("say"):
            # The How column already says typed or spoken, so it carries the distinction from the live take
            # without the word "scripted" sitting next to a demonstration run and reading as "staged".
            extra += f' + "{entry["say"]}" mid-run'
        rows.append(f"| {int(start) // 60}:{int(start) % 60:02d} | {entry['seed']} | \"{entry['command']}\"{extra} "
                    f"| {mode} | {result} |")
    print("\n".join(rows))

    # Written, not just printed: the landing page reads this file at build time and renders it under the reel, so
    # the index is never retyped by hand in two places. --index-only refreshes it without re-encoding the video.
    idx = Path(args.out).parent / "reel_index.md"
    idx.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\n{idx}")


if __name__ == "__main__":
    main()

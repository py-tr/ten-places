"""Tile evaluation videos into one grid with seed labels and PASS/FAIL stamps (Intel deliverable 4).

    python scripts/make_grid_video.py --dir out/eval/act_table_v1 --label best_torch_exec10 --out out/video/grid.mp4

Reads <label>_seed<N>.mp4 and <label>.csv from --dir; the stamp comes from the CSV's success column and
appears over the last second of each tile, with the CSV's cause column (if any) above a FAIL. Under each seed label,
that table's friction, light and colour (scene_table.sample, with the stress and shape in the dir's provenance.json).
Shorter clips hold their last frame.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.scene_table import sample  # noqa: E402


def stamp(frame, text, colour, big=False, line=0):
    """big: bottom left, `line` rows up from the bottom; otherwise top left."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28 if big else 16)
    except OSError:
        font = ImageFont.load_default()
    x, y = (10, img.height - 40 - 38 * line) if big else (6, 4)
    box = draw.textbbox((x, y), text, font=font)
    draw.rectangle([box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2], fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=colour)
    return np.asarray(img)


def chips(frame, p):
    """Under the seed label: the table's friction and light, and a swatch of its colour — the variation made visible."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    x, y = 6, 26
    box = draw.textbbox((x, y), f"friction {p.friction:.2f}  light {p.light_diffuse:.2f}", font=font)
    draw.rectangle([box[0] - 4, box[1] - 2, box[2] + 26, box[3] + 2], fill=(0, 0, 0))
    draw.text((x, y), f"friction {p.friction:.2f}  light {p.light_diffuse:.2f}", font=font, fill=(220, 220, 220))
    rgb = tuple(int(255 * c) for c in p.table_rgb)
    draw.rectangle([box[2] + 6, box[1] - 1, box[2] + 22, box[3] + 1], fill=rgb, outline=(128, 128, 128))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--speed", type=int, default=2, help="keep every Nth frame")
    ap.add_argument("--caption", default="held-out seeds passed", help="after the count, e.g. '9/10 <caption>'")
    ap.add_argument("--no-chips", action="store_true", help="leave out the friction/light/colour line under the seed")
    args = ap.parse_args()
    d = Path(args.dir)
    prov = json.loads((d / "provenance.json").read_text()) if (d / "provenance.json").is_file() else {}
    table = lambda s: sample(s, stress=prov.get("stress", 1.0), shape=prov.get("shape", 0.0))  # noqa: E731
    outcome, cause = {}, {}
    with open(d / f"{args.label}.csv") as f:
        for row in csv.DictReader(f):
            outcome[int(row["seed"])] = row["success"] == "True"
            cause[int(row["seed"])] = row.get("cause") or ""
    clips = sorted(d.glob(f"{args.label}_seed*.mp4"), key=lambda p: int(re.search(r"seed(\d+)", p.name).group(1)))
    if not clips:
        raise SystemExit(f"no {args.label}_seed*.mp4 in {d}")
    videos = [iio.mimread(p, memtest=False)[::args.speed] for p in clips]
    seeds = [int(re.search(r"seed(\d+)", p.name).group(1)) for p in clips]
    params = {s: table(s) for s in seeds}
    n = max(len(v) for v in videos)
    h, w = videos[0][0].shape[:2]
    rows = -(-len(videos) // args.cols)
    writer = iio.get_writer(args.out, fps=args.fps, macro_block_size=8)
    passed = sum(outcome.get(s, False) for s in seeds)
    for t in range(n):
        grid = np.zeros((rows * h, args.cols * w, 3), dtype=np.uint8)
        for i, (v, seed) in enumerate(zip(videos, seeds)):
            frame = v[min(t, len(v) - 1)]
            frame = stamp(frame, f"seed {seed}", (255, 255, 255))
            if not args.no_chips:
                frame = chips(frame, params[seed])
            if t >= n - args.fps:  # last second: outcome
                ok = outcome.get(seed, False)
                # A run counts as failed on either of the scorer's two conditions: a step the command asked for
                # was not physically done, or the robot refused something it was not asked about. The tally is the
                # same either way; the stamp says which, so a refusal is not read as a dropped object.
                refusal_only = not ok and cause.get(seed, "").strip() == "wrong refusal"
                word = "PASS" if ok else ("REFUSED" if refusal_only else "FAIL")
                colour = (80, 230, 80) if ok else ((235, 155, 60) if refusal_only else (240, 70, 70))
                frame = stamp(frame, word, colour, big=True)
                if not ok and cause.get(seed):
                    note = "table still set" if refusal_only else cause[seed]
                    frame = stamp(frame, note, (240, 210, 180) if refusal_only else (240, 200, 200), big=True, line=1)
            r, c = divmod(i, args.cols)
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = frame
        if t >= n - args.fps:
            grid = stamp(grid, f"{passed}/{len(seeds)} {args.caption}", (255, 255, 0), big=True)
        writer.append_data(grid)
    writer.close()
    print(args.out, f"{passed}/{len(seeds)} passed")


if __name__ == "__main__":
    main()

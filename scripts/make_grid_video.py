"""Tile evaluation videos into one grid with seed labels and PASS/FAIL stamps (Intel deliverable 4).

    python scripts/make_grid_video.py --dir out/eval/act_table_v1 --label best_torch_exec10 --out out/video/grid.mp4

Reads <label>_seed<N>.mp4 and <label>.csv from --dir; the stamp comes from the CSV's success column and
appears over the last second of each tile. Shorter clips hold their last frame.
"""
import argparse
import csv
import re
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def stamp(frame, text, colour, big=False):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28 if big else 16)
    except OSError:
        font = ImageFont.load_default()
    x, y = (10, img.height - 40) if big else (6, 4)
    box = draw.textbbox((x, y), text, font=font)
    draw.rectangle([box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2], fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=colour)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--speed", type=int, default=2, help="keep every Nth frame")
    args = ap.parse_args()
    d = Path(args.dir)
    outcome = {}
    with open(d / f"{args.label}.csv") as f:
        for row in csv.DictReader(f):
            outcome[int(row["seed"])] = row["success"] == "True"
    clips = sorted(d.glob(f"{args.label}_seed*.mp4"), key=lambda p: int(re.search(r"seed(\d+)", p.name).group(1)))
    if not clips:
        raise SystemExit(f"no {args.label}_seed*.mp4 in {d}")
    videos = [iio.mimread(p, memtest=False)[::args.speed] for p in clips]
    seeds = [int(re.search(r"seed(\d+)", p.name).group(1)) for p in clips]
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
            if t >= n - args.fps:  # last second: outcome
                ok = outcome.get(seed, False)
                frame = stamp(frame, "PASS" if ok else "FAIL", (80, 230, 80) if ok else (240, 70, 70), big=True)
            r, c = divmod(i, args.cols)
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = frame
        if t >= n - args.fps:
            grid = stamp(grid, f"{passed}/{len(seeds)} held-out seeds passed", (255, 255, 0), big=True)
        writer.append_data(grid)
    writer.close()
    print(args.out, f"{passed}/{len(seeds)} passed")


if __name__ == "__main__":
    main()

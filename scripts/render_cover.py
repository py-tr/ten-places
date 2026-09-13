"""Cover image: the scripted controller sets the full table, then the finished scene is rendered in high
resolution from a few free-camera angles (pick one for a cover image).

    python scripts/render_cover.py --seed 1003 --out out/cover
"""
import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.env_table import TableEpisode  # noqa: E402
from tenplaces.grader_table import grade_table  # noqa: E402
from tenplaces.oracle import table  # noqa: E402

ANGLES = [(90, -28, 1.05), (125, -32, 1.0), (55, -32, 1.0), (90, -55, 1.1)]  # azimuth, elevation, distance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1003)
    ap.add_argument("--out", default="out/cover")
    ap.add_argument("--size", type=int, nargs=2, default=[1920, 1080], metavar=("W", "H"))
    args = ap.parse_args()
    ep = TableEpisode(args.seed, render=False)
    table.run(ep.ctl, ep.params)
    print("graded:", {k: v for k, v in grade_table(ep.m, ep.d, ep.params).items() if isinstance(v, bool)})
    w, h = args.size
    ep.m.vis.global_.offwidth, ep.m.vis.global_.offheight = max(w, ep.m.vis.global_.offwidth), max(h, ep.m.vis.global_.offheight)
    r = mujoco.Renderer(ep.m, h, w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.05, 0.0, 0.05]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for i, (az, el, dist) in enumerate(ANGLES):
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        r.update_scene(ep.d, camera=cam)
        img = r.render()
        iio.imwrite(out / f"cover_seed{args.seed}_{i}.png", img)
        thumbs.append(img[::4, ::4])
    iio.imwrite(out / "contact_sheet.png", np.concatenate([np.concatenate(thumbs[:2], 1), np.concatenate(thumbs[2:], 1)], 0))
    r.close()
    print(out / "contact_sheet.png")


if __name__ == "__main__":
    main()

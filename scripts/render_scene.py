"""Render the scene from every camera for a few seeds: a quick visual check of layout and randomisation."""
import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces import randomize, scene  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rows = []
    for seed in args.seeds:
        _, m, d = scene.compile_scene(randomize.sample(seed))
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, 240, 320)
        tiles = []
        for cam in ("top", "front", "a_wrist", "b_wrist"):
            r.update_scene(d, camera=cam)
            tiles.append(r.render())
        rows.append(np.concatenate(tiles, axis=1))
        r.close()
    path = OUT / "scene_check.png"
    iio.imwrite(path, np.concatenate(rows, axis=0))
    print(path)


if __name__ == "__main__":
    main()

"""Labelled images for the task-state classifier, from scripted runs of random verified plans.

    python scripts/record_state_data.py --runs 150 --start 4000 --out data/state_v1
    python scripts/record_state_data.py --runs 400 --start 4000 --out data/state_v3 --short-frac 0.25 --drawer-enough 0.074

Each run executes a random subset of skills (verified: prerequisites added, canonical order) and samples
the top camera every SAMPLE_S seconds, labelled with the grader's simulator truth for all five skills.
Privileged labels are fine for training; at run time the classifier only sees the camera. Covers what
the full-sequence demos lack: subsets (e.g. the cup placed with no cutlery out) and idle time after a
skill finishes.

--drawer-enough relabels the drawer as done only once it is open far enough for the cutlery, not at the grader's
6 cm: the learned pull sometimes stalls at 6-7 cm, the camera called that "done", and the spoon then failed.
--short-frac runs pull the drawer short (3.5 cm up to the threshold) and idle, so those stalled states are seen
labelled "not done". Every shard also stores the drawer opening per frame (drawer_m), so labels can be redrawn.
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces import scene_table  # noqa: E402
from tenplaces.control import GRIP_OPEN, Bimanual, IKFailure  # noqa: E402
from tenplaces.env_table import IMAGE_HW, SKILLS  # noqa: E402
from tenplaces.grader_table import grade_table  # noqa: E402
from tenplaces.oracle import table  # noqa: E402
from tenplaces.oracle.handoff import HOME  # noqa: E402
from tenplaces.planner import verify  # noqa: E402

SAMPLE_S = 0.2
KEYS = ["drawer_open", "spoon", "plate", "fork", "cup"]
SKILL_NAMES = [s for s, _, _ in SKILLS]


def run(seed: int, rng, short: bool = False, drawer_enough: float | None = None):
    raw = [s for s in SKILL_NAMES if rng.random() < 0.6] or [rng.choice(SKILL_NAMES)]
    p = scene_table.sample(seed)
    if short:  # a stalled pull, then only what needs no cutlery
        p.drawer_open = float(rng.uniform(0.035, drawer_enough or 0.06))
        raw = ["drawer"] + [s for s in ("plate", "cup") if rng.random() < 0.5]
    steps, _ = verify(raw)
    _, m, d = scene_table.compile_scene(p)
    slide = m.joint("drawer_slide").qposadr[0]
    r = mujoco.Renderer(m, *IMAGE_HW)
    every = int(round(SAMPLE_S / m.opt.timestep))
    imgs, labels, drawer_m = [], [], []
    count = [0]

    def sample(m_, d_):
        count[0] += 1
        if count[0] % every == 0:
            r.update_scene(d_, camera="top")
            imgs.append(r.render().transpose(2, 0, 1).copy())
            g = grade_table(m_, d_, p)
            opening = float(-d_.qpos[slide])
            label = [g[k] for k in KEYS]
            if drawer_enough is not None:
                label[0] = opening >= drawer_enough
            labels.append(label)
            drawer_m.append(opening)

    ctl = Bimanual(m, d, on_step=sample)
    for arm in ("a_", "b_"):
        ctl.set_now(arm, HOME, GRIP_OPEN)
    mujoco.mj_forward(m, d)
    try:
        ctl.hold(0.5)
        table.run_plan(ctl, p, steps)
        ctl.hold(1.5)  # idle after the last skill: 'done' must hold while nothing moves
        ok = True
    except IKFailure:
        ok = False
    r.close()
    return np.asarray(imgs, np.uint8), np.asarray(labels, np.float32), np.asarray(drawer_m, np.float32), steps, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=150)
    ap.add_argument("--start", type=int, default=4000)
    ap.add_argument("--out", default="data/state_v1")
    ap.add_argument("--shard", type=int, default=25, help="runs per saved shard")
    ap.add_argument("--short-frac", type=float, default=0.0, help="fraction of runs whose drawer pull stops short")
    ap.add_argument("--drawer-enough", type=float, default=None,
                    help="drawer label threshold in m (default: the grader's DRAWER_OPEN_MIN)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.start)
    buf_x, buf_y, buf_d, t0 = [], [], [], time.time()
    for i in range(args.runs):
        short = bool(rng.random() < args.short_frac)
        x, y, dm, steps, ok = run(args.start + i, rng, short, args.drawer_enough)
        if ok:
            buf_x.append(x)
            buf_y.append(y)
            buf_d.append(dm)
        if (i + 1) % args.shard == 0 or i + 1 == args.runs:
            if buf_x:
                X, Y, D = np.concatenate(buf_x), np.concatenate(buf_y), np.concatenate(buf_d)
                np.savez_compressed(out / f"shard_{i // args.shard:03d}.npz", images=X, labels=Y, drawer_m=D)
                print(f"{i + 1}/{args.runs} runs, shard {len(X)} frames, positives {Y.mean(0).round(3).tolist()} "
                      f"({time.time() - t0:.0f} s)", flush=True)
            buf_x, buf_y, buf_d = [], [], []


if __name__ == "__main__":
    main()

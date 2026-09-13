"""Spike: how reliable is the scripted A->B spoon hand-off across randomised seeds?

    python scripts/spike_handoff.py --seeds 50 --video 3
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces import grader, randomize, scene  # noqa: E402
from tenplaces.control import GRIP_OPEN, Bimanual, IKFailure  # noqa: E402
from tenplaces.oracle import handoff  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402

FRAME_EVERY = 20  # physics steps per video frame (0.04 s -> 25 fps)


def episode(seed: int, video_path: Path | None = None):
    params = randomize.sample(seed)
    _, m, d = scene.compile_scene(params)
    spoon_z = []
    frames = []
    renderer = mujoco.Renderer(m, 360, 640) if video_path else None

    def on_step(m_, d_):
        spoon_z.append(d_.xpos[m_.body("spoon").id][2])
        if renderer is not None and len(spoon_z) % FRAME_EVERY == 0:
            renderer.update_scene(d_, camera="front")
            frames.append(renderer.render())

    ctl = Bimanual(m, d, on_step=on_step)
    for p in ("a_", "b_"):
        ctl.set_now(p, handoff.HOME, GRIP_OPEN)
    mujoco.mj_forward(m, d)
    ctl.hold(0.3)

    phases = []
    try:
        handoff.run(ctl, params.target_xy, phases)
        at = {name: spoon_z[min(step, len(spoon_z) - 1)] for name, step in phases}
        trace = {"a_carry_z": at.get("b_reach", 1.0), "a_release_z": at.get("b_place", 1.0)}
        result = grader.grade_handoff(m, d, params.target_xy, trace)
    except IKFailure as e:
        result = {"success": False, "failure": "ik_unreachable", "detail": str(e)}
    result.update(seed=seed, sim_seconds=round(ctl.steps * m.opt.timestep, 2))
    if renderer is not None:
        iio.mimsave(video_path, frames, fps=25, macro_block_size=8)
        renderer.close()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--start", type=int, default=1000, help="spike seeds are disjoint from eval seeds 0-99")
    ap.add_argument("--video", type=int, default=2, help="record the first N episodes")
    args = ap.parse_args()
    out = OUT / "spike_handoff"
    out.mkdir(parents=True, exist_ok=True)
    results, t0 = [], time.time()
    for i in range(args.seeds):
        seed = args.start + i
        r = episode(seed, out / f"seed{seed}.mp4" if i < args.video else None)
        results.append(r)
        print(f"seed {seed}: {'PASS' if r['success'] else 'FAIL ' + str(r['failure'])}", flush=True)
    n_ok = sum(r["success"] for r in results)
    summary = {"episodes": len(results), "success": n_ok, "rate": n_ok / len(results),
               "failures": Counter(r["failure"] for r in results if not r["success"]),
               "wall_seconds": round(time.time() - t0, 1)}
    (out / "results.json").write_text(json.dumps({"summary": summary, "episodes": results}, indent=1, default=str))
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()

"""Spike: the scripted full dinner-table oracle across randomised seeds, graded per sub-task.

    python scripts/spike_table.py --seeds 20 --video 2
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import imageio.v2 as iio
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces import scene_table  # noqa: E402
from tenplaces.control import GRIP_OPEN, Bimanual, IKFailure  # noqa: E402
from tenplaces.grader_table import grade_table  # noqa: E402
from tenplaces.oracle import table  # noqa: E402
from tenplaces.oracle.handoff import HOME  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402

FRAME_EVERY = 20


def episode(seed: int, video_path: Path | None = None):
    p = scene_table.sample(seed)
    _, m, d = scene_table.compile_scene(p)
    frames, renderer = [], (mujoco.Renderer(m, 360, 640) if video_path else None)
    count = [0]

    def on_step(m_, d_):
        count[0] += 1
        if renderer is not None and count[0] % FRAME_EVERY == 0:
            renderer.update_scene(d_, camera="front")
            frames.append(renderer.render())

    ctl = Bimanual(m, d, on_step=on_step)
    for arm in ("a_", "b_"):
        ctl.set_now(arm, HOME, GRIP_OPEN)
    mujoco.mj_forward(m, d)
    ctl.hold(0.5)
    phases, error = [], None
    try:
        table.run(ctl, p, phases)
    except IKFailure as e:
        error = str(e)
    result = grade_table(m, d, p)
    result.update(seed=seed, error=error, last_phase=phases[-1][0] if phases else None,
                  sim_seconds=round(ctl.steps * m.opt.timestep, 1))
    if renderer is not None:
        iio.mimsave(video_path, frames, fps=25, macro_block_size=8)
        renderer.close()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--start", type=int, default=1000)
    ap.add_argument("--video", type=int, default=2)
    args = ap.parse_args()
    out = OUT / "spike_table"
    out.mkdir(parents=True, exist_ok=True)
    results, t0 = [], time.time()
    for i in range(args.seeds):
        seed = args.start + i
        r = episode(seed, out / f"seed{seed}.mp4" if i < args.video else None)
        results.append(r)
        status = "PASS" if r["success"] else f"FAIL {r['failed']}" + (f" (IK: {r['error'][:70]})" if r["error"] else "")
        print(f"seed {seed}: {r['subtasks_done']}/5 {status}", flush=True)
    per = {s: sum(r[s] for r in results) for s in ("drawer_open", "spoon", "fork", "plate", "cup")}
    summary = {"episodes": len(results), "full_success": sum(r["success"] for r in results), "per_subtask": per,
               "failed_counts": Counter(f for r in results for f in r["failed"]),
               "ik_errors": sum(1 for r in results if r["error"]), "wall_seconds": round(time.time() - t0, 1)}
    (out / "results.json").write_text(json.dumps({"summary": summary, "episodes": results}, indent=1, default=str))
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()

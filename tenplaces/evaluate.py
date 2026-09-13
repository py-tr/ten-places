"""Closed-loop evaluation of any policy over a list of seeds.

A policy is an object with `reset()` and `select_action(obs) -> np.ndarray[12]`, where obs is the
dict produced by `Episode.observation()` plus `obs["task"]`. The grader reads simulator state only.
"""
import csv
import json
import time
from pathlib import Path

import imageio.v2 as iio
import numpy as np

from . import grader
from .env import FPS, TASK, Episode

MAX_SECONDS = 16.0


def run_episode(policy, seed: int, video_path: Path | None = None, max_seconds: float = MAX_SECONDS):
    ep = Episode(seed, render=True)
    policy.reset()
    obs = ep.observation()
    frames, latencies = [], []
    spoon = ep.m.body("spoon").id
    stats = {"z_max": 0.0, "a_lifted": False, "b_lifted": False}
    for _ in range(int(max_seconds * FPS)):
        obs["task"] = TASK
        t = time.perf_counter()
        action = np.asarray(policy.select_action(obs), dtype=np.float64)
        latencies.append(time.perf_counter() - t)
        obs = ep.step(action)
        z = float(ep.d.xpos[spoon][2])
        stats["z_max"] = max(stats["z_max"], z)
        if z > 0.02:
            stats["a_lifted"] |= grader.touching(ep.m, ep.d, "spoon", ("a_pad",))
            stats["b_lifted"] |= grader.touching(ep.m, ep.d, "spoon", ("b_pad",))
        if video_path is not None:
            frames.append(np.concatenate([obs["images"][c] for c in ("top", "a_wrist", "b_wrist")], axis=1))
    result = grader.grade_handoff_episode(ep.m, ep.d, ep.params.target_xy, stats)
    result.update(seed=seed, policy_ms_mean=1000 * float(np.mean(latencies)), policy_ms_p95=1000 * float(np.percentile(latencies, 95)))
    if video_path is not None:
        iio.mimsave(video_path, frames, fps=FPS, macro_block_size=8)
    ep.close()
    return result


def evaluate(policy, seeds, out_dir: Path, videos: int = 0, label: str = "policy"):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, seed in enumerate(seeds):
        r = run_episode(policy, seed, out_dir / f"{label}_seed{seed}.mp4" if i < videos else None)
        rows.append(r)
        print(f"[{label}] seed {seed}: {'PASS' if r['success'] else 'FAIL ' + str(r['failure'])} ({r['policy_ms_mean']:.1f} ms/step)", flush=True)
    with open(out_dir / f"{label}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "success", "failure", "place_err_m", "policy_ms_mean", "policy_ms_p95"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    k = sum(r["success"] for r in rows)
    summary = {"label": label, "episodes": n, "success": k, "rate": k / n, "wilson95": wilson(k, n),
               "policy_ms_mean": float(np.mean([r["policy_ms_mean"] for r in rows]))}
    (out_dir / f"{label}_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a success rate."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(float(centre - half), 3), round(float(centre + half), 3))

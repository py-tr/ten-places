"""Closed-loop evaluation of a skill-conditioned policy on the full dinner-table workflow.

A sequencer runs the skills in order (drawer, spoon, plate, fork, cup). For each skill it sets the skill
one-hot (obs["env_state"]), resets the policy's action queue, and runs a fixed time budget; switching
is time-based, never on simulator state, so the policy gets no privileged hints. Grading uses the
simulator only after the episode, via grade_table.
"""
import csv
import json
import os
import time
from pathlib import Path

import imageio.v2 as iio
import numpy as np

from .env import CAMERAS, FPS
from .env_table import SKILLS, TableEpisode, go_home
from .evaluate import wilson
from .grader_table import grade_table

# Frames per skill, a cap (the camera classifier ends a skill when it is done): the longest oracle segment in
# data/table_v1 plus ~15% (drawer includes one re-grip); spoon and fork raised from 360 after the chained run cut
# the fork off mid-placement (seed 108 video) — tuning seeds 100-109: fork 0 -> 3/10, full tables 0 -> 2/10.
DEFAULT_BUDGETS = {"drawer": 300, "spoon": 450, "plate": 210, "fork": 500, "cup": 230}
# How a skill ends. CAMERA_ENDS: may the camera classifier end it early ("done" + SETTLE frames of the policy to
# finish releasing and retreating), or does its policy run the whole budget? The drawer's "done" label is the
# grader's >= 6 cm, but the cutlery needs >= ~7.4 cm: ended early, the pull stopped at 6-7 cm on 7 of 20 tuning seeds
# and the spoon then failed on all 7. Same tuning seeds 100-119, drawer -> release + home -> spoon: camera end
# spoon 12/20, camera end + 90 settle frames 12/20, whole budget 16/20 -> the drawer runs its budget.
CAMERA_ENDS = {s: s != "drawer" for s in DEFAULT_BUDGETS}
# Experiments only: TENPLACES_CAMERA_ENDS="drawer=1,cup=0" overrides per skill. An env var, because parallel eval
# workers are spawned processes that re-import this module and inherit the environment, not the parent's objects.
for _item in filter(None, os.environ.get("TENPLACES_CAMERA_ENDS", "").split(",")):
    _skill, _value = _item.split("=")
    CAMERA_ENDS[_skill] = _value == "1"
SETTLE = {s: 30 for s in DEFAULT_BUDGETS}
# One more attempt, from home, when the camera still says "not done" at the end of a skill. Tuning seeds 100-149
# (scripts/eval_table_chain.py --retry): plate 3/7, fork 5/24, cup 1/4 retries recovered, none lost; spoon 0/23 (a
# spoon left in a short drawer is not a transient): not retried. Drawer: a whole-budget re-pull recovered 1/21; the
# camera-ended re-pull (RETRY_CAMERA_END) rescued 3 tables of 100 on seeds 100-199, spoon 6/7 after it: retried.
# The agent uses the same table (tenplaces/agent.py).
RETRY = {"drawer": True, "spoon": False, "plate": True, "fork": True, "cup": True}
# A retry or re-run of a budget-ended skill (the drawer) ends at the camera's "done" + settle instead of running its
# whole budget again from a half-open start (the camera-ended drawer re-pull).
RETRY_CAMERA_END = True
# Experiments only: TENPLACES_DRAWER_REPULL=1 turns the re-pull on (RETRY["drawer"] and RETRY_CAMERA_END), for spawned
# evaluation workers, which inherit the environment, not the parent's objects.
if os.environ.get("TENPLACES_DRAWER_REPULL") == "1":
    RETRY["drawer"], RETRY_CAMERA_END = True, True


def ends_on_camera(skill: str, attempt: int = 1, rerun: bool = False) -> bool:
    """May the camera end this run of the skill early? Camera-ended skills always; a budget-ended one only on a retry
    (attempt > 1) or a re-run of a re-queued step, and only with RETRY_CAMERA_END."""
    return CAMERA_ENDS[skill] or (RETRY_CAMERA_END and (attempt > 1 or rerun))


def run_episode(policy, seed: int, budgets=None, video_path: Path | None = None, checker=None,
                check_every: int = 10, min_frames: int = 40, settle_frames: int = 30, home_frames: int = 0,
                retry: bool = True):
    """checker(skill, top_image) -> done: when given, a skill ends as soon as the camera says it is done
    (checked every `check_every` frames after `min_frames`); the budget is then only a cap.
    home_frames > 0: between skills the arms return to the home pose (env_table.go_home), where every
    demonstration starts a skill. retry (needs a checker): a skill in RETRY that ends with the camera saying
    "not done" is run once more from home."""
    budgets = budgets or DEFAULT_BUDGETS
    ep = TableEpisode(seed, render=True)
    obs = ep.observation()
    frames, latencies = [], []
    record = (lambda e, o: frames.append(np.concatenate([o["images"][c] for c in CAMERAS], axis=1))) \
        if video_path is not None else None
    retries = []
    for k, (skill, _, text) in enumerate(SKILLS):
        retried = RETRY[skill] or skill in getattr(policy, "alternates", {})  # a second controller is also a retry
        attempts = 2 if retry and checker is not None and retried else 1
        for attempt in range(1, attempts + 1):
            if (k or attempt > 1) and home_frames:
                obs = go_home(ep, home_frames, on_frame=record)
            if hasattr(policy, "attempt"):
                policy.attempt(skill, attempt)
            policy.reset()
            onehot = np.zeros(len(SKILLS), dtype=np.float32)
            onehot[k] = 1.0
            seen_done = False
            for i in range(budgets[skill]):
                obs["task"], obs["env_state"], obs["skill"] = text, onehot, skill
                t = time.perf_counter()
                action = np.asarray(policy.select_action(obs), dtype=np.float64)
                latencies.append(time.perf_counter() - t)
                obs = ep.step(action)
                if record is not None:
                    record(ep, obs)
                if (checker is not None and ends_on_camera(skill, attempt) and i >= min_frames and i % check_every == 0
                        and checker(skill, obs["images"]["top"])):
                    # Let the policy finish releasing and retreating (the classifier sees "done" while still held).
                    for _ in range(max(settle_frames, SETTLE[skill])):
                        obs["task"], obs["env_state"], obs["skill"] = text, onehot, skill
                        obs = ep.step(np.asarray(policy.select_action(obs), dtype=np.float64))
                        if record is not None:
                            record(ep, obs)
                    seen_done = True
                    break
            if attempt == attempts or seen_done or checker(skill, obs["images"]["top"]):
                break
            retries.append(skill)
    result = grade_table(ep.m, ep.d, ep.params)
    result["retries"] = retries
    result.update(seed=seed, policy_ms_mean=1000 * float(np.mean(latencies)),
                  policy_ms_p95=1000 * float(np.percentile(latencies, 95)))
    if video_path is not None:
        iio.mimsave(video_path, frames, fps=FPS, macro_block_size=8)
    ep.close()
    return result


def evaluate(policy, seeds, out_dir: Path, videos: int = 0, label: str = "policy", budgets=None, checker=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, seed in enumerate(seeds):
        r = run_episode(policy, seed, budgets, out_dir / f"{label}_seed{seed}.mp4" if i < videos else None, checker=checker)
        rows.append(r)
        print(f"[{label}] seed {seed}: {r['subtasks_done']}/5 {'PASS' if r['success'] else 'failed ' + str(r['failed'])}"
              f" ({r['policy_ms_mean']:.1f} ms/step)", flush=True)
    steps = [s for s, _, _ in SKILLS]
    keys = ["drawer_open" if s == "drawer" else s for s in steps]
    with open(out_dir / f"{label}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "success", "subtasks_done", *keys, "policy_ms_mean"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    n, k = len(rows), sum(r["success"] for r in rows)
    summary = {"label": label, "episodes": n, "full_success": k, "rate": k / n, "wilson95": wilson(k, n),
               "per_subtask": {s: sum(r[key] for r in rows) for s, key in zip(steps, keys)},
               "mean_subtasks": float(np.mean([r["subtasks_done"] for r in rows])),
               "policy_ms_mean": float(np.mean([r["policy_ms_mean"] for r in rows]))}
    (out_dir / f"{label}_summary.json").write_text(json.dumps(summary, indent=1))
    return summary

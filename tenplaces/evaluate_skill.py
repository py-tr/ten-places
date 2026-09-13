"""Per-skill evaluation: how often does one skill's policy complete its skill from a realistic start?

For skill k on a seed, the scripted controller first performs the skills that come before k in the
canonical order (so the scene looks the way it would when k starts), then the learned policy runs skill k
until the camera classifier says it is done or the budget runs out. Only skill k is graded.
"""
import json
from pathlib import Path

import numpy as np

from .env_table import SKILLS, TableEpisode
from .evaluate import wilson
from .evaluate_table import CAMERA_ENDS, DEFAULT_BUDGETS, SETTLE
from .grader_table import grade_table
from .oracle import table

SKILL_NAMES = [s for s, _, _ in SKILLS]
KEY = {"drawer": "drawer_open", "spoon": "spoon", "plate": "plate", "fork": "fork", "cup": "cup"}


def verified_prefixes(skill: str):
    """Every start a verified plan can give this skill: subsets of the earlier skills that contain the skill's
    prerequisites and their own, in canonical order (the full-table prefix is the last one)."""
    from itertools import combinations

    from .planner import PREREQS

    earlier = SKILL_NAMES[:SKILL_NAMES.index(skill)]
    out = []
    for r in range(len(earlier) + 1):
        for sub in combinations(earlier, r):
            s = set(sub)
            if set(PREREQS.get(skill, ())) <= s and all(set(PREREQS.get(k, ())) <= s for k in s):
                out.append(list(sub))
    return out


def run_skill_episode(policy, skill: str, seed: int, checker=None, budget=None, check_every=10, min_frames=40,
                      settle_frames=30, before=None, drawer_open=None):
    """before: skills the scripted controller performs first (default: every earlier skill in canonical order;
    a verified subset plan can start a skill from fewer, e.g. the cup with the plate never moved).
    drawer_open: how far the scripted drawer is pulled (default: the scene's 0.09 m) — the learned drawer opens
    8.4-9.3 cm, and the spoon policy trained at exactly 9 cm scored 5/10 at 8 cm and 0/10 at 10 cm."""
    ep = TableEpisode(seed, render=True)
    if drawer_open is not None:
        ep.params.drawer_open = drawer_open
    if before is None:
        before = SKILL_NAMES[:SKILL_NAMES.index(skill)]
    if before:
        table.run_plan(ep.ctl, ep.params, before)  # the scripted controller sets the scene up
    obs = ep.observation()
    policy.reset()
    onehot = np.zeros(len(SKILLS), dtype=np.float32)
    onehot[SKILL_NAMES.index(skill)] = 1.0
    text = dict((s, t) for s, _, t in SKILLS)[skill]
    frames = 0
    for i in range(budget or DEFAULT_BUDGETS[skill]):
        obs["task"], obs["env_state"], obs["skill"] = text, onehot, skill
        obs = ep.step(np.asarray(policy.select_action(obs), dtype=np.float64))
        frames = i + 1
        if (checker is not None and CAMERA_ENDS[skill] and i >= min_frames and i % check_every == 0
                and checker(skill, obs["images"]["top"])):
            # "Done" is seen as soon as the object is in place, often while it is still held: let the policy
            # finish the release and retreat that ends every demo before moving on.
            for _ in range(max(settle_frames, SETTLE[skill])):
                obs["task"], obs["env_state"], obs["skill"] = text, onehot, skill
                obs = ep.step(np.asarray(policy.select_action(obs), dtype=np.float64))
            frames += settle_frames
            break
    g = grade_table(ep.m, ep.d, ep.params)
    ep.close()
    return {"seed": seed, "skill": skill, "success": bool(g[KEY[skill]]), "frames": frames,
            "err_m": g.get(f"{skill}_err_m")}


def evaluate_skill(policy, skill: str, seeds, out_dir: Path | None = None, checker=None, label=None, before=None,
                   budget=None, drawer_open=None):
    rows = [run_skill_episode(policy, skill, s, checker=checker, before=before, budget=budget, drawer_open=drawer_open)
            for s in seeds]
    k = sum(r["success"] for r in rows)
    summary = {"skill": skill, "label": label or skill, "episodes": len(rows), "success": k,
               "rate": k / len(rows), "wilson95": wilson(k, len(rows)),
               "mean_frames": float(np.mean([r["frames"] for r in rows]))}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{summary['label']}_skill.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    return summary

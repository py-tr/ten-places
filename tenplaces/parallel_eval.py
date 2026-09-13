"""Evaluation episodes in parallel worker processes. Each episode is a pure function of its seed (a fresh
TableEpisode per episode, policy.reset() per skill), so workers can run seeds in any order and the rows are
sorted by seed afterwards; serial and parallel runs give the same rows (tests/test_parallel_eval.py).

    rows = run_skill_parallel({"kind": "lerobot", "path": ck, "kwargs": {"n_action_steps": 10}}, "plate",
                              range(100, 150), workers=4)

Why: an episode is 10-20 s of single-threaded MuJoCo + rendering + a small policy, and honest comparisons need
40-60 seeds per setting (a 10-seed comparison cannot separate 60% from 80% success). On a 14-core CPU with one
GPU, 4 workers while something trains and 6-8 on an idle machine.

Windows: workers are spawned (not forked), so policies are built from a picklable spec inside each worker, and
the MuJoCo renderer (a GL context) and CUDA are only ever created there — never in the parent.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

_W = {}  # this worker's policy and checker, built once by _init


class StillPolicy:
    """Holds every joint where it is (tests and dry runs; needs no model)."""

    def reset(self):
        pass

    def select_action(self, obs):
        return obs["state"].astype("float64")


def build_policy(spec: dict):
    """spec: {"kind": "still"} | {"kind": "lerobot", "path": ..., "kwargs": {...}} |
    {"kind": "skills", "runs": [...], "exec_settings": {...}, "kwargs": {...}}. exec_settings is passed explicitly
    (SkillPolicies would otherwise re-read out/eval/exec_settings.json, which a selection run may be rewriting)."""
    kind = spec["kind"]
    if kind == "still":
        return StillPolicy()
    if kind == "lerobot":
        from .lerobot_policy import LeRobotPolicy

        return LeRobotPolicy(spec["path"], **spec.get("kwargs", {}))
    if kind == "skills":
        from .skill_policies import SkillPolicies

        return SkillPolicies(spec["runs"], exec_settings=spec.get("exec_settings", {}),
                             checkpoints=spec.get("checkpoints", {}), **spec.get("kwargs", {}))
    raise ValueError(f"unknown policy kind {kind!r}")


def _init(spec: dict, classifier_xml: str | None, threads: int):
    import torch

    torch.set_num_threads(threads)
    if torch.cuda.is_available():  # the same kernels in every worker and in a serial run
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    _W["policy"] = build_policy(spec)
    _W["checker"], _W["probs"] = None, None
    if classifier_xml:
        from .state_classifier import OVStateClassifier

        clf = OVStateClassifier(classifier_xml, ov_config={"INFERENCE_NUM_THREADS": threads})
        _W["checker"], _W["probs"] = clf.is_done, clf.probs


def _skill_job(job):
    from .evaluate_skill import run_skill_episode

    skill, seed, before, budget, drawer_open = job
    return run_skill_episode(_W["policy"], skill, seed, checker=_W["checker"], budget=budget, before=before,
                             drawer_open=drawer_open)


STEP_KEYS = ("drawer_open", "spoon", "plate", "fork", "cup")


def _failed_row(seed: int, error: Exception, **extra) -> dict:
    """A seed whose episode raised: counted as a failed table (never dropped), with the error kept, so one bad
    seed does not discard a whole 50-seed run."""
    return {"seed": seed, "success": False, "subtasks_done": 0, "failed": "error", "error": repr(error)[:300],
            **{k: False for k in STEP_KEYS}, **extra}


def _table_job(job):
    from .evaluate_table import run_episode

    from pathlib import Path

    seed, home_frames, budgets, video_path = job
    try:
        return run_episode(_W["policy"], seed, budgets=budgets, checker=_W["checker"], home_frames=home_frames,
                           video_path=Path(video_path) if video_path else None)
    except Exception as e:
        return _failed_row(seed, e)


class FixedPlanner:
    """Stands in for the VLM planner in execution-only evaluations: always plans `steps` (verified), never amends.
    For "set the table" the real planner returns all five skills (10/10 in scripts/eval_planner.py)."""

    def __init__(self, steps=("drawer", "spoon", "plate", "fork", "cup")):
        self.steps = list(steps)

    def plan(self, command, image, done=()):
        from .planner import verify

        steps, notes = verify(self.steps, done)
        return {"command": command, "proposed": list(self.steps), "steps": steps, "corrections": notes, "ms": 0.0}


def _agent_job(job):
    """One seed through the agent's own control loop (tenplaces.agent.run_command): check after each skill, one
    retry, one re-plan, re-check of finished skills before the next one — the system, not just the sequencer."""
    import time

    from .agent import run_command

    seed, steps, budgets, disturb, *rest = job
    look_first = rest[0] if rest else None  # the first look (run_command look_first); None: off

    def check(skill, image):
        t = time.perf_counter()
        return _W["checker"](skill, image), 1000 * (time.perf_counter() - t)

    if look_first is not None:
        check.probs = _W.get("probs")
    try:
        events, grade = run_command(_W["policy"], FixedPlanner(steps), "set the table", seed, budgets=budgets,
                                    checker=check, log=lambda *_: None, linger_s=0.0, disturb=disturb,
                                    look_first=look_first)
    except Exception as e:
        return _failed_row(seed, e, retries=0, replans=0, regressed=0, pushed=0, regressed_skills=[], seen=[])
    kinds = [e["kind"] for e in events]
    return {**grade, "seed": seed, "retries": sum(e["kind"] == "skill_start" and e.get("attempt", 1) > 1 for e in events),
            "replans": kinds.count("replan"), "regressed": kinds.count("regressed"), "pushed": kinds.count("pushed"),
            "regressed_skills": [e.get("skill") for e in events if e["kind"] == "regressed"],
            "seen": next((e.get("steps", []) for e in events if e["kind"] == "seen_done"), [])}


def run_agent_parallel(spec: dict, seeds, steps=("drawer", "spoon", "plate", "fork", "cup"), workers: int = 4,
                       classifier_xml: str = "models/state_classifier_v3/state_classifier.xml", threads: int = 1,
                       budgets: dict | None = None, disturb=(), look_first: float | None = None) -> list[dict]:
    """Every seed through tenplaces.agent.run_command with a fixed plan; rows (grade + retry counts) by seed.
    disturb: tenplaces.agent.parse_push entries applied to every episode (e.g. the plate slid off its mat).
    look_first: the first look's probability bar (None: off); the row's "seen" lists what it skipped."""
    jobs = [(int(s), list(steps), budgets, list(disturb), look_first) for s in sorted(int(x) for x in seeds)]
    with _pool(spec, classifier_xml, workers, threads) as pool:
        rows = list(pool.map(_agent_job, jobs))
    return sorted(rows, key=lambda r: r["seed"])


def _pool(spec, classifier_xml, workers, threads):
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init,
                               initargs=(spec, classifier_xml, threads))


def run_skill_parallel(spec: dict, skill: str, seeds, before=None, budget=None, workers: int = 4,
                       classifier_xml: str | None = "models/state_classifier_v3/state_classifier.xml",
                       threads: int = 1, drawer_open: float | None = None) -> list[dict]:
    """tenplaces.evaluate_skill.run_skill_episode for every seed; rows sorted by seed. drawer_open: how far the
    scripted drawer is pulled before the skill (default: the scene's)."""
    jobs = [(skill, int(s), None if before is None else list(before), budget, drawer_open) for s in seeds]
    with _pool(spec, classifier_xml, workers, threads) as pool:
        rows = list(pool.map(_skill_job, jobs))
    return sorted(rows, key=lambda r: r["seed"])


def run_table_parallel(spec: dict, seeds, workers: int = 4,
                       classifier_xml: str | None = "models/state_classifier_v3/state_classifier.xml",
                       threads: int = 1, home_frames: int = 0, budgets: dict | None = None, video_dir=None,
                       videos: int = 0, label: str = "policy") -> list[dict]:
    """tenplaces.evaluate_table.run_episode (full table, sequencer + classifier) for every seed; sorted by seed.
    home_frames > 0: the arms return home between skills (env_table.go_home). budgets: frames per skill
    (default evaluate_table.DEFAULT_BUDGETS). video_dir + videos: <label>_seed<N>.mp4 for the first `videos`
    seeds (the names scripts/make_grid_video.py reads)."""
    seeds = sorted(int(s) for s in seeds)
    jobs = [(s, home_frames, budgets, str(Path(video_dir) / f"{label}_seed{s}.mp4") if video_dir and i < videos else None)
            for i, s in enumerate(seeds)]
    with _pool(spec, classifier_xml, workers, threads) as pool:
        rows = list(pool.map(_table_job, jobs))
    return sorted(rows, key=lambda r: r["seed"])

"""One trained policy per skill behind the single-policy interface the evaluators and the agent use.

obs["skill"] names the active skill; reset() is called at every skill start, so each skill's policy starts
with an empty action queue.

Which checkpoint: by default the latest one of the last runs dir that has the skill. A checkpoint chosen by
closed-loop rollouts on tuning seeds (scripts/eval_skill_checkpoints.py — usually the earliest that holds up, not
the last) is listed in out/eval/selected_checkpoints.json and wins over that. Execution settings (how many actions
of each predicted chunk to run before re-planning, or temporal ensembling) can differ per skill: chosen on a
tuning seed split by `scripts/eval_skill_variants.py --select` and stored in out/eval/exec_settings.json. Every
entry point (evaluation, agent, live viewer) picks both files up by default; parallel workers get them passed
explicitly so a file rewritten mid-run cannot change what they load.
"""
import json
from pathlib import Path

from .env_table import SKILLS
from .lerobot_policy import LeRobotPolicy
from .paths import OUT

EXEC_SETTINGS = OUT / "eval" / "exec_settings.json"
SELECTED = OUT / "eval" / "selected_checkpoints.json"
SKILL_NAMES = [s for s, _, _ in SKILLS]


def latest_checkpoint(run: Path) -> Path:
    ckpts = sorted((p for p in (run / "checkpoints").iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    return ckpts[-1] / "pretrained_model"


def skill_run(runs_dirs, skill: str) -> Path:
    """The last runs dir with a finished checkpoint for this skill (later dirs override earlier ones; a run
    that was stopped before its first checkpoint is skipped)."""
    for d in reversed(runs_dirs):
        ck = Path(d) / skill / "checkpoints"
        if ck.is_dir() and any(p.name.isdigit() for p in ck.iterdir()):
            return Path(d) / skill
    raise FileNotFoundError(f"no run for {skill} in {runs_dirs}")


def _load(value, default_file: Path) -> dict:
    """None: the file if it exists; a path: that file; a dict: as given ({} disables the file)."""
    if value is None:
        return json.loads(default_file.read_text()) if default_file.exists() else {}
    if isinstance(value, (str, Path)):
        return json.loads(Path(value).read_text())
    return dict(value)


def load_exec_settings(exec_settings=None) -> dict:
    return _load(exec_settings, EXEC_SETTINGS)


def load_selected(checkpoints=None) -> dict:
    return _load(checkpoints, SELECTED)


def resolve_checkpoints(runs_dirs, step: int | None = None, selected: dict | None = None) -> dict:
    """skill -> pretrained_model dir: the selected checkpoint if listed, else `step` (or the latest) of the run."""
    out = {}
    for skill in SKILL_NAMES:
        if selected and skill in selected:
            out[skill] = Path(selected[skill])
            continue
        run = skill_run(runs_dirs, skill)
        out[skill] = run / "checkpoints" / f"{step:06d}" / "pretrained_model" if step else latest_checkpoint(run)
    return out


class SkillPolicies:
    def __init__(self, runs_dir, step: int | None = None, exec_settings=None, checkpoints=None, **policy_kwargs):
        runs_dirs = [runs_dir] if isinstance(runs_dir, (str, Path)) else list(runs_dir)
        self.exec_settings = load_exec_settings(exec_settings)
        self.selected = load_selected(checkpoints)
        self.sources = {s: str(p) for s, p in resolve_checkpoints(runs_dirs, step, self.selected).items()}
        self.policies = {s: LeRobotPolicy(p, **{**policy_kwargs, **self.exec_settings.get(s, {})})
                         for s, p in self.sources.items()}
        print(f"[skills] execution settings: {self.exec_settings or 'default for every skill'}; "
              f"selected checkpoints: {sorted(self.selected) or 'none (latest of each run)'}", flush=True)
        self.active = None

    def reset(self):
        for p in self.policies.values():
            p.reset()

    def select_action(self, obs):
        return self.policies[obs["skill"]].select_action(obs)

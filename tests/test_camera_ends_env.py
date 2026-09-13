"""TENPLACES_CAMERA_ENDS overrides how a skill ends, in a fresh process (as a spawned eval worker imports it)."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = "import json; from tenplaces.evaluate_table import CAMERA_ENDS; print(json.dumps(CAMERA_ENDS))"


def _camera_ends(env_value=None):
    env = {k: v for k, v in os.environ.items() if k != "TENPLACES_CAMERA_ENDS"}
    if env_value is not None:
        env["TENPLACES_CAMERA_ENDS"] = env_value
    out = subprocess.run([sys.executable, "-c", CODE], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_default_drawer_runs_its_budget():
    ends = _camera_ends()
    assert ends["drawer"] is False and all(ends[s] for s in ("spoon", "plate", "fork", "cup"))


def test_env_override_per_skill():
    ends = _camera_ends("drawer=1,cup=0")
    assert ends["drawer"] is True and ends["cup"] is False and ends["spoon"] is True

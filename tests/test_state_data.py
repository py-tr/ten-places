"""record_state_data.py: a short drawer pull is recorded, and --drawer-enough labels the drawer from its opening."""
import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "record_state_data.py"


def _module():
    spec = importlib.util.spec_from_file_location("record_state_data", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_short_pull_labelled_by_opening():
    mod = _module()
    x, y, drawer_m, steps, ok = mod.run(9100, np.random.default_rng(9100), short=True, drawer_enough=0.074)
    assert ok and steps[0] == "drawer"
    assert len(x) == len(y) == len(drawer_m) > 0
    assert drawer_m.max() < 0.074  # the pull stopped short of "enough"
    assert (y[:, 0] == (drawer_m >= 0.074)).all()  # so the drawer is never labelled done
    assert (drawer_m >= 0.06).any()  # yet it passed the grader's 6 cm: the states v2 called "done"

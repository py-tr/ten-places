"""Finished steps are re-checked and redone when they no longer hold (a "not done" must be confirmed a moment
later); scripted pushes really move objects. Stand-in policy and planner (see test_agent_voice.py)."""
import numpy as np

from tenplaces.agent import run_command
from tenplaces.env_table import TableEpisode
from test_agent_voice import ScriptedPlanner, StillPolicy, started


class FlakyChecker:
    """Says 'done' every time, except on the listed call numbers of `skill`."""

    def __init__(self, skill, fail_on):
        self.skill, self.fail_on, self.calls = skill, set(fail_on), {}

    def __call__(self, skill, image):
        self.calls[skill] = self.calls.get(skill, 0) + 1
        return not (skill == self.skill and self.calls[skill] in self.fail_on), 0.0


def run(planner, checker, **kw):
    events, _ = run_command(StillPolicy(), planner, "set the table", seed=0, checker=checker, log=lambda *_: None, **kw)
    return events


def test_a_step_that_no_longer_holds_is_redone_first():
    # drawer check 1: its own completion; 2: the re-check before the spoon; 3: the confirming second look
    events = run(ScriptedPlanner(["drawer", "spoon"]), FlakyChecker("drawer", fail_on={2, 3}))
    assert started(events) == ["drawer", "drawer", "spoon"]
    assert [e["skill"] for e in events if e["kind"] == "regressed"] == ["drawer"]
    assert events[-1]["done"] == ["drawer", "spoon"]


def test_one_bad_frame_does_not_trigger_a_redo():
    events = run(ScriptedPlanner(["drawer", "spoon"]), FlakyChecker("drawer", fail_on={2}))
    assert started(events) == ["drawer", "spoon"]
    assert not any(e["kind"] == "regressed" for e in events)


def test_repairs_are_bounded():
    class NeverHolds(FlakyChecker):  # done after each attempt (calls 1, 4, ...), gone at every re-check and look
        def __call__(self, skill, image):
            self.calls[skill] = self.calls.get(skill, 0) + 1
            return self.calls[skill] % 3 == 1, 0.0

    events = run(ScriptedPlanner(["drawer"]), NeverHolds("drawer", ()), max_repairs=1)
    assert started(events) == ["drawer", "drawer"]
    assert any(e["kind"] == "gave_up" and e["skill"] == "drawer" for e in events)


def test_no_recheck_when_disabled():
    events = run(ScriptedPlanner(["drawer", "spoon"]), FlakyChecker("drawer", fail_on={2, 3}), recheck=False)
    assert started(events) == ["drawer", "spoon"]


def test_push_moves_the_plate():
    start = TableEpisode(0, render=False)
    xy0 = start.d.xpos[start.m.body("plate").id][:2].copy()
    moved = {}

    def remember(ep, obs):
        moved["xy"] = ep.d.xpos[ep.m.body("plate").id][:2].copy()

    events, _ = run_command(StillPolicy(), ScriptedPlanner(["drawer"]), "open the drawer", seed=0,
                            checker=lambda s, img: (True, 0.0), on_frame=remember, log=lambda *_: None,
                            disturb=[(1.0, "plate", (0.0, -0.07), 0.25)])
    shift = np.linalg.norm(moved["xy"] - xy0)
    assert any(e["kind"] == "pushed" and e["body"] == "plate" for e in events)
    assert 0.05 < shift < 0.09, shift  # 7 cm asked; friction may add or take a little


def test_push_tied_to_a_step_waits_for_it():
    events, _ = run_command(StillPolicy(), ScriptedPlanner(["drawer", "spoon"]), "x", seed=0,
                            checker=lambda s, img: (True, 0.0), log=lambda *_: None,
                            disturb=[("after-drawer", "plate", (0.0, -0.07), 0.25)])
    drawer_done = next(e["t"] for e in events if e["kind"] == "check" and e["skill"] == "drawer")
    pushed = next(e["t"] for e in events if e["kind"] == "pushed")
    assert drawer_done + 0.9 <= pushed <= drawer_done + 1.2

"""The agent's control flow when the person speaks during the run — with a still policy, a scripted planner
and an instant camera check, so no model is loaded. Timing: a skill takes 40 frames + 30 settle = 2.8 s."""
import numpy as np

from tenplaces.agent import run_command
from tenplaces.listen import STOP, ScriptedVoice
from tenplaces.planner import verify


class StillPolicy:
    def reset(self):
        pass

    def select_action(self, obs):
        return obs["state"].astype(np.float64)  # hold every joint where it is


class ScriptedPlanner:
    def __init__(self, steps, amendments=None):
        self.steps, self.amendments, self.amend_calls = steps, amendments or {}, []

    def plan(self, command, image, done=()):
        steps, notes = verify(self.steps, done)
        return {"proposed": list(self.steps), "steps": steps, "corrections": notes, "ms": 0.0}

    def amend(self, command, heard, image, done=(), current=None, remaining=()):
        self.amend_calls.append({"heard": heard, "done": list(done), "current": current, "remaining": list(remaining)})
        proposed = self.amendments[heard]
        steps, notes = verify(proposed, list(done) + ([current] if current else []))
        return {"proposed": proposed, "steps": steps, "corrections": notes, "ms": 0.0}


def instant_check(skill, image):
    return True, 0.0


def run(planner, schedule, linger_s=4.0):
    events, _ = run_command(StillPolicy(), planner, "set the table", seed=0, checker=instant_check,
                            voice=ScriptedVoice(schedule), linger_s=linger_s, log=lambda *_: None)
    return events


def started(events):
    return [e["skill"] for e in events if e["kind"] == "skill_start"]


def test_stop_word():
    assert STOP.search("Stop!") and STOP.search("please halt now") and not STOP.search("unstoppable")


def test_spoken_change_applies_after_the_current_step():
    planner = ScriptedPlanner(["drawer", "spoon", "plate", "fork", "cup"],
                              {"Skip the fork.": ["spoon", "plate", "cup"]})
    events = run(planner, [(1.5, "Skip the fork.")])
    assert started(events) == ["drawer", "spoon", "plate", "cup"]
    assert planner.amend_calls[0]["current"] == "drawer" and planner.amend_calls[0]["done"] == []
    amend = next(e for e in events if e["kind"] == "amend")
    assert amend["after"] == "drawer" and amend["steps"] == ["spoon", "plate", "cup"]


def test_stop_ends_the_run_at_once():
    events = run(ScriptedPlanner(["drawer", "spoon", "plate"]), [(1.5, "Stop.")])
    assert started(events) == ["drawer"]
    assert not any(e["kind"] == "check" for e in events)
    assert events[-1]["kind"] == "finished" and events[-1]["stopped"]


def test_request_after_the_last_step_is_still_done():
    planner = ScriptedPlanner(["drawer"], {"Oh, and put the cup out too.": ["cup"]})
    events = run(planner, [(4.5, "Oh, and put the cup out too.")])
    assert started(events) == ["drawer", "cup"]
    assert events[-1]["done"] == ["drawer", "cup"]


def test_amendment_cannot_skip_physical_prerequisites():
    # "Just the fork" mid-drawer: the verifier re-adds the plate (it starts on the fork's spot).
    planner = ScriptedPlanner(["drawer", "spoon", "plate", "fork", "cup"], {"Just the fork.": ["fork"]})
    events = run(planner, [(1.5, "Just the fork.")])
    assert started(events) == ["drawer", "plate", "fork"]

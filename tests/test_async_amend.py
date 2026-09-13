"""Spoken changes are amended on a worker thread: the arms keep moving while the planner thinks, the change
applies at the next step boundary (holding still if the step ends first), "stop" still halts at once, and two
quick sentences chain. The planner stand-in sleeps 0.3 s of wall time per amend(), like a (much faster) VLM."""
import time

from tenplaces.agent import run_command
from tenplaces.listen import ScriptedVoice
from test_agent_voice import ScriptedPlanner, StillPolicy, instant_check, started

FULL = ["drawer", "spoon", "plate", "fork", "cup"]
SKIP_FORK = {"Skip the fork.": ["spoon", "plate", "cup"]}


class SlowPlanner(ScriptedPlanner):
    delay = 0.3

    def amend(self, command, heard, image, done=(), current=None, remaining=()):
        time.sleep(self.delay)
        if heard in self.amendments:
            return super().amend(command, heard, image, done, current, remaining)
        # "Skip the <skill>.": drop it from what was planned, so chaining is visible in the result
        steps = [s for s in remaining if s not in heard]
        self.amend_calls.append({"heard": heard, "done": list(done), "current": current, "remaining": list(remaining)})
        return {"proposed": steps, "steps": steps, "corrections": [], "ms": 1000 * self.delay}


class Tape:
    """Control steps so far, and the step count at which each event happened."""

    def __init__(self):
        self.frames, self.events = 0, []

    def on_frame(self, ep, obs):
        self.frames += 1

    def on_event(self, e):
        self.events.append((self.frames, e))

    def at(self, kind):
        return [n for n, e in self.events if e["kind"] == kind]


def run(planner, schedule, **kw):
    tape = Tape()
    events, _ = run_command(StillPolicy(), planner, "set the table", seed=0, checker=instant_check,
                            voice=ScriptedVoice(schedule), on_frame=tape.on_frame, on_event=tape.on_event,
                            log=lambda *_: None, **kw)
    return events, tape


def drawer_end():
    """Simulated time at which the drawer step's completion check happens (no speech)."""
    events, _ = run(ScriptedPlanner(["drawer"]), [], linger_s=0)
    return next(e["t"] for e in events if e["kind"] == "check")


def test_the_arms_keep_moving_while_the_planner_thinks():
    planner = SlowPlanner(FULL, SKIP_FORK)
    events, tape = run(planner, [(1.0, "Skip the fork.")])
    (thinking,), (amend,) = tape.at("thinking"), tape.at("amend")
    assert amend > thinking  # control steps ran during the 0.3 s amend()
    assert started(events) == ["drawer", "spoon", "plate", "cup"]
    kinds = [e["kind"] for e in events]
    spoon_start = [i for i, k in enumerate(kinds) if k == "skill_start"][1]
    assert kinds.index("amend") < spoon_start  # applied at the drawer/spoon boundary
    assert next(e for e in events if e["kind"] == "amend")["after"] == "drawer"


def test_the_arms_hold_still_at_the_boundary_until_the_amendment_arrives():
    planner = SlowPlanner(FULL, SKIP_FORK)
    events, tape = run(planner, [(drawer_end() - 0.1, "Skip the fork.")])
    amend = next(e for e in events if e["kind"] == "amend")
    assert amend["waited_s"] > 0
    (check,), (amended,) = tape.at("check")[:1], tape.at("amend")
    assert amended > check  # held (and stepped) between the end of the drawer and the new plan
    assert started(events) == ["drawer", "spoon", "plate", "cup"]


def test_stop_during_a_pending_amendment_halts_at_once():
    planner = SlowPlanner(FULL, SKIP_FORK)
    events, tape = run(planner, [(1.0, "Skip the fork."), (1.05, "Stop.")])
    kinds = [e["kind"] for e in events]
    assert "amend" not in kinds and "discarded" in kinds
    assert tape.at("stop") == [tape.frames]  # no control step after the stop
    assert started(events) == ["drawer"] and events[-1]["stopped"]


def test_two_sentences_chain_onto_the_pending_result():
    planner = SlowPlanner(FULL)
    events, _ = run(planner, [(1.0, "Skip the fork."), (1.05, "Skip the cup.")])
    assert planner.amend_calls[1]["remaining"] == ["spoon", "plate", "cup"]  # the first amendment's result
    assert [(e["kind"], e["heard"]) for e in events if e["kind"] in ("thinking", "amend")] == [
        ("thinking", "Skip the fork."), ("amend", "Skip the fork."), ("thinking", "Skip the cup."), ("amend", "Skip the cup.")]
    assert started(events) == ["drawer", "spoon", "plate"]


def test_synchronous_amend_is_still_available():
    events, _ = run(ScriptedPlanner(FULL, SKIP_FORK), [(1.5, "Skip the fork.")], async_amend=False)
    assert "thinking" not in [e["kind"] for e in events]
    assert started(events) == ["drawer", "spoon", "plate", "cup"]

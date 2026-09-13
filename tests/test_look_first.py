"""The first look (run_command look_first): steps the camera classifier already sees done are passed to the planner
as done and not planned again, the robot says so, and a prepared demo table is scored on everything it should end
with. A still policy and a scripted planner, so no model is loaded."""
import numpy as np
from test_agent_voice import ScriptedPlanner, StillPolicy

from scripts.score_demo import score
from tenplaces.agent import run_command
from tenplaces.speak import phrase_for

FULL = ["drawer", "spoon", "plate", "fork", "cup"]


def seeing(probs):
    """An instant camera check whose first look reports these probabilities (drawer, spoon, plate, fork, cup)."""
    def check(skill, image):
        return True, 0.0

    check.probs = lambda image: np.asarray(probs, dtype=float)
    return check


class RecordingPlanner(ScriptedPlanner):
    def plan(self, command, image, done=()):
        self.done_given = list(done)
        return super().plan(command, image, done)


def run(probs, look_first, steps=FULL):
    planner = RecordingPlanner(steps)
    events, _ = run_command(StillPolicy(), planner, "Set the table.", seed=0, checker=seeing(probs),
                            look_first=look_first, log=lambda *_: None)
    return planner, events


def test_seen_steps_are_given_to_the_planner_as_done():
    planner, events = run([0.99, 0.01, 0.98, 0.0, 0.0], 0.95)
    seen = next(e for e in events if e["kind"] == "seen_done")
    assert seen["steps"] == ["drawer", "plate"] and planner.done_given == ["drawer", "plate"]
    plan = next(e for e in events if e["kind"] == "plan")
    assert plan["steps"] == ["spoon", "fork", "cup"]


def test_below_the_bar_nothing_is_skipped():
    planner, events = run([0.9, 0.9, 0.9, 0.9, 0.9], 0.95)
    assert next(e for e in events if e["kind"] == "seen_done")["steps"] == [] and planner.done_given == []


def test_off_by_default():
    planner, events = run([0.99] * 5, None)
    assert not any(e["kind"] == "seen_done" for e in events) and planner.done_given == []


def test_the_robot_says_what_it_saw():
    assert phrase_for({"kind": "seen_done", "steps": ["drawer", "plate"]}) == \
        "The drawer is already open and the plate is already out — I'll skip those."
    assert phrase_for({"kind": "seen_done", "steps": ["cup"]}) == "The cup is already out — I'll skip it."
    assert phrase_for({"kind": "seen_done", "steps": []}) is None


def test_the_parallel_agent_job_carries_the_first_look():
    from tenplaces import parallel_eval

    parallel_eval._W.update(policy=StillPolicy(), checker=lambda skill, image: True,
                            probs=lambda image: np.asarray([0.99, 0.0, 0.0, 0.0, 0.97]))
    row = parallel_eval._agent_job((0, FULL, None, [], 0.95))
    assert row["seen"] == ["drawer", "cup"]
    assert parallel_eval._agent_job((0, FULL, None, []))["seen"] == []  # old job tuples: off


def test_a_prepared_table_is_scored_on_the_whole_result():
    entry = {"seed": 0, "mode": "typed", "command": "c", "expected_steps": ["cup"], "prepared": ["plate"]}
    grade = {"drawer_open": True, "spoon": False, "plate": True, "fork": False, "cup": True}
    run_ = {"command": "c", "grade": grade, "events": [{"kind": "seen_done", "steps": ["drawer", "plate"]},
                                                       {"kind": "plan", "steps": ["cup"], "ms": 7000}]}
    s = score(entry, run_)
    assert s["success"] and s["seen"] == ["drawer", "plate"]

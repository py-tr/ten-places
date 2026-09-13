"""The demonstration scorer (scripts/score_demo.py): a seed passes only when exactly the asked-for steps are done and
the refusals match the command. No simulator, no model: the run records are written by hand."""
from scripts.score_demo import score

FULL = ["drawer", "spoon", "plate", "fork", "cup"]


def run(done, unsupported=(), command="c"):
    grade = {"drawer_open": "drawer" in done, **{s: s in done for s in FULL if s != "drawer"}}
    return {"command": command, "grade": grade,
            "events": [{"kind": "plan", "steps": list(done), "ms": 7000, "unsupported": list(unsupported)}]}


def entry(steps, unsupported=None):
    e = {"seed": 0, "mode": "typed", "command": "c", "expected_steps": steps}
    if unsupported:
        e["expected_unsupported"] = unsupported
    return e


def test_exactly_what_was_asked_passes():
    assert score(entry(["spoon"]), run(["drawer", "spoon"]))["success"]  # the verifier's drawer counts on both sides


def test_an_extra_or_missing_step_fails():
    assert not score(entry(["spoon"]), run(["drawer", "spoon", "cup"]))["success"]
    assert not score(entry(FULL), run(["drawer", "spoon", "plate", "fork"]))["success"]


def test_the_expected_refusal_must_be_said():
    asked = entry(FULL, ["light a candle"])
    assert score(asked, run(FULL, ["light a candle"]))["success"]
    assert not score(asked, run(FULL))["success"]
    assert not score(asked, run(FULL, ["pour water"]))["success"]


def test_a_refusal_nobody_asked_for_fails():
    assert not score(entry(FULL), run(FULL, ["skip the cup"]))["success"]

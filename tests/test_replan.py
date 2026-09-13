"""A failed step is retried from what the person currently wants — the plan plus spoken changes — not from the
original command: the first full run brought back a fork the person had skipped. Stand-ins, no model."""
from tenplaces.agent import run_command
from tenplaces.listen import ScriptedVoice
from test_agent_voice import ScriptedPlanner, StillPolicy, started


def plate_never_lands(skill, image):
    return skill != "plate", 0.0


def test_replan_keeps_a_spoken_change():
    planner = ScriptedPlanner(["drawer", "spoon", "plate", "fork", "cup"], {"Skip the fork.": ["spoon", "plate", "cup"]})
    events, _ = run_command(StillPolicy(), planner, "set the table", seed=0, checker=plate_never_lands,
                            voice=ScriptedVoice([(1.5, "Skip the fork.")]), linger_s=0.0, log=lambda *_: None)
    assert "fork" not in started(events)
    replans = [e for e in events if e["kind"] == "replan"]
    assert len(replans) == 1 and replans[0]["steps"] == ["plate", "cup"]
    # 2 attempts in the plan + 2 in the one re-plan, then the plate is left and the cup is still set
    assert started(events) == ["drawer", "spoon", "plate", "plate", "plate", "plate", "cup"]


def test_replan_without_speech_retries_the_rest_of_the_plan():
    events, _ = run_command(StillPolicy(), ScriptedPlanner(["drawer", "plate", "cup"]), "x", seed=0,
                            checker=plate_never_lands, log=lambda *_: None)
    replans = [e for e in events if e["kind"] == "replan"]
    assert len(replans) == 1 and replans[0]["steps"] == ["plate", "cup"]
    assert started(events)[-1] == "cup"

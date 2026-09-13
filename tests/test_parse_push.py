"""tenplaces.agent.parse_push: the --push strings shared by run_agent.py and eval_agent_table.py."""
from tenplaces.agent import parse_push


def test_after_skill_default_duration():
    assert parse_push("after-plate:plate:0:-0.07") == ("after-plate", "plate", (0.0, -0.07), 0.25)


def test_absolute_time_and_duration():
    assert parse_push("20:cup:0.05:0:0.5") == (20.0, "cup", (0.05, 0.0), 0.5)

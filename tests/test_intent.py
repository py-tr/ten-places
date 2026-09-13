"""plan_intent() / plan_checked() / cannot_do() without the VLM: the model's answers are scripted, so only
assemble_intent(), the filters, the re-plan rule and the verifier run."""
import numpy as np

from tenplaces.planner import SKILL_NAMES, VLMPlanner, assemble_intent

IMAGE = np.zeros((8, 8, 3), np.uint8)


def intent(all_=False, include=(), exclude=()):
    return {"all": all_, "include": list(include), "exclude": list(exclude), "reason": ""}


class ScriptedVLM(VLMPlanner):
    def __init__(self, answers):  # skip loading the model
        self.answers = list(answers)

    def _ask(self, prompt, image, schema, max_new_tokens=120, idle=False):
        answer = self.answers.pop(0)
        return answer, str(answer), 5.0


def test_assemble_intent():
    assert assemble_intent(intent(all_=True)) == SKILL_NAMES
    assert assemble_intent(intent(all_=True, exclude=["cup"])) == ["drawer", "spoon", "plate", "fork"]
    assert assemble_intent(intent(include=["cup", "cup"])) == ["cup"]
    assert assemble_intent(intent()) == []


def test_leave_out_and_the_whole_table():
    p = ScriptedVLM([intent(all_=True, exclude=["cup"])]).plan_intent("Leave the cup out, set the rest.", IMAGE)
    assert p["steps"] == ["drawer", "spoon", "plate", "fork"]


def test_included_skills_get_their_prerequisites():
    p = ScriptedVLM([intent(include=["spoon", "fork"])]).plan_intent("Put out cutlery only.", IMAGE)
    assert p["steps"] == ["drawer", "spoon", "plate", "fork"]
    assert any("added 'plate' before 'fork'" in n for n in p["corrections"])


def test_unreadable_intent_plans_nothing():
    assert ScriptedVLM([None]).plan_intent("Hmm.", IMAGE)["steps"] == []


def test_cannot_do_filters():
    answers = [{"unsupported": ["everything else"]}, {"unsupported": ["skip the cup"]},
               {"unsupported": ["a cup of coffee"]}, {"unsupported": ["light a candle"]}]
    vlm = ScriptedVLM(answers)
    assert vlm.cannot_do("No spoon today, but everything else.", IMAGE)[0] == []
    assert vlm.cannot_do("Set it all up but skip the cup.", IMAGE)[0] == []
    assert vlm.cannot_do("Give me a cup of coffee.", IMAGE)[0] == ["a cup of coffee"]
    assert vlm.cannot_do("Set the table and light a candle.", IMAGE)[0] == ["light a candle"]


def test_empty_plan_is_redone_without_the_impossible_part():
    # plan_intent -> nothing; cannot_do -> "light a candle"; plan_intent on "Set the table." -> everything
    vlm = ScriptedVLM([intent(), {"unsupported": ["light a candle"]}, intent(all_=True)])
    p = vlm.plan_checked("Set the table and light a candle.", IMAGE, use_intent=True)
    assert p["steps"] == SKILL_NAMES and p["unsupported"] == ["light a candle"]
    assert p["replanned_from"] == "Set the table." and p["ms"] == 15.0

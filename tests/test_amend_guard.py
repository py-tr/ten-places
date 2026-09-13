"""planner.amend() without the VLM: the model's edit is scripted, so only apply_edit() and the verifier run."""
import numpy as np

from tenplaces.planner import VLMPlanner, apply_edit

IMAGE = np.zeros((8, 8, 3), np.uint8)
NO_CHANGE = {"add": [], "remove": [], "only": [], "nothing_more": False}


def edit(**kw):
    return {**NO_CHANGE, **kw}


class ScriptedVLM(VLMPlanner):
    def __init__(self, answers):  # skip loading the model
        self.answers, self.prompts = list(answers), []

    def _ask(self, prompt, image, schema, max_new_tokens=120, idle=False):
        self.prompts.append(prompt)
        answer = self.answers.pop(0)
        return answer, str(answer), 5.0


def test_apply_edit():
    planned = ["spoon", "plate", "fork", "cup"]
    assert apply_edit(edit(remove=["spoon", "cup"]), planned) == ["plate", "fork"]
    assert apply_edit(edit(add=["cup"]), ["fork"]) == ["fork", "cup"]
    assert apply_edit(edit(only=["cup"]), planned) == ["cup"]
    assert apply_edit(edit(nothing_more=True, add=["cup"]), planned) == []
    assert apply_edit(NO_CHANGE, planned) == planned


def test_removal_keeps_the_rest():
    a = ScriptedVLM([edit(remove=["spoon"])]).amend(
        "Set the table.", "No spoon, please.", IMAGE, done=[], current="drawer", remaining=["spoon", "plate", "fork", "cup"])
    assert a["steps"] == ["plate", "fork", "cup"]


def test_addition_keeps_the_rest_and_flags_the_order():
    a = ScriptedVLM([edit(add=["spoon"])]).amend(
        "Just the plate and the cup.", "Add the spoon as well.", IMAGE, done=["drawer"], current="plate", remaining=["cup"])
    assert a["steps"] == ["spoon", "cup"]
    assert any("'spoon' after 'plate'" in n for n in a["corrections"])


def test_nothing_more_and_unreadable_answers():
    vlm = ScriptedVLM([edit(nothing_more=True), None])
    assert vlm.amend("Set the table.", "That's enough.", IMAGE, ["drawer"], "spoon", ["plate", "cup"])["steps"] == []
    assert vlm.amend("Set the table.", "Mmm.", IMAGE, ["drawer"], "spoon", ["plate", "cup"])["steps"] == ["plate", "cup"]


def test_verifier_still_adds_prerequisites_after_a_change():
    a = ScriptedVLM([edit(remove=["plate"])]).amend(
        "Set the table.", "Don't bother with the plate.", IMAGE, done=["drawer"], current="spoon",
        remaining=["plate", "fork", "cup"])
    assert a["steps"] == ["plate", "fork", "cup"]
    assert any("added 'plate' before 'fork'" in n for n in a["corrections"])


def test_only_is_completed_by_the_verifier():
    a = ScriptedVLM([edit(only=["spoon", "fork"])]).amend(
        "Set the table.", "Only the cutlery, please.", IMAGE, done=[], current="drawer",
        remaining=["spoon", "plate", "fork", "cup"])
    assert a["steps"] == ["spoon", "plate", "fork"]

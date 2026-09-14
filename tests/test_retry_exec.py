"""SkillPolicies.attempt: a skill with an alternate execution setting runs it from its second attempt on; the
first attempt and every skill without one run as before."""
from tenplaces.skill_policies import SkillPolicies


class Fake:
    def __init__(self, name):
        self.name, self.resets = name, 0

    def reset(self):
        self.resets += 1

    def select_action(self, obs):
        return self.name


def policies(alternates):
    sp = SkillPolicies.__new__(SkillPolicies)  # no checkpoints: only the dispatch is under test
    sp.policies = {"spoon": Fake("spoon ensemble"), "plate": Fake("plate")}
    sp.alternates, sp.on_alternate = alternates, set()
    return sp


def test_second_attempt_runs_the_alternate():
    sp = policies({"spoon": Fake("spoon chunk")})
    assert sp.attempt("spoon", 1) is False and sp.select_action({"skill": "spoon"}) == "spoon ensemble"
    assert sp.attempt("spoon", 2) is True and sp.select_action({"skill": "spoon"}) == "spoon chunk"
    assert sp.attempt("spoon", 1) is False and sp.select_action({"skill": "spoon"}) == "spoon ensemble"


def test_skills_without_an_alternate_never_switch():
    sp = policies({"spoon": Fake("spoon chunk")})
    assert sp.attempt("plate", 2) is False and sp.select_action({"skill": "plate"}) == "plate"


def test_reset_clears_both_controllers():
    alt = Fake("spoon chunk")
    sp = policies({"spoon": alt})
    sp.reset()
    assert sp.policies["spoon"].resets == 1 and alt.resets == 1


def test_off_by_default():
    from tenplaces.skill_policies import RETRY_EXEC

    assert RETRY_EXEC == {}  # the deployed robot: every attempt as the first

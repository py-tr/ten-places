"""The symbolic verifier and the set of starts it can give each skill (no VLM, no simulator)."""
from itertools import combinations

import pytest

from tenplaces.evaluate_skill import SKILL_NAMES, verified_prefixes
from tenplaces.planner import PREREQS, verify

ALL_SUBSETS = [list(c) for r in range(1, len(SKILL_NAMES) + 1) for c in combinations(SKILL_NAMES, r)]


def test_adds_prerequisites_and_fixes_order():
    steps, notes = verify(["cup", "fork"])
    assert steps == ["drawer", "plate", "fork", "cup"]
    assert any("added 'drawer' before 'fork'" in n for n in notes)
    assert any("added 'plate' before 'fork'" in n for n in notes)
    assert any(n.startswith("reordered") for n in notes)


def test_leaves_out_what_is_done():
    assert verify(["spoon"], done=["drawer"]) == (["spoon"], [])
    steps, notes = verify(["drawer"], done=["drawer"])
    assert steps == [] and "already done" in notes[0]


def test_drops_duplicates_and_unknown_skills():
    steps, notes = verify(["plate", "plate", "knife"])
    assert steps == ["drawer", "plate"]
    assert any("duplicate 'plate'" in n for n in notes)
    assert any("unknown skill 'knife'" in n for n in notes)


@pytest.mark.parametrize("request_", ALL_SUBSETS, ids=lambda s: "+".join(s))
def test_every_subset_becomes_executable(request_):
    steps, _ = verify(request_)
    assert set(request_) <= set(steps)
    assert steps == sorted(steps, key=SKILL_NAMES.index)
    for i, s in enumerate(steps):
        assert set(PREREQS[s]) <= set(steps[:i])


def test_prefix_counts():
    assert {s: len(verified_prefixes(s)) for s in SKILL_NAMES} == {"drawer": 1, "spoon": 1, "plate": 2, "fork": 2, "cup": 7}


@pytest.mark.parametrize("request_", ALL_SUBSETS, ids=lambda s: "+".join(s))
def test_every_verified_plan_starts_each_skill_from_a_known_prefix(request_):
    # The context evaluation and the context demos cover exactly the starts a verified plan can produce.
    steps, _ = verify(request_)
    for i, s in enumerate(steps):
        assert steps[:i] in verified_prefixes(s)

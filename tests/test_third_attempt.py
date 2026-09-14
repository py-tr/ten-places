"""evaluate_table.attempts_for: 1 without a retry, 2 with one, 3 for a skill in EXTRA_ATTEMPT (off by default)."""
from tenplaces import evaluate_table as et


def test_off_by_default():
    assert et.EXTRA_ATTEMPT == set()
    assert et.attempts_for("plate", True) == 2 and et.attempts_for("spoon", False) == 1


def test_third_attempt_only_for_listed_retried_skills(monkeypatch):
    monkeypatch.setattr(et, "EXTRA_ATTEMPT", {"plate", "fork", "cup"})
    assert et.attempts_for("plate", True) == 3
    assert et.attempts_for("drawer", True) == 2
    assert et.attempts_for("plate", False) == 1  # never retried: a third attempt does not apply

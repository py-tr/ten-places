"""evaluate_table.ends_on_camera: camera-ended skills always; the budget-ended drawer only on a retry or re-run, and
only with RETRY_CAMERA_END (the camera-ended drawer re-pull)."""
from tenplaces import evaluate_table as et


def test_off_the_drawer_always_runs_its_budget(monkeypatch):
    monkeypatch.setattr(et, "RETRY_CAMERA_END", False)
    assert not et.ends_on_camera("drawer", 1)
    assert not et.ends_on_camera("drawer", 2)
    assert not et.ends_on_camera("drawer", 1, rerun=True)


def test_on_a_retry_or_rerun_ends_at_the_camera(monkeypatch):
    monkeypatch.setattr(et, "RETRY_CAMERA_END", True)
    assert not et.ends_on_camera("drawer", 1)  # the first pull still runs its budget
    assert et.ends_on_camera("drawer", 2)
    assert et.ends_on_camera("drawer", 1, rerun=True)


def test_camera_ended_skills_unchanged(monkeypatch):
    for flag in (False, True):
        monkeypatch.setattr(et, "RETRY_CAMERA_END", flag)
        assert all(et.ends_on_camera(s, 1) for s in ("spoon", "plate", "fork", "cup"))

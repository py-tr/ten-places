"""Which checkpoint each skill loads: a selected checkpoint wins, then the latest of the last runs dir that has a
finished checkpoint (a run stopped before its first checkpoint is skipped). Throwaway dirs, no model loaded."""
import json

from tenplaces.skill_policies import SKILL_NAMES, load_selected, resolve_checkpoints


def make_run(root, skill, steps):
    for s in steps:
        (root / skill / "checkpoints" / f"{s:06d}" / "pretrained_model").mkdir(parents=True)
    if not steps:
        (root / skill / "checkpoints").mkdir(parents=True)


def test_latest_of_the_last_runs_dir_wins(tmp_path):
    a, b = tmp_path / "v1", tmp_path / "t1"
    for skill in SKILL_NAMES:
        make_run(a, skill, [20000])
    make_run(b, "plate", [2500, 5000, 10000])
    make_run(b, "cup", [])  # stopped before its first checkpoint
    got = resolve_checkpoints([a, b])
    assert got["plate"] == b / "plate" / "checkpoints" / "010000" / "pretrained_model"
    assert got["cup"] == a / "cup" / "checkpoints" / "020000" / "pretrained_model"


def test_selected_checkpoint_wins(tmp_path):
    a, b = tmp_path / "v1", tmp_path / "t1"
    for skill in SKILL_NAMES:
        make_run(a, skill, [20000])
    make_run(b, "plate", [2500, 5000, 10000])
    chosen = b / "plate" / "checkpoints" / "005000" / "pretrained_model"
    f = tmp_path / "selected.json"
    f.write_text(json.dumps({"plate": str(chosen)}))
    got = resolve_checkpoints([a, b], selected=load_selected(f))
    assert got["plate"] == chosen
    assert load_selected({}) == {}

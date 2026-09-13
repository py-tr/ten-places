"""Parallel evaluation gives the same rows as a serial run (each episode depends only on its seed). Uses the
hold-still stand-in policy and no classifier, so no model is loaded; spawning workers takes a few seconds."""
from tenplaces.evaluate_skill import run_skill_episode
from tenplaces.parallel_eval import StillPolicy, run_skill_parallel, run_table_parallel

SHORT = {"drawer": 3, "spoon": 3, "plate": 3, "fork": 3, "cup": 3}  # a full-table pass in a few frames


def test_table_rows_are_sorted_and_videos_named_for_the_grid(tmp_path):
    rows = run_table_parallel({"kind": "still"}, [3, 0], workers=2, classifier_xml=None, budgets=SHORT,
                              video_dir=tmp_path, videos=1, label="torch")
    assert [r["seed"] for r in rows] == [0, 3]
    assert (tmp_path / "torch_seed0.mp4").exists() and not (tmp_path / "torch_seed3.mp4").exists()


def test_parallel_rows_match_serial_rows():
    seeds, before = [0, 3], ["drawer"]
    serial = [run_skill_episode(StillPolicy(), "cup", s, checker=None, budget=30, before=before) for s in seeds]
    parallel = run_skill_parallel({"kind": "still"}, "cup", reversed(seeds), before=before, budget=30, workers=2,
                                  classifier_xml=None)
    assert [r["seed"] for r in parallel] == seeds  # sorted by seed whatever order the workers finished in
    assert parallel == serial

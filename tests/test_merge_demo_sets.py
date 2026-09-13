"""scripts/merge_demo_sets.py: the merged manifest shifts each set's episodes past the sets before it."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "merge_demo_sets.py"


def _module():
    spec = importlib.util.spec_from_file_location("merge_demo_sets", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_episode_indices_are_shifted_in_order():
    mod = _module()
    chain = {"root": "data/chain", "fps": 25, "image_hw": [144, 192],
             "skills": {"spoon": {"instruction": "spoon", "episodes": [0, 1, 2]}},
             "episodes": [{"episode": 0, "seed": 9500}, {"episode": 1, "seed": 9501}, {"episode": 2, "seed": 9503}]}
    bands = {"root": "data/bands", "skills": {"spoon": {"instruction": "spoon", "episodes": [0, 1]}},
             "episodes": [{"episode": 0, "seed": 12000}, {"episode": 1, "seed": 12001}]}
    m = mod.merged_manifest([chain, bands], "data/merged", "local/x")
    assert m["skills"]["spoon"]["episodes"] == [0, 1, 2, 3, 4]
    assert m["total_episodes"] == 5 and m["merged_from"] == ["data/chain", "data/bands"]
    assert [e["episode"] for e in m["episodes"]] == [0, 1, 2, 3, 4]
    assert m["episodes"][3]["seed"] == 12000 and m["episodes"][3]["source"] == "data/bands"


def test_skills_from_different_sets_stay_separate():
    mod = _module()
    a = {"root": "a", "skills": {"drawer": {"instruction": "d", "episodes": [0, 1]}}, "episodes": []}
    b = {"root": "b", "skills": {"spoon": {"instruction": "s", "episodes": [0]}}, "episodes": []}
    m = mod.merged_manifest([a, b], "m", "local/x")
    assert m["skills"]["drawer"]["episodes"] == [0, 1] and m["skills"]["spoon"]["episodes"] == [2]

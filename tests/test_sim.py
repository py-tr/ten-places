"""Scene randomisation, the scripted controller and the grader, on a few seeds (a few seconds each)."""
import numpy as np
import pytest

from tenplaces import scene_table
from tenplaces.env_table import TableEpisode, record_skill_oracle
from tenplaces.grader_table import grade_table
from tenplaces.oracle import table
from tenplaces.oracle.handoff import HOME


def test_scene_is_determined_by_its_seed():
    assert scene_table.sample(7).targets() == scene_table.sample(7).targets()
    assert scene_table.sample(7).targets() != scene_table.sample(8).targets()


def test_untouched_table_grades_as_nothing_done():
    ep = TableEpisode(1000, render=False)
    g = grade_table(ep.m, ep.d, ep.params)
    assert not g["success"] and not any(g[k] for k in ("drawer_open", "spoon", "plate", "fork", "cup"))


@pytest.mark.parametrize("seed", [1000, 1001])
def test_scripted_full_table_succeeds(seed):
    ep = TableEpisode(seed, render=False)
    table.run(ep.ctl, ep.params)
    assert grade_table(ep.m, ep.d, ep.params)["success"]


def test_skill_recording_starts_at_home_after_a_subset_prefix():
    frames, result = record_skill_oracle(5000, "cup", ["drawer"])
    assert result["cup"] and result["drawer_open"] and result["error"] is None
    start = frames[0]["state"]
    np.testing.assert_allclose(start[:5], HOME, atol=2e-3)   # arm A joints
    np.testing.assert_allclose(start[6:11], HOME, atol=2e-3)  # arm B joints

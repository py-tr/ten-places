"""Robustness outside the training ranges (scene_table.sample stress): stress 1.0 is exactly the usual table, a
higher stress widens the physical and visual ranges about their centres, and placements never move."""
import numpy as np

from tenplaces.scene_table import sample

SEEDS = [0, 7, 123, 200, 249]
PLACEMENTS = ("drawer_xy", "spoon_offset", "fork_offset", "plate_xy", "cup_xy", "mat_xy")


def test_stress_one_is_the_usual_table(monkeypatch):
    monkeypatch.delenv("TENPLACES_STRESS", raising=False)
    for s in SEEDS:
        assert sample(s).to_dict() == sample(s, stress=1.0).to_dict()


def test_stress_widens_about_the_centre_and_keeps_placements():
    for s in SEEDS:
        base, wide = sample(s, stress=1.0), sample(s, stress=1.5)
        for k in PLACEMENTS:
            assert getattr(base, k) == getattr(wide, k)
        assert np.isclose(wide.friction - 1.0, 1.5 * (base.friction - 1.0))
        assert np.isclose(wide.plate_mass - 0.12, 1.5 * (base.plate_mass - 0.12))
        assert 0.1 <= wide.light_diffuse <= 1.0 and all(0.0 <= c <= 1.0 for c in wide.table_rgb)


def test_the_environment_sets_the_default(monkeypatch):
    monkeypatch.setenv("TENPLACES_STRESS", "1.5")
    assert sample(7).to_dict() == sample(7, stress=1.5).to_dict()

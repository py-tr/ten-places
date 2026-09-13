"""Evaluation plumbing: the confidence interval, the tuning-seed guard, and the grader's pass/fail edges."""
import sys

import mujoco
import pytest

from tenplaces.env_table import TableEpisode
from tenplaces.evaluate import wilson
from tenplaces.grader_table import grade_table


def test_wilson_matches_reported_values():
    assert wilson(10, 10) == pytest.approx((0.722, 1.0), abs=1e-3)
    assert wilson(5, 10) == pytest.approx((0.237, 0.763), abs=1e-3)
    assert wilson(17, 20) == pytest.approx((0.64, 0.948), abs=1e-3)


def test_checkpoint_selection_refuses_reporting_seeds(monkeypatch):
    import scripts.eval_skill_checkpoints as ecs

    monkeypatch.setattr(sys, "argv", ["x", "--run", "out/none/plate", "--skill", "plate", "--seeds", "0", "10"])
    with pytest.raises(SystemExit, match="tuning seeds"):
        ecs.main()


def _place_cup(ep, dx):
    """Put the cup at its target (shifted by dx metres in x), at rest on the table."""
    t = ep.params.targets()["cup"]
    adr = ep.m.jnt_qposadr[ep.m.body_jntadr[ep.m.body("cup").id]]
    ep.d.qpos[adr:adr + 2] = [t[0] + dx, t[1]]
    ep.d.qvel[:] = 0
    mujoco.mj_forward(ep.m, ep.d)


def test_cup_on_its_target_passes_and_3_cm_off_fails():
    ep = TableEpisode(3, render=False)
    _place_cup(ep, 0.0)
    assert grade_table(ep.m, ep.d, ep.params)["cup"]
    _place_cup(ep, 0.03)
    g = grade_table(ep.m, ep.d, ep.params)
    assert not g["cup"] and g["cup_err_m"] == pytest.approx(0.03, abs=2e-3)

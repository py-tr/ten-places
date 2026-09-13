"""go_home puts both arms in the state every demonstration starts a skill from: grippers opened to at least the
released opening (never closed further), then the joints back to the home pose."""
import numpy as np

from tenplaces.env_table import RELEASED, TableEpisode, go_home
from tenplaces.oracle.handoff import HOME


def displaced_episode(grip_a, grip_b):
    ep = TableEpisode(0, render=False)
    away = ep.command().astype(np.float64)
    away[0:5] += [0.3, -0.2, 0.25, -0.15, 0.4]
    away[6:11] += [-0.25, 0.2, -0.2, 0.3, -0.35]
    away[5], away[11] = grip_a, grip_b
    for _ in range(40):  # let the arms get there
        ep.step(away)
    return ep


def test_arms_return_home():
    ep = displaced_episode(0.4, 1.0)
    assert np.abs(ep.state()[0:5] - HOME).max() > 0.1
    go_home(ep, frames=20)
    for _ in range(10):  # settle on the final target
        ep.step(ep.command())
    np.testing.assert_allclose(ep.state()[0:5], HOME, atol=0.03)
    np.testing.assert_allclose(ep.state()[6:11], HOME, atol=0.03)


def test_grippers_are_released_but_never_closed_further():
    ep = displaced_episode(-0.15, 1.0)  # A still closed (on a handle), B fully open
    go_home(ep, frames=5)
    assert ep.command()[5] == np.float32(RELEASED) and ep.command()[11] == np.float32(1.0)

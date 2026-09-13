"""Takeover demos: the scripted controller finishes a skill from wherever a policy left the arms, and only that
continuation is recorded. A stand-in policy nudges arm B off its home pose; no model is loaded."""
import numpy as np

from tenplaces.env_table import record_skill_oracle
from tenplaces.oracle.handoff import HOME

B_PAN = 6  # index of b_shoulder_pan in the 12-joint vector


class Nudge:
    """Moves arm B's shoulder pan steadily away from home, holding every other joint."""

    def reset(self):
        pass

    def select_action(self, obs):
        a = obs["state"].astype(np.float64).copy()
        a[B_PAN] += 0.02
        return a


def test_takeover_records_from_where_the_policy_left_the_arm():
    frames, result = record_skill_oracle(5001, "cup", ["drawer"], policy=Nudge(), policy_frames=10)
    assert result["cup"] and result["drawer_open"] and result["error"] is None
    assert abs(frames[0]["state"][B_PAN] - HOME[0]) > 0.02  # not re-homed: recorded from the takeover state


def test_no_policy_is_the_plain_recording():
    frames, result = record_skill_oracle(5001, "cup", ["drawer"])
    assert result["cup"]
    np.testing.assert_allclose(frames[0]["state"][6:11], HOME, atol=2e-3)

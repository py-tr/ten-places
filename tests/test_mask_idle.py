"""LeRobotPolicy(mask_idle_std=...): joints the policy never saw move are fed their training mean. Uses the drawer
fine-tune whose idle arm B has near-zero spreads (skipped when that checkpoint is not on this machine)."""
from pathlib import Path

import numpy as np
import pytest

CKPT = Path(__file__).resolve().parent.parent / "out/train/chain_t1_drawer/drawer/checkpoints/010000/pretrained_model"


@pytest.mark.skipif(not CKPT.exists(), reason="needs the local drawer fine-tune checkpoint")
def test_idle_joints_are_masked_to_their_training_mean():
    from tenplaces.lerobot_policy import LeRobotPolicy

    pol = LeRobotPolicy(CKPT, device="cpu", mask_idle_std=1e-4)
    assert pol._idle is not None and not pol._idle[:6].any()  # arm A moves in drawer demos
    assert pol._idle[6:].any()  # arm B never does
    state = pol._idle_mean + 1e-3  # every joint 1 mrad off its training mean
    masked = pol.masked_state(state)
    np.testing.assert_array_equal(masked[pol._idle], pol._idle_mean[pol._idle])
    np.testing.assert_array_equal(masked[~pol._idle], state[~pol._idle])


@pytest.mark.skipif(not CKPT.exists(), reason="needs the local drawer fine-tune checkpoint")
def test_mask_is_off_by_default():
    from tenplaces.lerobot_policy import LeRobotPolicy

    pol = LeRobotPolicy(CKPT, device="cpu")
    state = np.arange(12, dtype=np.float32)
    np.testing.assert_array_equal(pol.masked_state(state), state)

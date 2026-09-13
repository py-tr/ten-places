"""Adapter: a trained LeRobot policy checkpoint as a `tenplaces.evaluate` policy (reset / select_action).

backend: "torch" (any torch device), "ov-fp32", "ov-w8" (INT8 weights; what the robot runs) or "ov-int8" (INT8 weights + activations; rejected, 7/20) (OpenVINO on the Intel CPU; only the ACT
network changes, normalisation / action queue / temporal ensembling stay LeRobot's).
Execution settings that need no retraining:
  n_action_steps  - how many actions of each predicted chunk to execute before re-planning
  temporal_coeff  - ACT temporal ensembling (re-plans every step and blends overlapping chunks)
"""
from pathlib import Path

import numpy as np
import torch

from .env import CAMERAS


class LeRobotPolicy:
    def __init__(self, pretrained_path: str | Path, device: str = "cuda", backend: str = "torch",
                 n_action_steps: int | None = None, temporal_coeff: float | None = None,
                 calib_cache: str | Path | None = None, ov_device: str = "CPU", ov_config: dict | None = None):
        """ov_config: OpenVINO compile properties for the ov-* backends (tenplaces.cores.control_config())."""
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        path = str(pretrained_path)
        if backend != "torch":
            device = "cpu"
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = device
        if temporal_coeff is not None:
            cfg.temporal_ensemble_coeff = temporal_coeff
            cfg.n_action_steps = 1
        elif n_action_steps is not None:
            cfg.n_action_steps = n_action_steps
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}})
        if backend.startswith("ov-"):
            from .ov_backend import compile_act

            self.policy.model = compile_act(self.policy, self._preprocess, backend.split("-", 1)[1], calib_cache,
                                            Path(path) / "openvino", device=ov_device, ov_config=ov_config)
        self.backend = backend

    def _preprocess(self, raw: dict) -> dict:
        return self.pre({**raw, "task": [""]})

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def select_action(self, obs: dict) -> np.ndarray:
        batch = {"observation.state": torch.from_numpy(obs["state"]).float().unsqueeze(0), "task": [obs["task"]]}
        if "env_state" in obs:  # skill one-hot for skill-conditioned policies
            batch["observation.environment_state"] = torch.from_numpy(obs["env_state"]).float().unsqueeze(0)
        for cam in CAMERAS:
            img = torch.from_numpy(obs["images"][cam]).permute(2, 0, 1).float().div_(255.0)
            batch[f"observation.images.{cam}"] = img.unsqueeze(0)
        action = self.post(self.policy.select_action(self.pre(batch)))
        return action.squeeze(0).detach().cpu().numpy()

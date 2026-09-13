"""Camera-only task-state classifier: which of the five skills are done, from the top camera image.

Replaces VLM yes/no questions for completion checks (faster and far more reliable). Labels come from
the demo structure, not from simulator state: within each oracle run (5 consecutive skill episodes),
skills before the current segment are done, and the current skill counts as done over the last
DONE_TAIL frames of its segment.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .env_table import SKILLS

N_SKILLS = len(SKILLS)
DONE_TAIL = 15  # frames (0.6 s at 25 Hz) at the end of a segment where its skill counts as done
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_model(pretrained: bool = True) -> nn.Module:
    import torchvision

    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = torchvision.models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, N_SKILLS)
    return net


def preprocess(images_uint8_nchw: torch.Tensor) -> torch.Tensor:
    return (images_uint8_nchw.float() / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def labels_from_structure(episode_index: np.ndarray, frame_index: np.ndarray) -> np.ndarray:
    """Per-frame done flags from the run/segment layout written by record_table_demos.py."""
    ep = episode_index.astype(np.int64)
    lengths = np.bincount(ep)
    skill = ep % N_SKILLS
    y = np.zeros((len(ep), N_SKILLS), dtype=np.float32)
    for k in range(N_SKILLS):
        y[:, k] = (skill > k).astype(np.float32)
    tail = frame_index >= (lengths[ep] - DONE_TAIL)
    y[np.arange(len(ep)), skill] = tail.astype(np.float32)
    return y


class OVStateClassifier:
    """OpenVINO runtime for the exported classifier: done flags from one top-camera frame (HWC uint8)."""

    def __init__(self, xml: str | Path, device: str = "CPU", threshold: float = 0.5, ov_config: dict | None = None):
        import openvino as ov

        self.compiled = ov.Core().compile_model(str(xml), device, {"PERFORMANCE_HINT": "LATENCY", **(ov_config or {})})
        self.threshold = threshold

    def probs(self, image_hwc_uint8: np.ndarray) -> np.ndarray:
        x = preprocess(torch.from_numpy(image_hwc_uint8).permute(2, 0, 1)[None]).numpy()
        logits = self.compiled([x])[0][0]
        return 1.0 / (1.0 + np.exp(-logits))

    def is_done(self, skill: str, image_hwc_uint8: np.ndarray) -> bool:
        k = [s for s, _, _ in SKILLS].index(skill)
        return bool(self.probs(image_hwc_uint8)[k] > self.threshold)

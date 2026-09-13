"""FastDataset serves the memmap cache in LeRobot's sample format and applies the dataset's image transforms
(augmentation was silently skipped before). A tiny hand-built cache and a stand-in dataset; no LeRobot I/O."""
import json
import types

import numpy as np
import torch

from tenplaces.fastdata import FastDataset

CAM = "observation.images.top"


def make_cache(tmp_path, n=4):
    img = np.lib.format.open_memmap(tmp_path / "observation__images__top.npy", mode="w+", dtype=np.uint8,
                                    shape=(n, 3, 2, 2))
    img[:] = 255
    img.flush()
    np.savez(tmp_path / "vectors.npz", action=np.arange(n * 2, dtype=np.float32).reshape(n, 2),
             timestamp=np.zeros(n, np.float32), frame_index=np.arange(n), episode_index=np.zeros(n, np.int64),
             index=np.arange(n), task_index=np.zeros(n, np.int64))
    (tmp_path / "cache_meta.json").write_text(json.dumps({"n": n, "cams": [CAM], "vec_keys": ["action"],
                                                          "shapes": {CAM: [n, 3, 2, 2]}}))


class Inner:
    def __init__(self, n, image_transforms=None):
        self.n, self.image_transforms = n, image_transforms
        self.meta = types.SimpleNamespace(episodes={"dataset_from_index": [0], "dataset_to_index": [n]},
                                          tasks=types.SimpleNamespace(index=["set the table"]))

    def __len__(self):
        return self.n


def test_sample_format(tmp_path):
    make_cache(tmp_path)
    item = FastDataset(Inner(4), tmp_path, {"action": [0, 1, 2]})[1]
    assert torch.allclose(item[CAM], torch.ones(3, 2, 2))
    assert item["action"].tolist() == [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]
    assert item["action_is_pad"].tolist() == [False, False, False]
    assert item["task"] == "set the table"


def test_chunk_past_the_episode_end_is_padded(tmp_path):
    make_cache(tmp_path)
    item = FastDataset(Inner(4), tmp_path, {"action": [0, 1, 2]})[3]
    assert item["action_is_pad"].tolist() == [False, True, True]


def test_image_transforms_are_applied(tmp_path):
    make_cache(tmp_path)
    item = FastDataset(Inner(4, image_transforms=lambda x: x * 0.5), tmp_path, None)[0]
    assert torch.allclose(item[CAM], torch.full((3, 2, 2), 0.5))

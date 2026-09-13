"""Fast training reads for a LeRobot image dataset: decode every PNG once into uint8 memmaps.

LeRobot 0.6 reads image datasets through Hugging Face `datasets`, decoding 3 PNGs per frame on every
sample; on Windows that kept the GPU ~70% idle. `build_cache` decodes once; `FastDataset` wraps the
original dataset (so metadata, stats, sampler and episode bookkeeping stay LeRobot's own) and serves
samples in exactly the format `DatasetReader.get_item` produces, including delta-index chunks and
their `<key>_is_pad` masks.
"""
import json
from pathlib import Path

import numpy as np
import torch

SCALARS = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def build_cache(dataset, cache_dir: str | Path, batch: int = 512) -> Path:
    """Decode `dataset` (a LeRobotDataset without delta timestamps) into `cache_dir`."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    hf = dataset.hf_dataset
    n = len(hf)
    cams = list(dataset.meta.camera_keys)
    vec_keys = [k for k, f in dataset.meta.features.items() if f["dtype"] in ("float32", "float64") and k not in SCALARS]
    shapes = {}
    for cam in cams:
        c, h, w = hf[0][cam].shape
        shapes[cam] = (n, c, h, w)
    mm = {cam: np.lib.format.open_memmap(cache / f"{_safe(cam)}.npy", mode="w+", dtype=np.uint8, shape=shapes[cam]) for cam in cams}
    vecs = {k: np.zeros((n, *dataset.meta.features[k]["shape"]), dtype=np.float32) for k in vec_keys}
    scal = {k: np.zeros(n, dtype=np.float32 if k == "timestamp" else np.int64) for k in SCALARS}
    for s in range(0, n, batch):
        rows = hf[s:s + batch]
        e = s + len(rows["index"])
        for cam in cams:
            mm[cam][s:e] = (torch.stack(rows[cam]) * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
        for k in vec_keys:
            vecs[k][s:e] = torch.stack(rows[k]).numpy()
        for k in SCALARS:
            scal[k][s:e] = torch.stack(rows[k]).numpy().reshape(-1)
        print(f"cache {e}/{n}", flush=True)
    for arr in mm.values():
        arr.flush()
    np.savez(cache / "vectors.npz", **{_safe(k): v for k, v in vecs.items()}, **scal)
    (cache / "cache_meta.json").write_text(json.dumps({"n": n, "cams": cams, "vec_keys": vec_keys, "shapes": shapes}))
    return cache


def _safe(key: str) -> str:
    return key.replace(".", "__")


class FastDataset(torch.utils.data.Dataset):
    """Drop-in replacement for a non-filtered LeRobotDataset during training."""

    def __init__(self, inner, cache_dir: str | Path, delta_indices: dict | None):
        self.inner = inner
        self.cache = Path(cache_dir)
        info = json.loads((self.cache / "cache_meta.json").read_text())
        # Episode-filtered datasets (e.g. one skill's episodes): map LeRobot's relative index to the cache's
        # absolute frame index; the cache always holds every frame of the dataset.
        a2r = getattr(inner, "absolute_to_relative_idx", None)
        if a2r is not None:
            self.rel_to_abs = np.empty(len(inner), dtype=np.int64)
            for abs_i, rel_i in a2r.items():
                self.rel_to_abs[rel_i] = abs_i
        else:
            if info["n"] != len(inner):
                raise ValueError("cache does not match the dataset")
            self.rel_to_abs = None
        self.cams, self.vec_keys = info["cams"], info["vec_keys"]
        self.delta_indices = delta_indices or {}
        eps = inner.meta.episodes
        self.ep_from = np.asarray(eps["dataset_from_index"], dtype=np.int64)
        self.ep_to = np.asarray(eps["dataset_to_index"], dtype=np.int64)
        self.task_names = list(inner.meta.tasks.index)
        self._arrays = None  # opened lazily in each worker process

    def _open(self):
        z = np.load(self.cache / "vectors.npz")
        self._arrays = {
            "img": {cam: np.load(self.cache / f"{_safe(cam)}.npy", mmap_mode="r") for cam in self.cams},
            "vec": {k: z[_safe(k)] for k in self.vec_keys},
            "scal": {k: z[k] for k in SCALARS},
        }

    def __getstate__(self):
        # Never pickle open memmaps to DataLoader workers: numpy pickles a memmap by copying its data.
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self):
        return len(self.inner)

    def __getattr__(self, name):
        # Everything the trainer reads besides samples (meta, episodes, num_frames, ...) is LeRobot's.
        if name in ("inner", "_arrays"):
            raise AttributeError(name)
        return getattr(self.inner, name)

    def __getitem__(self, idx: int) -> dict:
        if self._arrays is None:
            self._open()
        if self.rel_to_abs is not None:
            idx = int(self.rel_to_abs[idx])
        a = self._arrays
        item = {cam: torch.from_numpy(np.array(a["img"][cam][idx])).float().div_(255.0) for cam in self.cams}
        tf = getattr(self.inner, "image_transforms", None)
        if tf is not None:  # augmentation, as LeRobot's own get_item does (it was silently skipped here before)
            for cam in self.cams:
                item[cam] = tf(item[cam])
        for k in self.vec_keys:
            item[k] = torch.from_numpy(a["vec"][k][idx].copy())
        for k in SCALARS:
            item[k] = torch.tensor(a["scal"][k][idx])
        ep = int(a["scal"]["episode_index"][idx])
        abs_idx = int(a["scal"]["index"][idx])
        start, end = self.ep_from[ep], self.ep_to[ep]
        for key, deltas in self.delta_indices.items():
            q = [max(start, min(end - 1, abs_idx + d)) for d in deltas]
            item[key] = torch.from_numpy(a["vec"][key][q].copy())
            item[f"{key}_is_pad"] = torch.BoolTensor([bool(abs_idx + d < start or abs_idx + d >= end) for d in deltas])
        item["task"] = self.task_names[int(a["scal"]["task_index"][idx])]
        return item

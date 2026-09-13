"""Thin wrapper around `lerobot-train` for this project. All CLI arguments pass straight through.

1. Windows: LeRobot marks the newest checkpoint with a `checkpoints/last` symlink, which Windows
   refuses without admin rights or Developer Mode. Here it becomes a directory junction (no privilege
   needed), falling back to a `last.txt` pointer file.
2. Fast data: if the environment variable TENPLACES_CACHE points at a cache built by
   `scripts/build_cache.py`, training samples come from uint8 memmaps instead of per-sample PNG
   decoding (identical samples; verified by `scripts/build_cache.py --verify`).

    set TENPLACES_CACHE=data/handoff_v1_cache
    python scripts/train.py --dataset.repo_id=local/tenplaces_handoff --dataset.root=data/handoff_v1 \
        --policy.type=act --policy.device=cuda --policy.push_to_hub=false --output_dir=out/train/act_v1
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lerobot.common import train_utils  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.scripts import lerobot_train  # noqa: E402


def update_last_checkpoint(checkpoint_dir: Path) -> None:
    last = checkpoint_dir.parent / train_utils.LAST_CHECKPOINT_LINK
    if os.name != "nt":
        return train_utils.update_last_checkpoint(checkpoint_dir)
    if last.exists() or last.is_symlink():
        subprocess.run(["cmd", "/c", "rmdir", str(last)], check=False, capture_output=True)
    made = subprocess.run(["cmd", "/c", "mklink", "/J", str(last), str(checkpoint_dir.resolve())], capture_output=True)
    if made.returncode != 0:
        (checkpoint_dir.parent / "last.txt").write_text(checkpoint_dir.name)


_make_datasets = lerobot_train.make_train_eval_datasets


def make_train_eval_datasets(cfg):
    train, evald = _make_datasets(cfg)
    cache = os.environ.get("TENPLACES_CACHE")
    if cache:
        from tenplaces.fastdata import FastDataset

        dts = resolve_delta_timestamps(cfg.trainable_config, train.meta) or {}
        deltas = {k: [round(t * train.meta.fps) for t in v] for k, v in dts.items()}
        train = FastDataset(train, cache, deltas)
        print(f"[tenplaces] training from memmap cache {cache} (delta keys: {list(deltas)})", flush=True)
    return train, evald


lerobot_train.update_last_checkpoint = update_last_checkpoint
lerobot_train.make_train_eval_datasets = make_train_eval_datasets

if __name__ == "__main__":
    if os.environ.get("TENPLACES_AMP_DTYPE") == "bf16":
        import torch

        # With --policy.use_amp=true LeRobot autocasts to the device's default autocast dtype (fp16 + GradScaler);
        # bf16 needs no loss scaling and keeps ACT's exp(log_sigma) KL term in range. Ada GPUs support it.
        torch.set_autocast_dtype("cuda", torch.bfloat16)
    lerobot_train.main()

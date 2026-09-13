"""Build (and verify) the uint8 memmap training cache for a LeRobot image dataset.

    python scripts/build_cache.py --root data/handoff_v1 --cache data/handoff_v1_cache --verify

--verify compares random samples from FastDataset against LeRobotDataset (with the same ACT action
chunk) for exact equality, then times both through a DataLoader with the training settings.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from tenplaces.fastdata import FastDataset, build_cache  # noqa: E402


def loader_rate(ds, batch, workers, n_batches=30):
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                                     persistent_workers=False, multiprocessing_context="spawn" if workers else None)
    it = iter(dl)
    next(it)  # worker start-up
    t = time.perf_counter()
    for _ in range(n_batches):
        next(it)
    return (time.perf_counter() - t) / n_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--repo-id", default="local/tenplaces_handoff")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    if not args.skip_build:
        t = time.time()
        build_cache(LeRobotDataset(args.repo_id, root=args.root), args.cache)
        print(f"cache built in {time.time() - t:.0f} s")
    if not args.verify:
        return

    ref = LeRobotDataset(args.repo_id, root=args.root)
    fps = ref.meta.fps
    deltas = {"action": list(range(args.chunk))}
    ref = LeRobotDataset(args.repo_id, root=args.root, delta_timestamps={"action": [d / fps for d in deltas["action"]]})
    fast = FastDataset(ref, args.cache, deltas)
    g = torch.Generator().manual_seed(0)
    # Include episode boundaries so padding is exercised.
    ends = [int(e) - 1 for e in ref.meta.episodes["dataset_to_index"][:20]]
    idx = torch.randint(0, len(ref), (180,), generator=g).tolist() + ends
    for i in idx:
        a, b = ref[i], fast[i]
        for k in a:
            va, vb = a[k], b[k]
            same = va == vb if isinstance(va, str) else torch.equal(torch.as_tensor(va), torch.as_tensor(vb))
            if not same:
                sys.exit(f"MISMATCH at index {i}, key {k}: {va if isinstance(va, str) else torch.as_tensor(va).flatten()[:6]} vs "
                         f"{vb if isinstance(vb, str) else torch.as_tensor(vb).flatten()[:6]}")
        if set(a) != set(b):
            sys.exit(f"KEY MISMATCH at {i}: only-ref={set(a) - set(b)} only-fast={set(b) - set(a)}")
    print(f"verified {len(idx)} samples (incl. {len(ends)} episode ends): identical keys and values")
    t_fast = loader_rate(fast, args.batch, args.workers)
    t_ref = loader_rate(ref, args.batch, args.workers)
    print(f"DataLoader s/batch (batch {args.batch}, {args.workers} workers): LeRobot {t_ref:.3f} | cache {t_fast:.3f} "
          f"-> {t_ref / t_fast:.1f}x faster")


if __name__ == "__main__":
    main()

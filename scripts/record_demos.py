"""Record scripted-oracle demonstrations as a local LeRobot dataset (images stored as PNG, no video codec).

    python scripts/record_demos.py --episodes 200 --start 2000 --root data/handoff_v1

Seeds: training demos use 2000+, the hand-off spike used 1000-1049, evaluation uses 0-99. Only
episodes the grader passes are kept, so the dataset never teaches a failure.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from tenplaces.env import CAMERAS, FPS, IMAGE_HW, JOINTS, TASK, record_oracle  # noqa: E402


def features():
    h, w = IMAGE_HW
    feats = {
        "observation.state": {"dtype": "float32", "shape": (len(JOINTS),), "names": list(JOINTS)},
        "action": {"dtype": "float32", "shape": (len(JOINTS),), "names": list(JOINTS)},
    }
    for cam in CAMERAS:
        feats[f"observation.images.{cam}"] = {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--root", default="data/handoff_v1")
    ap.add_argument("--repo-id", default="local/tenplaces_handoff")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    if root.exists():
        if not args.overwrite:
            sys.exit(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)

    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=FPS, features=features(), root=root,
                               robot_type="bimanual_so101_sim", use_videos=False, image_writer_threads=8)
    kept, skipped, seed, t0 = [], [], args.start, time.time()
    while len(kept) < args.episodes:
        frames, result = record_oracle(seed)
        if not result["success"]:
            skipped.append({"seed": seed, "failure": result["failure"]})
        else:
            for f in frames:
                frame = {"observation.state": f["state"], "action": f["action"], "task": TASK}
                for cam in CAMERAS:
                    frame[f"observation.images.{cam}"] = f["images"][cam]
                ds.add_frame(frame)
            ds.save_episode()
            kept.append(seed)
            if len(kept) % 10 == 0:
                print(f"{len(kept)}/{args.episodes} episodes ({time.time() - t0:.0f} s)", flush=True)
        seed += 1
    ds.finalize()
    manifest = {"repo_id": args.repo_id, "root": str(root), "fps": FPS, "task": TASK, "seeds": kept,
                "skipped_oracle_failures": skipped, "wall_seconds": round(time.time() - t0, 1)}
    (root / "tenplaces_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != "seeds"}, indent=1))


if __name__ == "__main__":
    main()

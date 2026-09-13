"""Record full-table oracle demonstrations as a LeRobot dataset, one episode per skill segment.

    python scripts/record_table_demos.py --episodes 100 --start 3000 --root data/table_v1

Each oracle run is split into its skill segments (drawer, spoon, plate, fork, cup); every segment is
saved as its own episode with its own instruction as the task string. tenplaces_manifest.json lists,
per skill, the dataset episode indices, so one policy per skill can be trained with
--dataset.episodes=[...], or one policy on everything. Only runs where all five sub-tasks pass are kept.
Seeds: table demos use 3000+, the table spike 1000+, table evaluation 0-99.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from tenplaces.env import CAMERAS, FPS, JOINTS  # noqa: E402
from tenplaces.env_table import IMAGE_HW, SKILLS, record_table_oracle  # noqa: E402


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
    ap.add_argument("--episodes", type=int, default=100, help="full oracle runs to keep")
    ap.add_argument("--start", type=int, default=3000)
    ap.add_argument("--root", default="data/table_v1")
    ap.add_argument("--repo-id", default="local/tenplaces_table")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    if root.exists():
        if not args.overwrite:
            sys.exit(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)

    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=FPS, features=features(), root=root,
                               robot_type="bimanual_so101_sim", use_videos=False, image_writer_threads=8)
    by_skill = {skill: [] for skill, _, _ in SKILLS}
    kept, skipped, seed, ep_index, t0 = [], [], args.start, 0, time.time()
    while len(kept) < args.episodes:
        frames, segments, result = record_table_oracle(seed)
        if not result["success"] or len(segments) != len(SKILLS):
            skipped.append({"seed": seed, "failed": result["failed"], "error": result["error"]})
        else:
            for skill, text, a, b in segments:
                for f in frames[a:b]:
                    frame = {"observation.state": f["state"], "action": f["action"], "task": text}
                    for cam in CAMERAS:
                        frame[f"observation.images.{cam}"] = f["images"][cam]
                    ds.add_frame(frame)
                ds.save_episode()
                by_skill[skill].append(ep_index)
                ep_index += 1
            kept.append(seed)
            if len(kept) % 5 == 0:
                print(f"{len(kept)}/{args.episodes} runs ({ep_index} segments, {time.time() - t0:.0f} s)", flush=True)
        seed += 1
    ds.finalize()
    manifest = {"repo_id": args.repo_id, "root": str(root), "fps": FPS, "image_hw": IMAGE_HW, "seeds": kept,
                "skills": {s: {"instruction": t, "episodes": by_skill[s]} for s, _, t in SKILLS},
                "skipped_oracle_failures": skipped, "wall_seconds": round(time.time() - t0, 1)}
    (root / "tenplaces_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("seeds", "skills")}, indent=1))


if __name__ == "__main__":
    main()

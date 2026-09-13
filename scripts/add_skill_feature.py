"""Add a per-frame skill one-hot (observation.environment_state) to a table dataset, so one ACT policy can
be conditioned on the commanded sub-task (ACT feeds environment_state into its encoder as a token).

    python scripts/add_skill_feature.py --root data/table_v1 --out data/table_v1_skill
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lerobot.datasets.dataset_tools import add_features  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from tenplaces.env_table import SKILLS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default="local/tenplaces_table")
    args = ap.parse_args()
    ds = LeRobotDataset(args.repo_id, root=args.root)
    text_to_skill = {text: i for i, (_, _, text) in enumerate(SKILLS)}
    task_names = list(ds.meta.tasks.index)
    task_to_skill = np.array([text_to_skill[t] for t in task_names])
    task_index = np.asarray(ds.hf_dataset.data.column("task_index").to_numpy()).reshape(-1)
    onehot = np.eye(len(SKILLS), dtype=np.float32)[task_to_skill[task_index]]
    info = {"dtype": "float32", "shape": [len(SKILLS)], "names": [s for s, _, _ in SKILLS]}
    add_features(ds, {"observation.environment_state": (onehot, info)}, output_dir=args.out, repo_id=args.repo_id)
    manifest = Path(args.root) / "tenplaces_manifest.json"
    if manifest.exists():
        (Path(args.out) / "tenplaces_manifest.json").write_text(manifest.read_text())
    print(json.dumps({"frames": int(len(onehot)), "per_skill_frames": onehot.sum(0).astype(int).tolist()}))


if __name__ == "__main__":
    main()

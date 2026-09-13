"""Evaluate a trained LeRobot checkpoint in closed loop over a seed range.

    python scripts/eval_policy.py --checkpoint out/train/act_handoff_v1/checkpoints/last/pretrained_model \
        --seeds 0 10 --videos 2 --label act_v1

Evaluation seeds 0-99 are never used for training (demos use 2000+) or development (1000+).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.evaluate import evaluate  # noqa: E402
from tenplaces.lerobot_policy import LeRobotPolicy  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 10], metavar=("FIRST", "STOP"))
    ap.add_argument("--videos", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--label", default="policy")
    args = ap.parse_args()
    policy = LeRobotPolicy(args.checkpoint, device=args.device)
    summary = evaluate(policy, range(*args.seeds), OUT / "eval", videos=args.videos, label=args.label)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

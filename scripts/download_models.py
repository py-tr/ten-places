"""Fetch the weights a clean clone needs, into the paths every script expects:
the trained skill policies (with their OpenVINO IR) and the camera classifier from the project's model repo, and the
planner from Intel's public OpenVINO/Qwen3-VL-4B-Instruct-int4-ov (~4.4 GB in total).

    make models HF_SKILLS_REPO=<user>/<repo>
    python scripts/download_models.py --skills-repo <user>/<repo> [--skip-planner]
"""
import argparse
import os
import sys

PLANNER_REPO = "OpenVINO/Qwen3-VL-4B-Instruct-int4-ov"
PLANNER_DIR = "models/Qwen3-VL-4B-Instruct-int4-ov"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills-repo", default=os.environ.get("HF_SKILLS_REPO"),
                    help="the model repo made by scripts/upload_models.py (or set HF_SKILLS_REPO)")
    ap.add_argument("--skip-planner", action="store_true")
    args = ap.parse_args()
    if not args.skills_repo:
        sys.exit("set --skills-repo or HF_SKILLS_REPO to the project's Hugging Face model repo")
    from huggingface_hub import snapshot_download

    # The repo mirrors this checkout's layout (out/train/.../pretrained_model, models/state_classifier_v3).
    print(f"skills + classifier: {snapshot_download(args.skills_repo, local_dir='.')}", flush=True)
    if not args.skip_planner:
        print(f"planner: {snapshot_download(PLANNER_REPO, local_dir=PLANNER_DIR)}", flush=True)


if __name__ == "__main__":
    main()

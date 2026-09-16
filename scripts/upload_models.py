"""Publish the trained weights a clean clone needs (once, logged in with `hf auth login`).

    python scripts/upload_models.py --repo py-tr/ten-places --dry-run     # list what would be uploaded
    python scripts/upload_models.py --repo py-tr/ten-places               # create the model repo and upload

Uploads the five deployed skill checkpoints (out/eval/selected_checkpoints.json, plus the cup), each with its
OpenVINO IR, and the camera classifier, at the same relative paths, so `make models` puts them back where every
script expects them. The planner is not uploaded: it is Intel's public OpenVINO/Qwen3-VL-4B-Instruct-int4-ov.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.paths import OUT  # noqa: E402

CUP = "out/train/skills_ctx/cup/checkpoints/015000/pretrained_model"  # only when the selection names no cup
CLASSIFIER = "models/state_classifier_v3"


def artefacts() -> list[str]:
    selected = json.loads((OUT / "eval" / "selected_checkpoints.json").read_text())
    return [*sorted(selected.values()), *([] if "cup" in selected else [CUP]), CLASSIFIER]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="<user>/<repo> on the Hugging Face Hub")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    paths = artefacts()
    missing = [p for p in paths if not Path(p).is_dir()]
    if missing:
        sys.exit(f"missing: {missing}")
    total = sum(f.stat().st_size for p in paths for f in Path(p).rglob("*") if f.is_file())
    print(f"{len(paths)} folders, {total / 2**30:.2f} GB -> {args.repo}")
    for p in paths:
        print(f"  {p}")
    if args.dry_run:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    for p in paths:
        api.upload_folder(repo_id=args.repo, folder_path=p, path_in_repo=p, commit_message=f"Add {p}")
        print(f"uploaded {p}", flush=True)


if __name__ == "__main__":
    main()

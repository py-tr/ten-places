"""Merge demo sets into one LeRobot dataset + one tenplaces manifest, so one fine-tune can use them all.

    python scripts/merge_demo_sets.py --roots data/table_chain_spoon data/table_spoon_bands --out data/table_spoon_t2

scripts/train_skills.py takes one dataset and picks each skill's episodes from its tenplaces_manifest.json.
LeRobot's aggregate_datasets() concatenates the datasets in the order given; the manifests are merged the same
way, each set's episode indices shifted past the sets before it.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MANIFEST = "tenplaces_manifest.json"


def merged_manifest(manifests: list[dict], root: str, repo_id: str) -> dict:
    """One manifest for datasets concatenated in this order: per-skill episode lists shifted by the number of
    episodes in the sets before, the per-episode records kept with their source set."""
    skills, episodes, offset = {}, [], 0
    for i, m in enumerate(manifests):
        n = max((e for s in m["skills"].values() for e in s["episodes"]), default=-1) + 1
        for skill, entry in m["skills"].items():
            merged = skills.setdefault(skill, {"instruction": entry["instruction"], "episodes": []})
            merged["episodes"] += [offset + e for e in entry["episodes"]]
        for rec in m.get("episodes", []) if isinstance(m.get("episodes"), list) else []:
            episodes.append({**rec, "source": m.get("root", f"set{i}"),
                             **({"episode": offset + rec["episode"]} if "episode" in rec else {})})
        offset += n
    return {"repo_id": repo_id, "root": root, "fps": manifests[0].get("fps"), "image_hw": manifests[0].get("image_hw"),
            "merged_from": [m.get("root") for m in manifests], "skills": skills, "episodes": episodes,
            "total_episodes": offset}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="dataset roots, in the order to concatenate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default="local/tenplaces_table")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists")
    manifests = [json.loads((Path(r) / MANIFEST).read_text()) for r in args.roots]
    from lerobot.datasets.aggregate import aggregate_datasets

    aggregate_datasets(repo_ids=[args.repo_id] * len(args.roots), aggr_repo_id=args.repo_id,
                       roots=[Path(r) for r in args.roots], aggr_root=out)
    m = merged_manifest(manifests, str(out), args.repo_id)
    (out / MANIFEST).write_text(json.dumps(m, indent=1, default=str))
    print(json.dumps({"out": str(out), "episodes": m["total_episodes"],
                      "per_skill": {s: len(v["episodes"]) for s, v in m["skills"].items()}}))


if __name__ == "__main__":
    main()

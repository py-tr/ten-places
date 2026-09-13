"""Fine-tune with the parent policy's normalisation: copy the parent dataset's action and observation.state
statistics into a new demo set's meta/stats.json (the set's own statistics are kept as stats.own.json).

    python scripts/use_parent_stats.py --data data/table_chain_drawer --parent-data data/table_v1_skill

Why: in a demo set where one arm never moves (drawer demos: only arm A pulls), that arm's joints have a spread of
exactly 0, and MEAN_STD normalisation then divides its tiny numerical wobble by ~eps — the fine-tuned policy sees
enormous inputs. Tuning seeds 100-149: the drawer fine-tuned on such a set opened 0/50 against 44/50 for its
parent. The parent's statistics are also the ones its weights were trained with.
"""
import argparse
import json
import shutil
from pathlib import Path

KEYS = ("action", "observation.state")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the new demo set (its meta/stats.json is rewritten)")
    ap.add_argument("--parent-data", required=True, help="the dataset the parent checkpoint was trained on")
    args = ap.parse_args()
    stats, own = Path(args.data) / "meta" / "stats.json", Path(args.data) / "meta" / "stats.own.json"
    if not own.exists():
        shutil.copy(stats, own)
    s = json.loads(own.read_text())
    parent = json.loads((Path(args.parent_data) / "meta" / "stats.json").read_text())
    for k in KEYS:
        s[k] = parent[k]
    stats.write_text(json.dumps(s, indent=4))
    print(f"{stats}: {', '.join(KEYS)} statistics from {args.parent_data} (own statistics kept in {own.name})")


if __name__ == "__main__":
    main()

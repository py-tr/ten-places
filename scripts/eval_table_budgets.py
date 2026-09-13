"""Paired full-table comparison of time budgets per skill, on the same tuning seeds, in parallel workers.

    python scripts/eval_table_budgets.py --seeds 100 110 --long spoon=450 fork=500 --workers 5

Why: the fork needs ~300-330 of its 360 frames even alone; inside the chain a skill that runs out of time is cut
off mid-hand-off, the next skill's policy takes over, and the object is dropped. Budgets are a cap: the camera
classifier still ends a skill as soon as it is done. Tuning seeds only (100-149).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.evaluate import wilson  # noqa: E402
from tenplaces.evaluate_table import DEFAULT_BUDGETS  # noqa: E402
from tenplaces.parallel_eval import run_table_parallel  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402
from tenplaces.skill_policies import EXEC_SETTINGS, load_selected  # noqa: E402

KEYS = ["drawer_open", "spoon", "plate", "fork", "cup"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=[100, 110], metavar=("FIRST", "STOP"))
    ap.add_argument("--long", nargs="+", default=["spoon=450", "fork=500"], metavar="SKILL=FRAMES")
    ap.add_argument("--runs", nargs="+", default=["out/train/skills_v1", "out/train/skills_v2", "out/train/skills_ctx"])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--classifier", default="models/state_classifier_v3/state_classifier.xml")
    args = ap.parse_args()
    if args.seeds[0] < 100:
        sys.exit("comparisons use tuning seeds (100-149); 0-49 are for the final report")
    long = {**DEFAULT_BUDGETS, **{k: int(v) for k, v in (s.split("=") for s in args.long)}}
    exec_settings = json.loads(EXEC_SETTINGS.read_text())
    spec = {"kind": "skills", "runs": [r for r in args.runs if Path(r).is_dir()], "exec_settings": exec_settings,
            "checkpoints": load_selected(), "kwargs": {"device": "cuda", "n_action_steps": 10}}
    seeds = range(*args.seeds)
    results = {}
    for name, budgets in (("default", dict(DEFAULT_BUDGETS)), ("long", long)):
        rows = run_table_parallel(spec, seeds, workers=args.workers, classifier_xml=args.classifier, budgets=budgets)
        results[name] = rows
        k = sum(r["success"] for r in rows)
        print(json.dumps({"budgets": name, "full": f"{k}/{len(rows)}", "per_step": {s: sum(bool(r[s]) for r in rows) for s in KEYS},
                          "mean_steps": sum(r["subtasks_done"] for r in rows) / len(rows)}), flush=True)
    lines = [f"# Full table: default vs longer budgets (seeds {args.seeds[0]}-{args.seeds[1] - 1})", "",
             f"default {DEFAULT_BUDGETS}; long {long}", "",
             "| budgets | full tables | 95% CI | mean steps | " + " | ".join(KEYS) + " |", "|---|---|---|---|" + "---|" * 5]
    for name, rows in results.items():
        k = sum(r["success"] for r in rows)
        lines.append(f"| {name} | {k}/{len(rows)} | {wilson(k, len(rows))} | "
                     f"{sum(r['subtasks_done'] for r in rows) / len(rows):.2f} | "
                     + " | ".join(str(sum(bool(r[s]) for r in rows)) for s in KEYS) + " |")
    ra, rb = results["default"], results["long"]
    better = sum(y["subtasks_done"] > x["subtasks_done"] for x, y in zip(ra, rb))
    worse = sum(y["subtasks_done"] < x["subtasks_done"] for x, y in zip(ra, rb))
    lines += ["", f"Per seed, long vs default: more steps on {better}, fewer on {worse}, equal on {len(ra) - better - worse}."]
    out = OUT / "eval" / "budgets"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "rows.json").write_text(json.dumps(results, indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Paired full-table comparison: the chained skills as they are vs. with the arms returning home between skills
(env_table.go_home), on the same tuning seeds, run in parallel workers.

    python scripts/eval_table_home.py --seeds 100 110 --workers 5

Why: each skill alone (scripted earlier skills, arms at home) scores far above the same skill inside the chain;
one suspect is the start pose — every demonstration starts a skill at home, the chain starts it wherever the
previous learned skill left the arms. Tuning seeds only (100-149); the reporting seeds (0-49) are run once, at the
end, with whatever this selects.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.evaluate import wilson  # noqa: E402
from tenplaces.parallel_eval import run_table_parallel  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402
from tenplaces.skill_policies import EXEC_SETTINGS, load_selected  # noqa: E402

KEYS = ["drawer_open", "spoon", "plate", "fork", "cup"]
BACKENDS = {  # the same policy settings as the final report's rows (scripts/final_report.py ROWS)
    "torch": {"device": "cuda", "n_action_steps": 10},
    "ov_w8": {"backend": "ov-w8", "n_action_steps": 10, "ov_config": {"INFERENCE_NUM_THREADS": 2}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=[100, 110], metavar=("FIRST", "STOP"))
    ap.add_argument("--runs", nargs="+", default=["out/train/skills_v1", "out/train/skills_v2", "out/train/skills_ctx"])
    ap.add_argument("--home-frames", type=int, nargs="+", default=[0, 20])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--classifier", default="models/state_classifier_v3/state_classifier.xml")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="torch")
    ap.add_argument("--calib-cache", default="data/table_v1_skill_cache", help="OpenVINO rows: IR conversion inputs")
    ap.add_argument("--out", default=None, help="output dir (default out/eval/home_frames)")
    args = ap.parse_args()
    if args.seeds[0] < 100:
        sys.exit("comparisons use tuning seeds (100-149); 0-49 are for the final report")
    exec_settings = json.loads(EXEC_SETTINGS.read_text())  # frozen for this run, passed to every worker
    kwargs = dict(BACKENDS[args.backend], **({"calib_cache": args.calib_cache} if args.backend != "torch" else {}))
    spec = {"kind": "skills", "runs": [r for r in args.runs if Path(r).is_dir()], "exec_settings": exec_settings,
            "checkpoints": load_selected(), "kwargs": kwargs}
    seeds = range(*args.seeds)
    out = Path(args.out) if args.out else OUT / "eval" / "home_frames"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for hf in args.home_frames:
        rows = run_table_parallel(spec, seeds, workers=args.workers, classifier_xml=args.classifier, home_frames=hf)
        results[hf] = rows
        k = sum(r["success"] for r in rows)
        print(json.dumps({"home_frames": hf, "full": f"{k}/{len(rows)}", "wilson95": wilson(k, len(rows)),
                          "per_step": {s: sum(bool(r[s]) for r in rows) for s in KEYS},
                          "mean_steps": sum(r["subtasks_done"] for r in rows) / len(rows)}), flush=True)
    lines = [f"# Full table, arms home between skills or not (seeds {args.seeds[0]}-{args.seeds[1] - 1}, "
             f"exec settings {exec_settings})", "",
             "| home frames | full tables | 95% CI | mean steps | " + " | ".join(KEYS) + " |", "|---|---|---|---|" + "---|" * 5]
    for hf, rows in results.items():
        k = sum(r["success"] for r in rows)
        lines.append(f"| {hf} | {k}/{len(rows)} | {wilson(k, len(rows))} | "
                     f"{sum(r['subtasks_done'] for r in rows) / len(rows):.2f} | "
                     + " | ".join(str(sum(bool(r[s]) for r in rows)) for s in KEYS) + " |")
    if len(results) == 2:  # paired: on how many seeds did each variant do strictly more steps?
        (a, ra), (b, rb) = results.items()
        better = sum(y["subtasks_done"] > x["subtasks_done"] for x, y in zip(ra, rb))
        worse = sum(y["subtasks_done"] < x["subtasks_done"] for x, y in zip(ra, rb))
        lines += ["", f"Per seed, home_frames={b} vs {a}: more steps on {better} seeds, fewer on {worse}, "
                      f"equal on {len(ra) - better - worse}."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "rows.json").write_text(json.dumps({str(k): v for k, v in results.items()}, indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

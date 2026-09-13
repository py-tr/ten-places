"""Evaluate per-skill ACT policies on the full table: sequencer + camera classifier (skill ends when done).

    python scripts/eval_skills.py --runs out/train/skills_v1 --seeds 0 20 --videos 2

Rows: classifier switching (PyTorch), fixed time budgets (PyTorch, for comparison), and classifier switching
on OpenVINO FP32 / INT8-weights. Table evaluation seeds 0-99 are held out from demos (3000+).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.evaluate_table import evaluate  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402
from tenplaces.skill_policies import SkillPolicies  # noqa: E402
from tenplaces.state_classifier import OVStateClassifier  # noqa: E402

ROWS = [  # (label, policy kwargs, use classifier)
    ("torch_clf", dict(device="cuda", n_action_steps=10), True),
    ("torch_budget", dict(device="cuda", n_action_steps=10), False),
    ("ov_fp32_clf", dict(backend="ov-fp32", n_action_steps=10), True),
    ("ov_w8_clf", dict(backend="ov-w8", n_action_steps=10), True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["out/train/skills_v1"],
                    help="runs dirs; for each skill the last dir that has it wins (e.g. skills_v1 skills_v2)")
    ap.add_argument("--name", default=None, help="eval output name (default: last runs dir)")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest per skill)")
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 20], metavar=("FIRST", "STOP"))
    ap.add_argument("--videos", type=int, default=2)
    ap.add_argument("--classifier", default="models/state_classifier_v3/state_classifier.xml")
    ap.add_argument("--calib-cache", default="data/table_v1_skill_cache")
    ap.add_argument("--rows", nargs="*", default=None)
    args = ap.parse_args()
    clf = OVStateClassifier(args.classifier)
    out = OUT / "eval" / (args.name or Path(args.runs[-1]).name)
    seeds = list(range(*args.seeds))
    results = []
    for label, kw, use_clf in ROWS:
        if args.rows and label not in args.rows:
            continue
        try:
            pol = SkillPolicies(args.runs, step=args.step, calib_cache=args.calib_cache, **kw)
            s = evaluate(pol, seeds, out, videos=args.videos if label == "torch_clf" else 0, label=label,
                         checker=clf.is_done if use_clf else None)
        except Exception as e:
            s = {"label": label, "error": f"{type(e).__name__}: {e}"}
        results.append(s)
        print(json.dumps(s), flush=True)
    steps = ["drawer", "spoon", "plate", "fork", "cup"]
    lines = [f"# Per-skill policies — {(args.name or Path(args.runs[-1]).name)}", "", f"Held-out seeds {seeds[0]}–{seeds[-1]}.", "",
             "| row | full table | mean sub-tasks | " + " | ".join(steps) + " | policy ms/step |", "|---|---|---|" + "---|" * 6]
    for s in results:
        if "error" in s:
            lines.append(f"| {s['label']} | ERROR {s['error'][:60]} |")
        else:
            lines.append(f"| {s['label']} | {s['full_success']}/{s['episodes']} | {s['mean_subtasks']:.2f} | "
                         + " | ".join(str(s["per_subtask"][k]) for k in steps) + f" | {s['policy_ms_mean']:.1f} |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

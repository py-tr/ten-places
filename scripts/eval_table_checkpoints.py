"""Score every checkpoint of a table run on held-out seeds, then compare execution settings and OpenVINO
precisions on the best one. Writes out/eval/<run>[_tag]/summary.md and summary.json.

    python scripts/eval_table_checkpoints.py --run out/train/act_table_v1 --calib-cache data/table_v1_skill_cache

The sequencer runs drawer -> spoon -> plate -> fork -> cup with fixed time budgets (no privileged hints);
table evaluation seeds 0-99 are held out from demos (3000+) and development (1000+).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.evaluate_table import evaluate  # noqa: E402
from tenplaces.lerobot_policy import LeRobotPolicy  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402

VARIANTS = [  # run on the best checkpoint
    ("torch_exec10", dict(backend="torch", n_action_steps=10)),
    ("torch_exec50", dict(backend="torch")),
    ("torch_ensemble", dict(backend="torch", temporal_coeff=0.01)),
    ("ov_fp32_exec10", dict(backend="ov-fp32", n_action_steps=10)),
    ("ov_w8_exec10", dict(backend="ov-w8", n_action_steps=10)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--calib-cache", default=None)
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 20], metavar=("FIRST", "STOP"))
    ap.add_argument("--min-step", type=int, default=0)
    ap.add_argument("--videos", type=int, default=2)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    run = Path(args.run)
    out = OUT / "eval" / (run.name + (f"_{args.tag}" if args.tag else ""))
    seeds = list(range(*args.seeds))
    ckpts = sorted((p for p in (run / "checkpoints").iterdir() if p.name.isdigit() and int(p.name) >= args.min_step),
                   key=lambda p: int(p.name))
    rows, t0 = [], time.time()
    for ck in ckpts:
        s = evaluate(LeRobotPolicy(ck / "pretrained_model", device="cuda", n_action_steps=10), seeds, out, videos=0,
                     label=f"step{int(ck.name)}_exec10")
        rows.append({"checkpoint": int(ck.name), **s})
    best = max(rows, key=lambda r: (r["full_success"], r["mean_subtasks"], r["checkpoint"]))
    best_ck = run / "checkpoints" / f"{best['checkpoint']:06d}" / "pretrained_model"
    variants = []
    for label, kw in VARIANTS:
        try:
            s = evaluate(LeRobotPolicy(best_ck, device="cuda", calib_cache=args.calib_cache, **kw), seeds, out,
                         videos=args.videos if label == "torch_exec10" else 0, label=f"best_{label}")
        except Exception as e:  # keep going; record the failure
            s = {"label": f"best_{label}", "error": f"{type(e).__name__}: {e}"}
        variants.append({"variant": label, **s})
    summary = {"run": str(run), "seeds": seeds, "checkpoints": rows, "best_checkpoint": best["checkpoint"],
               "variants": variants, "wall_minutes": round((time.time() - t0) / 60, 1)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    steps = ["drawer", "spoon", "plate", "fork", "cup"]
    lines = [f"# Table evaluation — {run.name}", "", f"Held-out seeds {seeds[0]}–{seeds[-1]}; sequencer with fixed budgets.", "",
             "## Checkpoints (PyTorch GPU, re-plan every 10 actions)", "",
             "| step | full table | mean sub-tasks | " + " | ".join(steps) + " |", "|---|---|---|" + "---|" * len(steps)]
    for r in rows:
        lines.append(f"| {r['checkpoint']} | {r['full_success']}/{r['episodes']} | {r['mean_subtasks']:.2f} | "
                     + " | ".join(str(r["per_subtask"][s]) for s in steps) + " |")
    lines += ["", f"## Variants on step {best['checkpoint']}", "",
              "| variant | full table | mean sub-tasks | " + " | ".join(steps) + " | policy ms/step |", "|---|---|---|" + "---|" * len(steps) + "---|"]
    for v in variants:
        if "error" in v:
            lines.append(f"| {v['variant']} | ERROR: {v['error'][:70]} |")
        else:
            lines.append(f"| {v['variant']} | {v['full_success']}/{v['episodes']} | {v['mean_subtasks']:.2f} | "
                         + " | ".join(str(v["per_subtask"][s]) for s in steps) + f" | {v['policy_ms_mean']:.1f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

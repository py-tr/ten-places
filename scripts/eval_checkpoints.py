"""Score every checkpoint of a training run, then compare execution settings and OpenVINO precisions
on the best one. Writes <out>/summary.md and summary.json.

    python scripts/eval_checkpoints.py --run out/train/act_handoff_v1 --calib-cache data/handoff_v1_cache

Evaluation seeds 0-9 are held out from demos (2000+) and development (1000+).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.evaluate import evaluate  # noqa: E402
from tenplaces.lerobot_policy import LeRobotPolicy  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402

VARIANTS = [  # (label, kwargs) run on the best checkpoint
    ("torch_exec50", dict(backend="torch")),
    ("torch_exec10", dict(backend="torch", n_action_steps=10)),
    ("torch_ensemble", dict(backend="torch", temporal_coeff=0.01)),
    ("ov_fp32_exec10", dict(backend="ov-fp32", n_action_steps=10)),
    ("ov_int8_backbone_exec10", dict(backend="ov-a8w8_backbone", n_action_steps=10)),
    ("ov_w8_exec10", dict(backend="ov-w8", n_action_steps=10)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--calib-cache", default=None)
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 10], metavar=("FIRST", "STOP"))
    ap.add_argument("--videos", type=int, default=2)
    ap.add_argument("--min-step", type=int, default=0, help="only score checkpoints at or after this step")
    ap.add_argument("--tag", default="", help="suffix for summary files, e.g. after resuming a run")
    args = ap.parse_args()
    run = Path(args.run)
    out = OUT / "eval" / (run.name + (f"_{args.tag}" if args.tag else ""))
    seeds = list(range(*args.seeds))
    ckpts = sorted((p for p in (run / "checkpoints").iterdir() if p.name.isdigit() and int(p.name) >= args.min_step),
                   key=lambda p: int(p.name))
    if not ckpts:
        sys.exit(f"no checkpoints under {run / 'checkpoints'}")

    rows, t0 = [], time.time()
    for ck in ckpts:
        s = evaluate(LeRobotPolicy(ck / "pretrained_model", device="cuda"), seeds, out, videos=0, label=f"step{int(ck.name)}_torch_exec50")
        rows.append({"checkpoint": int(ck.name), **s})
    best = max(rows, key=lambda r: (r["rate"], r["checkpoint"]))
    best_ck = run / "checkpoints" / f"{best['checkpoint']:06d}" / "pretrained_model"
    print(f"best checkpoint: step {best['checkpoint']} ({best['success']}/{best['episodes']})", flush=True)

    variants = []
    for label, kw in VARIANTS:
        try:
            pol = LeRobotPolicy(best_ck, device="cuda", calib_cache=args.calib_cache, **kw)
            s = evaluate(pol, seeds, out, videos=args.videos, label=f"best_{label}")
        except Exception as e:  # keep the overnight chain going; record the failure
            s = {"label": f"best_{label}", "error": f"{type(e).__name__}: {e}"}
        variants.append({"variant": label, **s})

    summary = {"run": str(run), "seeds": seeds, "checkpoints": rows, "best_checkpoint": best["checkpoint"],
               "variants": variants, "wall_minutes": round((time.time() - t0) / 60, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = [f"# Evaluation — {run.name}", "", f"Held-out seeds {seeds[0]}–{seeds[-1]}.", "",
             "## Checkpoints (PyTorch, GPU, execute 50 actions per chunk)", "",
             "| step | success | 95% CI |", "|---|---|---|"]
    lines += [f"| {r['checkpoint']} | {r['success']}/{r['episodes']} | {r['wilson95']} |" for r in rows]
    lines += ["", f"## Variants on step {best['checkpoint']}", "", "| variant | success | 95% CI | policy ms/step (mean) |", "|---|---|---|---|"]
    for v in variants:
        if "error" in v:
            lines.append(f"| {v['variant']} | ERROR: {v['error'][:80]} | | |")
        else:
            lines.append(f"| {v['variant']} | {v['success']}/{v['episodes']} | {v['wilson95']} | {v['policy_ms_mean']:.1f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

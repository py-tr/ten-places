"""Closed-loop score of every saved checkpoint of one skill's run, so the earliest one that holds up is kept
(offline validation loss picks imitation-learning checkpoints badly — robomimic; ACT's success keeps improving
after its loss plateaus).

    python scripts/eval_skill_checkpoints.py --run out/train/plate_t1/plate --skill plate --seeds 100 120 \
        --compare out/train/skills_ctx/plate/checkpoints/015000/pretrained_model --workers 4
    python scripts/eval_skill_checkpoints.py --run out/train/cutlery_t1/spoon --skill spoon --seeds 100 110 \
        --exec ensemble --drawer-open 0.08 0.09 0.10 --workers 5     # the drawer pulled to each distance in turn

Uses the tuning seed range (100-149), never the reporting range (0-49). Every checkpoint sees the same seeds, so
checkpoints can be compared pairwise (McNemar on the per-seed rows in the JSON). --workers > 1 runs the seeds in
parallel processes (tenplaces.parallel_eval; the same rows as a serial run). --drawer-open repeats the seeds with
the scripted drawer pulled to each given distance (the spoon trained at exactly 9 cm: 5/10 at 8 cm, 0/10 at 10).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.evaluate import wilson  # noqa: E402
from tenplaces.evaluate_skill import SKILL_NAMES, run_skill_episode  # noqa: E402
from tenplaces.lerobot_policy import LeRobotPolicy  # noqa: E402
from tenplaces.parallel_eval import run_skill_parallel  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402
from tenplaces.state_classifier import OVStateClassifier  # noqa: E402

EXEC = {"exec10": {"n_action_steps": 10}, "exec50": {"n_action_steps": 50}, "ensemble": {"temporal_coeff": 0.01}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a skill's run dir (with checkpoints/<step>/pretrained_model)")
    ap.add_argument("--skill", required=True, choices=SKILL_NAMES)
    ap.add_argument("--seeds", type=int, nargs=2, default=[100, 120], metavar=("FIRST", "STOP"))
    ap.add_argument("--exec", default="exec10", choices=sorted(EXEC))
    ap.add_argument("--before", nargs="*", default=None, help="skills done first (default: every earlier one)")
    ap.add_argument("--compare", nargs="*", default=[], help="other pretrained_model dirs to score alongside")
    ap.add_argument("--drawer-open", type=float, nargs="*", default=None, help="scripted drawer distances to test")
    ap.add_argument("--classifier", default="models/state_classifier_v3/state_classifier.xml")
    ap.add_argument("--workers", type=int, default=1, help="parallel episode workers (4 while training, 6-8 idle)")
    args = ap.parse_args()
    if args.seeds[0] < 100:
        sys.exit("checkpoint selection uses tuning seeds (100-149); 0-49 are for the final report")
    run = Path(args.run)
    ckpts = sorted((p for p in (run / "checkpoints").iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    candidates = [(f"{run.parent.name}_{p.name}", p / "pretrained_model") for p in ckpts]
    candidates += [(f"compare_{i}_{Path(c).parts[-4] if len(Path(c).parts) >= 4 else i}", Path(c))
                   for i, c in enumerate(args.compare)]
    openings = args.drawer_open or [None]
    clf = OVStateClassifier(args.classifier) if args.workers <= 1 else None
    out = OUT / "eval" / "checkpoints" / f"{run.parent.name}_{args.skill}"
    out.mkdir(parents=True, exist_ok=True)
    head = [f"at {100 * o:.0f} cm" for o in openings] if args.drawer_open else []
    lines = [f"# {args.skill}: checkpoints of {run} ({args.exec}, seeds {args.seeds[0]}-{args.seeds[1] - 1})", "",
             "| checkpoint | success | 95% CI | " + " | ".join(head + ["mean frames"]) + " |",
             "|---|---|---|" + "---|" * (len(head) + 1)]
    for label, ck in candidates:
        pol = None if args.workers > 1 else LeRobotPolicy(ck, device="cuda", **EXEC[args.exec])
        rows, per = [], []
        for opening in openings:
            if args.workers > 1:  # the same rows as a serial run, several seeds at a time (tenplaces.parallel_eval)
                r = run_skill_parallel({"kind": "lerobot", "path": str(ck), "kwargs": {"device": "cuda", **EXEC[args.exec]}},
                                       args.skill, range(*args.seeds), before=args.before, workers=args.workers,
                                       classifier_xml=args.classifier, drawer_open=opening)
            else:
                r = [run_skill_episode(pol, args.skill, s, checker=clf.is_done, before=args.before, drawer_open=opening)
                     for s in range(*args.seeds)]
            for x in r:
                x["drawer_open"] = opening
            rows += r
            per.append(f"{sum(x['success'] for x in r)}/{len(r)}")
        k = sum(r["success"] for r in rows)
        s = {"skill": args.skill, "label": label, "checkpoint": str(ck), "episodes": len(rows), "success": k,
             "rate": k / len(rows), "wilson95": wilson(k, len(rows)), "per_opening": dict(zip(map(str, openings), per)),
             "mean_frames": float(np.mean([r["frames"] for r in rows]))}
        (out / f"{label}_skill.json").write_text(json.dumps({"summary": s, "rows": rows}, indent=1))
        print(json.dumps(s), flush=True)
        lines.append(f"| {label} | {k}/{len(rows)} | {s['wilson95']} | "
                     + " | ".join((per if args.drawer_open else []) + [f"{s['mean_frames']:.0f}"]) + " |")
        (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        del pol
    print("\n".join(lines))


if __name__ == "__main__":
    main()

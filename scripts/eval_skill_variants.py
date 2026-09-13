"""Execution settings that need no retraining, scored per skill (camera classifier ends the skill).

    python scripts/eval_skill_variants.py --skills spoon --seeds 100 110 --long     # explore
    python scripts/eval_skill_variants.py --skills spoon plate fork cup --deployed --seeds 100 150 \
        --select --select-to out/eval/exec_settings_seeds100-149.json

Settings: re-plan every 10 actions (the default), execute whole 50-action chunks, ACT temporal ensembling
(re-plan every step, blend overlapping chunks); --long adds the default with a 1.7x budget (is it only slow?).
--select writes the best setting per skill (ties keep the earlier setting in the list above) to --select-to, by
default out/eval/exec_settings.json; skills not scored keep their current setting. --deployed scores the robot's
own checkpoints (out/eval/selected_checkpoints.json). Select only on the tuning seeds (100-149): the reporting
seeds are 0-49, and --select refuses them. (Until 2026-09-13 spoon, plate, fork and cup were selected on seeds
20-29, a split chosen when only seeds 0-9 were reported.)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.evaluate_skill import evaluate_skill  # noqa: E402
from tenplaces.evaluate_table import DEFAULT_BUDGETS  # noqa: E402
from tenplaces.lerobot_policy import LeRobotPolicy  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402
from tenplaces.skill_policies import EXEC_SETTINGS, SELECTED, latest_checkpoint, skill_run  # noqa: E402
from tenplaces.state_classifier import OVStateClassifier  # noqa: E402

VARIANTS = [("exec10", {"n_action_steps": 10}), ("exec50", {"n_action_steps": 50}), ("ensemble", {"temporal_coeff": 0.01})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", default=["out/train/skills_v1", "out/train/skills_v2"])
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 10], metavar=("FIRST", "STOP"))
    ap.add_argument("--classifier", default="models/state_classifier_v3/state_classifier.xml")
    ap.add_argument("--long", action="store_true", help="also run the default setting with a 1.7x budget")
    ap.add_argument("--select", action="store_true", help="write the best setting per skill to --select-to")
    ap.add_argument("--select-to", default=str(EXEC_SETTINGS))
    ap.add_argument("--deployed", action="store_true", help=f"score the checkpoints in {SELECTED.name}")
    args = ap.parse_args()
    if args.select and args.seeds[0] < 100:
        sys.exit("--select only on the tuning seeds (100-149); seeds 0-49 are the reporting seeds")
    clf = OVStateClassifier(args.classifier)
    tag = f"seeds{args.seeds[0]}-{args.seeds[1] - 1}"
    out = OUT / "eval" / "variants" / tag
    select_to = Path(args.select_to)
    chosen = json.loads(EXEC_SETTINGS.read_text()) if args.select and EXEC_SETTINGS.exists() else {}
    selected = json.loads(SELECTED.read_text()) if args.deployed and SELECTED.exists() else {}
    for skill in args.skills:
        ck = Path(selected[skill]) if skill in selected else latest_checkpoint(skill_run(args.runs, skill))
        base = DEFAULT_BUDGETS[skill]
        variants = [(label, kw, base) for label, kw in VARIANTS]
        if args.long:
            variants.append(("exec10_long", {"n_action_steps": 10}, int(base * 1.7)))
        lines = [f"# {skill} execution settings, {tag} — {ck}", "",
                 "| setting | budget | success | 95% CI | mean frames |", "|---|---|---|---|---|"]
        scores = []
        for label, kw, budget in variants:
            pol = LeRobotPolicy(ck, device="cuda", **kw)
            s = evaluate_skill(pol, skill, range(*args.seeds), out, checker=clf.is_done, label=f"{skill}_{label}",
                               budget=budget)
            print(json.dumps(s), flush=True)
            lines.append(f"| {label} | {budget} | {s['success']}/{s['episodes']} | {s['wilson95']} | {s['mean_frames']:.0f} |")
            (out / f"{skill}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            if budget == base:
                scores.append((s["success"], label, kw))
            del pol
        best = max(scores, key=lambda x: x[0])  # max() keeps the first of equal scores
        print(f"{skill}: best {best[1]} ({best[0]}/{args.seeds[1] - args.seeds[0]})", flush=True)
        if args.select:
            chosen[skill] = best[2]
            select_to.parent.mkdir(parents=True, exist_ok=True)
            select_to.write_text(json.dumps(chosen, indent=1))
    if args.select:
        print(f"wrote {select_to}: {chosen}")


if __name__ == "__main__":
    main()

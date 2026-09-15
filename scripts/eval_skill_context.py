"""Does a skill's policy still work when a verified subset plan skipped some earlier skills?

    python scripts/eval_skill_context.py --skill cup --runs out/train/skills_v1 out/train/skills_v2 --seeds 0 10

The per-skill demos all come from full tables, so skill k always saw every earlier skill done. A subset
command ("just the drawer and the cup") starts the cup with the plate never moved. For every prefix the
verifier can produce (earlier skills closed under the physical prerequisites), the scripted controller
performs the prefix, the policy runs skill k with the camera classifier, and only skill k is graded.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.evaluate_skill import evaluate_skill, verified_prefixes  # noqa: E402
from tenplaces.lerobot_policy import LeRobotPolicy  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402
from tenplaces.skill_policies import latest_checkpoint, skill_run  # noqa: E402
from tenplaces.state_classifier import OVStateClassifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", required=True)
    ap.add_argument("--runs", nargs="+", default=["out/train/skills_v1", "out/train/skills_v2"])
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 10], metavar=("FIRST", "STOP"))
    ap.add_argument("--classifier", default="models/state_classifier_v3/state_classifier.xml")
    ap.add_argument("--name", default=None, help="output file name (default: the skill)")
    ap.add_argument("--deployed", action="store_true",
                    help="the robot's own policy (out/eval/selected_checkpoints.json and exec_settings.json) instead of "
                         "the run's latest checkpoint with 10-action chunks")
    ap.add_argument("--ckpt", default=None, metavar="DIR",
                    help="with --deployed: this checkpoint for the skill instead of the selected one (a candidate)")
    ap.add_argument("--prefix", default="all", choices=["all", "full"],
                    help="full: only the full-table start (every earlier skill done), as in the chain")
    ap.add_argument("--cup-shape", type=float, default=None, metavar="S",
                    help="only the cup's size varies: per seed the cup scale scene_table.sample(shape=S) gives it")
    ap.add_argument("--plate-shape", type=float, default=None, metavar="S",
                    help="only the plate's size varies: per seed the plate scale scene_table.sample(shape=S) gives it")
    ap.add_argument("--grader-ends", action="store_true",
                    help="diagnostic: the simulator's grade ends the skill instead of the camera classifier")
    args = ap.parse_args()
    cup_scales = None
    if args.cup_shape is not None:
        from tenplaces.scene_table import sample

        cup_scales = {s: sample(s, stress=1.0, shape=args.cup_shape).cup_scale for s in range(*args.seeds)}
    plate_scales = None
    if args.plate_shape is not None:
        from tenplaces.scene_table import sample

        plate_scales = {s: sample(s, stress=1.0, shape=args.plate_shape).plate_scale for s in range(*args.seeds)}
    if args.deployed:
        from tenplaces.skill_policies import SkillPolicies

        skills = SkillPolicies(args.runs, skills=[args.skill], device="cuda", n_action_steps=10,
                               checkpoints={args.skill: args.ckpt} if args.ckpt else None)
        pol, ck = skills.policies[args.skill], skills.sources[args.skill]
    else:
        ck = latest_checkpoint(skill_run(args.runs, args.skill))
        pol = LeRobotPolicy(ck, device="cuda", n_action_steps=10)
    print(f"{args.skill}: {ck}", flush=True)
    clf = OVStateClassifier(args.classifier)
    out = OUT / "eval" / "context"
    name = args.name or args.skill
    lines = [f"# {args.skill} from every verified prefix — {ck}", "",
             "| done before | success | 95% CI | mean frames |", "|---|---|---|---|"]
    prefixes = verified_prefixes(args.skill)
    for before in (prefixes[-1:] if args.prefix == "full" else prefixes):
        label = f"{name}_after_{'-'.join(before) or 'nothing'}"
        s = evaluate_skill(pol, args.skill, range(*args.seeds), out, checker=clf.is_done, label=label, before=before,
                           cup_scales=cup_scales, grader_ends=args.grader_ends, plate_scales=plate_scales)
        print(json.dumps(s), flush=True)
        lines.append(f"| {', '.join(before) or '(nothing)'} | {s['success']}/{s['episodes']} | {s['wilson95']} | "
                     f"{s['mean_frames']:.0f} |")
        (out / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

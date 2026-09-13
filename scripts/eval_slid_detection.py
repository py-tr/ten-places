"""Does the camera classifier notice a placed object that was knocked off its target?

    python scripts/eval_slid_detection.py --classifier models/state_classifier_v3/state_classifier.xml
    python scripts/eval_slid_detection.py --classifier models/state_classifier_v4/state_classifier.xml --name v4

For each seed the scripted controller performs the object's verified plan (the plate: drawer, plate), the table is
read once untouched (the classifier must say done), then the object is moved --dist metres in each of four
directions (+x, -x, +y, -y) from that same state and left to settle; the grader confirms it is off its target and
the classifier must say it is no longer done. Tuning seeds only (default 120-139).
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.env_table import TableEpisode, displace  # noqa: E402
from tenplaces.evaluate_skill import KEY, SKILL_NAMES  # noqa: E402
from tenplaces.grader_table import grade_table  # noqa: E402
from tenplaces.oracle import table  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402
from tenplaces.planner import verify  # noqa: E402
from tenplaces.state_classifier import OVStateClassifier  # noqa: E402

DIRS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", default="models/state_classifier_v3/state_classifier.xml")
    ap.add_argument("--body", default="plate", choices=["plate", "cup"])
    ap.add_argument("--seeds", type=int, nargs=2, default=[120, 140])
    ap.add_argument("--dist", type=float, default=0.07)
    ap.add_argument("--name", default="v3")
    args = ap.parse_args()
    if args.seeds[0] < 100:
        sys.exit("tuning seeds only (100-149)")
    clf = OVStateClassifier(args.classifier)
    k = SKILL_NAMES.index(args.body)
    plan = verify([args.body])[0]
    counts = {d: [0, 0] for d in DIRS}  # noticed, valid
    untouched_done, rows = 0, []
    for seed in range(*args.seeds):
        ep = TableEpisode(seed, render=True)
        table.run_plan(ep.ctl, ep.params, plan)
        ep.ctl.hold(0.5)
        p0 = float(clf.probs(ep.observation()["images"]["top"])[k])
        untouched_done += p0 > clf.threshold
        saved = (ep.d.qpos.copy(), ep.d.qvel.copy(), ep.d.act.copy(), ep.d.ctrl.copy(), ep.d.time)
        row = {"seed": seed, "untouched_p": round(p0, 3)}
        for name, (ux, uy) in DIRS.items():
            ep.d.qpos[:], ep.d.qvel[:], ep.d.act[:], ep.d.ctrl[:], ep.d.time = saved
            mujoco.mj_forward(ep.m, ep.d)
            displace(ep.m, ep.d, args.body, args.dist * ux, args.dist * uy)
            ep.ctl.hold(0.5)
            off = not grade_table(ep.m, ep.d, ep.params)[KEY[args.body]]
            p = float(clf.probs(ep.observation()["images"]["top"])[k])
            row[name] = {"p": round(p, 3), "off_target": off}
            if off:
                counts[name][1] += 1
                counts[name][0] += p <= clf.threshold
        rows.append(row)
        ep.close()
        print(json.dumps(row), flush=True)
    n = args.seeds[1] - args.seeds[0]
    out = OUT / "eval" / "slid_detection"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.body}_{args.name}_seeds{args.seeds[0]}-{args.seeds[1] - 1}"
    (out / f"{stem}.json").write_text(json.dumps({"classifier": args.classifier, "dist": args.dist, "counts": counts,
                                                  "untouched_done": untouched_done, "rows": rows}, indent=1))
    md = [f"# Knocked {args.body} noticed by the classifier — {args.name} ({args.classifier})", "",
          f"Seeds {args.seeds[0]}-{args.seeds[1] - 1}; moved {args.dist * 100:.0f} cm after a scripted placement. "
          f"Untouched table read as done: {untouched_done}/{n}.", "",
          "| direction | noticed (said not done) | off its target per the grader |", "|---|---|---|"]
    md += [f"| {d} | {c[0]}/{c[1]} | {c[1]}/{n} |" for d, c in counts.items()]
    (out / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

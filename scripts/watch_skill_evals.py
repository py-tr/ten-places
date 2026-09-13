"""Score each per-skill policy as soon as its training finishes (runs alongside scripts/train_skills.py).

    python scripts/watch_skill_evals.py --runs out/train/skills_v1 --final 20000 --seeds 0 10

For each skill, waits for checkpoints/<final>, then evaluates that skill alone from a realistic start
(scripted controller does the earlier skills) with the camera classifier ending the skill. Summaries go to
out/eval/<runs name>/<skill>_skill.json and a combined per_skill.md.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.env_table import SKILLS  # noqa: E402
from tenplaces.evaluate_skill import evaluate_skill  # noqa: E402
from tenplaces.lerobot_policy import LeRobotPolicy  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402
from tenplaces.state_classifier import OVStateClassifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="out/train/skills_v1")
    ap.add_argument("--final", type=int, default=20000)
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 10], metavar=("FIRST", "STOP"))
    ap.add_argument("--classifier", default="models/state_classifier_v3/state_classifier.xml")
    ap.add_argument("--skills", nargs="*", default=None)
    ap.add_argument("--drop-optimizer", action="store_true", help="delete each checkpoint's optimizer state once it lands")
    args = ap.parse_args()
    runs = Path(args.runs)
    out = OUT / "eval" / runs.name
    clf = OVStateClassifier(args.classifier)
    lines = ["| skill | success | 95% CI | mean frames |", "|---|---|---|---|"]
    for skill in args.skills or [s for s, _, _ in SKILLS]:
        ck = runs / skill / "checkpoints" / f"{args.final:06d}" / "pretrained_model"
        while not (ck / "model.safetensors").exists():
            time.sleep(30)
        time.sleep(20)  # let the checkpoint finish writing
        if args.drop_optimizer:  # ~400 MB per checkpoint; evaluation and fine-tuning from weights don't need it
            shutil.rmtree(ck.parent / "training_state", ignore_errors=True)
        pol = LeRobotPolicy(ck, device="cuda", n_action_steps=10)
        s = evaluate_skill(pol, skill, range(*args.seeds), out, checker=clf.is_done, label=f"{skill}_exec10_clf")
        print(json.dumps(s), flush=True)
        lines.append(f"| {skill} | {s['success']}/{s['episodes']} | {s['wilson95']} | {s['mean_frames']:.0f} |")
        (out / "per_skill.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        del pol


if __name__ == "__main__":
    main()

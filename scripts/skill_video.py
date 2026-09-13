"""Video of one skill run by its learned policy from a realistic start: the scripted controller does the earlier
skills first (as in tenplaces.evaluate_skill), then the policy runs until the camera classifier says done.
Front camera with the policy's top camera inset; graded at the end. For diagnosing failures and for the demo.

    python scripts/skill_video.py --skill plate --seed 2 --out out/video/plate_s2.mp4
"""
import argparse
import sys
from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.env_table import SKILLS, TableEpisode  # noqa: E402
from tenplaces.evaluate_skill import KEY, SKILL_NAMES  # noqa: E402
from tenplaces.evaluate_table import DEFAULT_BUDGETS  # noqa: E402
from tenplaces.grader_table import grade_table  # noqa: E402
from tenplaces.lerobot_policy import LeRobotPolicy  # noqa: E402
from tenplaces.oracle import table  # noqa: E402
from tenplaces.skill_policies import latest_checkpoint, skill_run  # noqa: E402
from tenplaces.state_classifier import OVStateClassifier  # noqa: E402

EXEC = {"exec10": {"n_action_steps": 10}, "exec50": {"n_action_steps": 50}, "ensemble": {"temporal_coeff": 0.01}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", required=True, choices=SKILL_NAMES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", nargs="+", default=["out/train/skills_v1", "out/train/skills_v2", "out/train/skills_ctx",
                                                  "out/train/skills_ctx2"])
    ap.add_argument("--exec", default="exec10", choices=sorted(EXEC))
    ap.add_argument("--before", nargs="*", default=None, help="skills done first (default: every earlier one)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ck = latest_checkpoint(skill_run([r for r in args.runs if Path(r).is_dir()], args.skill))
    print(f"{args.skill}: {ck} ({args.exec})", flush=True)
    policy = LeRobotPolicy(ck, device="cuda", **EXEC[args.exec])
    clf = OVStateClassifier("models/state_classifier_v3/state_classifier.xml")
    ep = TableEpisode(args.seed, render=True)
    before = SKILL_NAMES[:SKILL_NAMES.index(args.skill)] if args.before is None else args.before
    if before:
        table.run_plan(ep.ctl, ep.params, before)
    obs = ep.observation()
    policy.reset()
    onehot = np.zeros(len(SKILLS), np.float32)
    onehot[SKILL_NAMES.index(args.skill)] = 1.0
    text = dict((s, t) for s, _, t in SKILLS)[args.skill]
    renderer = mujoco.Renderer(ep.m, 540, 960)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(args.out, fps=25, macro_block_size=8)
    settle = None
    for i in range(DEFAULT_BUDGETS[args.skill]):
        obs["task"], obs["env_state"], obs["skill"] = text, onehot, args.skill
        obs = ep.step(np.asarray(policy.select_action(obs), dtype=np.float64))
        renderer.update_scene(ep.d, camera="front")
        frame = Image.fromarray(renderer.render())
        frame.paste(Image.fromarray(obs["images"]["top"]).resize((240, 180)), (0, 0))
        writer.append_data(np.asarray(frame))
        if settle is None and i >= 40 and i % 10 == 0 and clf.is_done(args.skill, obs["images"]["top"]):
            settle = i + 30  # the camera says done: let the policy release and retreat, then stop
        if settle is not None and i >= settle:
            break
    g = grade_table(ep.m, ep.d, ep.params)
    writer.close()
    renderer.close()
    ep.close()
    print(f"seed {args.seed}: success={g[KEY[args.skill]]} err={g.get(f'{args.skill}_err_m')} frames={i + 1} -> {args.out}")


if __name__ == "__main__":
    main()

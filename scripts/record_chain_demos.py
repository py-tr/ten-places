"""Takeover demonstrations from genuinely chained states: the LEARNED policies perform the earlier skills exactly as the
sequencer does (camera-ended or budget-ended, release + home between skills), then the scripted controller performs
the skill being recorded — optionally after k frames of that skill's own learned policy (HG-DAgger-style takeover).

    python scripts/record_chain_demos.py --skills spoon --episodes 2 --start 9000 --root <scratch>/chain_smoke --workers 1
    python scripts/record_chain_demos.py --skills drawer spoon fork --episodes 80 120 80 --start 9000 \
        --root data/table_chain_t1 --takeover-frac 0.5 --takeover-frames 10 60 --workers 4

Why: every existing demo starts a skill from the scripted controller's state (drawer at 8-10 cm, objects exactly where
the script left them, arms home). The chain starts a skill from where the learned skills left things: the drawer at
5-10 cm (stalls and over-pulls), the spoon placed by the learned policy up to 2 cm off, the plate a few mm off. The
per-skill policies score 90-100% from scripted starts and ~50% full tables in the chain.

Per skill the "prefix" is the canonical earlier skills, run by the learned policies:
  drawer  no prefix. --drawer-finish: the learned drawer policy runs its whole budget first (release + home), and the
          scripted controller then finishes the pull to a random opening in --drawer-open — demos of completing a
          stalled pull, and of leaving an already-open drawer alone (kept only when the learned pull ended < --drawer-min).
  spoon   learned drawer -> release + home -> [k frames learned spoon] -> scripted spoon
  plate   learned drawer, spoon -> ... -> scripted plate
  fork    learned drawer, spoon, plate -> ... -> scripted fork
  cup     learned drawer, spoon, plate, fork -> ... -> scripted cup
Episodes whose learned prefix left a state the scripted controller cannot finish (grade fails, IK error) are dropped
and listed in the manifest; --require-prefix drops episodes where a prefix skill's own grade failed (default: keep them
when the recorded skill still succeeds — e.g. the fork after a spoon that was never placed is a state the chain reaches).
Seeds 9000+ (table demos 3000+, classifier 4000+, context 5000+, plate_t1 8000+; evaluation 0-149).
"""
import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from tenplaces.control import GRIP_CLOSED, IKFailure  # noqa: E402
from tenplaces.env import CAMERAS, FPS  # noqa: E402
from tenplaces.env_table import IMAGE_HW, SKILLS, TableEpisode, go_home  # noqa: E402
from tenplaces.evaluate_skill import KEY, SKILL_NAMES  # noqa: E402
from tenplaces.evaluate_table import CAMERA_ENDS, DEFAULT_BUDGETS, SETTLE  # noqa: E402
from tenplaces.grader_table import grade_table  # noqa: E402
from tenplaces.oracle import table  # noqa: E402
from tenplaces.oracle.handoff import HOME  # noqa: E402
from tenplaces.oracle.table import HANDLE_GRIP_Z, PARTIAL_OPEN, _line  # noqa: E402

_W = {}


def finish_pull(ctl, target_open: float):
    """Scripted completion of a (possibly stalled) pull: grip the post where it is now and pull the tray to
    `target_open` (absolute opening), then release and retreat. Mirrors oracle.table.open_drawer's loop, but with the
    remaining distance instead of a fixed pull, so it is valid from any partially open state."""
    d = ctl.d
    jaw = np.array([-1.0, 0.0])
    slide = ctl.m.joint("drawer_slide").qposadr[0]
    for _ in range(3):
        remaining = target_open - float(-d.qpos[slide])
        if remaining <= 0.005:
            break
        h = d.site("drawer_grip").xpos.copy()
        h[:2] += 0.0035 * jaw
        end_x = h[0] - remaining
        ctl.move({"a_": ctl.solve("a_", [h[0], h[1], HANDLE_GRIP_Z + 0.05], jaw)}, {"a_": PARTIAL_OPEN}, 1.0)
        ctl.move({"a_": ctl.solve("a_", [h[0], h[1], HANDLE_GRIP_Z], jaw)}, duration=0.6)
        ctl.move({}, {"a_": GRIP_CLOSED}, 0.4)
        ctl.hold(0.2)
        _line(ctl, "a_", [h[0], h[1], HANDLE_GRIP_Z], [end_x, h[1], HANDLE_GRIP_Z], jaw, n=6, duration=max(0.6, 1.8 * remaining / 0.09))
        ctl.move({}, {"a_": PARTIAL_OPEN}, 0.3)
        ctl.move({"a_": ctl.solve("a_", [end_x, h[1], HANDLE_GRIP_Z + 0.05], jaw)}, duration=0.5)
    ctl.move({"a_": HOME}, duration=0.8)


def init_worker(cfg):
    import torch

    from tenplaces.skill_policies import SkillPolicies
    from tenplaces.state_classifier import OVStateClassifier

    torch.set_num_threads(1)
    _W["policy"] = SkillPolicies(cfg["runs"], device="cuda", n_action_steps=10)
    _W["clf"] = OVStateClassifier(cfg["classifier"], ov_config={"INFERENCE_NUM_THREADS": 1})
    _W["cfg"] = cfg


def run_learned(ep, obs, skill: str):
    """One skill by its learned policy with the sequencer's ending rules (evaluate_table.run_episode)."""
    policy, clf = _W["policy"], _W["clf"]
    policy.reset()
    onehot = np.zeros(len(SKILLS), dtype=np.float32)
    onehot[SKILL_NAMES.index(skill)] = 1.0
    text = dict((s, t) for s, _, t in SKILLS)[skill]
    for i in range(DEFAULT_BUDGETS[skill]):
        obs["task"], obs["env_state"], obs["skill"] = text, onehot, skill
        obs = ep.step(np.asarray(policy.select_action(obs), dtype=np.float64))
        if CAMERA_ENDS[skill] and i >= 40 and i % 10 == 0 and clf.is_done(skill, obs["images"]["top"]):
            for _ in range(SETTLE[skill]):
                obs["task"], obs["env_state"], obs["skill"] = text, onehot, skill
                obs = ep.step(np.asarray(policy.select_action(obs), dtype=np.float64))
            break
    return obs


def record_one(job):
    seed, skill, k, opening, finish = job  # finish: a drawer demo that starts after the learned pull
    frames, cmds, recording = [], [], [False]
    ep = None

    def on_step(m, d):
        if recording[0] and ep.ctl.steps % ep.substeps == 0:
            frames.append(ep.observation())
            cmds.append(ep.command())

    ep = TableEpisode(seed, render=True, on_step=on_step)
    slide = ep.m.joint("drawer_slide").qposadr[0]
    prefix = SKILL_NAMES[:SKILL_NAMES.index(skill)]
    info = {"seed": seed, "skill": skill, "takeover_frames": k, "opening": opening, "prefix": prefix, "finish": finish}
    error = None
    try:
        obs = ep.observation()
        for i, s in enumerate(prefix):
            if i:
                obs = go_home(ep, 20)
            obs = run_learned(ep, obs, s)
        g0 = grade_table(ep.m, ep.d, ep.params)
        info["prefix_grade"] = {s: bool(g0[KEY[s]]) for s in prefix}
        if skill == "drawer" and finish:
            obs = run_learned(ep, obs, "drawer")  # the learned pull, then the scripted finish is the demo
            info["learned_drawer_cm"] = round(float(-ep.d.qpos[slide]) * 100, 2)
        if prefix or (skill == "drawer" and finish):
            obs = go_home(ep, 20)
        info["drawer_cm_at_start"] = round(float(-ep.d.qpos[slide]) * 100, 2)
        if k:
            policy = _W["policy"]
            policy.reset()
            onehot = np.zeros(len(SKILLS), dtype=np.float32)
            onehot[SKILL_NAMES.index(skill)] = 1.0
            text = dict((s, t) for s, _, t in SKILLS)[skill]
            for _ in range(k):
                obs["task"], obs["env_state"], obs["skill"] = text, onehot, skill
                obs = ep.step(np.asarray(policy.select_action(obs), dtype=np.float64))
        ep.ctl.steps = 0
        recording[0] = True
        if skill == "drawer":
            if finish:
                finish_pull(ep.ctl, opening)
            else:
                ep.params.drawer_open = opening
                table.run_plan(ep.ctl, ep.params, ["drawer"])
        else:
            table.run_plan(ep.ctl, ep.params, [skill])
    except IKFailure as e:
        error = str(e)
    g = grade_table(ep.m, ep.d, ep.params)
    ep.close()
    info.update(error=error, ok=bool(g[KEY[skill]]) and error is None, failed=g["failed"],
                drawer_cm_end=round(float(-ep.d.qpos[slide]) * 100, 2), frames=len(frames))
    actions = cmds[1:] + cmds[-1:]
    for f, a in zip(frames, actions):
        f["action"] = a
    return info, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", nargs="+", default=["spoon"])
    ap.add_argument("--episodes", nargs="+", type=int, default=[80])
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--root", default="data/table_chain_t1")
    ap.add_argument("--repo-id", default="local/tenplaces_table")
    ap.add_argument("--runs", nargs="+", default=["out/train/skills_v1", "out/train/skills_v2", "out/train/skills_ctx"])
    ap.add_argument("--classifier", default="models/state_classifier_v3/state_classifier.xml")
    ap.add_argument("--takeover-frac", type=float, default=0.5)
    ap.add_argument("--takeover-frames", type=int, nargs=2, default=[10, 60], metavar=("MIN", "MAX"))
    ap.add_argument("--drawer-open", type=float, nargs=2, default=[0.085, 0.095], metavar=("MIN", "MAX"),
                    help="scripted drawer target opening (drawer demos)")
    ap.add_argument("--drawer-finish", action="store_true", help="drawer demos start after the learned pull (stalled states)")
    ap.add_argument("--drawer-finish-frac", type=float, default=1.0,
                    help="with --drawer-finish: this fraction of the drawer episodes are finish demos, the rest plain pulls "
                         "from a closed drawer in the same dataset, so a fine-tune keeps its first pull")
    ap.add_argument("--drawer-min", type=float, default=0.074, help="keep drawer-finish demos only when the learned pull ended below this")
    ap.add_argument("--require-prefix", action="store_true", help="drop episodes where a learned prefix skill failed")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--writer-threads", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if len(args.episodes) != len(args.skills):
        sys.exit("--episodes needs one count per skill")
    if args.start < 9000:
        sys.exit("use seeds >= 9000 (evaluation seeds are 0-149, earlier demo sets 3000-8999)")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from record_context_demos import features

    root = Path(args.root)
    if root.exists():
        if not args.overwrite:
            sys.exit(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)
    cfg = {"runs": [r for r in args.runs if Path(r).is_dir()], "classifier": args.classifier, "drawer_finish": args.drawer_finish}
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=FPS, features=features(), root=root,
                               robot_type="bimanual_so101_sim", use_videos=False, image_writer_threads=args.writer_threads)
    rng = np.random.default_rng(args.start)
    text = {s: t for s, _, t in SKILLS}
    by_skill = {s: [] for s in args.skills}
    episodes, skipped, ep_index, t0 = [], [], 0, time.time()
    seed = args.start
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"), initializer=init_worker,
                             initargs=(cfg,)) as pool:
        for skill, n in zip(args.skills, args.episodes):
            onehot = np.zeros(len(SKILLS), dtype=np.float32)
            onehot[SKILL_NAMES.index(skill)] = 1.0
            kept = 0
            finish_quota = round(n * args.drawer_finish_frac) if (skill == "drawer" and args.drawer_finish) else 0
            for finish, quota in ((True, finish_quota), (False, n - finish_quota)):  # finish demos first, then plain
                got = 0
                while got < quota:
                    batch = []
                    for _ in range(max(args.workers, 1) * 2):
                        k = int(rng.integers(args.takeover_frames[0], args.takeover_frames[1] + 1)) if rng.random() < args.takeover_frac else 0
                        batch.append((seed, skill, k, float(rng.uniform(*args.drawer_open)), finish))
                        seed += 1
                    for info, frames in pool.map(record_one, batch):
                        if got >= quota:
                            break
                        drop = (not info["ok"]) or info["error"] or not frames
                        if args.require_prefix and not all(info.get("prefix_grade", {}).values()):
                            drop = True
                        if finish and info.get("learned_drawer_cm", 0) >= 100 * args.drawer_min:
                            drop = True  # already open enough: nothing to demonstrate here
                        if drop:
                            skipped.append(info)
                            continue
                        for f in frames:
                            frame = {"observation.state": f["state"], "action": f["action"],
                                     "observation.environment_state": onehot, "task": text[skill]}
                            for cam in CAMERAS:
                                frame[f"observation.images.{cam}"] = f["images"][cam]
                            ds.add_frame(frame)
                        ds.save_episode()
                        by_skill[skill].append(ep_index)
                        episodes.append({"episode": ep_index, **info})
                        ep_index += 1
                        kept += 1
                        got += 1
                    print(f"{skill} ({'finish' if finish else 'plain'}): {got}/{quota} kept, {len(skipped)} skipped, "
                          f"{time.time() - t0:.0f} s", flush=True)
    ds.finalize()
    manifest = {"repo_id": args.repo_id, "root": str(root), "fps": FPS, "image_hw": IMAGE_HW, "prefix": "learned",
                "takeover_frac": args.takeover_frac, "takeover_frames": args.takeover_frames, "drawer_finish": args.drawer_finish,
                "skills": {s: {"instruction": text[s], "episodes": by_skill[s]} for s in args.skills},
                "episodes": episodes, "skipped": skipped, "wall_seconds": round(time.time() - t0, 1)}
    (root / "tenplaces_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(json.dumps({"root": str(root), "episodes": ep_index, "frames": sum(e["frames"] for e in episodes),
                      "skipped": len(skipped), "wall_s": manifest["wall_seconds"]}))


if __name__ == "__main__":
    main()

"""Per-skill demonstrations from every start a verified plan can produce, not only the full-table order.

    python scripts/record_context_demos.py --skills cup plate fork --episodes 80 60 60 --root data/table_ctx_v1
    python scripts/record_context_demos.py --skills plate --episodes 100 --start 8000 --plate-y-max 0.085 \
        --takeover-policy out/train/skills_ctx/plate/checkpoints/015000/pretrained_model --root data/table_plate_t1

Why: the full-table demos show each skill only after every earlier one. A subset command such as "open the
drawer and put the cup out" starts the cup with the plate still on its start spot, and the cup policy trained
on full tables drops from 10/10 to 5/10 there (scripts/eval_skill_context.py). Here each episode draws a
verified prefix uniformly (tenplaces.evaluate_skill.verified_prefixes, the full-table one included), the
scripted controller performs it unrecorded, then the skill itself is recorded. Same features as
data/table_v1_skill (with the skill one-hot), so the per-skill policies fine-tune on it directly.
--plate-y-max keeps only seeds whose plate starts at most that far from the placemat's side (the learned plate
grasp flips the plate with its open jaw on those layouts: y 0.071-0.082 fail, 0.080-0.098 pass on seeds 0-9).
--takeover-policy: in a share of episodes the learned policy starts the skill for a random number of frames and
the scripted controller takes over (env_table.record_skill_oracle) — recovery demonstrations from the states
the policy actually reaches. Seeds 5000+ (table demos 3000+, classifier data 4000+, evaluation 0-149).
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from tenplaces import scene_table  # noqa: E402
from tenplaces.env import CAMERAS, FPS, JOINTS  # noqa: E402
from tenplaces.control import IKFailure  # noqa: E402
from tenplaces.env_table import IMAGE_HW, SKILLS, policy_outcome, record_skill_oracle  # noqa: E402
from tenplaces.evaluate_table import DEFAULT_BUDGETS  # noqa: E402
from tenplaces.evaluate_skill import KEY, SKILL_NAMES, verified_prefixes  # noqa: E402


def features():
    h, w = IMAGE_HW
    feats = {
        "observation.state": {"dtype": "float32", "shape": (len(JOINTS),), "names": list(JOINTS)},
        "action": {"dtype": "float32", "shape": (len(JOINTS),), "names": list(JOINTS)},
        "observation.environment_state": {"dtype": "float32", "shape": (len(SKILLS),), "names": SKILL_NAMES},
    }
    for cam in CAMERAS:
        feats[f"observation.images.{cam}"] = {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", nargs="+", default=["cup", "plate", "fork"])
    ap.add_argument("--episodes", nargs="+", type=int, default=[80, 60, 60], help="episodes to keep, per skill")
    ap.add_argument("--start", type=int, default=5000)
    ap.add_argument("--root", default="data/table_ctx_v1")
    ap.add_argument("--repo-id", default="local/tenplaces_table")
    ap.add_argument("--writer-threads", type=int, default=4)
    ap.add_argument("--plate-y-max", type=float, default=None, help="only seeds whose plate starts at y <= this")
    ap.add_argument("--takeover-policy", default=None, help="checkpoint (pretrained_model dir) that starts the skill")
    ap.add_argument("--takeover-frac", type=float, default=0.4, help="share of episodes that are takeovers")
    ap.add_argument("--takeover-frames", type=int, nargs=2, default=[10, 30], metavar=("MIN", "MAX"),
                    help="policy frames before the scripted controller takes over")
    ap.add_argument("--drawer-open", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="scripted drawer opening drawn per episode (default: the scene's 0.09 m) — a learned drawer "
                         "opens 8.4-9.3 cm, so cutlery sits a few mm from where fixed-opening demos had it")
    ap.add_argument("--displace", default=None, metavar="SKILL",
                    help="disturbance repair: the prefix also performs this skill, the object is knocked "
                         "--displace-range metres in a random direction, and the recorded skill puts it back")
    ap.add_argument("--displace-range", type=float, nargs=2, default=[0.04, 0.09], metavar=("MIN", "MAX"))
    ap.add_argument("--cup-scale", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="cup size (radius and height) drawn per episode, uniformly — the policies had seen one cup")
    ap.add_argument("--cup-trained-frac", type=float, default=0.33,
                    help="with --cup-scale: share of episodes kept at the trained size, so it cannot regress")
    ap.add_argument("--takeover-on-stall", action="store_true",
                    help="plate/cup: the takeover policy runs until it stalls (env_table.record_skill_oracle until_stall; "
                         "at most the --takeover-frames MAX); only stalled attempts are kept, so the demonstrations "
                         "start from the policy's own failed grasps")
    ap.add_argument("--takeover-temporal-coeff", type=float, default=None,
                    help="run the takeover policy with temporal ensembling (the deployed plate: 0.01) instead of 10-action chunks")
    ap.add_argument("--takeover-n-action-steps", type=int, default=10,
                    help="the takeover policy's chunk execution (the deployed fork: 50)")
    ap.add_argument("--takeover-continue", action="store_true",
                    help="the script continues the skill from the state the policy left (spoon / fork: from who holds "
                         "the utensil; oracle.table.takeover_plan) instead of starting it over")
    ap.add_argument("--release-first", action="store_true",
                    help="with a takeover policy: it runs its whole budget, then both arms let go and go home, and the "
                         "script records the skill from there — a retry's start")
    ap.add_argument("--seeds-file", default=None,
                    help="JSON lines with seed / ok (e.g. out/data_fix/fork_pass1_*.jsonl): only the seeds with ok false, "
                         "in order, instead of counting up from --start")
    ap.add_argument("--full-prefix", action="store_true", help="only the full-table start (every earlier skill done)")
    ap.add_argument("--replay-manifest", default=None,
                    help="re-record this manifest's episodes of --skills with their own seeds, prefixes, drawer openings and "
                         "takeover frames (e.g. one skill's part of a mixed set, without the other skill's frames)")
    ap.add_argument("--plate-x-max", type=float, default=None, help="only seeds whose plate starts at x <= this")
    ap.add_argument("--stop-at", default=None, metavar="HH:MM",
                    help="stop drawing seeds at this local time (next occurrence) and finalise what was kept")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    stop_at = None
    if args.stop_at:
        import datetime as dt

        now = dt.datetime.now()
        hh, mm = map(int, args.stop_at.split(":"))
        t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        stop_at = (t if t > now else t + dt.timedelta(days=1)).timestamp()
    if len(args.episodes) != len(args.skills):
        sys.exit("--episodes needs one count per skill")
    if args.displace and args.skills != [args.displace]:
        sys.exit("--displace SKILL records that one skill: pass --skills SKILL")
    root = Path(args.root)
    if root.exists():
        if not args.overwrite:
            sys.exit(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)
    policy = None
    if args.takeover_policy:
        from tenplaces.lerobot_policy import LeRobotPolicy

        policy = LeRobotPolicy(args.takeover_policy, device="cuda", n_action_steps=args.takeover_n_action_steps,
                               temporal_coeff=args.takeover_temporal_coeff)
    seed_list = None
    if args.seeds_file:
        from glob import glob

        seed_list = sorted(json.loads(line)["seed"] for p in sorted(glob(args.seeds_file)) for line in open(p)
                           if json.loads(line).get("ok") is False)
        print(f"{len(seed_list)} seeds from {args.seeds_file}", flush=True)
    replay = None
    if args.replay_manifest:
        man = json.loads(Path(args.replay_manifest).read_text())
        replay = {s: [e for e in man["episodes"] if e["skill"] == s] for s in args.skills}
        print(f"replaying {({s: len(v) for s, v in replay.items()})} episodes from {args.replay_manifest}", flush=True)

    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=FPS, features=features(), root=root,
                               robot_type="bimanual_so101_sim", use_videos=False, image_writer_threads=args.writer_threads)
    rng = np.random.default_rng(args.start)
    size_rng = np.random.default_rng([args.start, 7])  # its own stream: the prefix draws stay as without sizes
    text = {s: t for s, _, t in SKILLS}
    by_skill = {s: [] for s in args.skills}
    episodes, skipped, seed, ep_index, t0 = [], [], args.start, 0, time.time()
    for skill, n in zip(args.skills, args.episodes):
        starts = verified_prefixes(skill)
        onehot = np.zeros(len(SKILLS), dtype=np.float32)
        onehot[SKILL_NAMES.index(skill)] = 1.0
        kept, next_i, next_r = 0, 0, 0
        while kept < n:
            if stop_at is not None and time.time() >= stop_at:
                print(f"{skill}: stopped at {args.stop_at} with {kept}/{n}", flush=True)
                break
            if seed_list is not None:
                if next_i >= len(seed_list):
                    print(f"{skill}: seeds file exhausted with {kept}/{n}", flush=True)
                    break
                seed, next_i = seed_list[next_i], next_i + 1
            plate_xy = scene_table.sample(seed).plate_xy
            if ((args.plate_y_max is not None and plate_xy[1] > args.plate_y_max)
                    or (args.plate_x_max is not None and plate_xy[0] > args.plate_x_max)):
                seed += 1
                continue
            before = starts[-1] if args.full_prefix else starts[int(rng.integers(len(starts)))]
            knock = None
            if args.displace:  # the skill is done once, then knocked off its target
                before = before + [skill]
                ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(*args.displace_range)
                knock = (float(dist * np.cos(ang)), float(dist * np.sin(ang)))
            k = 0
            if policy is not None and rng.random() < args.takeover_frac:
                k = int(rng.integers(args.takeover_frames[0], args.takeover_frames[1] + 1))
                if args.release_first:
                    k = DEFAULT_BUDGETS[skill]  # the policy's whole attempt, then release + home
                if args.takeover_on_stall:
                    k = args.takeover_frames[1]  # an upper bound: the stall ends the policy's run
                    # Only tables the policy fails on without ever lifting the object: a first, unrecorded pass
                    # runs it for its whole budget (a slip-like dip also happens during grasps that succeed).
                    try:
                        g_pol, lifted_pol = policy_outcome(seed, skill, before, policy, DEFAULT_BUDGETS[skill])
                    except IKFailure as e:
                        g_pol, lifted_pol = {KEY[skill]: True, "error": str(e)}, False
                    if g_pol[KEY[skill]] or lifted_pol:
                        skipped.append({"seed": seed, "skill": skill, "before": before, "policy_ok": bool(g_pol[KEY[skill]]),
                                        "policy_lifted": lifted_pol, "error": g_pol.get("error")})
                        seed += 1
                        continue
            opening = float(rng.uniform(*args.drawer_open)) if args.drawer_open else None
            cup_scale = None
            if args.cup_scale:
                cup_scale = 1.0 if size_rng.random() < args.cup_trained_frac else float(size_rng.uniform(*args.cup_scale))
            if replay is not None:
                if next_r >= len(replay[skill]):
                    print(f"{skill}: replay done with {kept}/{n}", flush=True)
                    break
                rec, next_r = replay[skill][next_r], next_r + 1
                seed, before, k, opening = rec["seed"], rec["before"], rec.get("takeover_frames", 0), rec.get("drawer_open")
            frames, result = record_skill_oracle(seed, skill, before, policy=policy if k else None, policy_frames=k,
                                                 drawer_open=opening, displace_body=skill if knock else None,
                                                 displace_xy=knock or (0.0, 0.0), cup_scale=cup_scale,
                                                 until_stall=bool(k and args.takeover_on_stall),
                                                 continue_takeover=bool(k and args.takeover_continue),
                                                 release_first=bool(k and args.release_first))
            if k and args.takeover_on_stall:
                k = result["policy_frames_run"]
            no_stall = bool(k and args.takeover_on_stall and (not result["stalled"] or result["lifted"]))
            if no_stall or result["error"] or not result[KEY[skill]] or not all(result[KEY[b]] for b in before):
                skipped.append({"seed": seed, "skill": skill, "before": before, "takeover_frames": k,
                                "cup_scale": cup_scale, "failed": result["failed"], "error": result["error"],
                                "stalled": result["stalled"], "lifted": result["lifted"], "no_stall": no_stall})
            else:
                for f in frames:
                    frame = {"observation.state": f["state"], "action": f["action"],
                             "observation.environment_state": onehot, "task": text[skill]}
                    for cam in CAMERAS:
                        frame[f"observation.images.{cam}"] = f["images"][cam]
                    ds.add_frame(frame)
                ds.save_episode()
                by_skill[skill].append(ep_index)
                episodes.append({"episode": ep_index, "skill": skill, "before": before, "seed": seed,
                                 "frames": len(frames), "takeover_frames": k, "drawer_open": opening,
                                 "displaced_xy": knock, "cup_scale": cup_scale, "stalled": result["stalled"]})
                ep_index += 1
                kept += 1
                if kept % 10 == 0:
                    print(f"{skill}: {kept}/{n} ({ep_index} episodes, {len(skipped)} skipped, "
                          f"{sum(e['takeover_frames'] > 0 for e in episodes)} takeovers, {time.time() - t0:.0f} s)",
                          flush=True)
            seed += 1
    ds.finalize()
    manifest = {"repo_id": args.repo_id, "root": str(root), "fps": FPS, "image_hw": IMAGE_HW,
                "plate_y_max": args.plate_y_max, "plate_x_max": args.plate_x_max, "takeover_policy": args.takeover_policy,
                "takeover_on_stall": args.takeover_on_stall, "takeover_temporal_coeff": args.takeover_temporal_coeff,
                "takeover_n_action_steps": args.takeover_n_action_steps, "takeover_continue": args.takeover_continue,
                "release_first": args.release_first, "seeds_file": args.seeds_file, "full_prefix": args.full_prefix,
                "skills": {s: {"instruction": text[s], "episodes": by_skill[s]} for s in args.skills},
                "episodes": episodes, "skipped_oracle_failures": skipped, "wall_seconds": round(time.time() - t0, 1)}
    (root / "tenplaces_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k in ("root", "wall_seconds")} |
                     {"episodes": ep_index, "frames": sum(e["frames"] for e in episodes), "skipped": len(skipped),
                      "takeovers": sum(e["takeover_frames"] > 0 for e in episodes)}))


if __name__ == "__main__":
    main()

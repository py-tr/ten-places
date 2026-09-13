"""Instrumented full-table chain on tuning seeds: the same sequencer as tenplaces.evaluate_table.run_episode (release +
home between skills, camera classifier ends a skill or its budget does), but every hand-over is recorded and a few
execution-level levers can be switched on without touching tenplaces/.

    python scripts/eval_table_chain.py --seeds 100 150 --workers 6 --name baseline --oracle-ref --frames
    python scripts/eval_table_chain.py --seeds 100 150 --workers 6 --name retry_drawer --retry drawer
    python scripts/eval_table_chain.py --seeds 100 150 --workers 6 --name fork_ens --exec fork=ensemble
    python scripts/eval_table_chain.py --seeds 100 150 --workers 6 --name spoon_avg \
        --avg spoon=out/train/cutlery_t1/spoon/checkpoints/007500/pretrained_model,out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model

Per seed the row holds, for every skill attempt: frames used, whether the camera or the budget ended it, the camera's
"done" probability at the end, the simulator grade of every object right after the skill (not only at the end), and
the state at the skill's start (arm joints, gripper commands, drawer opening, object poses). --oracle-ref adds the same
snapshots from the scripted controller on the same seed = the state the demonstrations start that skill from.
--frames saves a front-camera PNG at every skill start and end (out/eval/chain/<name>/frames/).

Levers (all off by default = the deployed configuration):
  --retry SKILL...      after a skill whose camera check says "not done" (budget ran out, or the drawer after its
                        budget): release + home, run the skill's policy once more (the agent's retry, in the sequencer)
  --exec SKILL=exec10|exec50|ensemble   execution setting per skill (default out/eval/exec_settings.json)
  --ckpt SKILL=DIR      checkpoint per skill (default out/eval/selected_checkpoints.json)
  --avg SKILL=DIR,DIR   average the actions of several checkpoints of one skill (each with the skill's exec setting)
  --budget SKILL=N      frame budget per skill
  --settle N            frames the policy keeps running after the camera's "done" (default 30)
  --home-hold N         frames of holding still after the arms reach home (demos hold 0.5 s = 12 frames after home)
  --release X           gripper command every arm is opened to before homing (default 0.35 as env_table.RELEASED)
  --threshold T         classifier "done" threshold (default 0.5)
Tuning seeds only (100-149); refuses seeds below 100.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.env import CAMERAS  # noqa: E402
from tenplaces.env_table import SKILLS, TableEpisode  # noqa: E402
from tenplaces.evaluate import wilson  # noqa: E402
from tenplaces.evaluate_table import CAMERA_ENDS, DEFAULT_BUDGETS  # noqa: E402
from tenplaces.grader import touching  # noqa: E402
from tenplaces.grader_table import PADS, grade_table  # noqa: E402
from tenplaces.oracle.handoff import HOME  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402

SKILL_NAMES = [s for s, _, _ in SKILLS]
SKILL_TEXT = {s: t for s, _, t in SKILLS}
KEY = {"drawer": "drawer_open", "spoon": "spoon", "plate": "plate", "fork": "fork", "cup": "cup"}
EXEC = {"exec10": {"n_action_steps": 10}, "exec50": {"n_action_steps": 50}, "ensemble": {"temporal_coeff": 0.01}}
ORACLE_MARK = {"drawer": "open_drawer", "spoon": "spoon_pick", "plate": "plate_pick", "fork": "fork_pick", "cup": "cup_pick"}
OBJECTS = ("spoon", "fork", "plate", "cup")
_W = {}


def snapshot(ep) -> dict:
    """Arm joints, gripper commands, drawer opening and object poses (evaluation-side instrumentation only)."""
    m, d = ep.m, ep.d
    slide = m.joint("drawer_slide").qposadr[0]
    q, cmd = ep.state(), ep.command()
    out = {"drawer_cm": round(float(-d.qpos[slide]) * 100, 2),
           "a_q": [round(float(x), 4) for x in q[0:5]], "b_q": [round(float(x), 4) for x in q[6:11]],
           "a_grip_q": round(float(q[5]), 3), "b_grip_q": round(float(q[11]), 3),
           "a_grip_cmd": round(float(cmd[5]), 3), "b_grip_cmd": round(float(cmd[11]), 3),
           "a_home_err": round(float(np.max(np.abs(q[0:5] - HOME))), 4),
           "b_home_err": round(float(np.max(np.abs(q[6:11] - HOME))), 4)}
    for name in OBJECTS:
        bid = m.body(name).id
        p = d.xpos[bid]
        R = d.xmat[bid].reshape(3, 3)
        out[name] = {"xy": [round(float(p[0]), 4), round(float(p[1]), 4)], "z": round(float(p[2]), 4),
                     "up": round(float(R[2, 2]), 3), "axis_x": round(float(abs(R[0, 0])), 3),
                     "held": bool(touching(m, d, name, PADS))}
    return out


class Snap(list):
    """Stands in for the oracle's `phases` list: snapshots the state at every phase mark (no oracle change needed)."""

    def __init__(self, ep):
        super().__init__()
        self.ep, self.snaps = ep, {}

    def append(self, item):
        name, _ = item
        self.snaps[name] = snapshot(self.ep)
        super().append(item)


def oracle_reference(seed: int) -> dict:
    """The scripted controller's full table on this seed: the state at the start of every skill = what the demos start
    from (tenplaces.env_table.record_table_oracle marks the same phases)."""
    from tenplaces.oracle import table

    ep = TableEpisode(seed, render=False)
    phases = Snap(ep)
    err = None
    try:
        table.run(ep.ctl, ep.params, phases)
    except Exception as e:  # noqa: BLE001 - diagnostic only
        err = str(e)
    g = grade_table(ep.m, ep.d, ep.params)
    ep.close()
    return {"starts": {s: phases.snaps.get(ORACLE_MARK[s]) for s in SKILL_NAMES}, "final": phases.snaps.get("done"),
            "success": g["success"], "error": err}


class AvgPolicy:
    """Actions averaged over several checkpoints of the same skill (each keeps its own action queue / ensemble)."""

    def __init__(self, policies):
        self.policies = policies

    def reset(self):
        for p in self.policies:
            p.reset()

    def select_action(self, obs):
        return np.mean([np.asarray(p.select_action(obs), dtype=np.float64) for p in self.policies], axis=0)


def build(cfg: dict):
    from tenplaces.lerobot_policy import LeRobotPolicy
    from tenplaces.skill_policies import SkillPolicies

    sp = SkillPolicies(cfg["runs"], exec_settings=cfg["exec_settings"], checkpoints=cfg["checkpoints"], **cfg["kwargs"])
    for skill, paths in cfg["avg"].items():
        exec_kw = cfg["exec_settings"].get(skill, {})
        sp.policies[skill] = AvgPolicy([LeRobotPolicy(p, **{**cfg["kwargs"], **exec_kw}) for p in paths])
    return sp


def init_worker(cfg: dict):
    import torch

    from tenplaces.state_classifier import OVStateClassifier

    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    _W["cfg"] = cfg
    _W["policy"] = build(cfg)
    _W["clf"] = OVStateClassifier(cfg["classifier"], threshold=cfg["threshold"], ov_config={"INFERENCE_NUM_THREADS": 1})


def go_home(ep, frames: int, release: float, release_frames: int = 8, hold: int = 0, on_frame=None, until: int = 0,
            tol: float = 0.05):
    """env_table.go_home with the release value and an optional hold after home exposed.
    until > 0: after the interpolation keep commanding home until every arm joint is within `tol` rad of it (at most
    `until` frames); if still off after half of them, open both grippers fully (an arm hung up on the drawer handle,
    seed 104) and keep going. Returns the last observation; the extra frames are counted in ep-level stats."""
    obs = None
    cmd = ep.command().astype(np.float64)
    cmd[5], cmd[11] = max(cmd[5], release), max(cmd[11], release)
    for _ in range(release_frames):
        obs = ep.step(cmd)
        if on_frame:
            on_frame(obs)
    start = cmd.copy()
    goal = start.copy()
    goal[0:5], goal[6:11] = HOME, HOME
    for i in range(1, frames + 1):
        s = 0.5 - 0.5 * np.cos(np.pi * i / frames)
        obs = ep.step(start + (goal - start) * s)
        if on_frame:
            on_frame(obs)
    for _ in range(hold):
        obs = ep.step(goal)
        if on_frame:
            on_frame(obs)
    for i in range(until):
        q = ep.state()
        if max(np.max(np.abs(q[0:5] - HOME)), np.max(np.abs(q[6:11] - HOME))) < tol:
            break
        if i == until // 2:
            goal[5], goal[11] = 1.0, 1.0
        obs = ep.step(goal)
        if on_frame:
            on_frame(obs)
    return obs


def run_one(seed: int) -> dict:
    import mujoco

    cfg, policy, clf = _W["cfg"], _W["policy"], _W["clf"]
    budgets = {**DEFAULT_BUDGETS, **cfg["budgets"]}
    ep = TableEpisode(seed, render=True)
    obs = ep.observation()
    front = mujoco.Renderer(ep.m, 288, 512) if (cfg["frames"] or seed in cfg["videos"]) else None
    frames_dir = Path(cfg["out"]) / "frames"
    video = []

    def shot(tag):
        if front is None:
            return
        front.update_scene(ep.d, camera="front")
        img = front.render().copy()
        if cfg["frames"]:
            import imageio.v2 as iio

            frames_dir.mkdir(parents=True, exist_ok=True)
            top = obs["images"]["top"]
            canvas = np.zeros((288, 512 + 192, 3), np.uint8)
            canvas[:, :512] = img
            canvas[:144, 512:] = top
            iio.imwrite(frames_dir / f"seed{seed}_{tag}.png", canvas)

    def on_frame(o):
        if seed in cfg["videos"] and front is not None:
            front.update_scene(ep.d, camera="front")
            img = front.render().copy()
            img[:144, :192] = o["images"]["top"]
            video.append(img)

    def run_skill(skill, obs, camera_end=None):
        policy.reset()
        onehot = np.zeros(len(SKILLS), dtype=np.float32)
        onehot[SKILL_NAMES.index(skill)] = 1.0
        text = SKILL_TEXT[skill]
        ended, done_at, lat = "budget", None, []
        camera_end = CAMERA_ENDS[skill] if camera_end is None else camera_end
        for i in range(budgets[skill]):
            obs["task"], obs["env_state"], obs["skill"] = text, onehot, skill
            t = time.perf_counter()
            a = np.asarray(policy.select_action(obs), dtype=np.float64)
            lat.append(time.perf_counter() - t)
            obs = ep.step(a)
            on_frame(obs)
            if camera_end and i >= cfg["min_frames"] and i % 10 == 0 and clf.is_done(skill, obs["images"]["top"]):
                ended, done_at = "camera", i
                for _ in range(cfg["settle"]):
                    obs["task"], obs["env_state"], obs["skill"] = text, onehot, skill
                    obs = ep.step(np.asarray(policy.select_action(obs), dtype=np.float64))
                    on_frame(obs)
                break
        frames_used = (done_at + 1 + cfg["settle"]) if done_at is not None else budgets[skill]
        prob = float(clf.probs(obs["images"]["top"])[SKILL_NAMES.index(skill)])
        return obs, {"ended": ended, "done_at": done_at, "frames": frames_used, "p_done_end": round(prob, 3),
                     "cam_done_end": prob > cfg["threshold"], "policy_ms": round(1000 * float(np.mean(lat)), 1)}

    row = {"seed": seed, "skills": {}}
    for k, skill in enumerate(SKILL_NAMES):
        attempts = []
        for attempt in range(1, 3):
            if k or attempt > 1:
                # A drawer retry starts from the drawer's own demo start state: both grippers at 1.0 (every drawer demo
                # starts there); release + home leaves them at 0.35, which the drawer policy has never seen.
                rel = cfg["drawer_retry_release"] if (skill == "drawer" and attempt > 1 and cfg["drawer_retry_release"]) else cfg["release"]
                obs = go_home(ep, cfg["home_frames"], rel, hold=cfg["home_hold"], on_frame=on_frame,
                              until=cfg["home_until"])
            start = snapshot(ep)
            shot(f"{skill}{attempt}_start")
            obs, info = run_skill(skill, obs, camera_end=True if (attempt > 1 and cfg["retry_camera_end"]) else None)
            end = snapshot(ep)
            shot(f"{skill}{attempt}_end")
            g = grade_table(ep.m, ep.d, ep.params)
            info.update(attempt=attempt, start=start, end=end, ok=bool(g[KEY[skill]]),
                        grade={s: bool(g[KEY[s]]) for s in SKILL_NAMES},
                        errs={o: g[f"{o}_err_m"] for o in OBJECTS})
            attempts.append(info)
            if info["cam_done_end"] or skill not in cfg["retry"]:
                break
        row["skills"][skill] = attempts
    g = grade_table(ep.m, ep.d, ep.params)
    row.update({k: g[k] for k in g})
    row["retries"] = sum(len(a) - 1 for a in row["skills"].values())
    row["first_fail"] = next((s for s in SKILL_NAMES if not g[KEY[s]]), None)
    if cfg["oracle_ref"]:
        row["oracle"] = oracle_reference(seed)
    if video:
        import imageio.v2 as iio

        vd = Path(cfg["out"]) / "video"
        vd.mkdir(parents=True, exist_ok=True)
        iio.mimsave(vd / f"seed{seed}.mp4", video, fps=25, macro_block_size=8)
    if front is not None:
        front.close()
    ep.close()
    return row


def summarize(rows, cfg) -> str:
    n, k = len(rows), sum(r["success"] for r in rows)
    lines = [f"# {cfg['name']}: seeds {rows[0]['seed']}-{rows[-1]['seed']} ({n}), backend torch/cuda", "",
             f"levers: retry={cfg['retry']} exec={cfg['exec_settings']} ckpt={cfg['checkpoints']} avg={cfg['avg']} "
             f"budgets={cfg['budgets']} settle={cfg['settle']} home_frames={cfg['home_frames']} home_hold={cfg['home_hold']} "
             f"release={cfg['release']} threshold={cfg['threshold']} camera_ends={CAMERA_ENDS} "
             f"home_until={cfg.get('home_until', 0)} retry_camera_end={cfg.get('retry_camera_end', False)} "
             f"drawer_retry_release={cfg.get('drawer_retry_release', 0)}", "",
             f"**full tables {k}/{n}** (Wilson 95% {wilson(k, n)}), mean steps "
             f"{np.mean([r['subtasks_done'] for r in rows]):.2f}", "",
             "| step | ok | ended by camera | retries | recovered by retry |", "|---|---|---|---|---|"]
    for s in SKILL_NAMES:
        ok = sum(r[KEY[s]] for r in rows)
        cam = sum(r["skills"][s][-1]["ended"] == "camera" for r in rows)
        retried = sum(len(r["skills"][s]) > 1 for r in rows)
        recovered = sum(len(r["skills"][s]) > 1 and r["skills"][s][-1]["ok"] and not r["skills"][s][0]["ok"] for r in rows)
        lines.append(f"| {s} | {ok}/{n} | {cam} | {retried} | {recovered} |")
    ff = {}
    for r in rows:
        ff[r["first_fail"]] = ff.get(r["first_fail"], 0) + 1
    lines += ["", "first failed step: " + ", ".join(f"{k or 'none'} {v}" for k, v in sorted(ff.items(), key=lambda x: -x[1])), "",
              "| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cells = []
        for s in SKILL_NAMES:
            a = r["skills"][s][-1]
            cells.append(f"{'ok' if r[KEY[s]] else 'X'} {a['ended'][0]}{a['frames']}" + (f" p{a['p_done_end']:.2f}" if not r[KEY[s]] else ""))
        dcm = r["skills"]["spoon"][0]["start"]["drawer_cm"]
        lines.append(f"| {r['seed']} | {'PASS' if r['success'] else 'fail'} | {r['subtasks_done']} | {dcm} | " + " | ".join(cells) + f" | {r['retries']} |")
    return "\n".join(lines) + "\n"


def parse_kv(items, conv=lambda v: v):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = conv(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=[100, 150], metavar=("FIRST", "STOP"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", nargs="+", default=["out/train/skills_v1", "out/train/skills_v2", "out/train/skills_ctx"])
    ap.add_argument("--classifier", default="models/state_classifier_v3/state_classifier.xml")
    ap.add_argument("--retry", nargs="*", default=[])
    ap.add_argument("--exec", nargs="*", default=[], metavar="SKILL=exec10|exec50|ensemble")
    ap.add_argument("--ckpt", nargs="*", default=[], metavar="SKILL=DIR")
    ap.add_argument("--avg", nargs="*", default=[], metavar="SKILL=DIR,DIR")
    ap.add_argument("--budget", nargs="*", default=[], metavar="SKILL=N")
    ap.add_argument("--settle", type=int, default=30)
    ap.add_argument("--min-frames", type=int, default=40)
    ap.add_argument("--home-frames", type=int, default=20)
    ap.add_argument("--home-hold", type=int, default=0)
    ap.add_argument("--home-until", type=int, default=0,
                    help="after homing, keep commanding home until every joint is within 0.05 rad (at most N frames; "
                         "grippers opened fully half-way through) — catches an arm hung up on the drawer handle")
    ap.add_argument("--drawer-retry-release", type=float, default=0.0,
                    help="gripper command both arms are opened to before a DRAWER retry (1.0 = the drawer demos' start "
                         "state; 0 = the usual 0.35 release, which the drawer policy never saw at a start)")
    ap.add_argument("--retry-camera-end", action="store_true",
                    help="retry attempts end at the camera's 'done' + settle even for budget-ended skills (the drawer): "
                         "a re-pull stops at 7.4 cm instead of running 300 frames and over-pulling")
    ap.add_argument("--release", type=float, default=0.35)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--oracle-ref", action="store_true")
    ap.add_argument("--frames", action="store_true")
    ap.add_argument("--videos", type=int, nargs="*", default=[])
    args = ap.parse_args()
    if args.seeds[0] < 100:
        sys.exit("tuning seeds only (100-149); 0-49 are the reporting seeds")
    from tenplaces.skill_policies import load_exec_settings, load_selected

    exec_settings = load_exec_settings()
    for s, v in parse_kv(args.exec).items():
        exec_settings[s] = dict(EXEC[v])
    checkpoints = load_selected()
    checkpoints.update(parse_kv(args.ckpt))
    out = OUT / "eval" / "chain" / args.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"name": args.name, "out": str(out), "runs": [r for r in args.runs if Path(r).is_dir()],
           "exec_settings": exec_settings, "checkpoints": checkpoints, "avg": parse_kv(args.avg, lambda v: v.split(",")),
           "kwargs": {"device": "cuda", "n_action_steps": 10}, "classifier": args.classifier, "threshold": args.threshold,
           "retry": list(args.retry), "budgets": parse_kv(args.budget, int), "settle": args.settle,
           "min_frames": args.min_frames, "home_frames": args.home_frames, "home_hold": args.home_hold,
           "release": args.release, "oracle_ref": args.oracle_ref, "frames": args.frames, "videos": list(args.videos),
           "home_until": args.home_until, "retry_camera_end": args.retry_camera_end,
           "drawer_retry_release": args.drawer_retry_release,
           "camera_ends": dict(CAMERA_ENDS), "env_camera_ends": os.environ.get("TENPLACES_CAMERA_ENDS", "")}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    seeds = list(range(*args.seeds))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"), initializer=init_worker,
                             initargs=(cfg,)) as pool:
        rows = list(pool.map(run_one, seeds))
    rows.sort(key=lambda r: r["seed"])
    (out / "rows.json").write_text(json.dumps(rows, indent=1, default=str))
    md = summarize(rows, cfg)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wall {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()

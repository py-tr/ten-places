"""Train one ACT policy per skill on its own demo segments (drawer, spoon, plate, fork, cup).

    python scripts/train_skills.py --data data/table_v1_skill --cache data/table_v1_skill_cache \
        --out out/train/skills_v1 --steps 20000 [--skills spoon fork]
    python scripts/train_skills.py --data data/table_plate_t1 --cache data/table_plate_t1_cache --skills plate \
        --init-from out/train/skills_ctx --init-step 15000 --steps 10000 --save-freq 2500 --amp --augment \
        --lr 2e-5 --out out/train/plate_t1 --drop-optimizer     # fast fine-tune

Why per skill: a single skill-conditioned ACT learned to ignore the skill input (it infers the phase from
the cameras), so a planner could not steer it. Each per-skill policy only knows its own skill; the
sequencer runs them and the camera classifier decides when one is done.

Fast fine-tunes: --amp trains in bf16 autocast (scripts/train.py), --save-freq keeps intermediate checkpoints so
the earliest one that holds up in closed loop can be kept (scripts/eval_skill_checkpoints.py; offline loss picks
checkpoints badly), --augment turns on LeRobot's image transforms (colour jitter + small affine shifts), --lr
overrides the transformer learning rate (ACT's 1e-5 was set for batch 8; we train at batch 32).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/table_v1_skill")
    ap.add_argument("--cache", default="data/table_v1_skill_cache")
    ap.add_argument("--repo-id", default="local/tenplaces_table")
    ap.add_argument("--out", default="out/train/skills_v1")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--skills", nargs="*", default=None)
    ap.add_argument("--init-from", default=None,
                    help="runs dir whose <skill>/checkpoints/<step> weights initialise training (fresh optimizer)")
    ap.add_argument("--init-step", type=int, default=20000)
    ap.add_argument("--save-freq", type=int, default=None, help="checkpoint every N steps (default: only the last)")
    ap.add_argument("--num-workers", type=int, default=6,
                    help="data-loading worker processes; each is a spawned Python with its own torch, so fewer workers "
                         "keep Windows from growing the page file (the uint8 cache is fast enough with 2)")
    ap.add_argument("--amp", action="store_true", help="bf16 mixed precision")
    ap.add_argument("--augment", action="store_true", help="LeRobot image transforms (colour jitter, small shifts)")
    ap.add_argument("--lr", type=float, default=None, help="transformer learning rate (default: the policy's)")
    ap.add_argument("--drop-optimizer", action="store_true",
                    help="delete the checkpoints' optimizer state (~400 MB each); evaluation and --init-from don't need it")
    args = ap.parse_args()
    manifest = json.loads((ROOT / args.data / "tenplaces_manifest.json").read_text())
    # A joint that never moves in the demo set (the idle arm) has a spread of ~0; MEAN_STD normalisation then divides
    # its numerical wobble by ~eps and the policy breaks (a drawer fine-tune: 0/50 against 44/50). Refuse, and say how.
    stats = json.loads((ROOT / args.data / "meta" / "stats.json").read_text())
    tiny = {k: [i for i, s in enumerate(stats[k]["std"]) if s < 1e-4] for k in ("action", "observation.state")}
    if any(tiny.values()):
        sys.exit(f"{args.data}: near-zero spread in {tiny} (a joint that never moves). Fine-tune with the parent's "
                 f"statistics: python scripts/use_parent_stats.py --data {args.data} --parent-data <parent's dataset>")
    skills = args.skills or list(manifest["skills"])
    env = {**os.environ, "TENPLACES_CACHE": args.cache, "PYTHONIOENCODING": "utf-8"}
    if args.amp:
        env["TENPLACES_AMP_DTYPE"] = "bf16"
    for skill in skills:
        episodes = manifest["skills"][skill]["episodes"]
        run = Path(args.out) / skill
        if run.exists():
            print(f"skip {skill}: {run} exists", flush=True)
            continue
        run.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "scripts" / "train.py"),
               f"--dataset.repo_id={args.repo_id}", f"--dataset.root={args.data}",
               f"--dataset.episodes=[{','.join(map(str, episodes))}]",
               *([f"--policy.path={Path(args.init_from) / skill / 'checkpoints' / f'{args.init_step:06d}' / 'pretrained_model'}"]
                 if args.init_from else ["--policy.type=act", "--policy.chunk_size=50", "--policy.n_action_steps=50"]),
               "--policy.device=cuda", "--policy.push_to_hub=false",
               *(["--policy.use_amp=true"] if args.amp else []),
               *([f"--policy.optimizer_lr={args.lr}"] if args.lr else []),
               *(["--dataset.image_transforms.enable=true"] if args.augment else []),
               # Checkpoints are ~600 MB each (weights + optimizer state): only the last one unless asked.
               "--batch_size=32", f"--steps={args.steps}", f"--save_freq={args.save_freq or args.steps}",
               "--log_freq=500", f"--num_workers={args.num_workers}", f"--output_dir={run}"]
        print(f"training {skill}: {len(episodes)} episodes -> {run}", flush=True)
        with open(f"{run}.log", "w", encoding="utf-8") as log:
            code = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"{skill}: exit {code}", flush=True)
        if code != 0:
            sys.exit(code)
        if args.drop_optimizer:
            for state in (run / "checkpoints").glob("*/training_state"):
                shutil.rmtree(state, ignore_errors=True)


if __name__ == "__main__":
    main()

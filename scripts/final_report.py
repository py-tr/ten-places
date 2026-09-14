"""The reported full-table result: the reporting seeds (0-49 by default; never used for tuning or selection), the
selections frozen before the run, PyTorch and OpenVINO rows, episodes in parallel workers; videos of the first
10 seeds for the grid (Intel deliverable 4: the 10 randomised seeds are the first 10 of these).

    python scripts/final_report.py --seeds 0 50 --rows torch ov_w8 --videos 10 --workers 6
    python scripts/make_grid_video.py --dir out/eval/final --label torch --out out/video/grid_10_seeds.mp4

Frozen: out/eval/exec_settings.json and out/eval/selected_checkpoints.json are read once and passed to every
worker; the resolved checkpoint of every skill, the seeds, versions and settings go to provenance.json. Policy
latency in this report is measured with parallel workers sharing the CPU; the latency figures to quote come from
scripts/bench_concurrency.py.
"""
import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.evaluate import wilson  # noqa: E402
from tenplaces.parallel_eval import run_table_parallel  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402
from tenplaces.skill_policies import load_exec_settings, load_selected, resolve_checkpoints  # noqa: E402

KEYS = ["drawer_open", "spoon", "plate", "fork", "cup"]
ROWS = {  # policy kwargs per row; OpenVINO workers get 2 inference threads each so they do not oversubscribe the CPU
    "torch": {"device": "cuda", "n_action_steps": 10},
    "ov_fp32": {"backend": "ov-fp32", "n_action_steps": 10, "ov_config": {"INFERENCE_NUM_THREADS": 2}},
    "ov_w8": {"backend": "ov-w8", "n_action_steps": 10, "ov_config": {"INFERENCE_NUM_THREADS": 2}},
}


def precompile(sources: dict, row: str, calib_cache: str):
    """Build each checkpoint's OpenVINO IR once, here, so parallel workers only load it (no racing writers)."""
    from tenplaces.lerobot_policy import LeRobotPolicy

    for skill, ck in sources.items():
        t = time.perf_counter()
        LeRobotPolicy(ck, backend=ROWS[row]["backend"], calib_cache=calib_cache)
        print(f"[{row}] {skill}: IR ready ({time.perf_counter() - t:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 50], metavar=("FIRST", "STOP"))
    ap.add_argument("--rows", nargs="+", default=["torch", "ov_w8"], choices=sorted(ROWS))
    ap.add_argument("--runs", nargs="+", default=["out/train/skills_v1", "out/train/skills_v2", "out/train/skills_ctx"])
    ap.add_argument("--videos", type=int, default=10, help="videos of the first N seeds (first row only)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--classifier", default="models/state_classifier_v3/state_classifier.xml")
    ap.add_argument("--calib-cache", default="data/table_v1_skill_cache")
    ap.add_argument("--home-frames", type=int, default=20,
                    help="release + return home between skills (env_table.go_home); 0 = no hand-over between skills")
    ap.add_argument("--out", default="out/eval/final")
    args = ap.parse_args()
    if 100 <= args.seeds[0] < 200:
        sys.exit("the report uses the reporting seeds (0-49); 100-149 are for tuning (200+: fresh seeds for stress tests)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [r for r in args.runs if Path(r).is_dir()]
    exec_settings, selected = load_exec_settings(), load_selected()
    sources = {s: str(p) for s, p in resolve_checkpoints(runs, selected=selected).items()}
    seeds = list(range(*args.seeds))
    import mujoco
    import openvino
    import torch

    from tenplaces.evaluate_table import CAMERA_ENDS, DEFAULT_BUDGETS, RETRY, SETTLE

    (out / "provenance.json").write_text(json.dumps({
        "seeds": [seeds[0], seeds[-1]], "rows": args.rows, "home_frames": args.home_frames,
        "stress": float(os.environ.get("TENPLACES_STRESS", "1.0")),  # scene_table.sample: ranges widened by this
        "shape": float(os.environ.get("TENPLACES_SHAPE", "0")),  # scene_table.sample: object sizes 1 ± shape
        "budgets": DEFAULT_BUDGETS, "camera_ends": CAMERA_ENDS, "settle": SETTLE, "retry": RETRY,
        "exec_settings": exec_settings, "selected": selected,
        "checkpoints": sources, "runs": runs, "classifier": args.classifier, "date": time.strftime("%Y-%m-%d %H:%M"),
        "versions": {"openvino": openvino.__version__, "torch": torch.__version__, "mujoco": mujoco.__version__,
                     "python": platform.python_version()}}, indent=1))
    stress = float(os.environ.get("TENPLACES_STRESS", "1.0"))
    shape = float(os.environ.get("TENPLACES_SHAPE", "0"))
    kind = "reporting seeds" if seeds[0] < 100 else "fresh seeds (stress test)"
    lines = [f"# Full table, {kind} {seeds[0]}-{seeds[-1]} ({len(seeds)} randomised tables)"
             + (f", ranges widened ×{stress}" if stress != 1.0 else "")
             + (f", object sizes ±{shape:.0%}" if shape else ""), "",
             "Selections frozen before the run: see provenance.json."
             + (" The first 10 seeds are the ones in the video grid." if seeds[0] < 100 else ""), "",
             "| row | full tables | 95% CI | seeds 0-9 | mean steps | " + " | ".join(KEYS) + " |",
             "|---|---|---|---|---|" + "---|" * len(KEYS)]
    for i, row in enumerate(args.rows):
        if row == "torch" and not torch.cuda.is_available():
            print("[torch] skipped: the PyTorch reference row needs a CUDA device; the OpenVINO rows run on any Intel CPU",
                  flush=True)
            continue
        if row != "torch":
            precompile(sources, row, args.calib_cache)
        spec = {"kind": "skills", "runs": runs, "exec_settings": exec_settings, "checkpoints": selected,
                "kwargs": {**ROWS[row], **({"calib_cache": args.calib_cache} if row != "torch" else {})}}
        t = time.perf_counter()
        rows = run_table_parallel(spec, seeds, workers=args.workers, classifier_xml=args.classifier,
                                  video_dir=out, videos=args.videos if i == 0 else 0, label=row,
                                  home_frames=args.home_frames)
        with open(out / f"{row}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["seed", "success", "subtasks_done", *KEYS, "policy_ms_mean"],
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        n, k = len(rows), sum(r["success"] for r in rows)
        first10 = [r for r in rows if r["seed"] < seeds[0] + 10]
        s = {"row": row, "episodes": n, "full": k, "wilson95": wilson(k, n), "first10_full": sum(r["success"] for r in first10),
             "mean_steps": sum(r["subtasks_done"] for r in rows) / n, "per_step": {c: sum(bool(r[c]) for r in rows) for c in KEYS},
             "wall_s": round(time.perf_counter() - t)}
        (out / f"{row}_summary.json").write_text(json.dumps(s, indent=1))
        print(json.dumps(s), flush=True)
        lines.append(f"| {row} | {k}/{n} | {s['wilson95']} | {s['first10_full']}/{len(first10)} | {s['mean_steps']:.2f} | "
                     + " | ".join(f"{s['per_step'][c]}/{n}" for c in KEYS) + " |")
        (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

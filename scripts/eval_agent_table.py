"""The full table through the agent's own control loop (tenplaces.agent.run_command) with a fixed plan, in parallel:
the check after every skill, one retry, one re-plan and the re-check of finished skills before the next one —
what the system does, which the bare sequencer (tenplaces.evaluate_table.run_episode, one attempt per skill)
does not.

    python scripts/eval_agent_table.py --seeds 100 120 --workers 6          # tuning seeds
    python scripts/eval_agent_table.py --seeds 0 50 --report --workers 6    # the reporting seeds, once

Why: on video the camera check sometimes ends a skill as "done" with the object still in the drawer; the bare
sequencer never looks again, the agent's re-check does. The plan is fixed to all five skills — what the VLM
planner returns for "set the table" (10/10, scripts/eval_planner.py) — so this measures execution only.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.evaluate import wilson  # noqa: E402
from tenplaces.parallel_eval import run_agent_parallel  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402
from tenplaces.skill_policies import load_exec_settings, load_selected, resolve_checkpoints  # noqa: E402

KEYS = ["drawer_open", "spoon", "plate", "fork", "cup"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=[100, 120], metavar=("FIRST", "STOP"))
    ap.add_argument("--report", action="store_true", help="allow the reporting seeds (0-49)")
    ap.add_argument("--runs", nargs="+", default=["out/train/skills_v1", "out/train/skills_v2", "out/train/skills_ctx"])
    ap.add_argument("--backend", default="torch", choices=["torch", "ov-w8", "ov-fp32"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--classifier", default="models/state_classifier_v3/state_classifier.xml")
    ap.add_argument("--name", default=None)
    ap.add_argument("--push", action="append", default=[], metavar="T:BODY:DX:DY[:DUR]",
                    help="disturb every episode, e.g. after-plate:plate:0:-0.07 (see scripts/run_agent.py)")
    ap.add_argument("--ckpt", action="append", default=[], metavar="SKILL=DIR",
                    help="a candidate checkpoint for one skill; the other skills keep out/eval/selected_checkpoints.json")
    args = ap.parse_args()
    if args.seeds[0] < 100 and not args.report:
        sys.exit("tuning seeds are 100-149; pass --report to run the reporting seeds (once, with frozen selections)")
    kwargs = ({"device": "cuda", "n_action_steps": 10} if args.backend == "torch" else
              {"backend": args.backend, "n_action_steps": 10, "ov_config": {"INFERENCE_NUM_THREADS": 2},
               "calib_cache": "data/table_v1_skill_cache"})
    overrides = dict(kv.split("=", 1) for kv in args.ckpt)
    spec = {"kind": "skills", "runs": [r for r in args.runs if Path(r).is_dir()], "exec_settings": load_exec_settings(),
            "checkpoints": {**load_selected(), **overrides}, "kwargs": kwargs}
    if args.backend != "torch":  # build each checkpoint's IR once here, so parallel workers only load it (no racing writers)
        from tenplaces.lerobot_policy import LeRobotPolicy

        for ck in resolve_checkpoints(spec["runs"], selected=spec["checkpoints"]).values():
            LeRobotPolicy(ck, backend=args.backend, calib_cache=kwargs["calib_cache"])
    from tenplaces.agent import parse_push

    disturb = [parse_push(p) for p in args.push]
    rows = run_agent_parallel(spec, range(*args.seeds), workers=args.workers, classifier_xml=args.classifier,
                              disturb=disturb)
    n, k = len(rows), sum(r["success"] for r in rows)
    s = {"seeds": args.seeds, "backend": args.backend, "full": f"{k}/{n}", "wilson95": wilson(k, n),
         "mean_steps": sum(r["subtasks_done"] for r in rows) / n, "per_step": {c: sum(bool(r[c]) for r in rows) for c in KEYS},
         "retries": sum(r["retries"] for r in rows), "replans": sum(r["replans"] for r in rows),
         "regressed": sum(r["regressed"] for r in rows)}
    if disturb:  # how often the push landed, was noticed, and was put right by the end
        pushed = [r for r in rows if r["pushed"]]
        bodies = {b for _, b, _, _ in disturb}
        s["push"] = args.push
        s["episodes_pushed"] = len(pushed)
        s["noticed"] = sum(any(sk in bodies for sk in r["regressed_skills"]) for r in pushed)  # skill = body name
        s["repaired"] = {b: f"{sum(bool(r[b]) for r in pushed)}/{len(pushed)}" for b in bodies if b in KEYS}
    out = OUT / "eval" / "agent_table"
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{args.backend}_seeds{args.seeds[0]}-{args.seeds[1] - 1}"
    (out / f"{name}.json").write_text(json.dumps({"summary": s, "spec": spec, "rows": rows}, indent=1, default=str))
    print(json.dumps(s))


if __name__ == "__main__":
    main()

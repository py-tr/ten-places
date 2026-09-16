"""Pass 1 of the fork takeover recipe: find the tables the deployed fork policy fails on, by running it alone.

The policy drives the fork step for its whole budget after a scripted drawer, spoon and plate, on demonstration
seeds. Every table's outcome is appended to --out as one JSON line, so pass 2 (scripts/record_context_demos.py
--takeover-continue) can replay exactly those tables and take over from the state the policy left the arms in.
Taking over anywhere else would teach recoveries from states that never occur.

    python scripts/fork_pass1.py --start 20000 --out out/fork_pass1.jsonl --stop-at 07:00
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.control import IKFailure  # noqa: E402
from tenplaces.cores import no_power_throttling  # noqa: E402
from tenplaces.env_table import policy_outcome  # noqa: E402
from tenplaces.evaluate_skill import KEY  # noqa: E402
from tenplaces.evaluate_table import DEFAULT_BUDGETS  # noqa: E402
from tenplaces.lerobot_policy import LeRobotPolicy  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True, help="first seed; the pass walks upward until --stop-at")
    ap.add_argument("--out", required=True, help="JSON lines, appended: one row per table")
    ap.add_argument("--exec", choices=["exec50", "ensemble"], default="exec50")
    ap.add_argument("--device", default="cuda", help="the mining pass only; the reported robot runs on OpenVINO")
    ap.add_argument("--stop-at", required=True, metavar="HH:MM", help="next occurrence of this wall-clock time")
    args = ap.parse_args()

    now = dt.datetime.now()
    hh, mm = map(int, args.stop_at.split(":"))
    stop = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    stop = (stop if stop > now else stop + dt.timedelta(days=1)).timestamp()

    no_power_throttling()
    ck = json.loads((OUT / "eval" / "selected_checkpoints.json").read_text())["fork"]
    kw = {"n_action_steps": 50} if args.exec == "exec50" else {"temporal_coeff": 0.01}
    pol = LeRobotPolicy(ck, device=args.device, **kw)

    out = Path(args.out)
    seed, n, fails = args.start, 0, 0
    while time.time() < stop:
        t = time.time()
        try:
            g, lifted = policy_outcome(seed, "fork", ["drawer", "spoon", "plate"], pol, DEFAULT_BUDGETS["fork"])
            row = {"seed": seed, "ok": bool(g[KEY["fork"]]), "failed": g["failed"], "lifted": lifted}
        except IKFailure as e:  # no scripted start on this table: not a result for any policy
            row = {"seed": seed, "ok": None, "error": str(e)}
        row.update(exec=args.exec, checkpoint=ck, s=round(time.time() - t, 1))
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        n += 1
        fails += row["ok"] is False
        if n % 10 == 0:
            print(f"{n} tables, {fails} fork failures", flush=True)
        seed += 1
    print(f"stopped at {args.stop_at}: {n} tables, {fails} fork failures", flush=True)


if __name__ == "__main__":
    main()

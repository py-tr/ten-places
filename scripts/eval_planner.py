"""Language-planning accuracy: does the VLM + verifier turn varied commands into the right skill plan, and does
it name what no skill can do (planner.plan_checked = plan + cannot_do + re-plan without the impossible part)?

    python scripts/eval_planner.py --method checked     # plan(): the VLM lists the steps
    python scripts/eval_planner.py --method intent      # plan_intent(): the VLM states the intent, code builds steps

Commands are phrased differently from the prompts' worked examples. A command counts as correct when the
verified plan equals the verified expected plan (the verifier adds physical prerequisites and fixes order).
Set history: "tests" was the first set (10/10); "dev" and "fresh" were each scored once and are development
data now; "fresh2" was written before either method was scored on it — the reported number.
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.agent import PLANNER_HW, planner_image  # noqa: E402
from tenplaces.env_table import TableEpisode  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402
from tenplaces.planner import VLMPlanner, verify  # noqa: E402

ALL = ["spoon", "plate", "fork", "cup"]
SETS = {  # (command, expected skills, should something be named as impossible?)
    "tests": [
        ("Set the table for dinner.", ALL, False),
        ("Put the plate and the fork out.", ["plate", "fork"], False),
        ("Just something to drink from, please.", ["cup"], False),
        ("Everything except the spoon, please.", ["plate", "fork", "cup"], False),
        ("Set it all up but skip the cup.", ["spoon", "plate", "fork"], False),
        ("I'll have a salad.", ["fork"], False),
        ("Open the drawer.", ["drawer"], False),
        ("Plate only.", ["plate"], False),
        ("Give me a spoon and a cup.", ["spoon", "cup"], False),
        ("Nothing for now, thanks.", [], False),
    ],
    "dev": [
        ("Set the table and light a candle.", ALL, True),
        ("Give me a cup of coffee.", ["cup"], True),
        ("Put the fork out and cut my steak for me.", ["fork"], True),
        ("Fold a napkin, please.", [], True),
        ("Lay the spoon, then wash the dishes.", ["spoon"], True),
    ],
    "fresh": [
        ("Set the table and turn on some music.", ALL, True),
        ("Just the cup, and bring me the salt.", ["cup"], True),
        ("Open the drawer and water the plants.", ["drawer"], True),
        ("Put out the fork and the plate, please.", ["plate", "fork"], False),
        ("No spoon today, but everything else.", ["plate", "fork", "cup"], False),
        ("Leave the cup out, set the rest.", ["spoon", "plate", "fork"], False),
    ],
    "fresh2": [
        ("Could you lay everything except the fork?", ["spoon", "plate", "cup"], False),
        ("I'm only having a drink.", ["cup"], False),
        ("Put out cutlery only.", ["spoon", "fork"], False),
        ("Full setting please, and dim the lights.", ALL, True),
        ("Set the table without the spoon and the cup.", ["plate", "fork"], False),
        ("Just open the drawer, nothing else.", ["drawer"], False),
        ("The spoon and the plate, and then tell me a joke.", ["spoon", "plate"], True),
        ("Everything apart from the cup.", ["spoon", "plate", "fork"], False),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["checked", "intent"], default="checked")
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    ap.add_argument("--seed", type=int, default=1201)
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()
    planner = VLMPlanner(device=args.device)
    ep = TableEpisode(args.seed, render=False)
    renderer = mujoco.Renderer(ep.m, *PLANNER_HW)
    image = planner_image(ep, renderer)

    def run(name, cases):
        rows = []
        for command, want, impossible in cases:
            res = planner.plan_checked(command, image, use_intent=args.method == "intent")
            expected = verify(want)[0]
            flagged = bool(res["unsupported"])
            rows.append({"command": command, "intent": res.get("intent"), "proposed": res["proposed"],
                         "plan": res["steps"], "expected": expected, "plan_ok": res["steps"] == expected,
                         "unsupported": res["unsupported"], "flag_ok": flagged == impossible,
                         "replanned_from": res.get("replanned_from"), "corrections": res["corrections"],
                         "ms": round(res["ms"])})
            r = rows[-1]
            print(f"{'OK ' if r['plan_ok'] and r['flag_ok'] else 'BAD'} [{name}] {command!r}: -> {res['steps']} "
                  f"unsupported={res['unsupported']}{' (re-planned)' if r['replanned_from'] else ''} ({res['ms']:.0f} ms)",
                  flush=True)
        return rows

    results = {name: run(name, SETS[name]) for name in args.sets}
    summary = {name: {"cases": len(rows), "plan_ok": sum(r["plan_ok"] for r in rows),
                      "flag_ok": sum(r["flag_ok"] for r in rows),
                      "both_ok": sum(r["plan_ok"] and r["flag_ok"] for r in rows)} for name, rows in results.items()}
    out = OUT / "planner"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"eval_planner_{args.method}.json").write_text(json.dumps({"method": args.method, "summary": summary,
                                                                     "rows": results}, indent=1))
    for name, s in summary.items():
        print(f"{args.method} {name}: plans {s['plan_ok']}/{s['cases']}, impossible-part flags {s['flag_ok']}/{s['cases']}, "
              f"both {s['both_ok']}/{s['cases']}")


if __name__ == "__main__":
    main()

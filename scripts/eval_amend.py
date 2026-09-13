"""Mid-run changes: does the VLM planner turn what the person says while the robot works into the right new plan?

    python scripts/eval_amend.py --set fresh      # the reported set: written after dev and heldout were seen
    python scripts/eval_amend.py --set heldout    # exposed the whole-list prompt (4/10); development data now
    python scripts/eval_amend.py --set dev        # the 10 cases the first prompt was developed on

Each case gives the original command, what is already done, the step in progress, what is still planned,
and the sentence heard; the expected answer is the verified list of steps to run after the current one
(the verifier re-adds physical prerequisites, e.g. the plate when the fork is kept). The planner sees a real
top-camera image of a seeded scene (evaluation seed range).
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.agent import PLANNER_HW  # noqa: E402
from tenplaces.env_table import TableEpisode  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402
from tenplaces.planner import VLMPlanner  # noqa: E402

ALL = ["spoon", "plate", "fork", "cup"]
# (original command, done, current, still planned, heard, expected steps after the current one)
CASES = {
    "dev": [
        ("Set the table.", ["drawer"], "spoon", ["plate", "fork", "cup"], "Skip the fork.", ["plate", "cup"]),
        ("Set the table.", ["drawer"], "spoon", ["plate", "fork", "cup"], "Actually, just the cup.", ["cup"]),
        ("Set the table.", ["drawer", "spoon", "plate"], "fork", ["cup"], "Forget the cup.", []),
        ("Just the plate, please.", ["drawer"], "plate", [], "Oh, and put the cup out too.", ["cup"]),
        ("Set the table, but no fork.", ["drawer", "spoon"], "plate", ["cup"],
         "I changed my mind, I want the fork too.", ["fork", "cup"]),
        ("Set the table.", ["drawer", "spoon"], "plate", ["fork", "cup"], "Looks great, carry on.", ["fork", "cup"]),
        ("Set the table.", [], "drawer", ALL, "No spoon, please.", ["plate", "fork", "cup"]),
        ("Only the cup.", [], "cup", [], "Thanks, that's all.", []),
        ("Lay a full place setting.", ["drawer"], "spoon", ["plate", "fork", "cup"],
         "We're having soup, skip the plate and the fork.", ["cup"]),
        ("Set the table.", ["drawer", "spoon", "plate"], "fork", ["cup"], "Can you leave out the cup and stop there?", []),
    ],
    "heldout": [
        ("Set the table.", [], "drawer", ALL, "Leave out the fork.", ["spoon", "plate", "cup"]),
        ("Set the table.", ["drawer"], "spoon", ["plate", "fork", "cup"], "Don't bother with the plate.",
         ["plate", "fork", "cup"]),  # the fork needs the plate moved: the verifier keeps it
        ("Just the plate and the cup.", ["drawer"], "plate", ["cup"], "Add the spoon as well.", ["spoon", "cup"]),
        ("Set the table.", ["drawer", "spoon", "plate", "fork"], "cup", [], "Perfect, thank you.", []),
        ("Set the table.", [], "drawer", ALL, "Only the cutlery, please.", ["spoon", "plate", "fork"]),
        ("Put the cup out.", [], "cup", [], "And the plate too, please.", ["drawer", "plate"]),
        ("Set the table.", ["drawer"], "spoon", ["plate", "fork", "cup"], "Hold on, no cup today.", ["plate", "fork"]),
        ("Set the table without the cup.", ["drawer", "spoon"], "plate", ["fork"], "Actually we do need the cup.",
         ["fork", "cup"]),
        ("Set the table.", [], "drawer", ALL, "Skip the spoon and the cup.", ["plate", "fork"]),
        ("Lay a full place setting.", ["drawer", "spoon"], "plate", ["fork", "cup"], "That's enough, no more.", []),
    ],
    # Written after dev and heldout had been seen (both are development data now); the reported set.
    "fresh": [
        ("Set the table.", ["drawer"], "spoon", ["plate", "fork", "cup"], "Let's not do the cup.", ["plate", "fork"]),
        ("Set the table.", [], "drawer", ALL, "Please add nothing but the spoon.", ["spoon"]),
        ("Just the spoon.", [], "drawer", ["spoon"], "Could you also set the cup down?", ["spoon", "cup"]),
        ("Set the table.", ["drawer", "spoon", "plate"], "fork", ["cup"], "Great, we're done after this one.", []),
        ("Set the table, no cup.", ["drawer"], "spoon", ["plate", "fork"], "Put the cup out after all.",
         ["plate", "fork", "cup"]),
        ("Set the table.", ["drawer"], "spoon", ["plate", "fork", "cup"], "No fork and no cup, thanks.", ["plate"]),
        ("Set the table.", [], "drawer", ALL, "Keep going, that's fine.", ["spoon", "plate", "fork", "cup"]),
        ("Only the plate.", [], "drawer", ["plate"], "Oh, and the fork too.", ["plate", "fork"]),
        ("Set the table.", ["drawer", "spoon"], "plate", ["fork", "cup"], "Change of plan, just the cup after this.",
         ["cup"]),
        ("Lay the table.", [], "drawer", ALL, "Drop the plate from the list.", ["spoon", "plate", "fork", "cup"]),
        # ^ the fork's spot is under the plate: the verifier puts the plate back (and says why)
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(CASES), default="dev")
    ap.add_argument("--seed", type=int, default=1201)
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()
    ep = TableEpisode(args.seed, render=False)
    r = mujoco.Renderer(ep.m, *PLANNER_HW)
    r.update_scene(ep.d, camera="top")
    image = r.render().copy()
    planner = VLMPlanner(device=args.device)
    rows, ok = [], 0
    for command, done, current, planned, heard, expected in CASES[args.set]:
        a = planner.amend(command, heard, image, done, current, planned)
        hit = a["steps"] == expected
        ok += hit
        rows.append({"heard": heard, "planned": planned, "proposed": a["proposed"], "steps": a["steps"],
                     "expected": expected, "ok": hit, "ms": round(a["ms"]), "corrections": a["corrections"],
                     "raw": a["raw"]})
        print(f"{'OK ' if hit else 'BAD'} {heard!r}: {planned} -> {a['steps']} (expected {expected}, {a['ms']:.0f} ms)",
              flush=True)
    out = OUT / "eval" / f"amend_{args.set}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"set": args.set, "correct": ok, "cases": len(rows), "rows": rows}, indent=1))
    print(f"{args.set}: {ok}/{len(rows)} correct -> {out}")


if __name__ == "__main__":
    main()

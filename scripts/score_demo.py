"""Score the 10 demonstration runs against their pre-registered commands (configs/demo_seeds.json).

A seed passes when every step its command asked for is physically done (the simulator's grade, not the camera),
nothing else was done, and — where the command asks for something no skill does — the robot said so.

    python scripts/score_demo.py                  # reads out/video/demo/seed<N>.json
    python scripts/make_grid_video.py --dir out/video/demo --label demo --out out/video/grid_demo.mp4

Writes out/video/demo/demo.csv (the grid's PASS/FAIL stamps), summary.md, and demo_seed<N>.mp4 links to the videos.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.planner import verify  # noqa: E402

GRADE_KEY = {"drawer": "drawer_open", "spoon": "spoon", "plate": "plate", "fork": "fork", "cup": "cup"}


def score(entry: dict, run: dict) -> dict:
    """One seed: what was asked, heard, planned and done, and whether that is what the command asked for."""
    ev = {e["kind"]: e for e in run["events"] if e["kind"] in ("plan", "finished")}
    said_unsupported = [i for e in run["events"] if e["kind"] in ("plan", "unsupported")
                        for i in (e.get("unsupported") or e.get("items") or [])]
    expected = verify(entry["expected_steps"])[0]
    done = [s for s in GRADE_KEY if run["grade"].get(GRADE_KEY[s])]
    # Every expected refusal must be named, and nothing refused that the command did not ask for.
    want = [w.lower() for w in entry.get("expected_unsupported", [])]
    said = " | ".join(said_unsupported).lower()
    refusals_ok = all(w in said for w in want) if want else not said_unsupported
    ok = done == expected and refusals_ok
    return {"seed": entry["seed"], "mode": entry["mode"], "command": entry["command"], "heard": run["command"],
            "say": entry.get("say", ""), "plan": ev.get("plan", {}).get("steps", []), "expected": expected,
            "done": done, "unsupported": said_unsupported, "plan_s": round(ev.get("plan", {}).get("ms", 0) / 1000, 1),
            "success": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo_seeds.json")
    ap.add_argument("--dir", default="out/video/demo")
    args = ap.parse_args()
    d = Path(args.dir)
    rows = []
    for entry in json.loads(Path(args.config).read_text(encoding="utf-8"))["seeds"]:
        f = d / f"seed{entry['seed']}.json"
        if not f.exists():
            print(f"seed {entry['seed']}: no run ({f})")
            continue
        rows.append(score(entry, json.loads(f.read_text(encoding="utf-8"))))
        video, link = d / f"seed{entry['seed']}.mp4", d / f"demo_seed{entry['seed']}.mp4"
        if video.exists():
            link.unlink(missing_ok=True)
            os.link(video, link)
    with open(d / "demo.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "success"])
        w.writerows([r["seed"], r["success"]] for r in rows)
    lines = [f"# Demonstration runs: {sum(r['success'] for r in rows)}/{len(rows)} did what was asked", "",
             "| seed | how | command (heard) | plan | done | said it cannot | first plan | result |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        heard = r["heard"] if r["heard"] == r["command"] else f"{r['command']} (heard: {r['heard']})"
        if r["say"]:
            heard += f" + said “{r['say']}”"
        lines.append(f"| {r['seed']} | {r['mode']} | {heard} | {' → '.join(r['plan']) or '–'} | "
                     f"{', '.join(r['done']) or '–'} | {', '.join(r['unsupported']) or '–'} | {r['plan_s']} s | "
                     f"{'PASS' if r['success'] else 'FAIL'} |")
    (d / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

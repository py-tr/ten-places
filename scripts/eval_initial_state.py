"""The camera classifier on the table as the robot first sees it: fresh tables must read as nothing done, and tables
where the scripted controller already did some steps must read exactly those steps. The probabilities are logged, so
a threshold for acting on the first look (skipping what is already done) is chosen from the margins.

    python scripts/eval_initial_state.py --classifier models/state_classifier_v3/state_classifier.xml --name v3

Tuning seeds only (default 100-149); ten verified prefixes in rotation.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.env_table import TableEpisode  # noqa: E402
from tenplaces.evaluate_skill import SKILL_NAMES  # noqa: E402
from tenplaces.oracle import table  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402
from tenplaces.state_classifier import OVStateClassifier  # noqa: E402

PREFIXES = [["drawer"], ["drawer", "spoon"], ["drawer", "plate"], ["drawer", "spoon", "plate"], ["drawer", "plate", "fork"],
            ["cup"], ["drawer", "spoon", "plate", "fork"], ["drawer", "cup"], ["drawer", "spoon", "cup"],
            ["drawer", "plate", "cup"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", default="models/state_classifier_v3/state_classifier.xml")
    ap.add_argument("--seeds", type=int, nargs=2, default=[100, 150])
    ap.add_argument("--name", default="v3")
    args = ap.parse_args()
    if args.seeds[0] < 100:
        sys.exit("tuning seeds only (100-149)")
    clf = OVStateClassifier(args.classifier)
    rows = []
    for i, seed in enumerate(range(*args.seeds)):
        ep = TableEpisode(seed, render=True)
        fresh = [round(float(v), 4) for v in clf.probs(ep.observation()["images"]["top"])]
        ep.close()
        prefix = PREFIXES[i % len(PREFIXES)]
        ep = TableEpisode(seed, render=True)
        table.run_plan(ep.ctl, ep.params, prefix)
        pre = [round(float(v), 4) for v in clf.probs(ep.observation()["images"]["top"])]
        ep.close()
        rows.append({"seed": seed, "fresh": dict(zip(SKILL_NAMES, fresh)), "prefix": prefix,
                     "prefixed": dict(zip(SKILL_NAMES, pre))})
        print(json.dumps(rows[-1]), flush=True)
    t = clf.threshold
    fresh_false = [r["seed"] for r in rows if any(p > t for p in r["fresh"].values())]
    exact = [r["seed"] for r in rows if {s for s, p in r["prefixed"].items() if p > t} == set(r["prefix"])]
    fresh_max = {s: max(r["fresh"][s] for r in rows) for s in SKILL_NAMES}
    done_min = {s: min((r["prefixed"][s] for r in rows if s in r["prefix"]), default=None) for s in SKILL_NAMES}
    undone_max = {s: max((r["prefixed"][s] for r in rows if s not in r["prefix"]), default=None) for s in SKILL_NAMES}
    n = len(rows)
    summary = {"classifier": args.classifier, "threshold": t, "fresh_false_done": len(fresh_false),
               "prefixed_exact": len(exact), "n": n, "fresh_max_p": fresh_max, "prefixed_done_min_p": done_min,
               "prefixed_not_done_max_p": undone_max}
    out = OUT / "eval" / "initial_state"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.name}_seeds{args.seeds[0]}-{args.seeds[1] - 1}"
    (out / f"{stem}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    fmt = lambda d: " · ".join(f"{s} {v:.3f}" if v is not None else f"{s} –" for s, v in d.items())  # noqa: E731
    md = [f"# First look at the table — classifier {args.name}", "",
          f"Seeds {args.seeds[0]}-{args.seeds[1] - 1}, threshold {t}. Fresh tables with a false 'done': "
          f"{len(fresh_false)}/{n}. Prefixed tables read exactly: {len(exact)}/{n}.", "",
          f"- highest probability on a fresh table: {fmt(fresh_max)}",
          f"- lowest probability for a step that was done: {fmt(done_min)}",
          f"- highest probability for a step that was not done: {fmt(undone_max)}"]
    (out / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

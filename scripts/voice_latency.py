"""Voice timings from recorded runs (run_agent.py --video writes <video>.json; --listen also <video>.live.json).

    python scripts/voice_latency.py out/video/demo_final out/video/demo_live/take1

Per run, from the logged stamps (no re-timing):
- command: final transcript after the end of speech (Speechmatics, spoken commands only);
- first plan: the planner's time for the first plan;
- change heard -> new plan: wall time from the "heard" event to its "amend" event, the planner's share, and how long
  the arms held waiting for it;
- live sentences: transcript final after the end of the sentence in the audio (Speechmatics timings + wall clock);
- "stop" heard -> "stop" handled (wall).
"""
import json
import statistics
import sys
from pathlib import Path


def runs(dirs):
    for d in dirs:
        for f in sorted(Path(d).glob("seed*.json")):
            if f.name.endswith(".live.json"):
                continue
            yield f, json.loads(f.read_text(encoding="utf-8"))


def main():
    rows = []
    for f, run in runs(sys.argv[1:] or ["out/video/demo_final"]):
        ev = run.get("events", [])
        r = {"run": f"{f.parent.name}/{f.stem}"}
        voice = run.get("voice") or {}
        if voice.get("ms_final_after_speech_end") is not None:
            r["command_final_s"] = voice["ms_final_after_speech_end"] / 1000
        plan = next((e for e in ev if e["kind"] == "plan"), None)
        if plan is not None and plan.get("ms") is not None:
            r["first_plan_s"] = plan["ms"] / 1000
        heard = [e for e in ev if e["kind"] == "heard" and "wall" in e]
        amends = [e for e in ev if e["kind"] == "amend" and "wall" in e]
        for h in heard:
            a = next((x for x in amends if x.get("heard") == h["text"] and x["wall"] >= h["wall"]), None)
            if a is not None:
                r.setdefault("heard_to_plan_s", []).append(round(a["wall"] - h["wall"], 2))
                r.setdefault("amend_planner_s", []).append(round(a.get("ms", 0) / 1000, 2))
                r.setdefault("arms_waited_s", []).append(a.get("waited_s"))
        stops = [e for e in ev if e["kind"] == "stop" and "wall" in e]
        for s in stops:
            h = next((x for x in heard if x["text"] == s.get("heard")), None)
            if h is not None:
                r.setdefault("stop_handled_s", []).append(round(s["wall"] - h["wall"], 3))
        live = f.with_suffix(".live.json")
        if live.exists():
            lj = json.loads(live.read_text(encoding="utf-8"))
            for u in lj.get("utterances", []):
                if u.get("audio_end") is not None:
                    r.setdefault("live_final_s", []).append(round(u["wall"] - (lj["t0_wall"] + u["audio_end"]), 2))
        if len(r) > 1:
            rows.append(r)
    keys = ["command_final_s", "first_plan_s", "heard_to_plan_s", "amend_planner_s", "arms_waited_s",
            "live_final_s", "stop_handled_s"]
    print("| run | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for r in rows:
        print(f"| {r['run']} | " + " | ".join(str(r.get(k, "–")) for k in keys) + " |")
    print()
    for k in keys:
        vals = [v for r in rows for v in (r[k] if isinstance(r.get(k), list) else [r[k]] if k in r else [])
                if v is not None]
        if vals:
            print(f"{k}: median {statistics.median(vals):.2f} s over {len(vals)} (min {min(vals):.2f}, max {max(vals):.2f})")


if __name__ == "__main__":
    main()

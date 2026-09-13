"""The first plan on the E-cores vs on every core (VLMPlanner idle_config), for the 10 demonstration commands on
their own tables (configs/demo_seeds.json). The arms are still until the first plan arrives, so it may use every
core; everything asked while they move stays on the E-cores. The order alternates per seed; both answers must
give the same plan.

    python scripts/bench_planner_placement.py      # -> out/benchmark/planner_placement.{json,md}
"""
import json
import statistics
import sys
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces import cores  # noqa: E402
from tenplaces.agent import PLANNER_HW, planner_image  # noqa: E402
from tenplaces.env_table import TableEpisode  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402
from tenplaces.planner import VLMPlanner  # noqa: E402


def main():
    planner = VLMPlanner(ov_config=cores.planner_config(), idle_config={})
    rows = []
    for c in json.loads(Path("configs/demo_seeds.json").read_text(encoding="utf-8"))["seeds"]:
        ep = TableEpisode(c["seed"], render=False)
        r = mujoco.Renderer(ep.m, *PLANNER_HW)
        image = planner_image(ep, r)
        r.close()
        order = (False, True) if c["seed"] % 2 == 0 else (True, False)
        res = {idle: planner.plan(c["command"], image, idle=idle) for idle in order}
        rows.append({"seed": c["seed"], "command": c["command"], "e_cores_ms": round(res[False]["ms"]),
                     "all_cores_ms": round(res[True]["ms"]), "same_plan": res[False]["steps"] == res[True]["steps"]})
        print(f"seed {c['seed']}: E-cores {res[False]['ms'] / 1000:.1f} s, all cores {res[True]['ms'] / 1000:.1f} s, "
              f"same plan {rows[-1]['same_plan']}", flush=True)
    e = statistics.median(r["e_cores_ms"] for r in rows) / 1000
    a = statistics.median(r["all_cores_ms"] for r in rows) / 1000
    summary = {"median_e_cores_s": round(e, 1), "median_all_cores_s": round(a, 1),
               "same_plan": sum(r["same_plan"] for r in rows), "n": len(rows),
               "e_cores_config": cores.planner_config(), "all_cores_config": "OpenVINO defaults"}
    out = OUT / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "planner_placement.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    lines = ["# First plan: E-cores vs every core", "", f"Median {e:.1f} s on the E-cores, {a:.1f} s on every core; "
             f"same plan {summary['same_plan']}/{len(rows)}.", "", "| seed | command | E-cores | all cores | same plan |",
             "|---|---|---|---|---|"]
    lines += [f"| {r['seed']} | {r['command']} | {r['e_cores_ms'] / 1000:.1f} s | {r['all_cores_ms'] / 1000:.1f} s | "
              f"{'yes' if r['same_plan'] else 'NO'} |" for r in rows]
    (out / "planner_placement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:3]))


if __name__ == "__main__":
    main()

"""The first plan on the E-cores vs on every core (VLMPlanner idle_config), for the 10 demonstration commands on
their own tables (configs/demo_seeds.json). The arms are still until the first plan arrives, so it may use every
core; everything asked while they move stays on the E-cores. The order alternates per seed; both answers must
give the same plan. Alongside the wall time, OpenVINO GenAI's own metrics for each answer: time to first token
(image encoding + prefill), time per output token and tokens/s.

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

PLACES = {False: "e_cores", True: "all_cores"}


def med(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(vals) if vals else None


def main():
    cores.no_power_throttling()
    planner = VLMPlanner(ov_config=cores.planner_config(), idle_config={})
    rows = []
    for c in json.loads(Path("configs/demo_seeds.json").read_text(encoding="utf-8"))["seeds"]:
        ep = TableEpisode(c["seed"], render=False)
        r = mujoco.Renderer(ep.m, *PLANNER_HW)
        image = planner_image(ep, r)
        r.close()
        order = (False, True) if c["seed"] % 2 == 0 else (True, False)
        res, perf = {}, {}
        for idle in order:
            res[idle] = planner.plan(c["command"], image, idle=idle)
            perf[idle] = getattr(planner, "last_perf", None) or {}
        row = {"seed": c["seed"], "command": c["command"], "same_plan": res[False]["steps"] == res[True]["steps"]}
        for idle, place in PLACES.items():
            row[f"{place}_ms"] = round(res[idle]["ms"])
            for k in ("ttft_ms", "tpot_ms", "tokens_per_s", "input_tokens", "output_tokens"):
                row[f"{place}_{k}"] = perf[idle].get(k)
        rows.append(row)
        print(f"seed {c['seed']}: E-cores {row['e_cores_ms'] / 1000:.1f} s, all cores {row['all_cores_ms'] / 1000:.1f} s, "
              f"same plan {row['same_plan']}; TTFT {row['e_cores_ttft_ms']} / {row['all_cores_ttft_ms']} ms", flush=True)
    summary = {"n": len(rows), "same_plan": sum(r["same_plan"] for r in rows),
               "e_cores_config": cores.planner_config(), "all_cores_config": "OpenVINO defaults"}
    for place in PLACES.values():
        for k in ("ms", "ttft_ms", "tpot_ms", "tokens_per_s", "input_tokens", "output_tokens"):
            summary[f"median_{place}_{k}"] = med(rows, f"{place}_{k}")
    out = OUT / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "planner_placement.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))

    def fmt(v, scale=1.0, nd=1):
        return "–" if v is None else f"{v * scale:.{nd}f}"

    s = summary
    lines = ["# First plan: E-cores vs every core", "",
             f"Qwen3-VL-4B INT4, OpenVINO GenAI, one top-camera image + command -> JSON plan; 10 demo commands; "
             f"same plan {s['same_plan']}/{s['n']}. Medians:", "",
             "| Placement | Plan (wall) | Time to first token | Per output token | Tokens/s | Input / output tokens |",
             "|---|---|---|---|---|---|"]
    for place, name in (("e_cores", "E-cores (while the arms move)"), ("all_cores", "Every core (arms still)")):
        lines.append(f"| {name} | {fmt(s[f'median_{place}_ms'], 1e-3)} s | {fmt(s[f'median_{place}_ttft_ms'], 1e-3, 2)} s | "
                     f"{fmt(s[f'median_{place}_tpot_ms'], 1, 0)} ms | {fmt(s[f'median_{place}_tokens_per_s'])} | "
                     f"{fmt(s[f'median_{place}_input_tokens'], 1, 0)} / {fmt(s[f'median_{place}_output_tokens'], 1, 0)} |")
    lines += ["", "| seed | command | E-cores | all cores | TTFT E / all | same plan |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['seed']} | {r['command']} | {r['e_cores_ms'] / 1000:.1f} s | {r['all_cores_ms'] / 1000:.1f} s | "
              f"{fmt(r['e_cores_ttft_ms'], 1e-3, 2)} / {fmt(r['all_cores_ttft_ms'], 1e-3, 2)} s | "
              f"{'yes' if r['same_plan'] else 'NO'} |" for r in rows]
    (out / "planner_placement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()

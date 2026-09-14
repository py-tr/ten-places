"""Power and energy per inference on the CPU, with HWiNFO64's CPU package power as the meter.

Windows exposes the CPU's package power (RAPL) only through a signed kernel driver, so HWiNFO64 (run as
administrator) logs it to a CSV while this script runs fixed, timestamped phases; `analyze` joins the two.

    python scripts/power_bench.py run --seconds 60                  # while HWiNFO is logging
    python scripts/power_bench.py analyze --csv out/power/hwinfo.csv

Phases, in order: idle (the baseline), PyTorch FP32 eager, OpenVINO FP32, OpenVINO INT8 weights (back to back, as
fast as possible), and OpenVINO INT8 weights paced at the 25 Hz control rate (what the robot actually runs). Run it
with nothing else loading the CPU. Package power is the CPU's own estimate, not wall power; energy above idle is
reported next to the total so the machine's idle draw is not charged to the model.
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CHECKPOINT = "out/train/skills_ctx/cup/checkpoints/015000/pretrained_model"  # has both FP32 and INT8-weights IRs
CONTROL_HZ = 25
DATE_FORMATS = ("%d.%m.%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y")
TIME_FORMATS = ("%H:%M:%S.%f", "%H:%M:%S")


def example_inputs(policy):
    """One observation of the right shapes (as scripts/benchmark.py): state, skill one-hot if used, camera images."""
    import torch

    cfg = policy.policy.config
    x = [torch.randn(1, cfg.robot_state_feature.shape[0])]
    if cfg.env_state_feature is not None:
        x.append(torch.zeros(1, cfg.env_state_feature.shape[0]))
        x[-1][0, 0] = 1
    x += [torch.randn(1, *cfg.image_features[k].shape) for k in cfg.image_features]
    return x


def run_phases(checkpoint: Path, seconds: float, out: Path, max_busy: float = 5.0):
    import openvino as ov
    import psutil
    import torch

    from tenplaces.lerobot_policy import LeRobotPolicy
    from tenplaces.ov_backend import ACTCore

    # A power number from a busy machine is not a power number (the first run's "idle" phase was 28% busy).
    busy = psutil.cpu_percent(interval=5.0)
    if busy > max_busy:
        raise SystemExit(f"CPU {busy:.0f}% busy before the run (limit {max_busy:.0f}%): stop other jobs and apps first")
    print(f"CPU {busy:.1f}% busy before the run: ok", flush=True)

    pol = LeRobotPolicy(checkpoint, device="cpu")
    x = example_inputs(pol)
    feed = [t.numpy() for t in x]
    net = ACTCore(pol.policy.model, has_env_state=pol.policy.config.env_state_feature is not None).eval()
    core = ov.Core()
    compiled = {name: core.compile_model(core.read_model(checkpoint / "openvino" / f"act_{name}.xml"), "CPU",
                                         {"PERFORMANCE_HINT": "LATENCY"}) for name in ("fp32", "w8")}

    def flat_out(fn):
        def phase():
            n, t_end = 0, time.perf_counter() + seconds
            while time.perf_counter() < t_end:
                fn()
                n += 1
            return n
        return phase

    def paced(fn, hz):
        def phase():
            n, period = 0, 1.0 / hz
            tick = t0 = time.perf_counter()
            while time.perf_counter() - t0 < seconds:
                fn()
                n += 1
                tick += period
                time.sleep(max(0.0, tick - time.perf_counter()))
            return n
        return phase

    def idle():
        time.sleep(seconds)
        return 0

    from tenplaces.cores import control_config

    # What the robot runs by default (scripts/run_agent.py --cores split): P-cores only, pinned, hyper-threading off.
    pinned = core.compile_model(core.read_model(checkpoint / "openvino" / "act_w8.xml"), "CPU",
                                {"PERFORMANCE_HINT": "LATENCY", **control_config()})
    with torch.no_grad():
        phases = [("idle", idle),
                  ("pytorch_fp32", flat_out(lambda: net(*x))),
                  ("openvino_fp32", flat_out(lambda: compiled["fp32"](feed))),
                  ("openvino_int8w", flat_out(lambda: compiled["w8"](feed))),
                  (f"openvino_int8w_{CONTROL_HZ}hz", paced(lambda: compiled["w8"](feed), CONTROL_HZ)),
                  (f"openvino_int8w_{CONTROL_HZ}hz_pcores_pinned", paced(lambda: pinned(feed), CONTROL_HZ))]
        # Warm up every variant so compilation and first-call costs stay out of the timed phases.
        for fn in (lambda: net(*x), lambda: compiled["fp32"](feed), lambda: compiled["w8"](feed), lambda: pinned(feed)):
            for _ in range(10):
                fn()
        rows = []
        for name, fn in phases:
            time.sleep(5.0)  # let the package power settle between phases
            psutil.cpu_percent(None)  # start the CPU-load window for this phase
            start = datetime.now()
            n = fn()
            end = datetime.now()
            cpu = psutil.cpu_percent(None)  # average load over the phase, all logical CPUs
            rows.append({"phase": name, "start": start.isoformat(), "end": end.isoformat(), "inferences": n,
                         "cpu_pct": cpu})
            print(f"{name}: {n} inferences in {(end - start).total_seconds():.1f} s, CPU {cpu:.0f}%", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": str(checkpoint), "cpu": core.get_property("CPU", "FULL_DEVICE_NAME"),
                               "seconds": seconds, "phases": rows}, indent=1))
    print(f"phases -> {out}")


def _parse_stamp(date_s: str, time_s: str) -> datetime | None:
    for df in DATE_FORMATS:
        for tf in TIME_FORMATS:
            try:
                return datetime.strptime(f"{date_s.strip()} {time_s.strip()}", f"{df} {tf}")
            except ValueError:
                continue
    return None


def read_hwinfo(csv_path: Path, column_hint: str = "CPU Package Power"):
    """(timestamps, watts) from a HWiNFO sensor log. Picks the first column whose header contains `column_hint`;
    skips the header HWiNFO repeats at the end of the file and any row that does not parse."""
    raw = None
    for enc in ("utf-8-sig", "cp1252"):
        try:
            raw = csv_path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    rows = list(csv.reader(raw.splitlines()))
    header = rows[0]
    cols = [i for i, h in enumerate(header) if column_hint.lower() in h.lower()]
    if not cols:
        raise SystemExit(f"no column containing {column_hint!r} in {csv_path}; columns: {header[:12]} ...")
    col = cols[0]
    stamps, watts = [], []
    for r in rows[1:]:
        if len(r) <= col:
            continue
        t = _parse_stamp(r[0], r[1])
        if t is None:
            continue
        try:
            w = float(r[col].replace(",", "."))
        except ValueError:
            continue
        stamps.append(t)
        watts.append(w)
    return stamps, watts, header[col]


def analyze(phases: dict, stamps, watts) -> list[dict]:
    """Mean package power per phase and energy per inference, total and above the idle phase's mean."""
    out = []
    for p in phases["phases"]:
        t0, t1 = datetime.fromisoformat(p["start"]), datetime.fromisoformat(p["end"])
        inside = [w for t, w in zip(stamps, watts) if t0 <= t <= t1]
        dur = (t1 - t0).total_seconds()
        out.append({"phase": p["phase"], "samples": len(inside), "seconds": round(dur, 1), "inferences": p["inferences"],
                    "mean_w": sum(inside) / len(inside) if inside else None, "cpu_pct": p.get("cpu_pct")})
    idle = next((r["mean_w"] for r in out if r["phase"] == "idle"), None)
    for r in out:
        n, w = r["inferences"], r["mean_w"]
        r["mj_per_inference"] = 1000 * w * r["seconds"] / n if n and w is not None else None
        r["mj_per_inference_above_idle"] = (1000 * (w - idle) * r["seconds"] / n
                                            if n and w is not None and idle is not None else None)
        r["inferences_per_s"] = n / r["seconds"] if n else None
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--checkpoint", default=CHECKPOINT)
    r.add_argument("--seconds", type=float, default=60.0)
    r.add_argument("--out", default="out/power/phases.json")
    a = sub.add_parser("analyze")
    a.add_argument("--csv", required=True)
    a.add_argument("--phases", default="out/power/phases.json")
    a.add_argument("--column", default="CPU Package Power")
    args = ap.parse_args()
    from tenplaces.cores import no_power_throttling

    no_power_throttling()
    if args.cmd == "run":
        run_phases(Path(args.checkpoint), args.seconds, Path(args.out))
        return
    phases = json.loads(Path(args.phases).read_text())
    stamps, watts, col = read_hwinfo(Path(args.csv), args.column)
    rows = analyze(phases, stamps, watts)
    fmt = lambda v, f: "–" if v is None else format(v, f)  # noqa: E731
    lines = [f"# Power per inference — {phases.get('cpu', '')}", "",
             f"Meter: HWiNFO64 `{col}` (CPU package, not wall power). Checkpoint: `{phases['checkpoint']}`.", "",
             "| phase | CPU busy | mean package power | inferences/s | energy per inference | above idle |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['phase']} | {fmt(r['cpu_pct'], '.0f')}% | {fmt(r['mean_w'], '.1f')} W ({r['samples']} samples) | "
                     f"{fmt(r['inferences_per_s'], '.0f')} | {fmt(r['mj_per_inference'], '.0f')} mJ | "
                     f"{fmt(r['mj_per_inference_above_idle'], '.0f')} mJ |")
    idle_cpu = next((r["cpu_pct"] for r in rows if r["phase"] == "idle"), None)
    if idle_cpu is not None and idle_cpu > 5:
        lines += ["", f"**Not valid: the idle phase was {idle_cpu:.0f}% busy — something else was running.**"]
    out = Path(args.phases).with_name("power.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.phases).with_name("power.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

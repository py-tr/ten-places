"""Intel inference benchmark for a trained ACT checkpoint (Intel deliverable 3).

Reports, per OpenVINO variant: precision, device, latency at batch 1 (median / p95, LATENCY hint),
throughput with async requests (THROUGHPUT hint), IR size, and a PyTorch CPU baseline; then sweeps the
CPU scheduling options for the chosen variant (performance vs efficiency cores, hyper-threading).
Task success per variant is read from an evaluation summary when given (success is measured by the
closed-loop evaluators, never here).

    python scripts/benchmark.py --checkpoint out/train/act_handoff_v1/checkpoints/075000/pretrained_model \
        --success out/eval/act_handoff_v1_resumed

Run it with nothing else loading the CPU: these are wall-clock numbers.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.lerobot_policy import LeRobotPolicy  # noqa: E402
from tenplaces.ov_backend import ACTCore  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402

VARIANTS = {  # IR name in <checkpoint>/openvino -> label
    "fp32": "FP32",
    "w8": "INT8 weights only",
    "a8w8_backbone": "INT8 image encoder (weights+activations), transformer float",
    "a8w8": "INT8 everything (weights+activations)",
}


def example_inputs(policy):
    cfg = policy.policy.config
    x = [torch.randn(1, cfg.robot_state_feature.shape[0])]
    if cfg.env_state_feature is not None:
        x.append(torch.zeros(1, cfg.env_state_feature.shape[0]))
        x[-1][0, 0] = 1
    x += [torch.randn(1, *cfg.image_features[k].shape) for k in cfg.image_features]
    return x


def timeit(fn, n, warmup=10):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        t.append(1000 * (time.perf_counter() - s))
    return float(np.median(t)), float(np.percentile(t, 95))


def throughput(core, model, feed, device, seconds=5.0, config=None):
    compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": "THROUGHPUT", **(config or {})})
    nreq = compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    queue = ov.AsyncInferQueue(compiled, nreq)
    done = [0]
    queue.set_callback(lambda req, _: done.__setitem__(0, done[0] + 1))
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        queue.start_async(feed)
    queue.wait_all()
    return done[0] / (time.perf_counter() - t0), nreq


def success_for(success_dir: Path | None, label_contains: str):
    if success_dir is None:
        return None
    for f in sorted(success_dir.glob("*_summary.json")):
        s = json.loads(f.read_text())
        if label_contains in s.get("label", ""):
            return f"{s['success']}/{s['episodes']}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="CPU", help="OpenVINO device: CPU (and GPU/NPU on a Core Ultra)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--success", default=None, help="evaluation output dir with *_summary.json")
    ap.add_argument("--sweep-variant", default="w8")
    ap.add_argument("--name", default=None,
                    help="output name (default <run or skill>_<step>: skill checkpoints share step folder names)")
    args = ap.parse_args()
    ck = Path(args.checkpoint)
    name = args.name or f"{ck.parents[2].name}_{ck.parent.name}"
    core = ov.Core()
    # Which OpenVINO devices this machine has: on a Core Ultra the list shows GPU and NPU, and --device picks one.
    available = {d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices}
    print(f"[openvino] devices on this machine: {available}; benchmarking on {args.device}", flush=True)
    dev_name = core.get_property(args.device, "FULL_DEVICE_NAME")
    if args.device == "GPU" and "Intel" not in dev_name:
        sys.exit(f"OpenVINO 'GPU' here is {dev_name!r}, not Intel graphics; refusing to report it as an Intel result.")

    pol = LeRobotPolicy(ck, device="cpu")
    x = example_inputs(pol)
    feed = [t.numpy() for t in x]
    net = ACTCore(pol.policy.model, has_env_state=pol.policy.config.env_state_feature is not None).eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    with torch.no_grad():
        pt_med, pt_p95 = timeit(lambda: net(*x), args.iters // 4)

    rows = [{"variant": "pytorch", "precision": "FP32 (PyTorch eager)", "device": "CPU", "latency_ms_median": pt_med,
             "latency_ms_p95": pt_p95, "throughput_ips": None, "ir_mb": None, "success": None}]
    models = {}
    for variant, label in VARIANTS.items():  # not `name`: that is the output file's name
        xml = ck / "openvino" / f"act_{variant}.xml"
        if not xml.exists():
            print(f"skip {variant}: {xml} not built (run an evaluation or int8_study with that variant)", flush=True)
            continue
        model = core.read_model(xml)
        models[variant] = model
        compiled = core.compile_model(model, args.device, {"PERFORMANCE_HINT": "LATENCY"})
        med, p95 = timeit(lambda: compiled(feed), args.iters)
        ips, nreq = throughput(core, model, feed, args.device)
        rows.append({"variant": variant, "precision": label, "device": args.device, "latency_ms_median": med,
                     "latency_ms_p95": p95, "throughput_ips": ips, "infer_requests": nreq,
                     "ir_mb": round(xml.with_suffix(".bin").stat().st_size / 2**20, 1),
                     "success": success_for(Path(args.success) if args.success else None,
                                            {"fp32": "ov_fp32", "w8": "ov_w8", "a8w8_backbone": "int8_backbone", "a8w8": "ov_int8"}[variant])})
        print(json.dumps(rows[-1]), flush=True)

    sweep = []
    if args.device == "CPU" and args.sweep_variant in models:
        for core_type in ("ANY_CORE", "PCORE_ONLY", "ECORE_ONLY"):
            for ht in (True, False):
                cfg = {"PERFORMANCE_HINT": "LATENCY", "SCHEDULING_CORE_TYPE": core_type, "ENABLE_HYPER_THREADING": ht}
                try:
                    c = core.compile_model(models[args.sweep_variant], "CPU", cfg)
                    med, p95 = timeit(lambda: c(feed), args.iters // 2)
                    sweep.append({"core_type": core_type, "hyper_threading": ht, "latency_ms_median": med, "latency_ms_p95": p95})
                except Exception as e:  # not every CPU supports every scheduling option
                    sweep.append({"core_type": core_type, "hyper_threading": ht, "error": str(e)[:80]})

    env = {"cpu": core.get_property("CPU", "FULL_DEVICE_NAME"), "device": dev_name, "openvino": ov.__version__,
           "available_devices": available, "torch": torch.__version__, "python": platform.python_version(),
           "os": platform.platform(), "checkpoint": str(ck)}
    out = OUT / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps({"env": env, "rows": rows, "cpu_sweep": sweep}, indent=1))
    lines = [f"# Benchmark — {name}", "", f"CPU: {env['cpu']} · OpenVINO {env['openvino']} · PyTorch {env['torch']}",
             "", "OpenVINO devices on this machine: "
             + ", ".join(f"{d} ({n}{'' if 'Intel' in n else ' — not Intel, not benchmarked'})" for d, n in available.items())
             + f" · measured on {args.device}", "", "| variant | precision | device | latency median / p95 (ms) | throughput (inf/s) | IR size | task success |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        thr = f"{r['throughput_ips']:.0f}" if r["throughput_ips"] else "–"
        lines.append(f"| {r['variant']} | {r['precision']} | {r['device']} | {r['latency_ms_median']:.1f} / {r['latency_ms_p95']:.1f} "
                     f"| {thr} | {r['ir_mb'] or '–'} MB | {r['success'] or '–'} |")
    if sweep:
        lines += ["", f"CPU scheduling sweep ({args.sweep_variant}, LATENCY hint):", "", "| cores | hyper-threading | latency median / p95 (ms) |", "|---|---|---|"]
        for s in sweep:
            lines.append(f"| {s['core_type']} | {s['hyper_threading']} | " + (f"{s['latency_ms_median']:.1f} / {s['latency_ms_p95']:.1f} |" if "error" not in s else f"n/a ({s['error']}) |"))
    (out / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

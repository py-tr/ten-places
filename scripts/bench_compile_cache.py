"""OpenVINO model caching: how long each model in the loop takes to get ready, compiled from its IR on every launch
(cold) versus loaded from OpenVINO's model cache (CACHE_DIR) once one launch has written it. The model files are read
once beforehand, so the disk is not what is measured; every timing is a fresh process with a fresh ov.Core.

    python scripts/bench_compile_cache.py        # -> out/benchmark/compile_cache.{json,md}
"""
import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.paths import OUT  # noqa: E402

CACHE = OUT / "ov_cache_bench"


def targets() -> dict:
    """name -> (kind, path): the five deployed policies (OpenVINO INT8 weights), the classifier, the planner."""
    sel = json.loads((OUT / "eval" / "selected_checkpoints.json").read_text())
    t = {s: ("ir", Path(p) / "openvino" / "act_w8.xml") for s, p in sorted(sel.items())}
    t["cup"] = ("ir", Path("out/train/skills_ctx/cup/checkpoints/015000/pretrained_model/openvino/act_w8.xml"))
    t["classifier"] = ("ir", Path("models/state_classifier_v3/state_classifier.xml"))
    t["planner (Qwen3-VL-4B INT4)"] = ("vlm", Path("models/Qwen3-VL-4B-Instruct-int4-ov"))
    return t


def child(kind: str, path: str, cache: str):
    """One launch: get the model ready on the CPU, print the milliseconds it took."""
    import openvino as ov

    t = time.perf_counter()
    if kind == "ir":
        ov.Core().compile_model(path, "CPU", {"PERFORMANCE_HINT": "LATENCY", **({"CACHE_DIR": cache} if cache else {})})
    else:
        import openvino_genai as og

        og.VLMPipeline(path, "CPU", **({"CACHE_DIR": cache} if cache else {}))
    print(json.dumps({"ms": 1000 * (time.perf_counter() - t)}))


def launch(kind: str, path: Path, cache: Path | None) -> float:
    out = subprocess.run([sys.executable, __file__, "--child", kind, str(path), str(cache or "")],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])["ms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--child", nargs=3, metavar=("KIND", "PATH", "CACHE"))
    args = ap.parse_args()
    from tenplaces.cores import no_power_throttling

    no_power_throttling()
    if args.child:
        return child(*args.child)
    rows = []
    for name, (kind, path) in targets().items():
        files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
        for f in files + [f.with_suffix(".bin") for f in files if f.suffix == ".xml"]:
            if f.exists():
                f.read_bytes()  # into the OS file cache: the timings compare compiling, not reading the disk
        cache = CACHE / name.split()[0]
        shutil.rmtree(cache, ignore_errors=True)
        cold = [launch(kind, path, None) for _ in range(args.repeats)]
        first = launch(kind, path, cache)  # writes the cache
        warm = [launch(kind, path, cache) for _ in range(args.repeats)]
        r = {"model": name, "cold_ms": round(statistics.median(cold)), "first_with_cache_ms": round(first),
             "cached_ms": round(statistics.median(warm)), "cache_mb": round(sum(p.stat().st_size for p in cache.rglob("*")
                                                                                   if p.is_file()) / 2**20, 1)}
        r["speedup"] = round(r["cold_ms"] / max(r["cached_ms"], 1), 1)
        rows.append(r)
        print(json.dumps(r), flush=True)
    shutil.rmtree(CACHE, ignore_errors=True)
    if CACHE.exists():  # a locked file on Windows can leave the ~3.7 GB cache behind; say so rather than fill the disk
        print(f"could not delete {CACHE}; remove it by hand", flush=True)
    out = OUT / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "compile_cache.json").write_text(json.dumps({"repeats": args.repeats, "rows": rows}, indent=1))
    md = ["# OpenVINO model cache: time to get each model ready on the CPU", "",
          f"Median of {args.repeats} fresh processes each; the model files are in the OS file cache beforehand.", "",
          "| model | compiled from IR (cold) | first launch, writing the cache | from the cache | speed-up | cache size |",
          "|---|---|---|---|---|---|"]
    md += [f"| {r['model']} | {r['cold_ms']} ms | {r['first_with_cache_ms']} ms | {r['cached_ms']} ms | {r['speedup']}× | "
           f"{r['cache_mb']} MB |" for r in rows]
    (out / "compile_cache.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

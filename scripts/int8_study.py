"""Which OpenVINO quantisation keeps task success? Compares, for one trained ACT checkpoint:

  fp32          reference IR
  w8            weight-only INT8 (activations stay float)
  a8w8          full INT8 post-training quantisation (weights + activations), 300 calibration frames
  a8w8_c1000    the same with 1000 calibration frames
  a8w8_acc      NNCF accuracy-controlled quantisation: layers are reverted to float, most sensitive first,
                until the action-chunk deviation from FP32 on validation frames is within --max-drop

For each: mean |action chunk - FP32| on validation frames (normalised units), CPU latency, and
closed-loop task success on held-out seeds. IRs land in <checkpoint>/openvino/act_<variant>.xml.

    python scripts/int8_study.py --checkpoint out/train/act_handoff_v1/checkpoints/040000/pretrained_model \
        --cache data/handoff_v1_cache --seeds 0 20
"""
import argparse
import json
import sys
import time
from pathlib import Path

import nncf
import numpy as np
import openvino as ov
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.evaluate import evaluate  # noqa: E402
from tenplaces.lerobot_policy import LeRobotPolicy  # noqa: E402
from tenplaces.ov_backend import ACTCore, calibration_batches  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 20], metavar=("FIRST", "STOP"))
    ap.add_argument("--max-drop", type=float, default=0.01)
    ap.add_argument("--variants", nargs="+", default=["fp32", "w8", "a8w8", "a8w8_c1000", "a8w8_acc"])
    ap.add_argument("--skip-closed-loop", action="store_true")
    args = ap.parse_args()
    ck = Path(args.checkpoint)
    ov_dir = ck / "openvino"
    ov_dir.mkdir(exist_ok=True)
    out = OUT / "int8_study" / ck.parent.name
    out.mkdir(parents=True, exist_ok=True)

    ref = LeRobotPolicy(ck, device="cpu")
    feats = list(ref.policy.config.image_features)
    to_inputs = lambda b: tuple([b["observation.state"]] + [b[k] for k in feats])  # noqa: E731
    calib300 = [to_inputs(b) for b in calibration_batches(ref._preprocess, args.cache, n=300, seed=0)]
    calib1000 = [to_inputs(b) for b in calibration_batches(ref._preprocess, args.cache, n=1000, seed=1)]
    val = [to_inputs(b) for b in calibration_batches(ref._preprocess, args.cache, n=150, seed=2)]

    net = ACTCore(ref.policy.model).eval()
    with torch.no_grad():
        fp32 = ov.convert_model(net, example_input=calib300[0])
    core = ov.Core()
    np_in = lambda t: [x.numpy() for x in t]  # noqa: E731
    fp32_c = core.compile_model(fp32, "CPU")
    val_items = [(np_in(v), fp32_c(np_in(v))[0]) for v in val]

    def deviation(compiled, items=val_items):
        return float(np.mean([np.abs(compiled(i)[0] - t).mean() for i, t in items]))

    def build(name):
        if name == "fp32":
            return fp32
        if name == "w8":
            return nncf.compress_weights(fp32.clone(), mode=nncf.CompressWeightsMode.INT8_ASYM)
        if name in ("a8w8", "a8w8_c1000"):
            cal = calib300 if name == "a8w8" else calib1000
            return nncf.quantize(fp32.clone(), nncf.Dataset([np_in(c) for c in cal]), model_type=nncf.ModelType.TRANSFORMER,
                                 subset_size=len(cal))
        if name == "a8w8_backbone":
            # Full INT8 for the ResNet vision backbone (most of the compute); transformer stays float.
            return nncf.quantize(fp32.clone(), nncf.Dataset([np_in(c) for c in calib300]), subset_size=len(calib300),
                                 ignored_scope=nncf.IgnoredScope(patterns=[r"^(?!.*backbone).*$"], validate=False))
        if name == "a8w8_acc":
            def validate(compiled, data):
                # 1 - deviation, so FP32 scores exactly 1.0 (NNCF also computes a relative drop).
                return 1.0 - float(np.mean([np.abs(compiled(i)[0] - t).mean() for i, t in data]))

            return nncf.quantize_with_accuracy_control(
                fp32.clone(), calibration_dataset=nncf.Dataset([np_in(c) for c in calib300]),
                validation_dataset=nncf.Dataset(val_items), validation_fn=validate, max_drop=args.max_drop,
                drop_type=nncf.DropType.ABSOLUTE, model_type=nncf.ModelType.TRANSFORMER, subset_size=len(calib300))
        raise ValueError(name)

    results = []
    for name in args.variants:
        t = time.time()
        xml = ov_dir / f"act_{name}.xml"
        if xml.exists() and name != "fp32":
            model = core.read_model(xml)  # built by an earlier run of this study
        else:
            model = build(name)
            ov.save_model(model, xml, compress_to_fp16=False)
        compiled = core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
        dev = deviation(compiled)
        x = np_in(val[0])
        for _ in range(5):
            compiled(x)
        lat = []
        for _ in range(40):
            s = time.perf_counter()
            compiled(x)
            lat.append(1000 * (time.perf_counter() - s))
        row = {"variant": name, "deviation_vs_fp32": dev, "latency_ms_median": float(np.median(lat)),
               "build_s": round(time.time() - t, 1)}
        print(json.dumps(row), flush=True)
        results.append(row)

    if not args.skip_closed_loop:
        for row in results:
            pol = LeRobotPolicy(ck, backend=f"ov-{row['variant']}")
            s = evaluate(pol, range(*args.seeds), out, videos=1, label=f"ov_{row['variant']}")
            row.update(success=s["success"], episodes=s["episodes"], wilson95=s["wilson95"])
    (out / "int8_study.json").write_text(json.dumps(results, indent=1))
    lines = ["| variant | mean abs dev vs FP32 | CPU latency ms (median) | task success |", "|---|---|---|---|"]
    for r in results:
        succ = f"{r['success']}/{r['episodes']} {r['wilson95']}" if "success" in r else "-"
        lines.append(f"| {r['variant']} | {r['deviation_vs_fp32']:.4f} | {r['latency_ms_median']:.1f} | {succ} |")
    (out / "int8_study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

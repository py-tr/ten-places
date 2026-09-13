"""Spike: does LeRobot ACT export to OpenVINO, match PyTorch, and survive full INT8 (NNCF) on this CPU?

    python scripts/spike_act_openvino.py --root data/smoke [--checkpoint path/to/pretrained_model]

Without --checkpoint the ACT is randomly initialised: that is enough to test export, numerical
parity, quantisation error and latency, but not task success (that comes after training).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_policy, make_policy_config  # noqa: E402
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE  # noqa: E402

from tenplaces.env import CAMERAS  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ACTCore(torch.nn.Module):
    """The bare ACT network with fixed-shape tensor inputs: normalised state + 3 normalised images."""

    def __init__(self, act_model):
        super().__init__()
        self.act = act_model

    def forward(self, state, top, a_wrist, b_wrist):
        actions, _ = self.act({OBS_STATE: state, OBS_IMAGES: [top, a_wrist, b_wrist]})
        return actions


def make_inputs(ds, idx, state_mean, state_std):
    x = ds[idx]
    state = ((x["observation.state"] - state_mean) / state_std).unsqueeze(0).float()
    imgs = [((x[f"observation.images.{c}"].unsqueeze(0).float()) - IMAGENET_MEAN) / IMAGENET_STD for c in CAMERAS]
    return (state, *imgs)


def bench(fn, inputs, n=50, warmup=5):
    for _ in range(warmup):
        fn(inputs)
    t = []
    for _ in range(n):
        s = time.perf_counter()
        fn(inputs)
        t.append(time.perf_counter() - s)
    return 1000 * float(np.median(t)), 1000 * float(np.percentile(t, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/smoke")
    ap.add_argument("--repo-id", default="local/tenplaces_handoff")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--calib", type=int, default=300)
    args = ap.parse_args()
    out = OUT / "spike_act_openvino"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    ds = LeRobotDataset(args.repo_id, root=args.root)
    cfg = make_policy_config("act", device="cpu", push_to_hub=False)
    if args.checkpoint:
        cfg.pretrained_path = args.checkpoint
    policy = make_policy(cfg, ds_meta=meta).eval()
    core = ACTCore(policy.model).eval()

    st = meta.stats["observation.state"]
    state_mean, state_std = torch.tensor(st["mean"]).float(), torch.tensor(st["std"]).float() + 1e-8
    example = make_inputs(ds, 0, state_mean, state_std)
    with torch.no_grad():
        ref = core(*example).numpy()
    print("ACT params (M):", round(sum(p.numel() for p in core.parameters()) / 1e6, 1), "| action chunk:", ref.shape)

    # 1. Convert to OpenVINO IR with static shapes (NPU-ready) and compare FP32 against PyTorch.
    ov_model = ov.convert_model(core, example_input=example)
    core_ov = ov.Core()
    ov.save_model(ov_model, out / "act_fp32.xml", compress_to_fp16=False)
    comp_fp32 = core_ov.compile_model(ov_model, "CPU")
    feed = lambda c, inp: c([t.numpy() for t in inp])[0]  # noqa: E731
    fp32 = feed(comp_fp32, example)
    parity = float(np.abs(fp32 - ref).max())

    # 2. Full INT8 post-training quantisation (weights + activations) with NNCF on real frames.
    import nncf

    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(args.calib, len(ds)), replace=False)
    calib = [tuple(t.numpy() for t in make_inputs(ds, int(i), state_mean, state_std)) for i in idx]
    q_model = nncf.quantize(ov_model, nncf.Dataset(calib), model_type=nncf.ModelType.TRANSFORMER, subset_size=len(calib))
    ov.save_model(q_model, out / "act_int8.xml")
    comp_int8 = core_ov.compile_model(q_model, "CPU")

    # 3. Output agreement on held-out frames + CPU latency.
    test_idx = [int(i) for i in rng.choice(len(ds), size=50, replace=False)]
    errs_fp32, errs_int8 = [], []
    for i in test_idx:
        inp = make_inputs(ds, i, state_mean, state_std)
        with torch.no_grad():
            r = core(*inp).numpy()
        errs_fp32.append(np.abs(feed(comp_fp32, inp) - r).mean())
        errs_int8.append(np.abs(feed(comp_int8, inp) - r).mean())
    torch.set_num_threads(max(1, torch.get_num_threads()))
    with torch.no_grad():
        pt_ms = bench(lambda inp: core(*inp), example)
    fp32_ms = bench(lambda inp: feed(comp_fp32, inp), example)
    int8_ms = bench(lambda inp: feed(comp_int8, inp), example)
    report = {
        "checkpoint": args.checkpoint or "random-init",
        "cpu": core_ov.get_property("CPU", "FULL_DEVICE_NAME"),
        "fp32_vs_torch_max_abs": parity,
        "mean_abs_err_vs_torch_fp32": float(np.mean(errs_fp32)),
        "mean_abs_err_vs_torch_int8": float(np.mean(errs_int8)),
        "latency_ms_median_p95": {"pytorch_fp32": pt_ms, "openvino_fp32": fp32_ms, "openvino_int8": int8_ms},
        "note": "errors are in normalised action units; task success is measured after training",
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

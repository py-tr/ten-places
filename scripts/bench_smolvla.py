"""SmolVLA (LeRobot's small vision-language-action model) timed on this CPU, for comparison with the deployed ACT
policies: one forward pass is one 50-action chunk from the camera images, the joint state and the instruction
(10 flow-matching steps). Stock PyTorch FP32 on the CPU, no OpenVINO export. Timing only: random images, the
published base checkpoint (lerobot/smolvla_base), not trained on this task.

    python scripts/bench_smolvla.py            # -> out/benchmark/smolvla.{json,md}
"""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.paths import OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/smolvla_base",
                    help="lerobot/smolvla_base (config.json + model.safetensors) downloaded here, or a hub id")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--command", default="Set the table.")
    args = ap.parse_args()
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
    from transformers import AutoProcessor

    # The SmolVLA checkpoint holds every weight, the backbone's too: build the backbone from its config instead of
    # downloading its separate weights as well.
    cfg = SmolVLAConfig.from_pretrained(args.checkpoint)
    cfg.load_vlm_weights, cfg.device = False, "cpu"
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint, config=cfg).to("cpu").float().eval()
    cfg = policy.config
    tok = AutoProcessor.from_pretrained(cfg.vlm_model_name).tokenizer
    enc = tok([args.command + "\n"], padding="max_length", max_length=cfg.tokenizer_max_length, truncation=True,
              return_tensors="pt")
    batch = {OBS_LANGUAGE_TOKENS: enc["input_ids"], OBS_LANGUAGE_ATTENTION_MASK: enc["attention_mask"].bool(),
             OBS_STATE: torch.zeros(1, cfg.robot_state_feature.shape[0])}
    for key, feat in cfg.image_features.items():
        batch[key] = torch.rand(1, *feat.shape)
    times = []
    for i in range(args.warmup + args.repeats):
        policy.reset()
        t = time.perf_counter()
        chunk = policy.predict_action_chunk(dict(batch))
        if i >= args.warmup:
            times.append(1000 * (time.perf_counter() - t))
    times.sort()
    res = {"checkpoint": args.checkpoint, "params_m": round(sum(p.numel() for p in policy.parameters()) / 1e6),
           "cameras": len(cfg.image_features), "image_shape": list(next(iter(cfg.image_features.values())).shape),
           "chunk": list(chunk.shape[1:]), "denoise_steps": cfg.num_steps, "torch_threads": torch.get_num_threads(),
           "median_ms": round(statistics.median(times), 1), "p90_ms": round(times[int(0.9 * (len(times) - 1))], 1),
           "repeats": len(times), "cpu": platform.processor(), "torch": torch.__version__}
    out = OUT / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "smolvla.json").write_text(json.dumps(res, indent=1))
    md = [f"# SmolVLA on the CPU ({res['cpu']})", "",
          f"`{res['checkpoint']}`, {res['params_m']}M parameters, {res['cameras']} cameras {res['image_shape']}, "
          f"{res['denoise_steps']} flow-matching steps, a chunk of {res['chunk'][0]} actions per pass. PyTorch "
          f"{res['torch']} FP32, {res['torch_threads']} threads, random inputs (timing only).", "",
          "| per pass, median | p90 | passes |", "|---|---|---|",
          f"| {res['median_ms']} ms | {res['p90_ms']} ms | {res['repeats']} |"]
    (out / "smolvla.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

"""OpenVINO backend for a trained LeRobot ACT policy.

Only the ACT network moves to OpenVINO; LeRobot's own pre/post-processors (normalisation), action
queue and temporal ensembler stay in charge, so PyTorch and OpenVINO runs differ in one place only.
IRs are cached next to the checkpoint: <checkpoint>/openvino/act_{fp32,int8}.xml.
"""
import json
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACTCore(torch.nn.Module):
    """Bare ACT network with positional tensor inputs: (state, [env_state,] *images in config order).

    env_state is the skill one-hot of skill-conditioned policies (ACT's environment_state token)."""

    def __init__(self, act_model, has_env_state: bool = False):
        super().__init__()
        self.act = act_model
        self.has_env_state = has_env_state

    def forward(self, state, *rest):
        batch = {OBS_STATE: state}
        if self.has_env_state:
            batch[OBS_ENV_STATE], rest = rest[0], rest[1:]
        batch[OBS_IMAGES] = list(rest)
        actions, _ = self.act(batch)
        return actions


class OVACT(torch.nn.Module):
    """Stands in for `ACTPolicy.model`: same call signature, runs a compiled OpenVINO model."""

    def __init__(self, compiled):
        super().__init__()
        self.compiled = compiled

    def forward(self, batch):
        inputs = [batch[OBS_STATE].numpy()]
        if OBS_ENV_STATE in batch:
            inputs.append(batch[OBS_ENV_STATE].numpy())
        inputs += [im.numpy() for im in batch[OBS_IMAGES]]
        return torch.from_numpy(self.compiled(inputs)[0]), (None, None)


def calibration_batches(preprocess, cache_dir: str | Path, n: int = 300, seed: int = 0):
    """Normalised model inputs for n random training frames, read from the fastdata memmap cache."""
    cache = Path(cache_dir)
    info = json.loads((cache / "cache_meta.json").read_text())
    z = np.load(cache / "vectors.npz")
    # Every observation vector the cache holds (state, and the skill one-hot for skill-conditioned data).
    vecs = {k: z[k.replace(".", "__")] for k in info["vec_keys"] if k.startswith("observation.")}
    imgs = {cam: np.load(cache / f"{cam.replace('.', '__')}.npy", mmap_mode="r") for cam in info["cams"]}
    idx = np.random.default_rng(seed).choice(info["n"], size=min(n, info["n"]), replace=False)
    for i in idx:
        raw = {k: torch.from_numpy(v[i].copy()).unsqueeze(0) for k, v in vecs.items()}
        for cam in info["cams"]:
            raw[cam] = torch.from_numpy(np.array(imgs[cam][i])).float().div_(255.0).unsqueeze(0)
        yield preprocess(raw)


def compile_act(policy, preprocess, precision: str, calib_cache: str | Path | None, out_dir: Path,
                device: str = "CPU", calib_frames: int = 300, ov_config: dict | None = None) -> OVACT:
    """ov_config: extra OpenVINO compile properties (e.g. tenplaces.cores.control_config()); None: latency hint only."""
    out_dir.mkdir(parents=True, exist_ok=True)
    xml = out_dir / f"act_{precision}.xml"
    core = ov.Core()
    feats = list(policy.config.image_features)
    has_env = policy.config.env_state_feature is not None
    to_inputs = lambda b: tuple([b[OBS_STATE]] + ([b[OBS_ENV_STATE]] if has_env else []) + [b[k] for k in feats])  # noqa: E731
    if xml.exists():
        model = core.read_model(xml)
    else:
        if calib_cache is None:
            raise ValueError("converting needs --calib-cache (a fastdata cache) for example/calibration inputs")
        batches = [to_inputs(b) for b in calibration_batches(preprocess, calib_cache, n=calib_frames)]
        net = ACTCore(policy.model, has_env_state=has_env).eval()
        with torch.no_grad():
            model = ov.convert_model(net, example_input=batches[0])
        import nncf

        calib = [tuple(t.numpy() for t in b) for b in batches]
        if precision in ("int8", "a8w8"):  # full INT8 (weights + activations): fastest, but costs task success
            model = nncf.quantize(model, nncf.Dataset(calib), model_type=nncf.ModelType.TRANSFORMER,
                                  subset_size=len(calib))
        elif precision == "a8w8_backbone":  # full INT8 on the ResNet image encoder only; transformer stays float
            model = nncf.quantize(model, nncf.Dataset(calib), subset_size=len(calib),
                                  ignored_scope=nncf.IgnoredScope(patterns=[r"^(?!.*backbone).*$"], validate=False))
        elif precision == "w8":  # weight-only INT8
            model = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)
        elif precision != "fp32":
            raise ValueError(f"unknown precision {precision}")
        ov.save_model(model, xml, compress_to_fp16=False)
    compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY", **(ov_config or {})})
    return OVACT(compiled)

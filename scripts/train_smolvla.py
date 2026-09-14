"""The VLA baseline: LeRobot's SmolVLA (450M parameters) fine-tuned on the cup, to compare with the ACT skill policies
on the same data, the same tuning tables and the same evaluation. Not deployed.

    python scripts/train_smolvla.py                       # 5k steps -> out/train/smolvla_cup
    python scripts/eval_skill_context.py --skill cup --deployed --ckpt out/train/smolvla_cup/checkpoints/005000/pretrained_model \
        --seeds 100 130 --name cup_smolvla005000_seeds100-129

Data: the 100 cup episodes of data/table_v1_skill (the full-table demonstrations the first ACT cup learned from; every
joint moves somewhere in that set, so MEAN_STD normalisation is safe — data/table_ctx_cup has four arm-A joints with
zero spread). Base checkpoint: models/smolvla_base (lerobot/smolvla_base: config, weights, processor files); its
backbone is built from the config (load_vlm_weights=false; the checkpoint holds every weight). The cameras are renamed
to the base model's. The instruction text of each frame is the language input. The camera names and the rename map
pass through Python, never a Windows shell (PowerShell 5.1 mangles the JSON's quotes).
"""
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    os.chdir(ROOT)
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/train/smolvla_cup")
    if out.exists():
        if any(out.rglob("model.safetensors")):
            sys.exit(f"{out} holds checkpoints; not overwriting")
        shutil.rmtree(out)  # an empty folder left by a failed start
    eps = json.loads(Path("data/table_v1_skill/tenplaces_manifest.json").read_text())["skills"]["cup"]["episodes"]
    rename = {"observation.images.top": "observation.images.camera1",
              "observation.images.a_wrist": "observation.images.camera2",
              "observation.images.b_wrist": "observation.images.camera3"}
    os.environ.update(TENPLACES_CACHE="data/table_v1_skill_cache", TENPLACES_AMP_DTYPE="bf16",
                      HF_HUB_DISABLE_XET="1", PYTHONIOENCODING="utf-8")
    sys.argv = ["scripts/train.py", "--dataset.repo_id=local/tenplaces_table", "--dataset.root=data/table_v1_skill",
                f"--dataset.episodes=[{','.join(map(str, eps))}]", "--policy.path=models/smolvla_base",
                "--policy.load_vlm_weights=false", "--policy.device=cuda", "--policy.push_to_hub=false",
                "--policy.use_amp=true", f"--rename_map={json.dumps(rename)}", "--batch_size=16", "--steps=5000",
                "--save_freq=2500", "--log_freq=100", "--num_workers=2", f"--output_dir={out}"]
    sys.path.insert(0, str(ROOT / "scripts"))
    runpy.run_path(str(ROOT / "scripts" / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()

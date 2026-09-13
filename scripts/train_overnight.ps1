# Overnight ACT training on the hand-off demos, reading from the verified memmap cache.
#   powershell -ExecutionPolicy Bypass -File scripts\train_overnight.ps1
# ~0.25 s/step on an RTX 4060 Ti -> 40k steps ~2.8 h. Checkpoints every 5k steps under $Run\checkpoints.
param(
    [int]$Steps = 40000,
    [string]$Run = "out/train/act_handoff_v1"
)
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path "data/handoff_v1_cache/cache_meta.json")) { throw "cache missing: run scripts/build_cache.py first" }
if (Test-Path $Run) { throw "$Run already exists; pick another -Run or remove it" }
New-Item -ItemType Directory -Force (Split-Path -Parent $Run) | Out-Null
$env:TENPLACES_CACHE = "data/handoff_v1_cache"
.\.venv\Scripts\python.exe scripts\train.py `
    --dataset.repo_id=local/tenplaces_handoff --dataset.root=data/handoff_v1 `
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false `
    --policy.chunk_size=50 --policy.n_action_steps=50 `
    --batch_size=32 --steps=$Steps --save_freq=5000 --log_freq=200 --num_workers=6 `
    --output_dir=$Run *> "$Run.log"

# Then score every checkpoint and compare execution settings / OpenVINO precisions on the best one.
# Results: out/eval/<run name>/summary.md
.\.venv\Scripts\python.exe scripts\eval_checkpoints.py --run $Run --calib-cache data/handoff_v1_cache `
    --seeds 0 10 --videos 2 *> "$Run-eval.log"

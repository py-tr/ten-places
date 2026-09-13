# Skill-conditioned ACT on the full dinner-table demos (skill one-hot in observation.environment_state).
#   powershell -ExecutionPolicy Bypass -File scripts\train_table.ps1
# ~117k frames; 70k steps x batch 32 ~ 19 epochs (the hand-off policy peaked after ~20), ~0.23 s/step on an
# RTX 4060 Ti with the memmap cache -> ~4.5 h. Checkpoints every 10k steps.
param(
    [int]$Steps = 70000,
    [string]$Run = "out/train/act_table_v1",
    [string]$Data = "data/table_v1_skill",
    [string]$Cache = "data/table_v1_skill_cache"
)
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path "$Cache/cache_meta.json")) { throw "cache missing: python scripts/build_cache.py --root $Data --cache $Cache" }
if (Test-Path $Run) { throw "$Run already exists; pick another -Run or remove it" }
New-Item -ItemType Directory -Force (Split-Path -Parent $Run) | Out-Null
$env:TENPLACES_CACHE = $Cache
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe scripts\train.py `
    --dataset.repo_id=local/tenplaces_table --dataset.root=$Data `
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false `
    --policy.chunk_size=50 --policy.n_action_steps=50 `
    --batch_size=32 --steps=$Steps --save_freq=10000 --log_freq=500 --num_workers=6 `
    --output_dir=$Run 2>&1 | Out-File -Encoding utf8 "$Run.log"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/zyb/infom"
PYTHON_BIN="/home/zyb/.local/opt/miniconda3/envs/elf/bin/python"
MODEL_DIR="$ROOT_DIR/models/opt-2.7b"

cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/models" "$ROOT_DIR/course_project/outputs"

# Use the plain HF download path instead of xet-backed transfers.
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1

echo "[download] Preparing facebook/opt-2.7b into $MODEL_DIR"
"$PYTHON_BIN" - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/opt-2.7b",
    local_dir="/home/zyb/infom/models/opt-2.7b",
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=4,
    allow_patterns=[
        "*.bin",
        "*.json",
        "*.txt",
        "*.model",
        "*.py",
    ],
)
PY

echo "[run] Starting watermark experiments on local OPT-2.7B"
CUDA_VISIBLE_DEVICES=7 "$PYTHON_BIN" -u course_project/scripts/run_experiments.py \
  --model_name_or_path "$MODEL_DIR" \
  --prompt_source prompts_file \
  --prompts_path course_project/data/prompts_c4_realnewslike_500.txt \
  --limit_prompts 500 \
  --output_csv course_project/outputs/opt27b_c4_500_results.csv \
  --max_new_tokens 100 \
  --use_gpu True \
  --fixed_deltas 0.5 1.0 2.0 3.0 \
  --adaptive_delta_min 0.5 \
  --adaptive_delta_max 3.0

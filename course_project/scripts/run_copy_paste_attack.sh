#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/opt-2.7b}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/course_project/outputs/additional/copy_paste}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-500}"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then echo "Model config not found: $MODEL_DIR/config.json" >&2; exit 1; fi
mkdir -p "$OUTPUT_DIR"
cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u course_project/scripts/run_copy_paste_attack.py \
  --manifest course_project/configs/copy_paste_sources.json --model_name_or_path "$MODEL_DIR" \
  --output_dir "$OUTPUT_DIR" --total_tokens 200 --ratios 0 0.25 0.50 0.75 1.0 \
  --positions beginning middle end --window_sizes 20,40,80,max --limit_samples "$LIMIT_SAMPLES" \
  --seed 1234 --use_gpu --load_fp16 --resume

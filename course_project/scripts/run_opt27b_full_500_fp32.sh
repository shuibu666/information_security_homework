#!/usr/bin/env bash
set -euo pipefail

# Full OPT-2.7B / C4-500 experiment runner.
# Defaults are conservative for the shared server:
# - use one visible GPU only
# - load OPT-2.7B in fp32
# - run generation/detection, result analysis, and PPL evaluation

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-2}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/opt-2.7b}"
PPL_MODEL_NAME_OR_PATH="${PPL_MODEL_NAME_OR_PATH:-$MODEL_DIR}"
PROMPTS_PATH="${PROMPTS_PATH:-$ROOT_DIR/course_project/data/prompts_c4_realnewslike_500.txt}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-opt27b_fp32_c4_500_full}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-0}"
RUN_PPL="${RUN_PPL:-1}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-4}"

OUTPUT_DIR="$ROOT_DIR/course_project/outputs"
RESULTS_CSV="$OUTPUT_DIR/${OUTPUT_PREFIX}_results.csv"
SUMMARY_MD="$OUTPUT_DIR/${OUTPUT_PREFIX}_summary.md"
ZSCORE_CHART="$OUTPUT_DIR/${OUTPUT_PREFIX}_zscore.png"
GREEN_CHART="$OUTPUT_DIR/${OUTPUT_PREFIX}_green.png"
TRADEOFF_CHART="$OUTPUT_DIR/${OUTPUT_PREFIX}_tradeoff.png"
WEIGHTED_AUC_CHART="$OUTPUT_DIR/${OUTPUT_PREFIX}_weighted_auc.png"
KL_DELTA_CHART="$OUTPUT_DIR/${OUTPUT_PREFIX}_kl_delta.png"
PPL_RESULTS_CSV="$OUTPUT_DIR/${OUTPUT_PREFIX}_ppl_results.csv"
PPL_SUMMARY_MD="$OUTPUT_DIR/${OUTPUT_PREFIX}_ppl_summary.md"

cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/models" "$OUTPUT_DIR"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export PYTHONUNBUFFERED=1

echo "[info] Root: $ROOT_DIR"
echo "[info] Python: $PYTHON_BIN"
echo "[info] GPU: $GPU_ID"
echo "[info] Model: $MODEL_DIR"
echo "[info] PPL scorer: $PPL_MODEL_NAME_OR_PATH"
echo "[info] Prompts: $PROMPTS_PATH"
echo "[info] Output prefix: $OUTPUT_PREFIX"
echo "[info] Download model: $DOWNLOAD_MODEL"

if [[ ! -f "$PROMPTS_PATH" ]]; then
  echo "[error] Prompt file not found: $PROMPTS_PATH" >&2
  exit 1
fi

if [[ "$DOWNLOAD_MODEL" == "1" ]]; then
  echo "[download] Preparing facebook/opt-2.7b into $MODEL_DIR"
  "$PYTHON_BIN" - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/opt-2.7b",
    local_dir="$MODEL_DIR",
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
elif [[ ! -f "$MODEL_DIR/config.json" || ! -f "$MODEL_DIR/pytorch_model.bin" ]]; then
  echo "[error] Model files not found in $MODEL_DIR" >&2
  echo "[error] Expected at least config.json and pytorch_model.bin." >&2
  echo "[error] If you want this script to download them, run with DOWNLOAD_MODEL=1." >&2
  exit 1
else
  echo "[download] Skipped; using existing model files in $MODEL_DIR"
fi

echo "[run] Starting full fp32 watermark experiment"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u course_project/scripts/run_experiments.py \
  --model_name_or_path "$MODEL_DIR" \
  --prompt_source prompts_file \
  --prompts_path "$PROMPTS_PATH" \
  --limit_prompts 500 \
  --output_csv "$RESULTS_CSV" \
  --max_new_tokens 100 \
  --use_gpu True \
  --load_fp16 False \
  --fixed_deltas 0.5 1.0 1.864 2.0 3.0 \
  --run_baseline True \
  --run_fixed True \
  --run_adaptive True \
  --adaptive_delta_min 0.5 \
  --adaptive_delta_max 3.0 \
  --run_cakl True \
  --kl_epsilon 0.50 \
  --cakl_delta_max 3.0 \
  --confidence_entropy_threshold 0.35 \
  --confidence_top1_threshold 0.85 \
  --candidate_top_p 0.95 \
  --window_sizes 20,40,80,max \
  --use_model_assisted_detector True

echo "[analyze] Writing summary and charts"
"$PYTHON_BIN" -u course_project/scripts/analyze_results.py \
  --input_csv "$RESULTS_CSV" \
  --summary_md "$SUMMARY_MD" \
  --zscore_chart "$ZSCORE_CHART" \
  --green_chart "$GREEN_CHART" \
  --tradeoff_chart "$TRADEOFF_CHART" \
  --weighted_auc_chart "$WEIGHTED_AUC_CHART" \
  --kl_delta_chart "$KL_DELTA_CHART"

if [[ "$RUN_PPL" == "1" ]]; then
  echo "[ppl] Evaluating continuation perplexity in fp32"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u course_project/scripts/evaluate_ppl.py \
    --input_csv "$RESULTS_CSV" \
    --output_csv "$PPL_RESULTS_CSV" \
    --summary_md "$PPL_SUMMARY_MD" \
    --model_name_or_path "$PPL_MODEL_NAME_OR_PATH" \
    --use_gpu True \
    --load_fp16 False \
    --batch_size "$PPL_BATCH_SIZE"
else
  echo "[ppl] Skipped because RUN_PPL=$RUN_PPL"
fi

echo "[done] Results: $RESULTS_CSV"
echo "[done] Summary: $SUMMARY_MD"
if [[ "$RUN_PPL" == "1" ]]; then
  echo "[done] PPL results: $PPL_RESULTS_CSV"
  echo "[done] PPL summary: $PPL_SUMMARY_MD"
fi

#!/usr/bin/env bash
set -euo pipefail

# Paper-aligned KGW baseline:
# - OPT-1.3B
# - C4/realnewslike
# - 500 shuffled prompts with fixed seed
# - prompt built by trimming the trailing completion from each source sample
# - target generated length 200 +- 5 tokens

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-1}"
MODEL_REPO="${MODEL_REPO:-facebook/opt-1.3b}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/opt-1.3b}"
PROMPTS_PATH="${PROMPTS_PATH:-$ROOT_DIR/course_project/data/prompts_c4_realnewslike_500_paper_t200.txt}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-opt13b_paper_c4_500_t200}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-0}"
RUN_PPL="${RUN_PPL:-1}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-8}"
DATASET_SEED="${DATASET_SEED:-1234}"

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
mkdir -p "$ROOT_DIR/models" "$OUTPUT_DIR" "$(dirname "$PROMPTS_PATH")"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export PYTHONUNBUFFERED=1

echo "[info] Root: $ROOT_DIR"
echo "[info] Python: $PYTHON_BIN"
echo "[info] GPU: $GPU_ID"
echo "[info] Model repo: $MODEL_REPO"
echo "[info] Model dir: $MODEL_DIR"
echo "[info] Prompts: $PROMPTS_PATH"
echo "[info] Output prefix: $OUTPUT_PREFIX"
echo "[info] Download model: $DOWNLOAD_MODEL"

if [[ "$DOWNLOAD_MODEL" == "1" ]]; then
  echo "[download] Preparing $MODEL_REPO into $MODEL_DIR"
  "$PYTHON_BIN" - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="$MODEL_REPO",
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
elif [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "[error] Model files not found in $MODEL_DIR" >&2
  echo "[error] If you want this script to download them, run with DOWNLOAD_MODEL=1." >&2
  exit 1
else
  echo "[download] Skipped; using existing model files in $MODEL_DIR"
fi

echo "[prepare] Exporting paper-style C4 prompts"
"$PYTHON_BIN" -u course_project/scripts/prepare_c4_prompts.py \
  --model_name_or_path "$MODEL_DIR" \
  --prompt_source hf_dataset \
  --save_loaded_prompts_path "$PROMPTS_PATH" \
  --dataset_name c4 \
  --dataset_config_name realnewslike \
  --shuffle_dataset True \
  --dataset_seed "$DATASET_SEED" \
  --min_prompt_tokens 50 \
  --max_new_tokens 200 \
  --target_new_tokens 200 \
  --length_tolerance 5 \
  --paper_style_prompt True \
  --limit_prompts 500

echo "[run] Starting paper-aligned OPT-1.3B baseline"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u course_project/scripts/run_experiments.py \
  --model_name_or_path "$MODEL_DIR" \
  --prompt_source prompts_file \
  --prompts_path "$PROMPTS_PATH" \
  --limit_prompts 500 \
  --output_csv "$RESULTS_CSV" \
  --max_new_tokens 200 \
  --target_new_tokens 200 \
  --length_tolerance 5 \
  --use_gpu True \
  --load_fp16 False \
  --shuffle_dataset True \
  --dataset_seed "$DATASET_SEED" \
  --fixed_deltas 2.0 \
  --run_baseline True \
  --run_fixed True \
  --run_adaptive False \
  --run_cakl False \
  --use_model_assisted_detector False

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
  echo "[ppl] Evaluating continuation perplexity"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u course_project/scripts/evaluate_ppl.py \
    --input_csv "$RESULTS_CSV" \
    --output_csv "$PPL_RESULTS_CSV" \
    --summary_md "$PPL_SUMMARY_MD" \
    --model_name_or_path "$MODEL_DIR" \
    --use_gpu True \
    --load_fp16 False \
    --batch_size "$PPL_BATCH_SIZE"
fi

echo "[done] Results: $RESULTS_CSV"
echo "[done] Summary: $SUMMARY_MD"
if [[ "$RUN_PPL" == "1" ]]; then
  echo "[done] PPL results: $PPL_RESULTS_CSV"
  echo "[done] PPL summary: $PPL_SUMMARY_MD"
fi

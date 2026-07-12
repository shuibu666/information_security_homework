#!/usr/bin/env bash
set -euo pipefail

# OPT-2.7B / C4-500 fp32 fixed-delta补充实验。
# 只跑 Fixed Delta 1.7，用于和 Full CA-KL-CG 的平均 delta≈1.69 做近似公平对比。
# 不重复跑 No Watermark / Adaptive / CA-KL；跑完自动生成 summary、图和 PPL。

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-2}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/opt-2.7b}"
PPL_MODEL_NAME_OR_PATH="${PPL_MODEL_NAME_OR_PATH:-$MODEL_DIR}"
PROMPTS_PATH="${PROMPTS_PATH:-$ROOT_DIR/course_project/data/prompts_c4_realnewslike_500.txt}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-opt27b_fp32_c4_500_fixed17}"
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
mkdir -p "$OUTPUT_DIR"
export PYTHONUNBUFFERED=1

echo "[info] Root: $ROOT_DIR"
echo "[info] Python: $PYTHON_BIN"
echo "[info] GPU: $GPU_ID"
echo "[info] Model: $MODEL_DIR"
echo "[info] PPL scorer: $PPL_MODEL_NAME_OR_PATH"
echo "[info] Prompts: $PROMPTS_PATH"
echo "[info] Output prefix: $OUTPUT_PREFIX"
echo "[info] Fixed delta: 1.7"

if [[ ! -f "$PROMPTS_PATH" ]]; then
  echo "[error] Prompt file not found: $PROMPTS_PATH" >&2
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" || ! -f "$MODEL_DIR/pytorch_model.bin" ]]; then
  echo "[error] Model files not found in $MODEL_DIR" >&2
  echo "[error] Expected at least config.json and pytorch_model.bin." >&2
  exit 1
fi

echo "[run] Starting fixed-delta 1.7 fp32 experiment"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u course_project/scripts/run_experiments.py \
  --model_name_or_path "$MODEL_DIR" \
  --prompt_source prompts_file \
  --prompts_path "$PROMPTS_PATH" \
  --limit_prompts 500 \
  --output_csv "$RESULTS_CSV" \
  --max_new_tokens 100 \
  --use_gpu True \
  --load_fp16 False \
  --fixed_deltas 1.7 \
  --run_baseline False \
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

echo "[ppl] Evaluating continuation perplexity in fp32"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u course_project/scripts/evaluate_ppl.py \
  --input_csv "$RESULTS_CSV" \
  --output_csv "$PPL_RESULTS_CSV" \
  --summary_md "$PPL_SUMMARY_MD" \
  --model_name_or_path "$PPL_MODEL_NAME_OR_PATH" \
  --use_gpu True \
  --load_fp16 False \
  --batch_size "$PPL_BATCH_SIZE"

echo "[done] Results: $RESULTS_CSV"
echo "[done] Summary: $SUMMARY_MD"
echo "[done] PPL results: $PPL_RESULTS_CSV"
echo "[done] PPL summary: $PPL_SUMMARY_MD"

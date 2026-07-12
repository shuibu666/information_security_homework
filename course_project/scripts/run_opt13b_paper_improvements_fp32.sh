#!/usr/bin/env bash
set -euo pipefail

# Paper-controlled comparison of every proposed improvement.  Unlike the
# OPT-2.7B exploratory table, every generation here uses OPT-1.3B with the
# same C4 prompts, paper-style 200-token completions, and sampling/watermark
# constants as the baseline reproduction. PPL is scored by OPT-2.7B.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"
MODEL_REPO="${MODEL_REPO:-facebook/opt-1.3b}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/opt-1.3b}"
PPL_MODEL_NAME_OR_PATH="${PPL_MODEL_NAME_OR_PATH:-$ROOT_DIR/models/opt-2.7b}"
PROMPTS_PATH="${PROMPTS_PATH:-$ROOT_DIR/course_project/data/prompts_c4_realnewslike_500_paper_t200.txt}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-opt13b_paper_c4_500_t200_improved_fp32}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-4}"
DATASET_SEED="${DATASET_SEED:-1234}"
LIMIT_PROMPTS="${LIMIT_PROMPTS:-500}"

OUTPUT_DIR="$ROOT_DIR/course_project/outputs/additional/paper_controlled"
RESULTS_CSV="$OUTPUT_DIR/${OUTPUT_PREFIX}_results.csv"
PPL_RESULTS_CSV="$OUTPUT_DIR/${OUTPUT_PREFIX}_ppl_results.csv"

cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/models" "$OUTPUT_DIR" "$(dirname "$PROMPTS_PATH")"
export PYTHONUNBUFFERED=1

if [[ ! -f "$MODEL_DIR/config.json" || ( ! -f "$MODEL_DIR/pytorch_model.bin" && ! -f "$MODEL_DIR/model.safetensors" ) ]]; then
  echo "[download] Downloading $MODEL_REPO to $MODEL_DIR"
  "$PYTHON_BIN" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$MODEL_REPO",
    local_dir="$MODEL_DIR",
    allow_patterns=["*.bin", "*.safetensors", "*.json", "*.txt", "*.model"],
)
PY
fi

if [[ ! -f "$PROMPTS_PATH" ]]; then
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
    --limit_prompts "$LIMIT_PROMPTS"
fi

echo "[run] OPT-1.3B paper-controlled improvements on GPU $GPU_ID"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u course_project/scripts/run_experiments.py \
  --model_name_or_path "$MODEL_DIR" \
  --prompt_source prompts_file \
  --prompts_path "$PROMPTS_PATH" \
  --limit_prompts "$LIMIT_PROMPTS" \
  --output_csv "$RESULTS_CSV" \
  --max_new_tokens 200 \
  --target_new_tokens 200 \
  --length_tolerance 5 \
  --use_gpu True \
  --load_fp16 False \
  --generation_seed 1234 \
  --use_sampling True \
  --sampling_temp 0.7 \
  --gamma 0.25 \
  --fixed_deltas 2.0 \
  --run_baseline True \
  --run_fixed True \
  --run_adaptive True \
  --adaptive_delta_min 0.5 \
  --adaptive_delta_max 3.0 \
  --adaptive_entropy_floor 0.20 \
  --adaptive_delta_exponent 0.5 \
  --run_cakl True \
  --kl_epsilon 0.50 \
  --cakl_delta_max 3.0 \
  --candidate_top_p 0.95 \
  --confidence_entropy_threshold 0.10 \
  --confidence_top1_threshold 0.95 \
  --window_sizes 20,40,80,max \
  --use_model_assisted_detector False

"$PYTHON_BIN" -u course_project/scripts/analyze_results.py \
  --input_csv "$RESULTS_CSV" \
  --summary_md "$OUTPUT_DIR/${OUTPUT_PREFIX}_summary.md" \
  --zscore_chart "$OUTPUT_DIR/${OUTPUT_PREFIX}_zscore.png" \
  --green_chart "$OUTPUT_DIR/${OUTPUT_PREFIX}_green.png" \
  --tradeoff_chart "$OUTPUT_DIR/${OUTPUT_PREFIX}_tradeoff.png" \
  --weighted_auc_chart "$OUTPUT_DIR/${OUTPUT_PREFIX}_weighted_auc.png" \
  --kl_delta_chart "$OUTPUT_DIR/${OUTPUT_PREFIX}_kl_delta.png"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u course_project/scripts/evaluate_ppl.py \
  --input_csv "$RESULTS_CSV" \
  --output_csv "$PPL_RESULTS_CSV" \
  --summary_md "$OUTPUT_DIR/${OUTPUT_PREFIX}_ppl_summary.md" \
  --model_name_or_path "$PPL_MODEL_NAME_OR_PATH" \
  --use_gpu True \
  --load_fp16 False \
  --batch_size "$PPL_BATCH_SIZE"

if [[ "$LIMIT_PROMPTS" -ge 5 ]]; then
  "$PYTHON_BIN" -u course_project/scripts/calibrate_detection.py \
    --input_csv "$RESULTS_CSV" \
    --ppl_csv "$PPL_RESULTS_CSV" \
    --output_dir "$OUTPUT_DIR/calibration" \
    --folds 5 --seed 1234 --bootstrap_repetitions 2000 --confidence_level 0.95
else
  echo "[calibration] Skipped because five-fold calibration requires at least 5 prompts."
fi

echo "[done] Results: $RESULTS_CSV"

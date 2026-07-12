#!/usr/bin/env bash
set -euo pipefail

# Full 500-prompt KL-budget scan.  Fresh files are written under
# outputs/additional/pareto/raw and never overwrite earlier formal results.
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/opt-2.7b}"
PROMPTS_PATH="${PROMPTS_PATH:-$ROOT_DIR/course_project/data/prompts_c4_realnewslike_500.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/course_project/outputs/additional/pareto}"
EPSILONS="${EPSILONS:-0.20 0.30 0.35 0.40 0.50}"

if [[ ! -f "$PROMPTS_PATH" ]]; then echo "Prompt file not found: $PROMPTS_PATH" >&2; exit 1; fi
if [[ ! -f "$MODEL_DIR/config.json" ]]; then echo "Model config not found: $MODEL_DIR/config.json" >&2; exit 1; fi
if [[ ! -f "$MODEL_DIR/pytorch_model.bin" && ! -f "$MODEL_DIR/model.safetensors" ]]; then echo "Model weights not found in $MODEL_DIR" >&2; exit 1; fi

mkdir -p "$OUTPUT_DIR/raw"
cd "$ROOT_DIR"
for epsilon in $EPSILONS; do
  tag="${epsilon/./}"
  prefix="$OUTPUT_DIR/raw/cakl_opt27b_c4_500_eps${tag}"
  echo "[run] epsilon=$epsilon -> $prefix"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u course_project/scripts/run_experiments.py \
    --model_name_or_path "$MODEL_DIR" --prompt_source prompts_file --prompts_path "$PROMPTS_PATH" \
    --limit_prompts 500 --output_csv "${prefix}_results.csv" --max_new_tokens 100 \
    --use_gpu True --load_fp16 False --run_baseline True --run_fixed False --run_adaptive False \
    --run_cakl True --kl_epsilon "$epsilon" --cakl_delta_max 3.0 --candidate_top_p 0.95 \
    --confidence_entropy_threshold 0.10 --confidence_top1_threshold 0.95 \
    --window_sizes 20,40,80,max --use_model_assisted_detector True
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u course_project/scripts/evaluate_ppl.py \
    --input_csv "${prefix}_results.csv" --output_csv "${prefix}_ppl_results.csv" \
    --summary_md "${prefix}_ppl_summary.md" --model_name_or_path "$MODEL_DIR" \
    --use_gpu True --load_fp16 False --batch_size 4
  "$PYTHON_BIN" -u course_project/scripts/calibrate_detection.py \
    --input_csv "${prefix}_results.csv" --ppl_csv "${prefix}_ppl_results.csv" \
    --output_dir "$OUTPUT_DIR/calibration_eps${tag}" --folds 5 --seed 1234 --bootstrap_repetitions 2000
done

"$PYTHON_BIN" -u course_project/scripts/prepare_pareto_manifest.py \
  --template_manifest course_project/configs/pareto_sources.json --raw_dir "$OUTPUT_DIR/raw" \
  --output_manifest "$OUTPUT_DIR/pareto_sources_complete.json" --epsilons $EPSILONS
"$PYTHON_BIN" -u course_project/scripts/build_pareto.py \
  --manifest "$OUTPUT_DIR/pareto_sources_complete.json" --output_dir "$OUTPUT_DIR"

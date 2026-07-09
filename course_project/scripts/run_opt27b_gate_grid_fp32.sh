#!/usr/bin/env bash
set -euo pipefail

# First-round gate tuning for OPT-2.7B CA-KL-CG.
# This fixes KL/candidate settings and sweeps confidence gate thresholds.
# By default it runs only baseline + CA-KL variants, without fixed/adaptive
# baselines or PPL, so the grid is much faster than the full experiment.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-2}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/opt-2.7b}"
PROMPTS_PATH="${PROMPTS_PATH:-$ROOT_DIR/course_project/data/prompts_c4_realnewslike_500.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/course_project/outputs/gate_grid_opt27b_fp32}"
LIMIT_PROMPTS="${LIMIT_PROMPTS:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
LOAD_FP16="${LOAD_FP16:-False}"
KL_EPSILON="${KL_EPSILON:-0.50}"
CAKL_DELTA_MAX="${CAKL_DELTA_MAX:-3.0}"
CANDIDATE_TOP_P="${CANDIDATE_TOP_P:-0.95}"
WINDOW_SIZES="${WINDOW_SIZES:-20,40,80,max}"
RUN_PPL="${RUN_PPL:-0}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-4}"

# Override like:
# GATE_GRID="0.10:0.95 0.20:0.95" bash course_project/scripts/run_opt27b_gate_grid_fp32.sh
GATE_GRID="${GATE_GRID:-0.10:0.95 0.20:0.95 0.25:0.90 0.30:0.90 0.35:0.85}"

cd "$ROOT_DIR"
mkdir -p "$OUTPUT_DIR"
export PYTHONUNBUFFERED=1

echo "[info] Root: $ROOT_DIR"
echo "[info] Python: $PYTHON_BIN"
echo "[info] GPU: $GPU_ID"
echo "[info] Model: $MODEL_DIR"
echo "[info] Prompts: $PROMPTS_PATH"
echo "[info] Output dir: $OUTPUT_DIR"
echo "[info] Limit prompts: $LIMIT_PROMPTS"
echo "[info] KL epsilon: $KL_EPSILON"
echo "[info] Candidate top-p: $CANDIDATE_TOP_P"
echo "[info] Gate grid: $GATE_GRID"

if [[ ! -f "$PROMPTS_PATH" ]]; then
  echo "[error] Prompt file not found: $PROMPTS_PATH" >&2
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" || ! -f "$MODEL_DIR/pytorch_model.bin" ]]; then
  echo "[error] Model files not found in $MODEL_DIR" >&2
  echo "[error] Expected at least config.json and pytorch_model.bin." >&2
  exit 1
fi

sanitize_float() {
  printf "%s" "$1" | tr -d "."
}

for pair in $GATE_GRID; do
  entropy_threshold="${pair%%:*}"
  top1_threshold="${pair##*:}"
  entropy_tag="$(sanitize_float "$entropy_threshold")"
  top1_tag="$(sanitize_float "$top1_threshold")"
  run_name="h${entropy_tag}_t${top1_tag}_n${LIMIT_PROMPTS}"

  results_csv="$OUTPUT_DIR/${run_name}_results.csv"
  summary_md="$OUTPUT_DIR/${run_name}_summary.md"
  zscore_chart="$OUTPUT_DIR/${run_name}_zscore.png"
  green_chart="$OUTPUT_DIR/${run_name}_green.png"
  tradeoff_chart="$OUTPUT_DIR/${run_name}_tradeoff.png"
  weighted_auc_chart="$OUTPUT_DIR/${run_name}_weighted_auc.png"
  kl_delta_chart="$OUTPUT_DIR/${run_name}_kl_delta.png"
  ppl_results_csv="$OUTPUT_DIR/${run_name}_ppl_results.csv"
  ppl_summary_md="$OUTPUT_DIR/${run_name}_ppl_summary.md"

  echo "[run] Gate entropy=$entropy_threshold top1=$top1_threshold -> $run_name"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u course_project/scripts/run_experiments.py \
    --model_name_or_path "$MODEL_DIR" \
    --prompt_source prompts_file \
    --prompts_path "$PROMPTS_PATH" \
    --limit_prompts "$LIMIT_PROMPTS" \
    --output_csv "$results_csv" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --use_gpu True \
    --load_fp16 "$LOAD_FP16" \
    --fixed_deltas 1.0 \
    --run_baseline True \
    --run_fixed False \
    --run_adaptive False \
    --run_cakl True \
    --kl_epsilon "$KL_EPSILON" \
    --cakl_delta_max "$CAKL_DELTA_MAX" \
    --confidence_entropy_threshold "$entropy_threshold" \
    --confidence_top1_threshold "$top1_threshold" \
    --candidate_top_p "$CANDIDATE_TOP_P" \
    --window_sizes "$WINDOW_SIZES" \
    --use_model_assisted_detector True

  echo "[analyze] $run_name"
  "$PYTHON_BIN" -u course_project/scripts/analyze_results.py \
    --input_csv "$results_csv" \
    --summary_md "$summary_md" \
    --zscore_chart "$zscore_chart" \
    --green_chart "$green_chart" \
    --tradeoff_chart "$tradeoff_chart" \
    --weighted_auc_chart "$weighted_auc_chart" \
    --kl_delta_chart "$kl_delta_chart"

  if [[ "$RUN_PPL" == "1" ]]; then
    echo "[ppl] $run_name"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u course_project/scripts/evaluate_ppl.py \
      --input_csv "$results_csv" \
      --output_csv "$ppl_results_csv" \
      --summary_md "$ppl_summary_md" \
      --model_name_or_path "$MODEL_DIR" \
      --use_gpu True \
      --load_fp16 "$LOAD_FP16" \
      --batch_size "$PPL_BATCH_SIZE"
  fi
done

echo "[aggregate] Writing compact grid table"
"$PYTHON_BIN" - <<PY
import csv
from collections import defaultdict
from pathlib import Path

output_dir = Path("$OUTPUT_DIR")
summary_csv = output_dir / "gate_grid_compact_summary.csv"
rows_out = []

def to_float(value):
    value = str(value or "").strip()
    return float(value) if value else None

def avg(rows, field):
    vals = [to_float(row.get(field)) for row in rows]
    vals = [value for value in vals if value is not None]
    return sum(vals) / len(vals) if vals else ""

for results_path in sorted(output_dir.glob("*_results.csv")):
    stem_parts = results_path.stem.split("_")
    entropy_tag = stem_parts[0].removeprefix("h")
    top1_tag = stem_parts[1].removeprefix("t")
    entropy = "0." + entropy_tag[1:] if entropy_tag.startswith("0") else entropy_tag
    top1 = "0." + top1_tag[1:] if top1_tag.startswith("0") else top1_tag

    with results_path.open(encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    for method, method_rows in sorted(grouped.items()):
        weighted_values = [str(row.get("weighted_prediction", "")).lower() for row in method_rows]
        rows_out.append(
            {
                "run": results_path.stem.removesuffix("_results"),
                "entropy_threshold": entropy,
                "top1_threshold": top1,
                "method": method,
                "samples": len(method_rows),
                "kgw_success_rate": sum(row.get("prediction") == "Watermarked" for row in method_rows) / len(method_rows),
                "weighted_success_rate": sum(value == "true" for value in weighted_values) / len(method_rows),
                "avg_z_score": avg(method_rows, "z_score"),
                "avg_weighted_z_score": avg(method_rows, "weighted_z_score"),
                "avg_winmax_weighted_z_score": avg(method_rows, "winmax_weighted_z_score"),
                "avg_kl": avg(method_rows, "avg_kl"),
                "avg_delta": avg(method_rows, "avg_delta"),
                "gate_pass_rate": avg(method_rows, "gate_pass_rate"),
                "avg_candidate_size": avg(method_rows, "avg_candidate_size"),
            }
        )

fieldnames = [
    "run",
    "entropy_threshold",
    "top1_threshold",
    "method",
    "samples",
    "kgw_success_rate",
    "weighted_success_rate",
    "avg_z_score",
    "avg_weighted_z_score",
    "avg_winmax_weighted_z_score",
    "avg_kl",
    "avg_delta",
    "gate_pass_rate",
    "avg_candidate_size",
]
with summary_csv.open("w", encoding="utf-8-sig", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)

print(f"Saved compact summary to {summary_csv}")
PY

echo "[done] Gate grid outputs are in $OUTPUT_DIR"
echo "[done] Compact summary: $OUTPUT_DIR/gate_grid_compact_summary.csv"

#!/usr/bin/env bash
set -euo pipefail

# Fixed-delta 1.7 comparison:
# - generate with OPT-1.3B
# - evaluate PPL with OPT-2.7B

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

export MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/opt-1.3b}"
export PPL_MODEL_NAME_OR_PATH="${PPL_MODEL_NAME_OR_PATH:-$ROOT_DIR/models/opt-2.7b}"
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-opt13b_gen_opt27b_ppl_c4_500_fixed17}"

exec bash "$ROOT_DIR/course_project/scripts/run_opt27b_fixed17_500_fp32.sh"

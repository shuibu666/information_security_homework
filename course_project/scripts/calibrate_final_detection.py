#!/usr/bin/env python3
"""Calibrate each final-v1 detector on validation human continuations only."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from course_project.final_calibration import calibrate_human_thresholds, evaluate_test_scores
from course_project.final_protocol import read_jsonl, write_jsonl_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final-v1 detector calibration with held-out human negatives.")
    parser.add_argument("--validation_scores_jsonl", required=True)
    parser.add_argument("--test_scores_jsonl", required=True)
    parser.add_argument("--thresholds_jsonl", required=True)
    parser.add_argument("--metrics_jsonl", required=True)
    parser.add_argument("--target_fprs", type=float, nargs="+", default=[0.01, 0.05])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = calibrate_human_thresholds(read_jsonl(args.validation_scores_jsonl), args.target_fprs)
    metrics = evaluate_test_scores(read_jsonl(args.test_scores_jsonl), thresholds)
    threshold_hash = write_jsonl_atomic(args.thresholds_jsonl, thresholds)
    metric_hash = write_jsonl_atomic(args.metrics_jsonl, metrics)
    print(f"wrote {len(thresholds)} thresholds ({threshold_hash}) and {len(metrics)} held-out metrics ({metric_hash})")


if __name__ == "__main__":
    main()

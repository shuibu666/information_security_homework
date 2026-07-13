#!/usr/bin/env python3
"""Offline standard-KGW scores for human, No-Watermark, and watermarked rows."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from course_project.final_detection import standard_kgw_score
from course_project.final_protocol import config_hash, detector_config_id, read_jsonl, write_jsonl_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score final-v1 records with the matched standard KGW detector.")
    parser.add_argument("--manifest_jsonl", required=True)
    parser.add_argument("--generations_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--tokenizer_name_or_path", default="facebook/opt-1.3b")
    parser.add_argument("--gamma", type=float, default=.25)
    parser.add_argument("--hash_key", type=int, default=15485863)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit_records", type=int, default=None, help="Use the first N fixed-manifest records (Phase 1 smoke only).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    vocab_size = AutoConfig.from_pretrained(args.tokenizer_name_or_path).vocab_size
    config = {"detector_id": "standard_kgw", "gamma": args.gamma, "hash_key": args.hash_key, "seeding_scheme": "simple_1", "tokenizer": args.tokenizer_name_or_path, "vocab_size": vocab_size}
    detector_id, detector_hash = detector_config_id(config), config_hash(config)
    manifest_rows = [row for row in read_jsonl(args.manifest_jsonl) if row["split"] == args.split]
    if args.limit_records is not None:
        if args.limit_records < 1:
            raise ValueError("limit_records must be positive when provided")
        manifest_rows = manifest_rows[:args.limit_records]
    manifest = {str(row["source_id"]): row for row in manifest_rows}
    generations = [row for row in read_jsonl(args.generations_jsonl) if row["split"] == args.split]
    rows: list[dict[str, object]] = []
    green_mask_cache = {}
    # Human completions are the only calibration negatives.  They are scored
    # separately for every detector configuration, never replaced by model-null.
    for source_id, record in manifest.items():
        score = standard_kgw_score(record["prompt_token_ids"], record["human_completion_token_ids"], vocab_size, args.gamma, args.hash_key, args.device, green_mask_cache)
        rows.append({"generation_id": f"human-{source_id}", "prompt_id": source_id, "split": args.split, "role": "human_completion", "detector_config_id": detector_id, "detector_config_hash": detector_hash, **score})
    for record in generations:
        source_id = str(record["source_id"])
        if source_id not in manifest:
            raise ValueError(f"generation {record['generation_id']} is absent from the fixed manifest")
        score = standard_kgw_score(record["prompt_token_ids"], record["continuation_token_ids"], vocab_size, args.gamma, args.hash_key, args.device, green_mask_cache)
        role = "no_watermark" if record["generator_id"] == "no_watermark" else "watermarked"
        rows.append({"generation_id": record["generation_id"], "prompt_id": source_id, "split": args.split, "role": role, "generator_id": record["generator_id"], "parameter_id": record["parameter_id"], "base_seed": record["base_seed"], "detector_config_id": detector_id, "detector_config_hash": detector_hash, **score})
    digest = write_jsonl_atomic(args.output_jsonl, rows)
    print(f"wrote {len(rows)} standard KGW score rows to {args.output_jsonl}; sha256={digest}")


if __name__ == "__main__":
    main()

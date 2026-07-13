#!/usr/bin/env python3
"""Score final-v1 records with one explicitly matched CA-KL detector config."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from course_project.final_protocol import config_hash, detector_config_id, read_jsonl, write_jsonl_atomic
from course_project.processors.cakl_watermark_processor import CAKLModelAssistedDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final-v1 matched weighted/global or WinMax detector scoring.")
    parser.add_argument("--manifest_jsonl", required=True)
    parser.add_argument("--generations_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--model_name_or_path", default="facebook/opt-1.3b")
    parser.add_argument("--model_revision", default="main")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--gamma", type=float, default=.25)
    parser.add_argument("--candidate_top_p", type=float, default=.95)
    parser.add_argument("--entropy_threshold", type=float, default=.35)
    parser.add_argument("--top1_threshold", type=float, default=.85)
    parser.add_argument("--window_sizes", default="20,40,80,200")
    parser.add_argument("--greenlist_mode", choices=["full", "candidate"], required=True)
    parser.add_argument("--gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--detector_id", choices=["weighted_global", "weighted_winmax"], required=True)
    parser.add_argument("--limit_records", type=int, default=None, help="Use the first N fixed-manifest records (Phase 1 smoke only).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype] if str(args.device).startswith("cuda") else None
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, revision=args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, revision=args.model_revision, torch_dtype=dtype).to(args.device).eval()
    config = {
        "detector_id": args.detector_id, "gamma": args.gamma, "seeding_scheme": "simple_1",
        "greenlist_mode": args.greenlist_mode, "candidate_top_p": args.candidate_top_p if args.greenlist_mode == "candidate" else None,
        "gate": args.gate, "entropy_threshold": args.entropy_threshold if args.gate else None,
        "top1_threshold": args.top1_threshold if args.gate else None, "window_sizes": args.window_sizes if args.detector_id == "weighted_winmax" else None,
        "model": args.model_name_or_path, "model_revision": args.model_revision,
    }
    detector_id, detector_hash = detector_config_id(config), config_hash(config)
    detector = CAKLModelAssistedDetector(
        vocab=list(range(model.get_output_embeddings().weight.shape[0])), gamma=args.gamma, delta=3.0, seeding_scheme="simple_1",
        model=model, tokenizer=tokenizer, device=torch.device(args.device), candidate_top_p=args.candidate_top_p,
        use_candidate_greenlist=args.greenlist_mode == "candidate", use_confidence_gate=args.gate,
        entropy_threshold=args.entropy_threshold, top1_threshold=args.top1_threshold, window_sizes=args.window_sizes,
    )
    manifest_rows = [row for row in read_jsonl(args.manifest_jsonl) if row["split"] == args.split]
    if args.limit_records is not None:
        if args.limit_records < 1:
            raise ValueError("limit_records must be positive when provided")
        manifest_rows = manifest_rows[:args.limit_records]
    manifest = {str(row["source_id"]): row for row in manifest_rows}
    generations = [row for row in read_jsonl(args.generations_jsonl) if row["split"] == args.split]
    score_key = "weighted_z_score" if args.detector_id == "weighted_global" else "winmax_weighted_z_score"
    rows: list[dict[str, object]] = []
    def append_score(generation_id: str, source_id: str, role: str, prompt_ids, continuation_ids, **extra) -> None:
        details = detector.detect(torch.tensor(prompt_ids, dtype=torch.long), torch.tensor(continuation_ids, dtype=torch.long))
        rows.append({"generation_id": generation_id, "prompt_id": source_id, "split": args.split, "role": role,
                     "detector_config_id": detector_id, "detector_config_hash": detector_hash, "score": details[score_key],
                     "score_field": score_key, **details, **extra})
    for source_id, record in manifest.items():
        append_score(f"human-{source_id}", source_id, "human_completion", record["prompt_token_ids"], record["human_completion_token_ids"])
    for record in generations:
        source_id = str(record["source_id"])
        if source_id not in manifest:
            raise ValueError(f"generation {record['generation_id']} is absent from the fixed manifest")
        append_score(str(record["generation_id"]), source_id, "no_watermark" if record["generator_id"] == "no_watermark" else "watermarked",
                     record["prompt_token_ids"], record["continuation_token_ids"], generator_id=record["generator_id"],
                     parameter_id=record["parameter_id"], base_seed=record["base_seed"])
    digest = write_jsonl_atomic(args.output_jsonl, rows)
    print(f"wrote {len(rows)} {args.detector_id} rows ({detector_id}) to {args.output_jsonl}; sha256={digest}")


if __name__ == "__main__":
    main()

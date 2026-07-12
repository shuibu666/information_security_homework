#!/usr/bin/env python3
"""Generate one final-v1 generator/parameter/shard without detector coupling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from course_project.final_protocol import config_hash, generation_id, read_jsonl, stable_sample_seed, tokenizer_vocab_fingerprint, validate_generation_records, write_json_atomic, write_jsonl_atomic
from course_project.processors.cakl_watermark_processor import CAKLWatermarkLogitsProcessor
from watermark_processor import WatermarkLogitsProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final-v1 single-generator JSONL generation shard.")
    parser.add_argument("--manifest_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--generator_id", choices=["no_watermark", "kgw_fixed", "cakl_base", "cakl_candidate", "cakl_gate", "cakl_cg"], required=True)
    parser.add_argument("--parameter_id", required=True)
    parser.add_argument("--model_name_or_path", default="facebook/opt-1.3b")
    parser.add_argument("--model_revision", default="main")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--base_seed", type=int, required=True)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--limit_records", type=int, default=None, help="Use the first N fixed-manifest records (Phase 1 smoke only).")
    parser.add_argument("--gamma", type=float, default=.25)
    parser.add_argument("--fixed_delta", type=float, default=None)
    parser.add_argument("--kl_epsilon", type=float, default=None)
    parser.add_argument("--delta_max", type=float, default=3.0)
    parser.add_argument("--candidate_top_p", type=float, default=.95)
    parser.add_argument("--entropy_threshold", type=float, default=.35)
    parser.add_argument("--top1_threshold", type=float, default=.85)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_new_tokens", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--suppress_eos_until_min_length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def dtype_for_args(args: argparse.Namespace) -> torch.dtype | None:
    if args.device == "cpu":
        return None
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]


def generator_flags(generator_id: str) -> tuple[bool, bool]:
    return generator_id in {"cakl_candidate", "cakl_cg"}, generator_id in {"cakl_gate", "cakl_cg"}


def validate_args(args: argparse.Namespace) -> None:
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    if args.limit_records is not None and args.limit_records < 1:
        raise ValueError("limit_records must be positive when provided")
    if args.min_new_tokens != args.max_new_tokens or args.max_new_tokens != 200:
        raise ValueError("final-v1 requires min_new_tokens=max_new_tokens=200")
    if not (args.temperature == 1.0 and args.top_k == 0 and args.top_p == 1.0):
        raise ValueError("final-v1 main protocol requires temperature=1, top_k=0, top_p=1")
    if args.generator_id == "kgw_fixed" and args.fixed_delta is None:
        raise ValueError("kgw_fixed requires --fixed_delta")
    if args.generator_id.startswith("cakl_") and args.kl_epsilon is None:
        raise ValueError("CA-KL generators require --kl_epsilon")


def build_processor(args: argparse.Namespace, vocab_ids: list[int]):
    if args.generator_id == "no_watermark":
        return None
    if args.generator_id == "kgw_fixed":
        return WatermarkLogitsProcessor(vocab=vocab_ids, gamma=args.gamma, delta=args.fixed_delta, seeding_scheme="simple_1")
    use_candidate, use_gate = generator_flags(args.generator_id)
    return CAKLWatermarkLogitsProcessor(
        vocab=vocab_ids, gamma=args.gamma, delta=args.delta_max, seeding_scheme="simple_1",
        kl_epsilon=args.kl_epsilon, delta_max=args.delta_max, candidate_top_p=args.candidate_top_p,
        use_candidate_greenlist=use_candidate, use_confidence_gate=use_gate,
        entropy_threshold=args.entropy_threshold, top1_threshold=args.top1_threshold,
    )


def config_for_args(args: argparse.Namespace, manifest_path: Path) -> dict[str, object]:
    return {key: value for key, value in vars(args).items() if key not in {"output_jsonl", "resume"}} | {"manifest_path": str(manifest_path), "protocol_version": "final_v1"}


def main() -> None:
    args = parse_args()
    validate_args(args)
    manifest_path = Path(args.manifest_jsonl)
    run_config = config_for_args(args, manifest_path)
    fingerprint = config_hash(run_config)
    output_path = Path(args.output_jsonl)
    metadata_path = output_path.with_suffix(output_path.suffix + ".complete.json")
    if args.resume and output_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("config_hash") == fingerprint:
            print(f"completed shard already exists with matching config hash: {output_path}")
            return
        raise RuntimeError("existing output has a different config hash; choose a new output path")
    records = [row for row in read_jsonl(manifest_path) if row["split"] == args.split]
    if args.limit_records is not None:
        records = records[:args.limit_records]
    records = [row for index, row in enumerate(records) if index % args.shard_count == args.shard_index]
    if not records:
        raise ValueError("selected shard contains no manifest records")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, revision=args.model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, revision=args.model_revision, torch_dtype=dtype_for_args(args)).to(args.device).eval()
    tokenizer_fingerprint = tokenizer_vocab_fingerprint(tokenizer)
    max_context = getattr(model.config, "max_position_embeddings", None)
    vocab_ids = list(range(len(tokenizer)))
    output: list[dict[str, object]] = []
    for record in records:
        prompt_ids = [int(value) for value in record["prompt_token_ids"]]
        if max_context and len(prompt_ids) + args.max_new_tokens > max_context:
            raise ValueError(f"{record['source_id']} would require context truncation ({len(prompt_ids)}+200>{max_context})")
        processor = build_processor(args, vocab_ids)
        sample_seed = stable_sample_seed(args.base_seed, str(record["source_id"]))
        # Generation APIs do not consistently accept a per-call Generator
        # across Transformers releases, so reset the process RNG immediately
        # before each one-record generation.  Shards remain order-invariant.
        torch.manual_seed(sample_seed)
        if str(args.device).startswith("cuda"):
            torch.cuda.manual_seed_all(sample_seed)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
        kwargs = {
            "input_ids": input_ids, "do_sample": True, "temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
            "min_new_tokens": args.min_new_tokens, "max_new_tokens": args.max_new_tokens, "pad_token_id": tokenizer.pad_token_id,
        }
        if processor is not None:
            kwargs["logits_processor"] = LogitsProcessorList([processor])
        if args.suppress_eos_until_min_length and tokenizer.eos_token_id is not None:
            kwargs["eos_token_id"] = tokenizer.eos_token_id
        start = time.perf_counter()
        with torch.no_grad():
            tokens = model.generate(**kwargs)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
        continuation_ids = tokens[len(prompt_ids):].detach().cpu().tolist()
        if len(continuation_ids) != 200:
            raise RuntimeError(f"{record['source_id']} generated {len(continuation_ids)} tokens; final-v1 rejects attrition")
        summary = processor.get_generation_summary() if isinstance(processor, CAKLWatermarkLogitsProcessor) else {}
        output.append({
            "generation_id": generation_id(args.split, str(record["source_id"]), args.base_seed, args.generator_id, args.parameter_id),
            "prompt_id": str(record["source_id"]), "source_id": record["source_id"], "split": args.split,
            "base_seed": args.base_seed, "sample_seed": sample_seed, "generator_id": args.generator_id,
            "parameter_id": args.parameter_id, "generator_config": run_config, "generation_tokenizer_id": args.model_name_or_path,
            "generation_tokenizer_vocab_hash": tokenizer_fingerprint,
            "prompt_token_ids": prompt_ids, "continuation_token_ids": continuation_ids,
            "continuation_text": tokenizer.decode(continuation_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            "generated_token_count": len(continuation_ids), "finish_reason": "length", "runtime_ms": elapsed_ms,
            "peak_memory_mb": (torch.cuda.max_memory_allocated(args.device) / 1024**2) if str(args.device).startswith("cuda") else None,
            "config_hash": fingerprint, **summary,
        })
    validate_generation_records(output)
    digest = write_jsonl_atomic(output_path, output)
    write_json_atomic(metadata_path, {"config_hash": fingerprint, "records": len(output), "sha256": digest, "complete": True})
    print(f"wrote {len(output)} generation records to {output_path}; sha256={digest}")


if __name__ == "__main__":
    main()

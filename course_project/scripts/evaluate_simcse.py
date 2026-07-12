#!/usr/bin/env python3
"""Continuation-only, prompt/seed-paired SimCSE evaluation for final-v1 JSONL."""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
import sys

import torch
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from course_project.final_protocol import read_jsonl, write_jsonl_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pair final-v1 continuations against No-Watermark using SimCSE.")
    parser.add_argument("--generations_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model_name_or_path", default="princeton-nlp/sup-simcse-roberta-base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def pairing_key(row: dict[str, object]) -> tuple[str, int]:
    try:
        return str(row["prompt_id"]), int(row["base_seed"])
    except KeyError as error:
        raise ValueError(f"generation record lacks required pairing field: {error.args[0]}") from error


def pair_rows(rows: list[dict[str, object]]) -> list[tuple[dict[str, object], dict[str, object]]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[pairing_key(row)].append(row)
    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    for key, group in sorted(grouped.items()):
        baseline = [row for row in group if row.get("generator_id") == "no_watermark"]
        if len(baseline) != 1:
            raise ValueError(f"{key} must have exactly one no_watermark generation, found {len(baseline)}")
        for row in group:
            if row is not baseline[0]:
                if not str(row.get("continuation_text", "")).strip():
                    raise ValueError(f"{row.get('generation_id', key)} has empty continuation_text")
                pairs.append((baseline[0], row))
    if not pairs:
        raise ValueError("no watermarked/no-watermark pairs were found")
    return pairs


def embed_texts(model, tokenizer, texts: list[str], batch_size: int, device: torch.device) -> torch.Tensor:
    embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(texts[start : start + batch_size], padding=True, truncation=True, return_tensors="pt")
            encoded = {name: value.to(device) for name, value in encoded.items()}
            outputs = model(**encoded, return_dict=True)
            pooled = outputs.pooler_output
            if pooled is None:
                raise RuntimeError("SimCSE model did not return pooler_output")
            embeddings.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
    return torch.cat(embeddings, dim=0)


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.generations_jsonl)
    pairs = pair_rows(rows)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path).to(device).eval()
    baseline_embeddings = embed_texts(model, tokenizer, [str(b["continuation_text"]) for b, _ in pairs], args.batch_size, device)
    watermarked_embeddings = embed_texts(model, tokenizer, [str(w["continuation_text"]) for _, w in pairs], args.batch_size, device)
    scores = (baseline_embeddings * watermarked_embeddings).sum(dim=-1).tolist()
    output = []
    for (baseline, watermarked), cosine in zip(pairs, scores):
        output.append({
            "generation_id": watermarked["generation_id"],
            "baseline_generation_id": baseline["generation_id"],
            "prompt_id": watermarked["prompt_id"],
            "base_seed": watermarked["base_seed"],
            "generator_id": watermarked["generator_id"],
            "parameter_id": watermarked["parameter_id"],
            "simcse_model": args.model_name_or_path,
            "simcse_cosine": float(cosine),
        })
    digest = write_jsonl_atomic(args.output_jsonl, output)
    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
    print(f"wrote {len(output)} paired SimCSE scores to {args.output_jsonl}; sha256={digest}; mean={mean_score:.6f}; std={math.sqrt(variance):.6f}")


if __name__ == "__main__":
    main()

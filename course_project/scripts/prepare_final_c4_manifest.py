#!/usr/bin/env python3
"""Build the frozen final-v1 validation/test C4 manifest.

Unlike the legacy prompt exporter, this keeps source identity and exact OPT
token IDs.  It intentionally fails rather than writing a partial 1,000-row
evaluation corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from course_project.final_protocol import sha256_text, validate_eval_manifest, write_jsonl_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final-v1 C4/realnewslike evaluation JSONL.")
    parser.add_argument("--output_jsonl", default="course_project/outputs/final_v1/data/eval_manifest.jsonl")
    parser.add_argument("--audit_json", default="course_project/outputs/final_v1/data/dataset_audit.json")
    parser.add_argument("--audit_md", default="course_project/outputs/final_v1/data/dataset_audit.md")
    parser.add_argument("--model_name_or_path", default="facebook/opt-1.3b")
    parser.add_argument("--model_revision", default="main")
    parser.add_argument("--dataset_name", default="c4")
    parser.add_argument("--dataset_config_name", default="realnewslike")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--dataset_revision", default="main", help="Pinned Hugging Face revision/commit, not an implicit default.")
    parser.add_argument("--selection_seed", type=int, default=1234)
    parser.add_argument("--max_scan_records", type=int, default=100_000)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def source_identifier(example: dict[str, object], raw_text: str) -> str:
    for field in ("_id", "id", "url"):
        value = example.get(field)
        if value:
            return f"{field}:{value}"
    return f"text-sha256:{sha256_text(raw_text)}"


def deterministic_rank(seed: int, source_id: str) -> str:
    return hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).hexdigest()


def decode_ids(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def build_record(example: dict[str, object], tokenizer, args: argparse.Namespace, counters: Counter[str]) -> dict[str, object] | None:
    raw_text = example.get("text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        counters["empty_text"] += 1
        return None
    if "\ufffd" in raw_text:
        counters["replacement_character"] += 1
        return None
    token_ids = tokenizer(raw_text, add_special_tokens=False)["input_ids"]
    if not 500 <= len(token_ids) <= 1000:
        counters["source_length"] += 1
        return None
    prompt_ids, completion_ids = token_ids[:-200], token_ids[-200:]
    if not 300 <= len(prompt_ids) <= 800:
        counters["prompt_length"] += 1
        return None
    prompt_text, completion_text = decode_ids(tokenizer, prompt_ids), decode_ids(tokenizer, completion_ids)
    if not prompt_text or not completion_text or "\ufffd" in prompt_text or "\ufffd" in completion_text:
        counters["decoded_text_invalid"] += 1
        return None
    source_id = source_identifier(example, raw_text)
    return {
        "dataset_id": f"{args.dataset_name}/{args.dataset_config_name}",
        "dataset_revision": args.dataset_revision,
        "source_id": source_id,
        "split": "",  # assigned after deterministic global ranking
        "raw_text_sha256": sha256_text(raw_text),
        "prompt_text": prompt_text,
        "prompt_token_ids": token_ids[:-200],
        "prompt_token_count": len(prompt_ids),
        "human_completion_text": completion_text,
        "human_completion_token_ids": completion_ids,
        "human_completion_token_count": len(completion_ids),
        "source_token_count": len(token_ids),
        "tokenizer_id": args.model_name_or_path,
        "tokenizer_revision": args.model_revision,
        "selection_seed": args.selection_seed,
    }


def write_audit(args: argparse.Namespace, records: list[dict[str, object]], counters: Counter[str], manifest_sha256: str) -> None:
    stats: dict[str, object] = {
        "dataset_id": f"{args.dataset_name}/{args.dataset_config_name}",
        "dataset_revision": args.dataset_revision,
        "selection_seed": args.selection_seed,
        "max_scan_records": args.max_scan_records,
        "accepted": len(records),
        "filter_counts": dict(sorted(counters.items())),
        "split_counts": {split: sum(row["split"] == split for row in records) for split in ("validation", "test")},
        "manifest_sha256": manifest_sha256,
        "lengths": {},
    }
    for name, field in (("source", "source_token_count"), ("prompt", "prompt_token_count"), ("human_completion", "human_completion_token_count")):
        values = [int(row[field]) for row in records]
        stats["lengths"][name] = {key: percentile(values, q) for key, q in (("min", 0), ("p25", .25), ("p50", .5), ("p75", .75), ("p90", .9), ("max", 1))}
    Path(args.audit_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_json).write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Final-v1 C4 manifest audit", "", f"- Dataset: `{stats['dataset_id']}` @ `{args.dataset_revision}`", f"- Manifest SHA-256: `{manifest_sha256}`", f"- Accepted: {len(records)} (validation={stats['split_counts']['validation']}, test={stats['split_counts']['test']})", "", "## Filter counts", ""]
    lines += [f"- {name}: {count}" for name, count in stats["filter_counts"].items()]
    lines += ["", "## Token lengths", "", "| Segment | Min | P25 | P50 | P75 | P90 | Max |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in stats["lengths"].items():
        lines.append("| {} | {} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {} |".format(name, int(row["min"]), row["p25"], row["p50"], row["p75"], row["p90"], int(row["max"])))
    Path(args.audit_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, revision=args.model_revision if Path(args.model_name_or_path).exists() is False else None)
    dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.dataset_split, revision=args.dataset_revision, streaming=args.streaming)
    counters: Counter[str] = Counter()
    candidates: list[dict[str, object]] = []
    seen_source_ids: set[str] = set()
    for source_index, example in enumerate(dataset):
        if source_index >= args.max_scan_records:
            break
        record = build_record(dict(example), tokenizer, args, counters)
        if record is None:
            continue
        if record["source_id"] in seen_source_ids:
            counters["duplicate_source_id"] += 1
            continue
        seen_source_ids.add(str(record["source_id"]))
        candidates.append(record)
    candidates.sort(key=lambda row: deterministic_rank(args.selection_seed, str(row["source_id"])))
    if len(candidates) < 1000:
        raise RuntimeError(f"only found {len(candidates)} valid records after scanning {args.max_scan_records}; refusing partial manifest")
    records = candidates[:1000]
    for index, record in enumerate(records):
        record["split"] = "validation" if index < 500 else "test"
    validate_eval_manifest(records)
    manifest_hash = write_jsonl_atomic(args.output_jsonl, records)
    write_audit(args, records, counters, manifest_hash)
    print(f"wrote {len(records)} rows to {args.output_jsonl}; sha256={manifest_hash}")


if __name__ == "__main__":
    main()

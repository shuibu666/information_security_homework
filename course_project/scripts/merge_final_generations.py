#!/usr/bin/env python3
"""Strict final-v1 shard merge with the document's required preflight checks."""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from course_project.final_protocol import config_hash, read_jsonl, validate_generation_records, write_json_atomic, write_jsonl_atomic


SHARD_FIELDS = {"shard_index", "shard_count", "limit_records"}


def experiment_config(row: dict[str, object]) -> dict[str, object]:
    config = dict(row.get("generator_config", {}))
    if not config:
        raise ValueError(f"{row.get('generation_id')} lacks generator_config")
    return {key: value for key, value in config.items() if key not in SHARD_FIELDS}


def merge_rows(manifest_rows: list[dict[str, object]], rows: list[dict[str, object]], split: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    validate_generation_records(rows)
    manifest = {str(row["source_id"]): row for row in manifest_rows if row["split"] == split}
    if len(manifest) != 500:
        raise ValueError(f"expected 500 {split} manifest rows, got {len(manifest)}")
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        if row["split"] != split:
            raise ValueError(f"mixed split row {row['generation_id']}")
        groups.setdefault((str(row["generator_id"]), str(row["parameter_id"]), int(row["base_seed"])), []).append(row)
    merged: list[dict[str, object]] = []
    group_summary = []
    expected_sources = set(manifest)
    for key, group in sorted(groups.items()):
        sources = {str(row["source_id"]) for row in group}
        if sources != expected_sources:
            raise ValueError(f"{key} has {len(sources)}/500 manifest sources; missing={len(expected_sources - sources)}, unexpected={len(sources - expected_sources)}")
        configs = {config_hash(experiment_config(row)) for row in group}
        if len(configs) != 1:
            raise ValueError(f"{key} has inconsistent base experiment configuration hashes")
        base_hash = next(iter(configs))
        for row in group:
            source = str(row["source_id"])
            if list(row["prompt_token_ids"]) != list(manifest[source]["prompt_token_ids"]):
                raise ValueError(f"{row['generation_id']} prompt IDs differ from fixed manifest")
            if not str(row.get("continuation_text", "")).strip():
                raise ValueError(f"{row['generation_id']} has empty decoded continuation")
            row = dict(row)
            row["experiment_config_hash"] = base_hash
            merged.append(row)
        group_summary.append({"generator_id": key[0], "parameter_id": key[1], "base_seed": key[2], "records": len(group), "experiment_config_hash": base_hash})
    return merged, {"split": split, "groups": group_summary, "records": len(merged)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge final-v1 generation shards after strict preflight.")
    parser.add_argument("--manifest_jsonl", required=True)
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--completion_json", required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    args = parser.parse_args()
    paths = sorted(Path(path) for path in glob.glob(args.input_glob))
    if not paths:
        raise ValueError("input_glob matched no shards")
    rows = [row for path in paths for row in read_jsonl(path)]
    merged, summary = merge_rows(read_jsonl(args.manifest_jsonl), rows, args.split)
    digest = write_jsonl_atomic(args.output_jsonl, merged)
    summary.update({"input_shards": len(paths), "output_sha256": digest, "complete": True})
    write_json_atomic(args.completion_json, summary)
    print(f"merged {len(paths)} shards / {len(merged)} rows; sha256={digest}")


if __name__ == "__main__":
    main()

"""Small, dependency-light primitives shared by the final-v1 experiment tools.

The legacy CSV files deliberately remain readable, but final-v1 uses JSONL with
stable identifiers so generation, detection, and evaluation cannot silently
duplicate or mix records.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def config_hash(config: Mapping[str, object]) -> str:
    return sha256_text(canonical_json(dict(config)))


def detector_config_id(config: Mapping[str, object]) -> str:
    """A detector identifier that changes whenever any scoring rule changes."""
    return f"det-{config_hash(config)[:16]}"


def stable_sample_seed(base_seed: int, prompt_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{prompt_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def generation_id(split: str, prompt_id: str, base_seed: int, generator_id: str, parameter_id: str) -> str:
    payload = {
        "base_seed": base_seed,
        "generator_id": generator_id,
        "parameter_id": parameter_id,
        "prompt_id": prompt_id,
        "split": split,
    }
    return f"gen-{config_hash(payload)[:20]}"


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def write_jsonl_atomic(path: str | Path, records: Iterable[Mapping[str, object]]) -> str:
    """Write a complete shard atomically and return its SHA-256 hash."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False, prefix=f".{target.name}.", suffix=".tmp") as handle:
        temp_path = Path(handle.name)
        digest = hashlib.sha256()
        for record in records:
            line = canonical_json(dict(record)) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    os.replace(temp_path, target)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, value: Mapping[str, object]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(dict(value)) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False, prefix=f".{target.name}.", suffix=".tmp") as handle:
        temp_path = Path(handle.name)
        handle.write(payload)
    os.replace(temp_path, target)
    return sha256_text(payload)


MANIFEST_FIELDS = {
    "dataset_id", "dataset_revision", "source_id", "split", "raw_text_sha256",
    "prompt_text", "prompt_token_ids", "prompt_token_count", "human_completion_text",
    "human_completion_token_ids", "human_completion_token_count", "source_token_count",
    "tokenizer_id", "tokenizer_revision", "selection_seed",
}


def validate_eval_manifest(records: Iterable[Mapping[str, object]], expected_per_split: int | None = 500) -> None:
    records = list(records)
    by_split: dict[str, list[Mapping[str, object]]] = {}
    seen_source_ids: set[str] = set()
    seen_prompt_hashes: set[str] = set()
    for index, record in enumerate(records):
        missing = MANIFEST_FIELDS - set(record)
        if missing:
            raise ValueError(f"manifest record {index} lacks fields: {sorted(missing)}")
        split = str(record["split"])
        if split not in {"validation", "test"}:
            raise ValueError(f"manifest record {index} has invalid split {split!r}")
        source_id = str(record["source_id"])
        if not source_id or source_id in seen_source_ids:
            raise ValueError(f"duplicate or empty source_id: {source_id!r}")
        seen_source_ids.add(source_id)
        prompt = str(record["prompt_text"])
        completion = str(record["human_completion_text"])
        if not prompt or not completion or "\ufffd" in prompt or "\ufffd" in completion:
            raise ValueError(f"record {index} has empty/replacement-character text")
        prompt_hash = sha256_text(prompt)
        if prompt_hash in seen_prompt_hashes:
            raise ValueError(f"duplicate prompt text at record {index}")
        seen_prompt_hashes.add(prompt_hash)
        if len(record["prompt_token_ids"]) != int(record["prompt_token_count"]):
            raise ValueError(f"record {index} prompt token count mismatch")
        if len(record["human_completion_token_ids"]) != int(record["human_completion_token_count"]):
            raise ValueError(f"record {index} completion token count mismatch")
        if int(record["source_token_count"]) != int(record["prompt_token_count"]) + int(record["human_completion_token_count"]):
            raise ValueError(f"record {index} source token count mismatch")
        if not 500 <= int(record["source_token_count"]) <= 1000:
            raise ValueError(f"record {index} source length is outside [500, 1000]")
        if not 300 <= int(record["prompt_token_count"]) <= 800:
            raise ValueError(f"record {index} prompt length is outside [300, 800]")
        if int(record["human_completion_token_count"]) != 200:
            raise ValueError(f"record {index} completion must contain exactly 200 tokens")
        by_split.setdefault(split, []).append(record)
    if expected_per_split is not None:
        observed = {split: len(rows) for split, rows in by_split.items()}
        expected = {"validation": expected_per_split, "test": expected_per_split}
        if observed != expected:
            raise ValueError(f"split sizes must be {expected}, got {observed}")

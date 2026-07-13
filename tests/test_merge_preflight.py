from __future__ import annotations

import pytest

from course_project.scripts.merge_final_generations import merge_rows


def manifest_row(source_id: str) -> dict[str, object]:
    return {"source_id": source_id, "split": "validation", "prompt_token_ids": [1, 2], "prompt_text": "prompt"}


def generation_row(source_id: str, config: dict[str, object]) -> dict[str, object]:
    return {
        "generation_id": f"g-{source_id}", "split": "validation", "prompt_id": source_id, "source_id": source_id,
        "base_seed": 1234, "generator_id": "kgw_fixed", "parameter_id": "delta=1.0",
        "prompt_token_ids": [1, 2], "continuation_token_ids": list(range(200)), "continuation_text": "completion",
        "generated_token_count": 200, "config_hash": "job", "generator_config": config,
    }


def test_merge_requires_complete_common_manifest_sources():
    manifest = [manifest_row("a"), manifest_row("b")]
    rows = [generation_row("a", {"gamma": .25})]
    with pytest.raises(ValueError, match="expected 500"):
        merge_rows(manifest, rows, "validation")


def test_merge_rejects_shard_independent_config_mismatch_after_normalization(monkeypatch):
    # Use 500 records so the test reaches the configuration check.
    manifest = [manifest_row(str(index)) for index in range(500)]
    rows = [generation_row(str(index), {"gamma": .25, "shard_index": index % 2, "shard_count": 2}) for index in range(500)]
    merged, summary = merge_rows(manifest, rows, "validation")
    assert len(merged) == 500
    assert summary["groups"][0]["records"] == 500

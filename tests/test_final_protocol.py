from __future__ import annotations

import json

import pytest

from course_project.cakl_math import green_mass_after_bias, kl_q_to_p
from course_project.final_calibration import calibrate_human_thresholds, conservative_threshold, evaluate_test_scores
from course_project.final_protocol import (
    detector_config_id,
    generation_id,
    stable_sample_seed,
    validate_generation_records,
    write_json_atomic,
    validate_eval_manifest,
    write_jsonl_atomic,
)


def valid_record(source_id: str, split: str) -> dict[str, object]:
    prompt_ids = list(range(300))
    completion_ids = list(range(300, 500))
    return {
        "dataset_id": "c4/realnewslike",
        "dataset_revision": "pinned-revision",
        "source_id": source_id,
        "split": split,
        "raw_text_sha256": "a" * 64,
        "prompt_text": f"Prompt {source_id}",
        "prompt_token_ids": prompt_ids,
        "prompt_token_count": len(prompt_ids),
        "human_completion_text": f"Completion {source_id}",
        "human_completion_token_ids": completion_ids,
        "human_completion_token_count": len(completion_ids),
        "source_token_count": len(prompt_ids) + len(completion_ids),
        "tokenizer_id": "facebook/opt-1.3b",
        "tokenizer_revision": "pinned-revision",
        "selection_seed": 1234,
    }


def test_generation_and_seed_are_stable():
    assert stable_sample_seed(1234, "p-1") == stable_sample_seed(1234, "p-1")
    assert generation_id("test", "p-1", 1234, "cakl_base", "eps=0.2") == generation_id("test", "p-1", 1234, "cakl_base", "eps=0.2")
    assert detector_config_id({"gamma": .25, "kind": "standard"}) != detector_config_id({"gamma": .25, "kind": "weighted"})


def test_manifest_rejects_duplicate_source_and_bad_lengths():
    records = [valid_record("a", "validation"), valid_record("a", "test")]
    with pytest.raises(ValueError, match="source_id"):
        validate_eval_manifest(records, expected_per_split=None)
    record = valid_record("b", "validation")
    record["human_completion_token_count"] = 199
    with pytest.raises(ValueError, match="completion token count mismatch"):
        validate_eval_manifest([record], expected_per_split=None)


def test_eval_manifest_has_exact_split_sizes_and_rejects_replacement_characters():
    records = [valid_record(f"validation-{index}", "validation") for index in range(500)]
    records += [valid_record(f"test-{index}", "test") for index in range(500)]
    validate_eval_manifest(records)
    records[0]["prompt_text"] = "bad \ufffd text"
    with pytest.raises(ValueError, match="replacement-character"):
        validate_eval_manifest(records)


def test_jsonl_writer_is_atomic_and_hashes_content(tmp_path):
    destination = tmp_path / "records.jsonl"
    digest = write_jsonl_atomic(destination, [{"b": 2, "a": 1}])
    assert len(digest) == 64
    assert json.loads(destination.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    metadata_hash = write_json_atomic(tmp_path / "complete.json", {"config_hash": "abc", "complete": True})
    assert len(metadata_hash) == 64
    assert json.loads((tmp_path / "complete.json").read_text(encoding="utf-8"))["config_hash"] == "abc"


def test_duplicate_generation_id_fails():
    row = {
        "generation_id": "g1", "split": "validation", "prompt_id": "p1", "base_seed": 1234,
        "generator_id": "cakl_base", "parameter_id": "eps=0.2", "prompt_token_ids": [1],
        "continuation_token_ids": list(range(200)), "generated_token_count": 200, "config_hash": "cfg",
    }
    with pytest.raises(ValueError, match="generation_id"):
        validate_generation_records([row, dict(row)])


def test_cakl_closed_form_is_nonnegative_and_monotone():
    previous = -1.0
    for delta in (0.0, 0.1, 0.5, 1.0, 2.0, 3.0):
        value = kl_q_to_p(delta, 0.25)
        assert value >= previous
        previous = value
    assert kl_q_to_p(0.0, 0.25) == 0.0
    assert 0.25 < green_mass_after_bias(0.25, 1.0) < 1.0


def test_calibration_is_human_only_conservative_and_hash_matched():
    validation = [
        {"generation_id": f"human-{index}", "prompt_id": f"p{index}", "detector_config_id": "d", "detector_config_hash": "h", "role": "human_completion", "score": score}
        for index, score in enumerate((0.0, 1.0, 2.0, 3.0))
    ]
    thresholds = calibrate_human_thresholds(validation, [0.25])
    assert thresholds[0]["threshold"] == 2.0
    assert thresholds[0]["calibration_empirical_fpr"] == 0.25
    test_rows = [
        {"generation_id": "test-human", "prompt_id": "p", "detector_config_id": "d", "detector_config_hash": "h", "role": "human_completion", "score": 2.0},
        {"generation_id": "test-positive", "prompt_id": "p", "detector_config_id": "d", "detector_config_hash": "h", "role": "watermarked", "score": 3.0},
    ]
    metrics = evaluate_test_scores(test_rows, thresholds)
    assert {row["role"] for row in metrics} == {"human_completion", "watermarked"}
    for row in test_rows:
        row["detector_config_hash"] = "wrong"
    with pytest.raises(ValueError, match="match calibration detector hash"):
        evaluate_test_scores(test_rows, thresholds)


def test_calibration_respects_target_fpr_with_ties():
    scores = [0.0, 2.0, 2.0, 2.0]
    threshold = conservative_threshold(scores, .25)
    assert threshold == 2.0
    assert sum(score > threshold for score in scores) / len(scores) <= .25

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from course_project.scripts.evaluate_simcse import pair_rows


def test_simcse_identity_and_pairing():
    rows = [
        {"generation_id": "base", "prompt_id": "p1", "base_seed": 1234, "generator_id": "no_watermark", "continuation_text": "same text"},
        {"generation_id": "wm", "prompt_id": "p1", "base_seed": 1234, "generator_id": "cakl_base", "continuation_text": "same text"},
    ]
    pairs = pair_rows(rows)
    assert len(pairs) == 1
    assert pairs[0][0]["generation_id"] == "base"
    assert pairs[0][1]["generation_id"] == "wm"


def test_simcse_requires_exactly_one_baseline_per_prompt_seed():
    rows = [
        {"generation_id": "wm", "prompt_id": "p1", "base_seed": 1234, "generator_id": "cakl_base", "continuation_text": "text"},
    ]
    with pytest.raises(ValueError, match="exactly one no_watermark"):
        pair_rows(rows)

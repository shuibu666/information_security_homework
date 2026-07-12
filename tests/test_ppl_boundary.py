from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from course_project.scripts.evaluate_ppl import build_scoring_example
from course_project.scripts.evaluate_ppl import resolve_io
from course_project.final_protocol import tokenizer_vocab_fingerprint


class Tokenizer:
    name_or_path = "same-tokenizer"
    def get_vocab(self):
        return {"a": 0, "b": 1}


def test_ppl_scores_only_continuation_ids():
    row = {
        "generation_id": "g", "generation_tokenizer_id": "same-tokenizer",
        "prompt_token_ids": "[10, 11, 12]", "continuation_token_ids": "[13, 14, 15]",
    }
    example = build_scoring_example(row, Tokenizer(), None)
    assert example["input_ids"] == [10, 11, 12, 13, 14, 15]
    assert example["labels"] == [-100, -100, -100, 13, 14, 15]
    assert example["scored_tokens"] == 3


def test_ppl_rejects_cross_tokenizer_boundary_guessing():
    row = {
        "generation_id": "g", "generation_tokenizer_id": "generation-tokenizer",
        "prompt_token_ids": "[10]", "continuation_token_ids": "[11]",
    }
    with pytest.raises(ValueError, match="identical generation/oracle tokenizer"):
        build_scoring_example(row, Tokenizer(), None)


def test_ppl_accepts_differently_named_checkpoint_with_identical_vocab():
    row = {
        "generation_id": "g", "generation_tokenizer_id": "facebook/opt-1.3b",
        "generation_tokenizer_vocab_hash": tokenizer_vocab_fingerprint(Tokenizer()),
        "prompt_token_ids": "[10]", "continuation_token_ids": "[11]",
    }
    assert build_scoring_example(row, Tokenizer(), None)["scored_tokens"] == 1


def test_ppl_final_jsonl_io_contract():
    class Args:
        input_csv = None
        input_jsonl = "raw.jsonl"
        output_csv = None
        output_jsonl = "scores.jsonl"
    assert resolve_io(Args()) == ("raw.jsonl", "scores.jsonl")

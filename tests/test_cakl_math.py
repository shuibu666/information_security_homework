from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from course_project.cakl_math import kl_q_to_p
from course_project.final_detection import standard_kgw_score
from course_project.processors.cakl_watermark_processor import CAKLWatermarkLogitsProcessor


def test_cakl_closed_form_kl_matches_torch():
    logits = torch.tensor([0.2, -0.4, 1.0, 0.6, -1.2], dtype=torch.float64)
    green_mask = torch.tensor([True, False, True, False, False])
    delta = 1.3
    p = torch.softmax(logits, dim=-1)
    q = torch.softmax(logits + green_mask.to(torch.float64) * delta, dim=-1)
    direct = torch.sum(q * (torch.log(q) - torch.log(p))).item()
    assert abs(kl_q_to_p(delta, float(p[green_mask].sum())) - direct) < 1e-10


def test_cakl_epsilon_zero_is_identity_and_respects_maximum():
    processor = CAKLWatermarkLogitsProcessor(vocab=list(range(32)), gamma=.25, delta=3.0, kl_epsilon=0.0, delta_max=3.0)
    scores = torch.randn(1, 32)
    result = processor(torch.tensor([[1, 2, 3]]), scores.clone())
    assert torch.equal(result, scores)
    assert processor.step_history[0]["actual_sampling_kl"] == 0.0
    processor = CAKLWatermarkLogitsProcessor(vocab=list(range(32)), gamma=.25, delta=0.7, kl_epsilon=10.0, delta_max=.7)
    processor(torch.tensor([[1, 2, 3]]), torch.randn(1, 32))
    assert processor.step_history[0]["delta"] <= .7


def test_kgw_z_matches_manual_count():
    prompt, continuation = [1, 2, 3], [4, 5, 6, 7]
    result = standard_kgw_score(prompt, continuation, vocab_size=32, gamma=.25)
    expected = (result["num_green_tokens"] - .25 * 4) / (4 * .25 * .75) ** .5
    assert result["z_score"] == expected

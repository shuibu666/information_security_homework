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


def test_standard_detector_uses_generation_device_rng():
    from watermark_processor import WatermarkLogitsProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = WatermarkLogitsProcessor(vocab=list(range(32)), gamma=.25, delta=2.0)
    prefix = torch.tensor([[1, 2, 3]], device=device)
    raw_scores = torch.zeros(1, 32, device=device)
    biased = processor(prefix, raw_scores.clone())
    green_id = int(torch.argmax(biased[0]).item())
    result = standard_kgw_score([1, 2, 3], [green_id], vocab_size=32, gamma=.25, device=device)
    assert result["num_green_tokens"] == 1


def test_standard_detector_cache_preserves_score():
    kwargs = {"vocab_size": 32, "gamma": .25, "device": "cpu"}
    uncached = standard_kgw_score([1, 2, 3], [4, 5, 4, 5], **kwargs)
    cached = standard_kgw_score([1, 2, 3], [4, 5, 4, 5], green_mask_cache={}, **kwargs)
    assert cached == uncached


def test_generation_and_detector_step_stats_match():
    from course_project.processors.cakl_watermark_processor import CAKLModelAssistedDetector

    class DummyModel:
        pass

    processor = CAKLWatermarkLogitsProcessor(
        vocab=list(range(32)), gamma=.25, delta=3.0, kl_epsilon=.2, delta_max=3.0,
        candidate_top_p=.9, use_candidate_greenlist=True, use_confidence_gate=True,
        entropy_threshold=0.0, top1_threshold=1.0,
    )
    detector = CAKLModelAssistedDetector(
        vocab=list(range(32)), gamma=.25, delta=3.0, model=DummyModel(), tokenizer=None,
        device=torch.device("cpu"), candidate_top_p=.9, use_candidate_greenlist=True,
        use_confidence_gate=True, entropy_threshold=0.0, top1_threshold=1.0,
    )
    prefix, logits = torch.tensor([1, 2, 3]), torch.randn(32)
    processor.rng = torch.Generator(device="cpu")
    _, generation = processor._step_greenlist_and_stats(prefix, logits)
    detection = detector._score_step(prefix, logits, target_id=4)
    assert detection["gate_passed"] == generation["gate_passed"]
    assert detection["candidate_size"] == generation["candidate_size"]
    assert detection["p_green_mass"] == pytest.approx(generation["p_green_mass"], abs=1e-6)

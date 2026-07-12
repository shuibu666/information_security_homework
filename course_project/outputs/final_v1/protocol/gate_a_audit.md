# Gate A audit — final-v1

Date: 2026-07-12

## Prerequisites

- P0 code paths are covered by the h102 project test suite: **19 passed**.
- GPU integration ran on an NVIDIA RTX 6000D and verified a finite CA-KL output with `actual_sampling_kl=0.1999955285 <= epsilon=0.2`.
- The frozen manifest contains 1,000 rows: validation=500 and test=500.
- Manifest SHA-256: `d87e0960e762ae30cc9c1655d0446aa3292e3e76af0ef740634566035848885f`.

## Phase 1 10-prompt smoke

Protocol: validation manifest records 1–10; OPT-1.3B FP16; `temperature=1`, `top_k=0`, `top_p=1`; exactly 200 generated tokens; base seed 1234.

| Generator / parameter | Records | Result |
| --- | ---: | --- |
| No Watermark | 10 | passed |
| KGW delta=0, 1, 2, 3 | 40 | passed |
| CA-KL epsilon=0, 0.1, 0.3, 0.5 | 40 | passed |

Smoke-generation merged SHA-256: `04882575d0a3c8a872eb4749817a7ef55b23a9bf289fa90f179580030626f48d`.

## Required invariants

- All 90 generation records contain exactly 200 continuation token IDs.
- Generation IDs and semantic generation keys are unique.
- CA-KL epsilon=0 continuation token IDs are exactly identical to No-Watermark under the same prompt/seed.
- Every recorded CA-KL `actual_kl_max` is finite and satisfies `actual_kl_max <= epsilon + 1e-4`.
- A repeated No-Watermark run produced identical generation IDs, sample seeds, and continuation token IDs for all 10 prompts.
- The separate standard-KGW detector, continuation-only PPL, paired SimCSE, and human-completion calibration interfaces all completed successfully.

## Smoke artifacts

- Raw records: `raw/validation/phase1_merged.jsonl` (90 rows).
- Standard detector: `scores/phase1_standard_kgw.jsonl` (100 rows, including 10 human continuations).
- PPL: `scores/phase1_ppl.jsonl` (90 rows).
- SimCSE: `scores/phase1_simcse.jsonl` (80 paired watermarked rows).
- Calibration smoke: `calibration/phase1_smoke_thresholds.jsonl` and `calibration/phase1_smoke_metrics.jsonl`.

The calibration smoke intentionally reuses the 10 rows on both inputs to validate the interface only; it is non-reportable and is not a held-out result.

## Decision

**Gate A: Go.** The next permitted step is the document’s Phase 2 50-prompt numerical/performance pilot. Validation-500 and test-500 remain unstarted.

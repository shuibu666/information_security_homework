# Gate A audit — final-v3

## Code and environment checks

- Commit: `4cce5ec`.
- Project test suite: 24 passed, including full-vocabulary candidate/gate and detector-device regression tests.
- Full-vocabulary repair: greenlist size derives from OPT output embeddings / `AutoConfig.vocab_size=50272`, not tokenizer length 50265.
- Frozen manifest SHA-256: `d87e0960e762ae30cc9c1655d0446aa3292e3e76af0ef740634566035848885f`.

## Phase 1 smoke

Validation records 1–10, base seed 1234, OPT-1.3B FP16, temperature 1, top-k 0, top-p 1, 200 tokens:

- 90 raw generation rows across No Watermark, KGW delta=0/1/2/3, and CA-KL epsilon=0/0.1/0.3/0.5.
- All continuations have 200 tokens and unique generation IDs.
- CA-KL epsilon=0 token IDs exactly equal No-Watermark output under the same seeds.
- Every CA-KL actual KL is finite and within epsilon + 1e-4.
- Deterministic same-seed rerun passed.
- Standard detector (100 rows), PPL (90 rows), SimCSE (80 pairs), and calibration smoke (20 rows) all completed.

## Decision

**Gate A: Go.** Continue to Phase 2 only. The smoke calibration reuses validation records for interface verification and is non-reportable.

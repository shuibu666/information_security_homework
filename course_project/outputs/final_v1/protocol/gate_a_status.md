# Gate A status — 2026-07-12

## Completed P0 foundations

- Added strict final-v1 JSONL schema, stable generation IDs/seeds, configuration hashes, atomic shard writes, and completion metadata.
- Added `prepare_final_c4_manifest.py`, which creates a deterministic 500 validation + 500 test manifest with exact prompt/human token IDs and rejects partial or malformed data.
- Added a single-generator final-v1 runner. It enforces `temperature=1`, `top_k=0`, `top_p=1`, and exactly 200 generated tokens; it writes no detector results.
- Added offline standard-KGW exact-token scoring, with human completions preserved as calibration negatives.
- Added detector-hash-matched human-only calibration using a conservative strict `score > threshold` rule.
- Changed PPL evaluation to require saved prompt/continuation IDs rather than re-tokenizing concatenated text.
- Added continuation-only, No-Watermark-paired SimCSE evaluation.
- Added CA-KL KL summaries and direct KL validation helpers.

## Executed checks

```text
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /home/zyb/.venvs/lm-watermarking/bin/python -m pytest -q tests/test_final_protocol.py tests/test_cakl_math.py
8 passed in 3.08s
```

The GPU integration check also produced a finite CA-KL output with:

```text
GPU: NVIDIA RTX 6000D
actual_sampling_kl: 0.1999955285
epsilon: 0.2
```

## Remaining P0 work

- Implement matched candidate/no-gate and full-vocabulary/gate detectors, and verify detector step statistics against generation.
- Replace the legacy Pareto source/aggregation path; it still contains mixed source IDs and in-sample weighted TPR fields.
- Add the remaining data, sharding, SimCSE pairing, PPL, calibration-tie, and Pareto preflight tests.
- Run the manifest builder and verify its frozen dataset audit.
- Run the full 10-prompt GPU smoke path through generation, standard detection, PPL, SimCSE, and calibration.

## Decision

**Gate A remains No-Go.** No validation-500 or test generation has been started.

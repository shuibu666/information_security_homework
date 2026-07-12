# Phase 2 — 50-prompt numerical and performance pilot

Date: 2026-07-12

## Protocol

- Frozen validation manifest records 1–50; base seed 1234.
- OPT-1.3B FP16 generation, `temperature=1`, `top_k=0`, `top_p=1`, exactly 200 tokens.
- No Watermark; KGW `delta=[0.5,1.0,1.5,1.7,2.0,2.5,3.0,5.0]`; CA-KL `epsilon=[0.0,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.70]`, `delta_max=3.0`.
- 900 raw generations; raw merged SHA-256: `5440eeea22abd341829ed8ac4dfaad73682c18cd7c168511631522c1d4681ce9`.

## CA-KL mechanism diagnostics

| epsilon | Mean actual KL | Mean delta | delta-max saturation |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.000 | 0.000 | 0.0% |
| 0.02 | 0.019 | 0.788 | 5.7% |
| 0.05 | 0.047 | 1.090 | 8.5% |
| 0.10 | 0.092 | 1.386 | 11.1% |
| 0.20 | 0.180 | 1.741 | 14.0% |
| 0.30 | 0.262 | 2.032 | 18.2% |
| 0.40 | 0.343 | 2.249 | 21.9% |
| 0.50 | 0.413 | 2.454 | 27.9% |
| 0.70 | 0.553 | 2.753 | 39.0% |

The expected monotone mechanism trend is present. No point exceeds the plan's 50% saturation trigger.

## Correctness incident and remediation

The first pilot detector pass used CPU RNG while generation used CUDA RNG. Since legacy KGW `torch.randperm` is device-local, this incorrectly produced near-zero watermark scores. Parameter selection was paused, the detector was changed to use the generation device and cache exact masks by previous token, and a GPU regression test was added. The corrected detector output SHA-256 is `9df9b4cc5fb2ab9126e8c63dda74b45142c91e6903a7964f1830bff21b578967`.

All diagnostic detection values below use the corrected detector. The 1% threshold is reused on the pilot data only to inspect ranges; it is non-reportable and not held-out.

## Parameter decision for Phase 3

The retained validation scan is exactly the document's main grid:

- KGW: `delta=[0.5,1.0,1.5,1.7,2.0,2.5,3.0]`.
- CA-KL base: `epsilon=[0.02,0.05,0.10,0.20,0.30,0.40,0.50]`, `delta_max=3.0`.
- Additional sanity point: KGW `gamma=0.5`, `delta=2.0`.

`delta=5.0`, `epsilon=0`, and `epsilon=0.70` are retained as pilot diagnostics only and will not enter the main Pareto grid.

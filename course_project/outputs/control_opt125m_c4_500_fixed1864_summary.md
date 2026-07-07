# Watermark Experiment Summary

| Method | Samples | Avg z-score | Avg Green Fraction | Detection Success Rate | Avg Word Count | Avg Distinct-1 | Avg Distinct-2 | Avg Repetition Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 500 | 0.0763 | 0.2528 | 0.40% | 74.71 | 0.6468 | 0.8750 | 0.3532 |
| Fixed Delta 1.864 | 500 | 7.5689 | 0.5842 | 97.00% | 76.13 | 0.6509 | 0.8612 | 0.3471 |
| Current Adaptive Delta | 500 | 7.9256 | 0.5987 | 96.60% | 76.28 | 0.6449 | 0.8482 | 0.3531 |

## CA-KL-CG Diagnostics

| Method | Avg Weighted z | Avg WinMax Weighted z | Avg KL | Avg Adaptive Delta | Avg Gate Pass Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 0.1636 | 1.7336 | - | - | 0.7715 |
| Fixed Delta 1.864 | - | - | - | - | - |
| Current Adaptive Delta | 8.8480 | 8.8851 | - | - | 0.7874 |

## Calibrated Detection Quality

| Score | Method | AUC | TPR@1%FPR | TPR@5%FPR |
| --- | --- | ---: | ---: | ---: |
| weighted_z_score | Current Adaptive Delta | 0.9985 | 99.60% | 99.80% |
| winmax_weighted_z_score | Current Adaptive Delta | 0.9963 | 98.20% | 98.80% |

## Notes

- `Detection Success Rate` counts the share of outputs predicted as `Watermarked`.
- `Distinct-1` and `Distinct-2` provide a lightweight diversity estimate for generated text.
- `Avg Repetition Rate` is `1 - unique_words / total_words`; lower values generally indicate less repetition.
- Calibrated detection quality uses `No Watermark` scores as the negative class.
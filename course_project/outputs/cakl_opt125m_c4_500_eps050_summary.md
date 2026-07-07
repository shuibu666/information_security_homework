# Watermark Experiment Summary

| Method | Samples | Avg z-score | Avg Green Fraction | Detection Success Rate | Avg Word Count | Avg Distinct-1 | Avg Distinct-2 | Avg Repetition Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 500 | 0.0763 | 0.2528 | 0.40% | 74.71 | 0.6468 | 0.8750 | 0.3532 |
| Fixed Delta 1.0 | 500 | 3.9266 | 0.4236 | 47.20% | 75.07 | 0.6472 | 0.8706 | 0.3528 |
| Fixed Delta 2.0 | 500 | 8.2140 | 0.6115 | 97.20% | 76.82 | 0.6528 | 0.8579 | 0.3452 |
| Fixed Delta 3.0 | 500 | 11.5481 | 0.7606 | 99.00% | 77.60 | 0.6536 | 0.8381 | 0.3444 |
| Current Adaptive Delta | 500 | 7.9256 | 0.5987 | 96.60% | 76.28 | 0.6449 | 0.8482 | 0.3531 |
| CA-KL | 500 | 8.8348 | 0.6388 | 97.60% | 77.27 | 0.6501 | 0.8517 | 0.3479 |
| CA-KL + Candidate Greenlist | 500 | 8.3465 | 0.6182 | 96.80% | 76.37 | 0.6570 | 0.8590 | 0.3410 |
| CA-KL + Weighted Detector | 500 | 8.8348 | 0.6388 | 97.60% | 77.27 | 0.6501 | 0.8517 | 0.3479 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 500 | 7.9897 | 0.6033 | 95.40% | 76.25 | 0.6578 | 0.8579 | 0.3402 |

## CA-KL-CG Diagnostics

| Method | Avg Weighted z | Avg WinMax Weighted z | Avg KL | Avg Adaptive Delta | Avg Gate Pass Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 0.1636 | 1.7336 | - | - | 0.7715 |
| Fixed Delta 1.0 | 4.3772 | 4.6820 | - | - | 0.7759 |
| Fixed Delta 2.0 | 8.7719 | 8.8126 | - | - | 0.7932 |
| Fixed Delta 3.0 | 11.7191 | 11.7255 | - | - | 0.7959 |
| Current Adaptive Delta | 8.8480 | 8.8851 | - | - | 0.7874 |
| CA-KL | - | - | 0.4099 | 2.4414 | 1.0000 |
| CA-KL + Candidate Greenlist | 9.8781 | 9.9010 | 0.4061 | 2.4372 | 0.7939 |
| CA-KL + Weighted Detector | 9.3858 | 9.4112 | 0.4099 | 2.4414 | 1.0000 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 9.7985 | 9.8208 | 0.3826 | 1.8376 | 0.7957 |

## Calibrated Detection Quality

| Score | Method | AUC | TPR@1%FPR | TPR@5%FPR |
| --- | --- | ---: | ---: | ---: |
| weighted_z_score | Fixed Delta 1.0 | 0.9930 | 92.40% | 96.60% |
| weighted_z_score | Fixed Delta 2.0 | 0.9985 | 99.40% | 99.80% |
| weighted_z_score | Fixed Delta 3.0 | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | Current Adaptive Delta | 0.9985 | 99.60% | 99.80% |
| weighted_z_score | CA-KL + Candidate Greenlist | 0.9978 | 99.20% | 99.40% |
| weighted_z_score | CA-KL + Weighted Detector | 0.9987 | 99.60% | 99.80% |
| weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 0.9978 | 99.20% | 99.40% |
| winmax_weighted_z_score | Fixed Delta 1.0 | 0.9822 | 86.60% | 91.80% |
| winmax_weighted_z_score | Fixed Delta 2.0 | 0.9962 | 98.40% | 98.80% |
| winmax_weighted_z_score | Fixed Delta 3.0 | 0.9997 | 99.80% | 99.80% |
| winmax_weighted_z_score | Current Adaptive Delta | 0.9963 | 98.20% | 98.80% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist | 0.9943 | 98.80% | 99.00% |
| winmax_weighted_z_score | CA-KL + Weighted Detector | 0.9975 | 99.40% | 99.40% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 0.9943 | 98.80% | 99.00% |

## Notes

- `Detection Success Rate` counts the share of outputs predicted as `Watermarked`.
- `Distinct-1` and `Distinct-2` provide a lightweight diversity estimate for generated text.
- `Avg Repetition Rate` is `1 - unique_words / total_words`; lower values generally indicate less repetition.
- Calibrated detection quality uses `No Watermark` scores as the negative class.
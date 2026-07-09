# Watermark Experiment Summary

| Method | Samples | Avg z-score | Avg Green Fraction | Detection Success Rate | Avg Word Count | Avg Distinct-1 | Avg Distinct-2 | Avg Repetition Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 500 | -0.3780 | 0.2327 | 0.20% | 73.13 | 0.6998 | 0.9193 | 0.3002 |
| CA-KL | 500 | 7.5682 | 0.5857 | 95.60% | 75.16 | 0.7022 | 0.9000 | 0.2978 |
| CA-KL + Candidate Greenlist | 500 | 6.8102 | 0.5544 | 92.40% | 75.62 | 0.7086 | 0.9135 | 0.2914 |
| CA-KL + Weighted Detector | 500 | 7.5682 | 0.5857 | 95.60% | 75.16 | 0.7022 | 0.9000 | 0.2978 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 500 | 6.3873 | 0.5345 | 88.80% | 75.59 | 0.7057 | 0.9098 | 0.2943 |

## CA-KL-CG Diagnostics

| Method | Avg Weighted z | Avg WinMax Weighted z | Avg KL | Avg Adaptive Delta | Avg Gate Pass Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 0.1427 | 1.7069 | - | - | 0.7084 |
| CA-KL | - | - | 0.3828 | 2.5108 | 1.0000 |
| CA-KL + Candidate Greenlist | 9.1410 | 9.1789 | 0.3904 | 2.4861 | 0.7302 |
| CA-KL + Weighted Detector | 8.8399 | 8.8749 | 0.3828 | 2.5108 | 1.0000 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 9.1174 | 9.1552 | 0.3481 | 1.6893 | 0.7230 |

## Calibrated Detection Quality

| Score | Method | AUC | TPR@1%FPR | TPR@5%FPR |
| --- | --- | ---: | ---: | ---: |
| weighted_z_score | CA-KL + Candidate Greenlist | 0.9988 | 99.00% | 99.20% |
| weighted_z_score | CA-KL + Weighted Detector | 0.9994 | 99.40% | 99.60% |
| weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 0.9992 | 99.60% | 99.60% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist | 0.9938 | 97.60% | 98.40% |
| winmax_weighted_z_score | CA-KL + Weighted Detector | 0.9952 | 98.00% | 98.20% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 0.9970 | 98.20% | 99.20% |

## Notes

- `Detection Success Rate` counts the share of outputs predicted as `Watermarked`.
- `Distinct-1` and `Distinct-2` provide a lightweight diversity estimate for generated text.
- `Avg Repetition Rate` is `1 - unique_words / total_words`; lower values generally indicate less repetition.
- Calibrated detection quality uses `No Watermark` scores as the negative class.
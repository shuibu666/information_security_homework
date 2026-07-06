# Watermark Experiment Summary

| Method | Samples | Avg z-score | Avg Green Fraction | Detection Success Rate | Avg Word Count | Avg Distinct-1 | Avg Distinct-2 | Avg Repetition Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 1 | 2.1170 | 0.5556 | 0.00% | 4.00 | 1.0000 | 1.0000 | 0.0000 |
| Fixed Delta 1.0 | 1 | 4.4264 | 0.8889 | 100.00% | 5.00 | 1.0000 | 1.0000 | 0.0000 |
| Current Adaptive Delta | 1 | 5.1962 | 1.0000 | 100.00% | 5.00 | 1.0000 | 1.0000 | 0.0000 |
| CA-KL | 1 | 2.8868 | 0.6667 | 0.00% | 5.00 | 1.0000 | 1.0000 | 0.0000 |
| CA-KL + Candidate Greenlist | 1 | 2.8868 | 0.6667 | 0.00% | 5.00 | 1.0000 | 1.0000 | 0.0000 |
| CA-KL + Weighted Detector | 1 | 2.8868 | 0.6667 | 0.00% | 5.00 | 1.0000 | 1.0000 | 0.0000 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 1 | 2.8868 | 0.6667 | 0.00% | 5.00 | 1.0000 | 1.0000 | 0.0000 |

## CA-KL-CG Diagnostics

| Method | Avg Weighted z | Avg WinMax Weighted z | Avg KL | Avg Adaptive Delta | Avg Gate Pass Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 1.9508 | 1.9508 | - | - | 1.0000 |
| Fixed Delta 1.0 | 4.1802 | 4.1802 | - | - | 1.0000 |
| Current Adaptive Delta | 4.9234 | 4.9234 | - | - | 1.0000 |
| CA-KL | - | - | 0.0200 | 0.4335 | 1.0000 |
| CA-KL + Candidate Greenlist | 2.6939 | 2.6939 | 0.0200 | 0.4390 | 1.0000 |
| CA-KL + Weighted Detector | 3.2865 | 3.2865 | 0.0200 | 0.4335 | 1.0000 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 2.6939 | 2.6939 | 0.0200 | 0.4390 | 1.0000 |

## Calibrated Detection Quality

| Score | Method | AUC | TPR@1%FPR | TPR@5%FPR |
| --- | --- | ---: | ---: | ---: |
| weighted_z_score | Fixed Delta 1.0 | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | Current Adaptive Delta | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | CA-KL + Candidate Greenlist | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | CA-KL + Weighted Detector | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 1.0000 | 100.00% | 100.00% |
| winmax_weighted_z_score | Fixed Delta 1.0 | 1.0000 | 100.00% | 100.00% |
| winmax_weighted_z_score | Current Adaptive Delta | 1.0000 | 100.00% | 100.00% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist | 1.0000 | 100.00% | 100.00% |
| winmax_weighted_z_score | CA-KL + Weighted Detector | 1.0000 | 100.00% | 100.00% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 1.0000 | 100.00% | 100.00% |

## Notes

- `Detection Success Rate` counts the share of outputs predicted as `Watermarked`.
- `Distinct-1` and `Distinct-2` provide a lightweight diversity estimate for generated text.
- `Avg Repetition Rate` is `1 - unique_words / total_words`; lower values generally indicate less repetition.
- Calibrated detection quality uses `No Watermark` scores as the negative class.
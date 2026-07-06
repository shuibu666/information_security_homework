# Watermark Experiment Summary

| Method | Samples | Avg z-score | Avg Green Fraction | Detection Success Rate | Avg Word Count | Avg Distinct-1 | Avg Distinct-2 | Avg Repetition Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 1 | -0.9623 | 0.1111 | 0.00% | 5.00 | 1.0000 | 1.0000 | 0.0000 |
| Fixed Delta 1.0 | 1 | -0.1925 | 0.2222 | 0.00% | 6.00 | 1.0000 | 1.0000 | 0.0000 |
| Current Adaptive Delta | 1 | 0.5774 | 0.3333 | 0.00% | 6.00 | 1.0000 | 1.0000 | 0.0000 |
| CA-KL | 1 | -0.9623 | 0.1111 | 0.00% | 5.00 | 1.0000 | 1.0000 | 0.0000 |
| CA-KL + Candidate Greenlist | 1 | -0.9623 | 0.1111 | 0.00% | 5.00 | 1.0000 | 1.0000 | 0.0000 |
| CA-KL + Weighted Detector | 1 | -0.9623 | 0.1111 | 0.00% | 5.00 | 1.0000 | 1.0000 | 0.0000 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 1 | -0.9623 | 0.1111 | 0.00% | 5.00 | 1.0000 | 1.0000 | 0.0000 |

## CA-KL-CG Diagnostics

| Method | Avg Weighted z | Avg WinMax Weighted z | Avg KL | Avg Adaptive Delta | Avg Gate Pass Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Watermark | -0.5601 | 0.0000 | - | - | 0.2000 |
| Fixed Delta 1.0 | -0.5601 | 0.0000 | - | - | 0.2000 |
| Current Adaptive Delta | -0.5601 | 0.0000 | - | - | 0.2000 |
| CA-KL | - | - | 0.0200 | 0.8106 | 1.0000 |
| CA-KL + Candidate Greenlist | -0.5601 | 0.0000 | 0.0200 | 0.7505 | 0.2000 |
| CA-KL + Weighted Detector | -0.3067 | 0.0000 | 0.0200 | 0.8106 | 1.0000 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | -0.5601 | 0.0000 | 0.0040 | 0.1035 | 0.2000 |

## Calibrated Detection Quality

| Score | Method | AUC | TPR@1%FPR | TPR@5%FPR |
| --- | --- | ---: | ---: | ---: |
| weighted_z_score | Fixed Delta 1.0 | 0.5000 | 100.00% | 100.00% |
| weighted_z_score | Current Adaptive Delta | 0.5000 | 100.00% | 100.00% |
| weighted_z_score | CA-KL + Candidate Greenlist | 0.5000 | 100.00% | 100.00% |
| weighted_z_score | CA-KL + Weighted Detector | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 0.5000 | 100.00% | 100.00% |
| winmax_weighted_z_score | Fixed Delta 1.0 | 0.5000 | 100.00% | 100.00% |
| winmax_weighted_z_score | Current Adaptive Delta | 0.5000 | 100.00% | 100.00% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist | 0.5000 | 100.00% | 100.00% |
| winmax_weighted_z_score | CA-KL + Weighted Detector | 0.5000 | 100.00% | 100.00% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 0.5000 | 100.00% | 100.00% |

## Notes

- `Detection Success Rate` counts the share of outputs predicted as `Watermarked`.
- `Distinct-1` and `Distinct-2` provide a lightweight diversity estimate for generated text.
- `Avg Repetition Rate` is `1 - unique_words / total_words`; lower values generally indicate less repetition.
- Calibrated detection quality uses `No Watermark` scores as the negative class.
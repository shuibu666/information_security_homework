# Watermark Experiment Summary

| Method | Samples | Avg z-score | Avg Green Fraction | Detection Success Rate | Avg Word Count | Avg Distinct-1 | Avg Distinct-2 | Avg Repetition Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 50 | -0.4041 | 0.2324 | 0.00% | 74.54 | 0.7050 | 0.9289 | 0.2950 |
| CA-KL | 50 | 7.3784 | 0.5827 | 92.00% | 74.70 | 0.7195 | 0.9178 | 0.2805 |
| CA-KL + Candidate Greenlist | 50 | 6.5272 | 0.5431 | 88.00% | 74.50 | 0.6981 | 0.9024 | 0.3019 |
| CA-KL + Weighted Detector | 50 | 7.3784 | 0.5827 | 92.00% | 74.70 | 0.7195 | 0.9178 | 0.2805 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 50 | 5.8227 | 0.5158 | 78.00% | 70.96 | 0.7172 | 0.9028 | 0.2828 |

## CA-KL-CG Diagnostics

| Method | Avg Weighted z | Avg WinMax Weighted z | Avg KL | Avg Adaptive Delta | Avg Gate Pass Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 0.2031 | 1.7580 | - | - | 0.7068 |
| CA-KL | - | - | 0.3894 | 2.5034 | 1.0000 |
| CA-KL + Candidate Greenlist | 8.8936 | 8.9460 | 0.3921 | 2.4912 | 0.7281 |
| CA-KL + Weighted Detector | 8.7523 | 8.7844 | 0.3894 | 2.5034 | 1.0000 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 8.6790 | 8.7157 | 0.3412 | 1.6579 | 0.7066 |

## Calibrated Detection Quality

| Score | Method | AUC | TPR@1%FPR | TPR@5%FPR |
| --- | --- | ---: | ---: | ---: |
| weighted_z_score | CA-KL + Candidate Greenlist | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | CA-KL + Weighted Detector | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 1.0000 | 100.00% | 100.00% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist | 0.9964 | 96.00% | 96.00% |
| winmax_weighted_z_score | CA-KL + Weighted Detector | 0.9924 | 94.00% | 94.00% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 0.9964 | 96.00% | 96.00% |

## Notes

- `Detection Success Rate` counts the share of outputs predicted as `Watermarked`.
- `Distinct-1` and `Distinct-2` provide a lightweight diversity estimate for generated text.
- `Avg Repetition Rate` is `1 - unique_words / total_words`; lower values generally indicate less repetition.
- Calibrated detection quality uses `No Watermark` scores as the negative class.
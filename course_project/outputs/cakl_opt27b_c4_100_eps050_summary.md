# Watermark Experiment Summary

| Method | Samples | Avg z-score | Avg Green Fraction | Detection Success Rate | Avg Word Count | Avg Distinct-1 | Avg Distinct-2 | Avg Repetition Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 100 | -0.3840 | 0.2335 | 0.00% | 74.74 | 0.6946 | 0.9225 | 0.3054 |
| Fixed Delta 1.0 | 100 | 2.8658 | 0.3792 | 25.00% | 74.33 | 0.6942 | 0.9148 | 0.3058 |
| Fixed Delta 2.0 | 100 | 6.6279 | 0.5460 | 91.00% | 75.58 | 0.7002 | 0.9121 | 0.2998 |
| Fixed Delta 3.0 | 100 | 10.2261 | 0.7096 | 100.00% | 75.36 | 0.6879 | 0.8727 | 0.3121 |
| Current Adaptive Delta | 100 | 6.5619 | 0.5433 | 91.00% | 74.71 | 0.6969 | 0.9062 | 0.3031 |
| CA-KL | 100 | 7.6077 | 0.5927 | 96.00% | 74.81 | 0.7057 | 0.9073 | 0.2943 |
| CA-KL + Candidate Greenlist | 100 | 6.6770 | 0.5524 | 95.00% | 74.24 | 0.7031 | 0.9075 | 0.2969 |
| CA-KL + Weighted Detector | 100 | 7.6077 | 0.5927 | 96.00% | 74.81 | 0.7057 | 0.9073 | 0.2943 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 100 | 6.3767 | 0.5408 | 87.00% | 73.14 | 0.7256 | 0.9257 | 0.2744 |

## CA-KL-CG Diagnostics

| Method | Avg Weighted z | Avg WinMax Weighted z | Avg KL | Avg Adaptive Delta | Avg Gate Pass Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 0.2157 | 1.8328 | - | - | 0.7111 |
| Fixed Delta 1.0 | 3.9588 | 4.2782 | - | - | 0.7062 |
| Fixed Delta 2.0 | 7.8605 | 7.9198 | - | - | 0.7279 |
| Fixed Delta 3.0 | 10.7661 | 10.7782 | - | - | 0.7341 |
| Current Adaptive Delta | 7.8868 | 7.9440 | - | - | 0.7336 |
| CA-KL | - | - | 0.3884 | 2.4966 | 1.0000 |
| CA-KL + Candidate Greenlist | 9.1218 | 9.1594 | 0.3924 | 2.4846 | 0.7326 |
| CA-KL + Weighted Detector | 8.9367 | 8.9518 | 0.3884 | 2.4966 | 1.0000 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 9.1943 | 9.2487 | 0.3591 | 1.7384 | 0.7434 |

## Calibrated Detection Quality

| Score | Method | AUC | TPR@1%FPR | TPR@5%FPR |
| --- | --- | ---: | ---: | ---: |
| weighted_z_score | Fixed Delta 1.0 | 0.9895 | 92.00% | 94.00% |
| weighted_z_score | Fixed Delta 2.0 | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | Fixed Delta 3.0 | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | Current Adaptive Delta | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | CA-KL + Candidate Greenlist | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | CA-KL + Weighted Detector | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 1.0000 | 100.00% | 100.00% |
| winmax_weighted_z_score | Fixed Delta 1.0 | 0.9592 | 83.00% | 86.00% |
| winmax_weighted_z_score | Fixed Delta 2.0 | 0.9972 | 97.00% | 97.00% |
| winmax_weighted_z_score | Fixed Delta 3.0 | 1.0000 | 100.00% | 100.00% |
| winmax_weighted_z_score | Current Adaptive Delta | 0.9972 | 97.00% | 97.00% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist | 0.9984 | 98.00% | 98.00% |
| winmax_weighted_z_score | CA-KL + Weighted Detector | 0.9964 | 97.00% | 97.00% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 0.9984 | 98.00% | 98.00% |

## Notes

- `Detection Success Rate` counts the share of outputs predicted as `Watermarked`.
- `Distinct-1` and `Distinct-2` provide a lightweight diversity estimate for generated text.
- `Avg Repetition Rate` is `1 - unique_words / total_words`; lower values generally indicate less repetition.
- Calibrated detection quality uses `No Watermark` scores as the negative class.
# Watermark Experiment Summary

| Method | Samples | Avg z-score | Avg Green Fraction | Detection Success Rate | Avg Word Count | Avg Distinct-1 | Avg Distinct-2 | Avg Repetition Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 20 | -0.0990 | 0.2439 | 0.00% | 38.10 | 0.8032 | 0.9583 | 0.1968 |
| Fixed Delta 1.0 | 20 | 2.2710 | 0.4017 | 5.00% | 38.20 | 0.8032 | 0.9623 | 0.1968 |
| Fixed Delta 2.0 | 20 | 4.7619 | 0.5557 | 65.00% | 37.85 | 0.8147 | 0.9656 | 0.1853 |
| Fixed Delta 3.0 | 20 | 7.2192 | 0.7034 | 95.00% | 39.90 | 0.7777 | 0.9321 | 0.2223 |
| Current Adaptive Delta | 20 | 4.5639 | 0.5435 | 60.00% | 37.25 | 0.8257 | 0.9745 | 0.1743 |
| CA-KL | 20 | 4.8444 | 0.5608 | 80.00% | 38.10 | 0.8211 | 0.9693 | 0.1789 |
| CA-KL + Candidate Greenlist | 20 | 4.0943 | 0.5118 | 50.00% | 37.40 | 0.8045 | 0.9603 | 0.1955 |
| CA-KL + Weighted Detector | 20 | 4.8444 | 0.5608 | 80.00% | 38.10 | 0.8211 | 0.9693 | 0.1789 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 20 | 3.7149 | 0.4884 | 35.00% | 37.60 | 0.7890 | 0.9462 | 0.2110 |

## CA-KL-CG Diagnostics

| Method | Avg Weighted z | Avg WinMax Weighted z | Avg KL | Avg Adaptive Delta | Avg Gate Pass Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 0.1588 | 1.2396 | - | - | 0.7640 |
| Fixed Delta 1.0 | 2.9819 | 3.4782 | - | - | 0.7314 |
| Fixed Delta 2.0 | 5.4470 | 5.5347 | - | - | 0.7594 |
| Fixed Delta 3.0 | 8.1754 | 8.2417 | - | - | 0.7820 |
| Current Adaptive Delta | 5.3834 | 5.4514 | - | - | 0.7614 |
| CA-KL | - | - | 0.3364 | 2.2964 | 1.0000 |
| CA-KL + Candidate Greenlist | 5.6508 | 5.7302 | 0.3292 | 2.2885 | 0.7569 |
| CA-KL + Weighted Detector | 5.8232 | 5.8604 | 0.3364 | 2.2964 | 1.0000 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 5.6567 | 5.7589 | 0.2957 | 1.6005 | 0.7559 |

## Calibrated Detection Quality

| Score | Method | AUC | TPR@1%FPR | TPR@5%FPR |
| --- | --- | ---: | ---: | ---: |
| weighted_z_score | Fixed Delta 1.0 | 0.9600 | 90.00% | 90.00% |
| weighted_z_score | Fixed Delta 2.0 | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | Fixed Delta 3.0 | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | Current Adaptive Delta | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | CA-KL + Candidate Greenlist | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | CA-KL + Weighted Detector | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 1.0000 | 100.00% | 100.00% |
| winmax_weighted_z_score | Fixed Delta 1.0 | 0.9425 | 70.00% | 70.00% |
| winmax_weighted_z_score | Fixed Delta 2.0 | 0.9975 | 95.00% | 95.00% |
| winmax_weighted_z_score | Fixed Delta 3.0 | 1.0000 | 100.00% | 100.00% |
| winmax_weighted_z_score | Current Adaptive Delta | 0.9975 | 95.00% | 95.00% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist | 0.9975 | 95.00% | 95.00% |
| winmax_weighted_z_score | CA-KL + Weighted Detector | 0.9975 | 95.00% | 95.00% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 0.9975 | 95.00% | 95.00% |

## Notes

- `Detection Success Rate` counts the share of outputs predicted as `Watermarked`.
- `Distinct-1` and `Distinct-2` provide a lightweight diversity estimate for generated text.
- `Avg Repetition Rate` is `1 - unique_words / total_words`; lower values generally indicate less repetition.
- Calibrated detection quality uses `No Watermark` scores as the negative class.
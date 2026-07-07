# Watermark Experiment Summary

| Method | Samples | Avg z-score | Avg Green Fraction | Detection Success Rate | Avg Word Count | Avg Distinct-1 | Avg Distinct-2 | Avg Repetition Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 100 | -0.0828 | 0.2462 | 1.00% | 74.43 | 0.6656 | 0.8910 | 0.3344 |
| Fixed Delta 1.0 | 100 | 4.0220 | 0.4254 | 51.00% | 76.33 | 0.6572 | 0.8840 | 0.3428 |
| Fixed Delta 2.0 | 100 | 8.0559 | 0.5982 | 97.00% | 77.56 | 0.6495 | 0.8584 | 0.3405 |
| Fixed Delta 3.0 | 100 | 11.5462 | 0.7561 | 98.00% | 76.84 | 0.6471 | 0.8338 | 0.3429 |
| Current Adaptive Delta | 100 | 7.7565 | 0.5876 | 95.00% | 75.80 | 0.6444 | 0.8555 | 0.3456 |
| CA-KL | 100 | 8.7443 | 0.6282 | 97.00% | 77.79 | 0.6431 | 0.8425 | 0.3469 |
| CA-KL + Candidate Greenlist | 100 | 8.4769 | 0.6194 | 96.00% | 76.97 | 0.6564 | 0.8665 | 0.3336 |
| CA-KL + Weighted Detector | 100 | 8.7443 | 0.6282 | 97.00% | 77.79 | 0.6431 | 0.8425 | 0.3469 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 100 | 8.0047 | 0.5995 | 94.00% | 76.03 | 0.6506 | 0.8574 | 0.3394 |

## CA-KL-CG Diagnostics

| Method | Avg Weighted z | Avg WinMax Weighted z | Avg KL | Avg Adaptive Delta | Avg Gate Pass Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 0.0615 | 1.7761 | - | - | 0.7745 |
| Fixed Delta 1.0 | 4.4507 | 4.6884 | - | - | 0.7834 |
| Fixed Delta 2.0 | 8.9580 | 8.9748 | - | - | 0.8054 |
| Fixed Delta 3.0 | 11.8049 | 11.8114 | - | - | 0.7963 |
| Current Adaptive Delta | 8.6914 | 8.7310 | - | - | 0.7916 |
| CA-KL | - | - | 0.4111 | 2.4398 | 1.0000 |
| CA-KL + Candidate Greenlist | 10.0640 | 10.0757 | 0.4145 | 2.4201 | 0.8123 |
| CA-KL + Weighted Detector | 9.4318 | 9.4376 | 0.4111 | 2.4398 | 1.0000 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 9.7782 | 9.7935 | 0.3841 | 1.8337 | 0.7970 |

## Calibrated Detection Quality

| Score | Method | AUC | TPR@1%FPR | TPR@5%FPR |
| --- | --- | ---: | ---: | ---: |
| weighted_z_score | Fixed Delta 1.0 | 0.9869 | 92.00% | 94.00% |
| weighted_z_score | Fixed Delta 2.0 | 0.9936 | 99.00% | 99.00% |
| weighted_z_score | Fixed Delta 3.0 | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | Current Adaptive Delta | 0.9936 | 99.00% | 99.00% |
| weighted_z_score | CA-KL + Candidate Greenlist | 0.9936 | 99.00% | 99.00% |
| weighted_z_score | CA-KL + Weighted Detector | 0.9945 | 99.00% | 99.00% |
| weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 0.9936 | 99.00% | 99.00% |
| winmax_weighted_z_score | Fixed Delta 1.0 | 0.9658 | 74.00% | 90.00% |
| winmax_weighted_z_score | Fixed Delta 2.0 | 0.9886 | 98.00% | 98.00% |
| winmax_weighted_z_score | Fixed Delta 3.0 | 0.9985 | 99.00% | 99.00% |
| winmax_weighted_z_score | Current Adaptive Delta | 0.9879 | 97.00% | 97.00% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist | 0.9886 | 98.00% | 98.00% |
| winmax_weighted_z_score | CA-KL + Weighted Detector | 0.9880 | 98.00% | 98.00% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 0.9886 | 98.00% | 98.00% |

## Notes

- `Detection Success Rate` counts the share of outputs predicted as `Watermarked`.
- `Distinct-1` and `Distinct-2` provide a lightweight diversity estimate for generated text.
- `Avg Repetition Rate` is `1 - unique_words / total_words`; lower values generally indicate less repetition.
- Calibrated detection quality uses `No Watermark` scores as the negative class.
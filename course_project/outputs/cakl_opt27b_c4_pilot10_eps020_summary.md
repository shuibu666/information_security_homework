# Watermark Experiment Summary

| Method | Samples | Avg z-score | Avg Green Fraction | Detection Success Rate | Avg Word Count | Avg Distinct-1 | Avg Distinct-2 | Avg Repetition Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 10 | -0.2474 | 0.2347 | 0.00% | 38.60 | 0.8000 | 0.9567 | 0.2000 |
| Fixed Delta 1.0 | 10 | 2.5238 | 0.4061 | 10.00% | 41.30 | 0.7936 | 0.9641 | 0.2064 |
| Fixed Delta 2.0 | 10 | 5.0642 | 0.5633 | 70.00% | 39.70 | 0.8151 | 0.9629 | 0.1849 |
| Fixed Delta 3.0 | 10 | 7.4176 | 0.7173 | 100.00% | 41.10 | 0.8080 | 0.9654 | 0.1920 |
| Current Adaptive Delta | 10 | 4.7673 | 0.5449 | 70.00% | 40.50 | 0.8055 | 0.9659 | 0.1945 |
| CA-KL | 10 | 3.9095 | 0.4918 | 50.00% | 40.60 | 0.8178 | 0.9739 | 0.1822 |
| CA-KL + Candidate Greenlist | 10 | 3.3816 | 0.4592 | 20.00% | 40.40 | 0.7828 | 0.9650 | 0.2172 |
| CA-KL + Weighted Detector | 10 | 3.9095 | 0.4918 | 50.00% | 40.60 | 0.8178 | 0.9739 | 0.1822 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 10 | 3.0847 | 0.4408 | 10.00% | 40.10 | 0.7971 | 0.9705 | 0.2029 |

## CA-KL-CG Diagnostics

| Method | Avg Weighted z | Avg WinMax Weighted z | Avg KL | Avg Adaptive Delta | Avg Gate Pass Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 0.2258 | 1.4043 | - | - | 0.7940 |
| Fixed Delta 1.0 | 3.2496 | 3.6860 | - | - | 0.7300 |
| Fixed Delta 2.0 | 5.5914 | 5.7079 | - | - | 0.7580 |
| Fixed Delta 3.0 | 8.1261 | 8.2142 | - | - | 0.7943 |
| Current Adaptive Delta | 5.5716 | 5.6356 | - | - | 0.7780 |
| CA-KL | - | - | 0.1850 | 1.7860 | 1.0000 |
| CA-KL + Candidate Greenlist | 4.8323 | 4.9556 | 0.1767 | 1.7551 | 0.7820 |
| CA-KL + Weighted Detector | 4.8229 | 5.0099 | 0.1850 | 1.7860 | 1.0000 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 4.9977 | 5.0752 | 0.1538 | 1.1638 | 0.7700 |

## Calibrated Detection Quality

| Score | Method | AUC | TPR@1%FPR | TPR@5%FPR |
| --- | --- | ---: | ---: | ---: |
| weighted_z_score | Fixed Delta 1.0 | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | Fixed Delta 2.0 | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | Fixed Delta 3.0 | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | Current Adaptive Delta | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | CA-KL + Candidate Greenlist | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | CA-KL + Weighted Detector | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 1.0000 | 100.00% | 100.00% |
| winmax_weighted_z_score | Fixed Delta 1.0 | 0.9500 | 80.00% | 80.00% |
| winmax_weighted_z_score | Fixed Delta 2.0 | 1.0000 | 100.00% | 100.00% |
| winmax_weighted_z_score | Fixed Delta 3.0 | 1.0000 | 100.00% | 100.00% |
| winmax_weighted_z_score | Current Adaptive Delta | 1.0000 | 100.00% | 100.00% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist | 1.0000 | 100.00% | 100.00% |
| winmax_weighted_z_score | CA-KL + Weighted Detector | 1.0000 | 100.00% | 100.00% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 1.0000 | 100.00% | 100.00% |

## Notes

- `Detection Success Rate` counts the share of outputs predicted as `Watermarked`.
- `Distinct-1` and `Distinct-2` provide a lightweight diversity estimate for generated text.
- `Avg Repetition Rate` is `1 - unique_words / total_words`; lower values generally indicate less repetition.
- Calibrated detection quality uses `No Watermark` scores as the negative class.
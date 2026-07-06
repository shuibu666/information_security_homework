# Watermark Experiment Summary

| Method | Samples | Avg z-score | Avg Green Fraction | Detection Success Rate | Avg Word Count | Avg Distinct-1 | Avg Distinct-2 | Avg Repetition Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 10 | -0.2474 | 0.2347 | 0.00% | 38.60 | 0.8000 | 0.9567 | 0.2000 |
| Fixed Delta 1.0 | 10 | 2.5238 | 0.4061 | 10.00% | 41.30 | 0.7936 | 0.9641 | 0.2064 |
| Fixed Delta 2.0 | 10 | 5.0642 | 0.5633 | 70.00% | 39.70 | 0.8151 | 0.9629 | 0.1849 |
| Fixed Delta 3.0 | 10 | 7.4176 | 0.7173 | 100.00% | 41.10 | 0.8080 | 0.9654 | 0.1920 |
| Current Adaptive Delta | 10 | 4.7673 | 0.5449 | 70.00% | 40.50 | 0.8055 | 0.9659 | 0.1945 |
| CA-KL | 10 | 0.9732 | 0.3102 | 0.00% | 37.60 | 0.8426 | 0.9853 | 0.1574 |
| CA-KL + Candidate Greenlist | 10 | 0.5101 | 0.2815 | 0.00% | 38.00 | 0.8036 | 0.9714 | 0.1964 |
| CA-KL + Weighted Detector | 10 | 0.9732 | 0.3102 | 0.00% | 37.60 | 0.8426 | 0.9853 | 0.1574 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 10 | -0.0227 | 0.2482 | 0.00% | 35.70 | 0.8306 | 0.9737 | 0.1694 |

## CA-KL-CG Diagnostics

| Method | Avg Weighted z | Avg WinMax Weighted z | Avg KL | Avg Adaptive Delta | Avg Gate Pass Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 0.2580 | 1.5125 | - | - | 0.2180 |
| Fixed Delta 1.0 | 2.4207 | 2.9673 | - | - | 0.1920 |
| Fixed Delta 2.0 | 3.6149 | 3.7861 | - | - | 0.2200 |
| Fixed Delta 3.0 | 5.0919 | 5.2114 | - | - | 0.2100 |
| Current Adaptive Delta | 3.7155 | 3.8436 | - | - | 0.2160 |
| CA-KL | - | - | 0.0194 | 0.7982 | 1.0000 |
| CA-KL + Candidate Greenlist | 0.6407 | 2.0239 | 0.0185 | 0.8719 | 0.1844 |
| CA-KL + Weighted Detector | 1.5118 | 2.3660 | 0.0194 | 0.7982 | 1.0000 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 0.7133 | 1.9335 | 0.0038 | 0.0885 | 0.1898 |

## Calibrated Detection Quality

| Score | Method | AUC | TPR@1%FPR | TPR@5%FPR |
| --- | --- | ---: | ---: | ---: |
| weighted_z_score | Fixed Delta 1.0 | 0.9400 | 70.00% | 70.00% |
| weighted_z_score | Fixed Delta 2.0 | 0.9800 | 90.00% | 90.00% |
| weighted_z_score | Fixed Delta 3.0 | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | Current Adaptive Delta | 1.0000 | 100.00% | 100.00% |
| weighted_z_score | CA-KL + Candidate Greenlist | 0.6100 | 0.00% | 0.00% |
| weighted_z_score | CA-KL + Weighted Detector | 0.8400 | 40.00% | 40.00% |
| weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 0.6350 | 10.00% | 10.00% |
| winmax_weighted_z_score | Fixed Delta 1.0 | 0.8600 | 20.00% | 20.00% |
| winmax_weighted_z_score | Fixed Delta 2.0 | 0.9200 | 50.00% | 50.00% |
| winmax_weighted_z_score | Fixed Delta 3.0 | 1.0000 | 100.00% | 100.00% |
| winmax_weighted_z_score | Current Adaptive Delta | 0.9300 | 50.00% | 50.00% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist | 0.6250 | 10.00% | 10.00% |
| winmax_weighted_z_score | CA-KL + Weighted Detector | 0.7400 | 0.00% | 0.00% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 0.6150 | 0.00% | 0.00% |

## Notes

- `Detection Success Rate` counts the share of outputs predicted as `Watermarked`.
- `Distinct-1` and `Distinct-2` provide a lightweight diversity estimate for generated text.
- `Avg Repetition Rate` is `1 - unique_words / total_words`; lower values generally indicate less repetition.
- Calibrated detection quality uses `No Watermark` scores as the negative class.
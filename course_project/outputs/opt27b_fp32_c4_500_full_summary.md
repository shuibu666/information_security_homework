# Watermark Experiment Summary

| Method | Samples | Avg z-score | Avg Green Fraction | Detection Success Rate | Avg Word Count | Avg Distinct-1 | Avg Distinct-2 | Avg Repetition Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 500 | -0.3780 | 0.2327 | 0.20% | 73.13 | 0.6998 | 0.9193 | 0.3002 |
| Fixed Delta 0.5 | 500 | 1.2469 | 0.3051 | 1.60% | 74.21 | 0.6943 | 0.9169 | 0.3057 |
| Fixed Delta 1.0 | 500 | 2.9412 | 0.3812 | 25.40% | 73.48 | 0.7031 | 0.9222 | 0.2969 |
| Fixed Delta 1.864 | 500 | 6.3353 | 0.5305 | 89.60% | 74.62 | 0.6967 | 0.8986 | 0.3033 |
| Fixed Delta 2.0 | 500 | 6.8200 | 0.5512 | 92.80% | 75.66 | 0.6935 | 0.8962 | 0.3065 |
| Fixed Delta 3.0 | 500 | 10.0975 | 0.6972 | 98.40% | 76.35 | 0.6958 | 0.8785 | 0.3042 |
| Current Adaptive Delta | 500 | 6.2743 | 0.5283 | 88.60% | 74.17 | 0.7037 | 0.9061 | 0.2963 |
| CA-KL | 500 | 7.5682 | 0.5857 | 95.60% | 75.16 | 0.7022 | 0.9000 | 0.2978 |
| CA-KL + Candidate Greenlist | 500 | 6.8102 | 0.5544 | 92.40% | 75.62 | 0.7086 | 0.9135 | 0.2914 |
| CA-KL + Weighted Detector | 500 | 7.5682 | 0.5857 | 95.60% | 75.16 | 0.7022 | 0.9000 | 0.2978 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 500 | 2.2154 | 0.3477 | 15.80% | 73.66 | 0.7012 | 0.9163 | 0.2988 |

## CA-KL-CG Diagnostics

| Method | Avg Weighted z | Avg WinMax Weighted z | Avg KL | Avg Adaptive Delta | Avg Gate Pass Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 0.0356 | 1.6096 | - | - | 0.2015 |
| Fixed Delta 0.5 | 1.2586 | 2.3518 | - | - | 0.2030 |
| Fixed Delta 1.0 | 2.5773 | 3.2200 | - | - | 0.2016 |
| Fixed Delta 1.864 | 4.9311 | 5.1085 | - | - | 0.2029 |
| Fixed Delta 2.0 | 5.2573 | 5.4193 | - | - | 0.2018 |
| Fixed Delta 3.0 | 6.6656 | 6.7191 | - | - | 0.2010 |
| Current Adaptive Delta | 5.5091 | 5.6278 | - | - | 0.2057 |
| CA-KL | - | - | 0.3828 | 2.5108 | 1.0000 |
| CA-KL + Candidate Greenlist | 5.5622 | 5.6717 | 0.3904 | 2.4861 | 0.2109 |
| CA-KL + Weighted Detector | 8.8399 | 8.8749 | 0.3828 | 2.5108 | 1.0000 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 5.6473 | 5.7747 | 0.1003 | 0.4332 | 0.2003 |

## Calibrated Detection Quality

| Score | Method | AUC | TPR@1%FPR | TPR@5%FPR |
| --- | --- | ---: | ---: | ---: |
| weighted_z_score | Fixed Delta 0.5 | 0.7975 | 12.20% | 37.20% |
| weighted_z_score | Fixed Delta 1.0 | 0.9373 | 48.80% | 76.20% |
| weighted_z_score | Fixed Delta 1.864 | 0.9872 | 89.80% | 96.20% |
| weighted_z_score | Fixed Delta 2.0 | 0.9907 | 92.80% | 96.80% |
| weighted_z_score | Fixed Delta 3.0 | 0.9951 | 96.60% | 98.80% |
| weighted_z_score | Current Adaptive Delta | 0.9923 | 93.20% | 97.80% |
| weighted_z_score | CA-KL + Candidate Greenlist | 0.9914 | 93.80% | 98.00% |
| weighted_z_score | CA-KL + Weighted Detector | 0.9994 | 99.00% | 99.80% |
| weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 0.9922 | 94.00% | 97.40% |
| winmax_weighted_z_score | Fixed Delta 0.5 | 0.7321 | 6.40% | 26.40% |
| winmax_weighted_z_score | Fixed Delta 1.0 | 0.8877 | 33.20% | 61.00% |
| winmax_weighted_z_score | Fixed Delta 1.864 | 0.9653 | 82.40% | 92.00% |
| winmax_weighted_z_score | Fixed Delta 2.0 | 0.9765 | 86.40% | 94.00% |
| winmax_weighted_z_score | Fixed Delta 3.0 | 0.9837 | 92.80% | 95.80% |
| winmax_weighted_z_score | Current Adaptive Delta | 0.9742 | 89.20% | 92.80% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist | 0.9764 | 89.00% | 95.00% |
| winmax_weighted_z_score | CA-KL + Weighted Detector | 0.9956 | 98.00% | 98.60% |
| winmax_weighted_z_score | CA-KL + Candidate Greenlist + Weighted/WinMax | 0.9802 | 89.80% | 94.60% |

## Notes

- `Detection Success Rate` counts the share of outputs predicted as `Watermarked`.
- `Distinct-1` and `Distinct-2` provide a lightweight diversity estimate for generated text.
- `Avg Repetition Rate` is `1 - unique_words / total_words`; lower values generally indicate less repetition.
- Calibrated detection quality uses `No Watermark` scores as the negative class.
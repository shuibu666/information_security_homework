# Watermark Experiment Summary

| Method | Samples | Avg z-score | Avg Green Fraction | Detection Success Rate | Avg Tokens Counted | Avg Word Count | Avg Distinct-1 | Avg Distinct-2 | Avg Repetition Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 499 | -0.1264 | 0.2462 | 1.00% | 203.81 | 160.36 | 0.5587 | 0.8479 | 0.4413 |
| Fixed Delta 2.0 | 499 | 10.6572 | 0.5732 | 97.39% | 203.86 | 165.96 | 0.5291 | 0.7826 | 0.4709 |
| Current Adaptive Delta | 499 | 9.9697 | 0.5523 | 96.79% | 203.88 | 166.41 | 0.5315 | 0.7915 | 0.4685 |
| CA-KL | 499 | 12.1205 | 0.6176 | 98.40% | 203.85 | 166.94 | 0.5272 | 0.7725 | 0.4728 |
| CA-KL + Candidate Greenlist | 499 | 10.9289 | 0.5814 | 98.00% | 203.93 | 167.11 | 0.5474 | 0.8087 | 0.4526 |
| CA-KL + Weighted Detector | 499 | 12.1205 | 0.6176 | 98.40% | 203.85 | 166.94 | 0.5272 | 0.7725 | 0.4728 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 499 | 10.4528 | 0.5670 | 97.19% | 203.90 | 166.64 | 0.5386 | 0.7945 | 0.4614 |

## CA-KL-CG Diagnostics

| Method | Avg Weighted z | Avg WinMax Weighted z | Avg KL | Avg Adaptive Delta | Avg Gate Pass Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Watermark | - | - | - | - | - |
| Fixed Delta 2.0 | - | - | - | - | - |
| Current Adaptive Delta | - | - | - | - | - |
| CA-KL | - | - | 0.3642 | 2.5401 | 1.0000 |
| CA-KL + Candidate Greenlist | - | - | 0.3757 | 2.5134 | 1.0000 |
| CA-KL + Weighted Detector | - | - | 0.3642 | 2.5401 | 1.0000 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | - | - | 0.3327 | 1.6304 | 0.6944 |

## Notes

- `Detection Success Rate` counts the share of outputs predicted as `Watermarked`.
- `Distinct-1` and `Distinct-2` provide a lightweight diversity estimate for generated text.
- `Avg Repetition Rate` is `1 - unique_words / total_words`; lower values generally indicate less repetition.
- Calibrated detection quality uses `No Watermark` scores as the negative class.
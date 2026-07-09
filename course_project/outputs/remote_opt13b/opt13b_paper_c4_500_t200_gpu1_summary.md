# Watermark Experiment Summary

| Method | Samples | Avg z-score | Avg Green Fraction | Detection Success Rate | Avg Tokens Counted | Avg Word Count | Avg Distinct-1 | Avg Distinct-2 | Avg Repetition Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 499 | -0.2836 | 0.2414 | 0.20% | 203.82 | 159.53 | 0.5571 | 0.8456 | 0.4429 |
| Fixed Delta 2.0 | 499 | 10.6006 | 0.5714 | 96.19% | 203.96 | 164.22 | 0.5374 | 0.7900 | 0.4626 |

## Notes

- `Detection Success Rate` counts the share of outputs predicted as `Watermarked`.
- `Distinct-1` and `Distinct-2` provide a lightweight diversity estimate for generated text.
- `Avg Repetition Rate` is `1 - unique_words / total_words`; lower values generally indicate less repetition.
- Calibrated detection quality uses `No Watermark` scores as the negative class.
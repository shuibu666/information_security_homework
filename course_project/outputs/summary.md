# Watermark Experiment Summary

| Method | Samples | Avg z-score | Avg Green Fraction | Detection Success Rate | Avg Word Count | Avg Distinct-1 | Avg Distinct-2 | Avg Repetition Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 12 | 0.5292 | 0.2729 | 0.00% | 74.50 | 0.6143 | 0.8491 | 0.3857 |
| Fixed Delta 0.5 | 12 | 1.6291 | 0.3264 | 0.00% | 71.17 | 0.6383 | 0.8590 | 0.3617 |
| Fixed Delta 1.0 | 12 | 4.4561 | 0.4450 | 66.67% | 77.33 | 0.6135 | 0.8461 | 0.3865 |
| Fixed Delta 2.0 | 12 | 8.6330 | 0.6313 | 100.00% | 77.50 | 0.6155 | 0.8185 | 0.3845 |
| Fixed Delta 3.0 | 12 | 13.2299 | 0.8258 | 100.00% | 84.33 | 0.5318 | 0.7171 | 0.4682 |
| Adaptive Delta | 12 | 6.1467 | 0.5231 | 83.33% | 72.58 | 0.5994 | 0.8238 | 0.4006 |

## Notes

- `Detection Success Rate` counts the share of outputs predicted as `Watermarked`.
- `Distinct-1` and `Distinct-2` provide a lightweight diversity estimate for generated text.
- `Avg Repetition Rate` is `1 - unique_words / total_words`; lower values generally indicate less repetition.
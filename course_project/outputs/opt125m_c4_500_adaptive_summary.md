# Watermark Experiment Summary

| Method | Samples | Avg z-score | Avg Green Fraction | Detection Success Rate | Avg Word Count | Avg Distinct-1 | Avg Distinct-2 | Avg Repetition Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 500 | 0.0763 | 0.2528 | 0.40% | 74.71 | 0.6468 | 0.8750 | 0.3532 |
| Fixed Delta 0.5 | 500 | 1.7881 | 0.3285 | 4.00% | 76.16 | 0.6449 | 0.8770 | 0.3551 |
| Fixed Delta 1.0 | 500 | 3.9266 | 0.4236 | 47.20% | 75.07 | 0.6472 | 0.8706 | 0.3528 |
| Fixed Delta 2.0 | 500 | 8.2140 | 0.6115 | 97.20% | 76.82 | 0.6528 | 0.8579 | 0.3452 |
| Fixed Delta 3.0 | 500 | 11.5481 | 0.7606 | 99.00% | 77.60 | 0.6536 | 0.8381 | 0.3444 |
| Adaptive Delta | 500 | 7.9256 | 0.5987 | 96.60% | 76.28 | 0.6449 | 0.8482 | 0.3531 |

## Notes

- `Detection Success Rate` counts the share of outputs predicted as `Watermarked`.
- `Distinct-1` and `Distinct-2` provide a lightweight diversity estimate for generated text.
- `Avg Repetition Rate` is `1 - unique_words / total_words`; lower values generally indicate less repetition.
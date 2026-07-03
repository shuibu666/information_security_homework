# Watermark Experiment Summary

| Method | Samples | Avg z-score | Avg Green Fraction | Detection Success Rate | Avg Word Count | Avg Distinct-1 | Avg Distinct-2 | Avg Repetition Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 500 | -0.3518 | 0.2339 | 0.20% | 73.20 | 0.7021 | 0.9214 | 0.2979 |
| Fixed Delta 0.5 | 500 | 1.2485 | 0.3051 | 1.40% | 74.28 | 0.6939 | 0.9168 | 0.3061 |
| Fixed Delta 1.0 | 500 | 2.9343 | 0.3810 | 24.40% | 73.36 | 0.7031 | 0.9220 | 0.2969 |
| Fixed Delta 2.0 | 500 | 6.8227 | 0.5512 | 93.40% | 75.73 | 0.6950 | 0.8994 | 0.3050 |
| Fixed Delta 3.0 | 500 | 10.0818 | 0.6970 | 98.40% | 76.31 | 0.6991 | 0.8829 | 0.3009 |
| Adaptive Delta | 500 | 6.2677 | 0.5281 | 89.00% | 74.29 | 0.7031 | 0.9068 | 0.2969 |

## Notes

- `Detection Success Rate` counts the share of outputs predicted as `Watermarked`.
- `Distinct-1` and `Distinct-2` provide a lightweight diversity estimate for generated text.
- `Avg Repetition Rate` is `1 - unique_words / total_words`; lower values generally indicate less repetition.
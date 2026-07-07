# Continuation Perplexity Summary

| Method | Samples | Scored Tokens | Mean Token NLL | Corpus PPL | Mean Row PPL | Median Row PPL | P90 Row PPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 500 | 48775 | 1.6778 | 5.3536 | 5.7837 | 5.7614 | 8.4058 |
| Fixed Delta 1.864 | 500 | 48451 | 1.9834 | 7.2675 | 7.8888 | 7.7888 | 11.6029 |
| Current Adaptive Delta | 500 | 48707 | 1.9501 | 7.0291 | 7.6320 | 7.4521 | 11.5404 |

## Notes

- PPL is computed only on the generated continuation, conditioned on the prompt.
- `Corpus PPL = exp(total continuation NLL / total scored continuation tokens)` is the most stable aggregate.
- The scoring model is the unwatermarked model specified by `--model_name_or_path`.
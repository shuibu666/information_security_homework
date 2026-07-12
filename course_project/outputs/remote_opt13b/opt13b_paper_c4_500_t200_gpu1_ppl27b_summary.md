# Continuation Perplexity Summary

| Method | Samples | Scored Tokens | Mean Token NLL | Corpus PPL | Mean Row PPL | Median Row PPL | P90 Row PPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 499 | 102184 | 1.4941 | 4.4554 | 4.8588 | 4.9489 | 6.8304 |
| Fixed Delta 2.0 | 499 | 102241 | 1.7548 | 5.7826 | 6.4205 | 6.4619 | 9.5436 |

## Notes

- PPL is computed only on the generated continuation, conditioned on the prompt.
- `Corpus PPL = exp(total continuation NLL / total scored continuation tokens)` is the most stable aggregate.
- The scoring model is the unwatermarked model specified by `--model_name_or_path`.
# Continuation Perplexity Summary

| Method | Samples | Scored Tokens | Mean Token NLL | Corpus PPL | Mean Row PPL | Median Row PPL | P90 Row PPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 500 | 48379 | 1.4169 | 4.1244 | 4.3719 | 4.3140 | 6.1133 |
| CA-KL | 500 | 48347 | 1.7607 | 5.8167 | 6.2729 | 6.1912 | 9.0014 |
| CA-KL + Weighted Detector | 500 | 48347 | 1.7607 | 5.8166 | 6.2728 | 6.1912 | 9.0014 |
| CA-KL + Candidate Greenlist | 500 | 47904 | 1.8011 | 6.0565 | 6.5694 | 6.4928 | 9.1913 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 500 | 48229 | 1.7241 | 5.6073 | 6.1721 | 6.1066 | 8.7802 |

## Notes

- PPL is computed only on the generated continuation, conditioned on the prompt.
- `Corpus PPL = exp(total continuation NLL / total scored continuation tokens)` is the most stable aggregate.
- The scoring model is the unwatermarked model specified by `--model_name_or_path`.
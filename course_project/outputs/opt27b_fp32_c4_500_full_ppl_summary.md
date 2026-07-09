# Continuation Perplexity Summary

| Method | Samples | Scored Tokens | Mean Token NLL | Corpus PPL | Mean Row PPL | Median Row PPL | P90 Row PPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 500 | 48379 | 1.4170 | 4.1245 | 4.3720 | 4.3140 | 6.1133 |
| Fixed Delta 0.5 | 500 | 48555 | 1.4377 | 4.2111 | 4.5072 | 4.4205 | 6.4031 |
| Fixed Delta 1.0 | 500 | 48196 | 1.5040 | 4.4996 | 4.7783 | 4.6095 | 6.7793 |
| Fixed Delta 1.864 | 500 | 48405 | 1.6497 | 5.2054 | 5.6269 | 5.5428 | 8.1376 |
| Fixed Delta 2.0 | 500 | 48631 | 1.6920 | 5.4301 | 5.8904 | 5.6195 | 8.6085 |
| Fixed Delta 3.0 | 500 | 48517 | 1.9487 | 7.0199 | 7.7218 | 7.4727 | 11.7209 |
| Current Adaptive Delta | 500 | 48352 | 1.6395 | 5.1528 | 5.5532 | 5.5497 | 8.0900 |
| CA-KL | 500 | 48347 | 1.7607 | 5.8167 | 6.2729 | 6.1912 | 9.0003 |
| CA-KL + Weighted Detector | 500 | 48347 | 1.7607 | 5.8168 | 6.2729 | 6.1912 | 9.0014 |
| CA-KL + Candidate Greenlist | 500 | 47904 | 1.8011 | 6.0565 | 6.5694 | 6.4928 | 9.1913 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 500 | 48838 | 1.4823 | 4.4030 | 4.7326 | 4.5957 | 6.8218 |

## Notes

- PPL is computed only on the generated continuation, conditioned on the prompt.
- `Corpus PPL = exp(total continuation NLL / total scored continuation tokens)` is the most stable aggregate.
- The scoring model is the unwatermarked model specified by `--model_name_or_path`.
# Continuation Perplexity Summary

| Method | Samples | Scored Tokens | Mean Token NLL | Corpus PPL | Mean Row PPL | Median Row PPL | P90 Row PPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 499 | 102167 | 1.5087 | 4.5207 | 4.9782 | 5.0568 | 7.1795 |
| Fixed Delta 2.0 | 499 | 102199 | 1.7479 | 5.7426 | 6.3846 | 6.4693 | 9.5250 |
| Current Adaptive Delta | 499 | 102211 | 1.7227 | 5.5995 | 6.1490 | 6.2142 | 8.9391 |
| CA-KL | 499 | 102196 | 1.8422 | 6.3104 | 7.0085 | 7.1410 | 10.4091 |
| CA-KL + Weighted Detector | 499 | 102196 | 1.8422 | 6.3104 | 7.0085 | 7.1410 | 10.4091 |
| CA-KL + Candidate Greenlist | 499 | 102204 | 1.8983 | 6.6746 | 7.2726 | 7.4071 | 10.4015 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 499 | 102225 | 1.8243 | 6.1984 | 6.8984 | 6.9945 | 10.6298 |

## Notes

- PPL is computed only on the generated continuation, conditioned on the prompt.
- `Corpus PPL = exp(total continuation NLL / total scored continuation tokens)` is the most stable aggregate.
- The scoring model is the unwatermarked model specified by `--model_name_or_path`.
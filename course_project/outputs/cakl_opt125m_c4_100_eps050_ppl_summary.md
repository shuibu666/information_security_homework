# Continuation Perplexity Summary

| Method | Samples | Scored Tokens | Mean Token NLL | Corpus PPL | Mean Row PPL | Median Row PPL | P90 Row PPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 100 | 9756 | 1.6881 | 5.4091 | 5.7190 | 5.6748 | 8.1148 |
| Fixed Delta 1.0 | 100 | 9816 | 1.7896 | 5.9872 | 6.3888 | 6.1652 | 9.2530 |
| Fixed Delta 2.0 | 100 | 9785 | 2.0395 | 7.6871 | 8.1489 | 7.8831 | 11.0386 |
| Fixed Delta 3.0 | 100 | 9664 | 2.2877 | 9.8520 | 10.8298 | 10.4111 | 17.0220 |
| Current Adaptive Delta | 100 | 9638 | 1.9542 | 7.0584 | 7.5188 | 7.1512 | 10.9567 |
| CA-KL | 100 | 9787 | 2.0549 | 7.8061 | 8.3825 | 7.8776 | 13.1434 |
| CA-KL + Weighted Detector | 100 | 9787 | 2.0549 | 7.8061 | 8.3825 | 7.8776 | 13.1434 |
| CA-KL + Candidate Greenlist | 100 | 9651 | 2.1417 | 8.5141 | 9.3068 | 8.7455 | 13.7946 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 100 | 9561 | 2.0821 | 8.0211 | 8.9429 | 8.5515 | 13.2844 |

## Notes

- PPL is computed only on the generated continuation, conditioned on the prompt.
- `Corpus PPL = exp(total continuation NLL / total scored continuation tokens)` is the most stable aggregate.
- The scoring model is the unwatermarked model specified by `--model_name_or_path`.
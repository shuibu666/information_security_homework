# Continuation Perplexity Summary

| Method | Samples | Scored Tokens | Mean Token NLL | Corpus PPL | Mean Row PPL | Median Row PPL | P90 Row PPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Watermark | 500 | 48775 | 1.6778 | 5.3536 | 5.7837 | 5.7614 | 8.4058 |
| Fixed Delta 1.0 | 500 | 48716 | 1.7583 | 5.8027 | 6.3572 | 6.1506 | 9.5722 |
| Fixed Delta 2.0 | 500 | 48693 | 2.0038 | 7.4175 | 8.0951 | 7.7725 | 12.2415 |
| Fixed Delta 3.0 | 500 | 48535 | 2.2714 | 9.6925 | 11.0199 | 10.9666 | 16.8275 |
| Current Adaptive Delta | 500 | 48707 | 1.9501 | 7.0291 | 7.6320 | 7.4521 | 11.5404 |
| CA-KL | 500 | 48716 | 2.0557 | 7.8122 | 8.5805 | 8.4373 | 12.5955 |
| CA-KL + Weighted Detector | 500 | 48716 | 2.0557 | 7.8122 | 8.5805 | 8.4373 | 12.5955 |
| CA-KL + Candidate Greenlist | 500 | 48505 | 2.0767 | 7.9782 | 8.8218 | 8.7575 | 13.2304 |
| CA-KL + Candidate Greenlist + Weighted/WinMax | 500 | 48262 | 2.0432 | 7.7155 | 8.5510 | 8.4795 | 12.3579 |

## Notes

- PPL is computed only on the generated continuation, conditioned on the prompt.
- `Corpus PPL = exp(total continuation NLL / total scored continuation tokens)` is the most stable aggregate.
- The scoring model is the unwatermarked model specified by `--model_name_or_path`.
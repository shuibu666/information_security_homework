# Vocabulary-size incident — invalidates final_v1 generation results

Date: 2026-07-12

## Finding

`len(OPTTokenizer)` is smaller than the OPT causal-LM output embedding / logits
dimension. The final-v1 runner constructed greenlists with `range(len(tokenizer))`
rather than the exact score-vector vocabulary size. This leaves the additional
output IDs outside every greenlist and violates the full-vocabulary KGW/CA-KL
definition.

The issue became visible when the candidate-aware processor attempted to index a
candidate ID from model logits into a tokenizer-sized mask and CUDA raised an
out-of-bounds assertion.

## Consequence

All `final_v1` watermarked raw generations and their derived detector, PPL,
SimCSE, calibration, and selection artifacts are **invalid for final analysis**.
They are retained unchanged as an audit trail and must not be deleted or used in
tables, figures, parameter selection, or test configuration.

## Remediation

- Generation and both detector paths now derive their vocabulary size from the
  model output embedding / `AutoConfig.vocab_size`, not `len(tokenizer)`.
- Candidate/gate generation has a regression test for the fixed-delta control.
- The strict protocol restarts in a new versioned result directory; validation
  and test outputs will not reuse final_v1 results.

# Final-v3 restart audit

Date: 2026-07-12

## Starting checkout

- Repository commit before staging current fixes: `e3d070bd6081c92e8efeadea5d193fad4e983dd1`.
- Existing result directories: `final_v1`, `final_v2`; neither will be overwritten or used as final-v3 input.
- Current code contains the uncommitted full-vocabulary repair described below.

## Mandatory repair retained for the restart

OPT-1.3B has tokenizer length 50,265 but output vocabulary size 50,272. A
greenlist built from tokenizer length is not a full-vocabulary KGW greenlist and
causes candidate-aware CUDA indexing failure. The final-v3 runner and detector
must derive their vocabulary size from model output embeddings / model config.

## Restart sequence

1. Stage the vocabulary repair and merge-preflight code, then run the project tests.
2. Freeze a copy of the already validated C4 manifest in `final_v3/data/` and verify its SHA-256.
3. Run the documented 10-prompt Phase 1 smoke and all required invariants.
4. Only after Gate A passes, run the 50-prompt pilot and continue through the remaining document phases.

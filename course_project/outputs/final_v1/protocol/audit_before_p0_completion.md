# P0 completion audit

Date: 2026-07-12

## Checkout examined

- Repository: `shuibu666/information_security_homework`
- Commit: `a2e6270558069a019259f7835e57d39165a824d0`
- Checkout: `/home/zyb/inform`
- Historical files under `course_project/outputs/`: retained and treated as legacy only.

## Completed earlier foundations

The checkout already contains an initial final-v1 JSONL schema, a structured C4 manifest builder, an isolated raw-generation runner, exact-token standard KGW scoring, exact-token PPL boundary checks, human-only calibration primitives, SimCSE pairing code, and basic CPU/GPU tests.

## Remaining required P0 work, in document order

1. **P0-01** — Run and verify the final 500/500 manifest builder; record frozen dataset revision and audit. The legacy TXT route remains available only for legacy runs.
2. **P0-02/P0-03** — Exercise the new runner against its 200-token/context/CLI invariants and add reproducibility tests; do not use the legacy runner for final-v1.
3. **P0-04** — Add direct actual-sampling KL checks over generated steps, not only a processor unit check.
4. **P0-05** — Verify generation and detection tables are independent and add unique-generation tests.
5. **P0-06** — Implement and test the missing matched candidate/no-gate and full-vocabulary/gate detector configurations. The standard detector path is available, but the complete detector matrix is not.
6. **P0-07** — Test continuation-only PPL with final-v1 records and tokenizer identity handling.
7. **P0-08** — Test SimCSE identity, pairing, and model execution.
8. **P0-09** — Add tie/duplicate/common-ID calibration preflight tests and wire the final score schema through it.
9. **P0-10** — Replace the legacy Pareto source manifest and aggregation path with final-v1 held-out inputs; no in-sample metric may occupy a reportable field.
10. **P0-11** — Add full config-fingerprint/resume tests and freeze the actual environment/model/dataset metadata after manifest construction.

## Pre-Gate decision

This is not a Gate A result. P0 is incomplete, so the only permitted next work is P0 implementation and testing. No smoke, validation, or test generation has been started.

# Final v1 pre-change audit

Date: 2026-07-12

## Audited checkout

- Repository: `shuibu666/information_security_homework`
- Commit: `37b9cc81d01024b9b95cd5a64c32a65a0c682ba7`
- Branch: `main`
- Server checkout: `/home/zyb/inform`
- Working tree at audit time: clean

## Findings against the execution plan

| Plan item | Status | Evidence / required correction |
| --- | --- | --- |
| P0-01 structured manifest | fails | `prompt_utils.py` returns only strings and `prepare_c4_prompts.py` writes TXT. Human completions, source IDs, token IDs, dataset revision, and validation/test split are absent. |
| P0-02 exact generation IDs | fails | `run_experiments.py` reloads prompt text rather than consuming a structured token-ID manifest. |
| P0-03 sampling CLI | fails | Generation configuration has legacy defaults and lacks the complete final-protocol parameter record. |
| P0-04 CA-KL KL verification | partial | The closed form `delta*q_G-log(Z)` is present, but actual post-processor sampling KL is not independently recorded or tested. |
| P0-05 generator/detector separation | fails | The runner writes generator variants and weighted-detector fields into the same result records. |
| P0-06 matched detector negatives | fails | Calibration keys solely on score fields and uses `No Watermark`; detector configuration identity is not carried through scores. |
| P0-07 continuation-only PPL | fails | `evaluate_ppl.py` re-tokenizes text and cannot prove its scored token boundary matches generation IDs. |
| P0-08 SimCSE | fails | No continuation-to-continuation SimCSE script exists. |
| P0-09 held-out human calibration | fails | `calibrate_detection.py` derives folds from No-Watermark rows rather than validation human completions. |
| P0-10 Pareto provenance | fails | `pareto_sources.json` reuses `source_id: full` across distinct fixed deltas; `build_pareto.py` writes in-sample weighted TPR into the reportable field. |
| P0-11 reproducibility | fails | Requirements are unpinned; manifests use legacy path conventions; no shard completion/fingerprint scheme exists. |
| Test gate | fails | There is no project-level `tests/` directory. |

## Historical-result boundary

Existing files in `course_project/outputs/` are retained unchanged and are legacy exploratory artifacts. They must not enter `final_v1` main tables or figures.

## Gate A decision

**No-Go.** No validation or test generation may start until P0 code changes and CPU/GPU tests pass. Work proceeds with the data manifest, deterministic generation schema, matched calibration, PPL, SimCSE, and test scaffolding.

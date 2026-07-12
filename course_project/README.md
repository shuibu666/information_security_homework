# Course Project

This folder contains all coursework-specific content for the improved watermarking project.

- `processors/`: adaptive watermark implementation
- `scripts/`: experiment runner, C4 prompt preparation, and result analysis scripts
- `data/`: reproducible prompt files, including the exported `C4/realnewslike` 500-sample subset
- `outputs/`: generated CSV files, summaries, charts, and `additional/` evidence-chain experiments
- `docs/`: submission-oriented documentation

Start with:

```bash
python course_project/scripts/run_experiments.py --help
```

Recommended workflow for paper-style evaluation:

```bash
python course_project/scripts/prepare_c4_prompts.py \
  --model_name_or_path ./models/opt-125m \
  --prompt_source hf_dataset \
  --save_loaded_prompts_path course_project/data/prompts_c4_realnewslike_500.txt \
  --limit_prompts 500
```

Then run experiments on the saved prompt file for fully reproducible results:

```bash
python course_project/scripts/run_experiments.py \
  --model_name_or_path ./models/opt-125m \
  --prompt_source prompts_file \
  --prompts_path course_project/data/prompts_c4_realnewslike_500.txt \
  --limit_prompts 500
```

## Interactive CA-KL-CG demo

The Gradio demo compares No Watermark, Fixed Delta, CA-KL, and CA-KL-CG for a
single prompt. It displays standard KGW, weighted, and WinMax detector scores,
along with the average KL, adaptive delta, and confidence-gate pass rate.

```bash
CUDA_VISIBLE_DEVICES=4 python -m course_project.scripts.demo_cakl \
  --model_name_or_path /home/zyb/inform/models/opt-1.3b \
  --server_name 127.0.0.1 \
  --server_port 7861
```

The model path and GPU index should be adjusted for the target machine. The
default front-end settings match the controlled CA-KL-CG presentation setup:
`gamma=0.25`, `epsilon=0.50`, candidate top-p `0.95`, and gate `(0.10, 0.95)`.

## Additional evidence-chain experiments

Use the same Python environment that contains PyTorch, Transformers, and the
local `models/opt-2.7b` checkpoint. All commands below are run from the
repository root and create new files without overwriting the prior formal runs.

### KL-budget Pareto scan

```bash
bash course_project/scripts/run_opt27b_pareto_500_fp32.sh
```

The script evaluates `epsilon = 0.20, 0.30, 0.35, 0.40, 0.50` on the fixed
500 C4 prompts, computes continuation PPL, performs five-fold calibration, and
writes `course_project/outputs/additional/pareto/`. The four report figures are
`ppl_vs_kgw_detection.pdf`, `ppl_vs_tpr1.pdf`, `epsilon_vs_avg_delta.pdf`, and
`epsilon_vs_avg_kl.pdf`. The `pareto_summary.csv` file is the source for the
Pareto table; its `*_in_sample` columns are diagnostics only, while the
`calibration_eps*/calibrated_metrics.csv` files contain reportable held-out-fold
TPR/FPR values and bootstrap confidence intervals.

### Copy-paste local-mixing attack

```bash
bash course_project/scripts/run_copy_paste_attack.sh
```

This reuses the stored `No Watermark`, `Fixed Delta 2.0`, `CA-KL`, and
`CA-KL-CG` generations. It tokenizes legacy CSVs before concatenating them,
constructs 200-token attacks at 0/25/50/75/100% watermark ratios and three
positions, and scores standard KGW, global weighted, and WinMax detectors.
Outputs are stored in `course_project/outputs/additional/copy_paste/`.

### Independent calibration and confidence intervals

```bash
python course_project/scripts/calibrate_detection.py \
  --input_csv course_project/outputs/opt27b_fp32_c4_500_gate010_095_results.csv \
  --ppl_csv course_project/outputs/opt27b_fp32_c4_500_gate010_095_ppl_results.csv \
  --output_dir course_project/outputs/additional/calibration \
  --folds 5 --seed 1234 --bootstrap_repetitions 2000 --confidence_level 0.95
```

This command fixes the previous in-sample threshold issue: each fold uses 400
No Watermark samples for threshold calibration and reports actual FPR/TPR on
the remaining 100 samples. Outputs include `fold_assignments.csv`,
`fold_thresholds.csv`, `out_of_fold_scores.csv`, `calibrated_metrics.csv`,
`bootstrap_metrics.csv`, and `experiment_config.json`.

## Runtime prerequisites

The local machine used for this audit has CPU-only PyTorch environments and no
OPT-2.7B checkpoint, so it cannot run the FP32 formal scripts. The required
remote CUDA environment, model layout, and override variables are documented
in [`docs/RUN_ENVIRONMENT.md`](docs/RUN_ENVIRONMENT.md).

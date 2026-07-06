# Reproduction and Submission Guide

This repository is organized for coursework submission:

- Root directory: official `lm-watermarking` code and your baseline reproduction environment.
- `course_project/`: all coursework-specific improvement code, prompts, outputs, and documentation.

## 1. Recommended directory view

```text
lm-watermarking-adaptive/
├── demo_watermark.py
├── watermark_processor.py
├── models/
├── course_project/
│   ├── data/
│   │   ├── prompts.txt
│   │   └── prompts_c4_realnewslike_500.txt
│   ├── docs/
│   │   └── README_SUBMISSION.md
│   ├── outputs/
│   ├── processors/
│   │   └── adaptive_watermark_processor.py
│   └── scripts/
│       ├── prepare_c4_prompts.py
│       ├── run_experiments.py
│       └── analyze_results.py
└── requirements.txt
```

## 2. Create and activate the environment

```bash
conda create -n inform python=3.10 -y
conda activate inform
```

## 3. Install dependencies

Run this from the repository root:

```bash
pip install -r requirements.txt
```

## 4. Prepare local models

Store local models in:

```text
models/
```

Recommended:

- `./models/tiny-gpt2` for smoke tests
- `./models/opt-125m` for course experiments
- `course_project/data/prompts.txt` remains the small manual demo prompt set
- `course_project/data/prompts_c4_realnewslike_500.txt` is the exported 500-sample subset used for formal experiments

## 5. Run the official baseline demo

```bash
python demo_watermark.py \
  --model_name_or_path ./models/tiny-gpt2 \
  --use_gpu True \
  --run_gradio True \
  --max_new_tokens 100
```

## 6. Prepare the C4/realnewslike 500-sample subset

This follows the paper-style data source more closely than the old 12 manual prompts:

```bash
python course_project/scripts/prepare_c4_prompts.py \
  --model_name_or_path ./models/opt-125m \
  --prompt_source hf_dataset \
  --dataset_name c4 \
  --dataset_config_name realnewslike \
  --limit_prompts 500 \
  --min_prompt_tokens 50 \
  --max_new_tokens 100 \
  --save_loaded_prompts_path course_project/data/prompts_c4_realnewslike_500.txt
```

## 7. Run the improved batch experiments

```bash
python course_project/scripts/run_experiments.py \
  --model_name_or_path ./models/opt-125m \
  --prompt_source prompts_file \
  --prompts_path course_project/data/prompts_c4_realnewslike_500.txt \
  --limit_prompts 500 \
  --output_csv course_project/outputs/results.csv \
  --max_new_tokens 100 \
  --use_gpu True \
  --fixed_deltas 0.5 1.0 2.0 3.0 \
  --adaptive_delta_min 0.5 \
  --adaptive_delta_max 3.0
```

Quick smoke test:

```bash
python course_project/scripts/run_experiments.py \
  --model_name_or_path ./models/tiny-gpt2 \
  --prompt_source prompts_file \
  --prompts_path course_project/data/prompts.txt \
  --output_csv course_project/outputs/smoke_results.csv \
  --max_new_tokens 20 \
  --use_gpu False \
  --fixed_deltas 0.5 2.0 \
  --adaptive_delta_min 0.5 \
  --adaptive_delta_max 3.0 \
  --limit_prompts 2
```

## 8. Run the presentation frontend

```bash
python course_project/scripts/demo_adaptive.py \
  --model_name_or_path ./models/opt-125m \
  --use_gpu True \
  --max_new_tokens 120 \
  --delta 2.0 \
  --adaptive_delta_min 0.5 \
  --adaptive_delta_max 3.0
```

This page keeps the official demo's overall interaction pattern, but adds:

- side-by-side comparison of `No Watermark`, `Fixed Delta`, and `Adaptive Delta`
- a summary table for video presentation
- adaptive delta statistics for the generated sample
- prompt presets loaded from `course_project/data/prompts.txt` for easier demo narration

## 9. Analyze experiment results

```bash
python course_project/scripts/analyze_results.py \
  --input_csv course_project/outputs/results.csv \
  --summary_md course_project/outputs/summary.md \
  --zscore_chart course_project/outputs/zscore_comparison.png \
  --green_chart course_project/outputs/green_fraction_comparison.png
```

## 9. Coursework improvement summary

The main improvement is an entropy-based adaptive watermark bias:

```text
delta_t = delta_min + (delta_max - delta_min) * H_norm
```

Where:

- `H_norm` is the normalized entropy of the next-token probability distribution.
- Higher entropy means the model is more uncertain, so the watermark bias becomes stronger.
- Lower entropy means the model is more certain, so the watermark bias becomes weaker.

This keeps the official greenlist sampling logic unchanged and only replaces the fixed bias with a dynamic bias.

## 10. CA-KL-CG final experiment workflow

The final course-project method is `CA-KL-CG Watermark`, an inference-time watermark that combines:

- KL-constrained adaptive `delta`
- confidence gating with normalized entropy and top-1 probability
- candidate-aware greenlists over top-p tokens
- model-assisted weighted and windowed detection

Recommended formal run:

```bash
python course_project/scripts/run_experiments.py \
  --model_name_or_path ./models/opt-2.7b \
  --prompt_source prompts_file \
  --prompts_path course_project/data/prompts_c4_realnewslike_500.txt \
  --limit_prompts 500 \
  --output_csv course_project/outputs/cakl_cg_opt27b_c4_500_results.csv \
  --max_new_tokens 100 \
  --use_gpu True \
  --fixed_deltas 1.0 2.0 3.0 \
  --run_cakl True \
  --kl_epsilon 0.50 \
  --cakl_delta_max 3.0 \
  --confidence_entropy_threshold 0.35 \
  --confidence_top1_threshold 0.85 \
  --candidate_top_p 0.95 \
  --window_sizes 20,40,80,max \
  --use_model_assisted_detector True
```

Summarize the final results:

```bash
python course_project/scripts/analyze_results.py \
  --input_csv course_project/outputs/cakl_cg_opt27b_c4_500_results.csv \
  --summary_md course_project/outputs/cakl_cg_opt27b_c4_500_summary.md \
  --zscore_chart course_project/outputs/cakl_cg_zscore.png \
  --green_chart course_project/outputs/cakl_cg_green.png \
  --tradeoff_chart course_project/outputs/cakl_cg_tradeoff.png \
  --weighted_auc_chart course_project/outputs/cakl_cg_weighted_auc.png \
  --kl_delta_chart course_project/outputs/cakl_cg_kl_delta.png
```

Quick smoke test:

```bash
python course_project/scripts/run_experiments.py \
  --model_name_or_path ./models/tiny-gpt2 \
  --prompt_source prompts_file \
  --prompts_path course_project/data/prompts.txt \
  --output_csv course_project/outputs/cakl_smoke_results.csv \
  --max_new_tokens 20 \
  --use_gpu False \
  --fixed_deltas 1.0 2.0 \
  --limit_prompts 2 \
  --run_cakl True \
  --use_model_assisted_detector True
```

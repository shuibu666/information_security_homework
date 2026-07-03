# Course Project

This folder contains all coursework-specific content for the improved watermarking project.

- `processors/`: adaptive watermark implementation
- `scripts/`: experiment runner, C4 prompt preparation, and result analysis scripts
- `data/`: reproducible prompt files, including the exported `C4/realnewslike` 500-sample subset
- `outputs/`: generated CSV files, summaries, and charts
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

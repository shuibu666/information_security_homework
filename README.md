# information_security_homework

本仓库用于复现《信息安全技术》课程项目，主题为：

- 论文：`A Watermark for Large Language Models`
- 官方仓库：`jwkirchenbauer/lm-watermarking`

仓库内容包括：

- 原始论文复现所需的基础代码
- 我们实现的自适应 `delta` 改进方案
- 基于 `C4/realnewslike` 500 条样本的正式实验数据
- 批量实验脚本
- 结果分析脚本
- 用于演示的前端脚本

本仓库不包含：

- 本地下载的大模型文件
- 本地缓存文件
- 实验报告 `.md` 和 `.pdf`
- 课程要求 PDF

## 1. 环境准备

建议使用 `conda` 创建独立环境：

```bash
conda create -n inform python=3.10 -y
conda activate inform
pip install -r requirements.txt
```

## 2. 模型准备

请将本地模型放到：

```text
models/
```

推荐使用：

- `./models/tiny-gpt2`
- `./models/opt-125m`
- `./models/opt-2.7b`

## 3. 运行官方复现 Demo

```bash
python demo_watermark.py \
  --model_name_or_path ./models/tiny-gpt2 \
  --use_gpu True \
  --run_gradio True \
  --max_new_tokens 100
```

## 4. 运行改进版前端

```bash
python course_project/scripts/demo_adaptive.py \
  --model_name_or_path ./models/opt-125m \
  --use_gpu True \
  --max_new_tokens 120 \
  --delta 2.0 \
  --adaptive_delta_min 0.5 \
  --adaptive_delta_max 3.0
```

该前端支持同时对比：

- `No Watermark`
- `Fixed Delta`
- `Adaptive Delta`

## 5. 运行批量实验

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

然后运行正式实验：

```bash
python course_project/scripts/run_experiments.py \
  --model_name_or_path ./models/opt-125m \
  --prompt_source prompts_file \
  --prompts_path course_project/data/prompts_c4_realnewslike_500.txt \
  --output_csv course_project/outputs/results.csv \
  --max_new_tokens 100 \
  --use_gpu True \
  --fixed_deltas 0.5 1.0 2.0 3.0 \
  --adaptive_delta_min 0.5 \
  --adaptive_delta_max 3.0
```

## 6. 分析实验结果

```bash
python course_project/scripts/analyze_results.py \
  --input_csv course_project/outputs/results.csv \
  --summary_md course_project/outputs/summary.md \
  --zscore_chart course_project/outputs/zscore_comparison.png \
  --green_chart course_project/outputs/green_fraction_comparison.png
```

## 7. 目录说明

```text
information_security_homework/
├── demo_watermark.py
├── watermark_processor.py
├── requirements.txt
├── models/
└── course_project/
    ├── data/
    │   ├── prompts.txt
    │   └── prompts_c4_realnewslike_500.txt
    ├── docs/
    │   └── README_SUBMISSION.md
    ├── outputs/
    ├── processors/
    │   └── adaptive_watermark_processor.py
    ├── prompt_utils.py
    └── scripts/
        ├── prepare_c4_prompts.py
        ├── run_experiments.py
        ├── analyze_results.py
        ├── demo_adaptive.py
        └── run_opt27b_remote.sh
```

## 8. 说明

更详细的运行说明请参考：

```text
course_project/docs/README_SUBMISSION.md
```

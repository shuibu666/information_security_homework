# coding=utf-8
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from course_project.prompt_utils import load_prompts, str2bool


def parse_args():
    parser = argparse.ArgumentParser(description="Export reproducible prompts from C4 realnewslike.")
    parser.add_argument("--model_name_or_path", type=str, default="./models/opt-125m")
    parser.add_argument("--prompt_source", type=str, choices=["prompts_file", "hf_dataset"], default="hf_dataset")
    parser.add_argument("--prompts_path", type=str, default="course_project/data/prompts.txt")
    parser.add_argument("--save_loaded_prompts_path", type=str, default="course_project/data/prompts_c4_realnewslike_500.txt")
    parser.add_argument("--dataset_name", type=str, default="c4")
    parser.add_argument("--dataset_config_name", type=str, default="realnewslike")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_text_field", type=str, default="text")
    parser.add_argument("--dataset_streaming", type=str2bool, default=True)
    parser.add_argument("--trust_remote_code", type=str2bool, default=True)
    parser.add_argument("--shuffle_dataset", type=str2bool, default=False)
    parser.add_argument("--dataset_seed", type=int, default=1234)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10_000)
    parser.add_argument("--dataset_skip_examples", type=int, default=0)
    parser.add_argument("--min_prompt_tokens", type=int, default=50)
    parser.add_argument("--min_source_tokens", type=int, default=150)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--limit_prompts", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    prompts = load_prompts(args, tokenizer)
    output_path = args.save_loaded_prompts_path if args.prompt_source == "hf_dataset" else args.prompts_path
    print(f"Saved {len(prompts)} prompts to {output_path}")


if __name__ == "__main__":
    main()

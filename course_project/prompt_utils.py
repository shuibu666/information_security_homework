# coding=utf-8
from __future__ import annotations

import argparse
import re
from pathlib import Path


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"yes", "true", "t", "y", "1"}:
        return True
    if lowered in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def read_prompts(prompts_path: str, limit_prompts: int | None) -> list[str]:
    prompts = []
    for raw_line in Path(prompts_path).read_text(encoding="utf-8").splitlines():
        prompt = raw_line.strip()
        if prompt and not prompt.startswith("#"):
            prompts.append(prompt)
    if limit_prompts is not None:
        prompts = prompts[:limit_prompts]
    if not prompts:
        raise ValueError("No prompts were loaded from the prompts file.")
    return prompts


def save_prompts(prompts: list[str], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(prompts) + "\n", encoding="utf-8")


def resolve_completion_trim_tokens(args) -> int:
    if getattr(args, "paper_completion_tokens", None) is not None:
        return max(1, int(args.paper_completion_tokens))
    if getattr(args, "target_new_tokens", None) is not None:
        return max(1, int(args.target_new_tokens))
    return max(1, int(args.max_new_tokens))


def build_prompt_from_dataset_text(
    raw_text: str,
    tokenizer,
    min_prompt_tokens: int,
    min_source_tokens: int,
    paper_style_prompt: bool,
    completion_trim_tokens: int,
) -> str | None:
    token_ids = tokenizer(raw_text, add_special_tokens=False)["input_ids"]
    if len(token_ids) < min_source_tokens:
        return None

    if paper_style_prompt:
        if len(token_ids) <= completion_trim_tokens:
            return None
        prompt_ids = token_ids[:-completion_trim_tokens]
        if len(prompt_ids) < min_prompt_tokens:
            return None
    else:
        prompt_ids = token_ids[:min_prompt_tokens]

    prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_ids, skip_special_tokens=True)
    prompt = tokenizer.convert_tokens_to_string(prompt_tokens).strip()
    prompt = re.sub(r"\s+", " ", prompt).strip()
    return prompt or None


def load_hf_dataset_prompts(args, tokenizer) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("The `datasets` package is required to load prompts from Hugging Face datasets.") from exc

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        split=args.dataset_split,
        streaming=args.dataset_streaming,
        trust_remote_code=getattr(args, "trust_remote_code", True),
    )
    if args.shuffle_dataset:
        dataset = dataset.shuffle(seed=args.dataset_seed, buffer_size=args.shuffle_buffer_size)

    completion_trim_tokens = resolve_completion_trim_tokens(args)
    min_source_tokens = args.min_source_tokens or (args.min_prompt_tokens + completion_trim_tokens)
    prompts: list[str] = []
    skipped = 0

    for example in dataset:
        raw_text = example.get(args.dataset_text_field)
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue
        if skipped < args.dataset_skip_examples:
            skipped += 1
            continue

        prompt = build_prompt_from_dataset_text(
            raw_text=raw_text,
            tokenizer=tokenizer,
            min_prompt_tokens=args.min_prompt_tokens,
            min_source_tokens=min_source_tokens,
            paper_style_prompt=getattr(args, "paper_style_prompt", False),
            completion_trim_tokens=completion_trim_tokens,
        )
        if prompt is None:
            continue

        prompts.append(prompt)
        if args.limit_prompts is not None and len(prompts) >= args.limit_prompts:
            break

    if not prompts:
        raise ValueError("No valid prompts were extracted from the Hugging Face dataset.")

    if args.save_loaded_prompts_path:
        save_prompts(prompts, args.save_loaded_prompts_path)

    return prompts


def load_prompts(args, tokenizer) -> list[str]:
    if args.prompt_source == "prompts_file":
        return read_prompts(args.prompts_path, args.limit_prompts)
    if args.prompt_source == "hf_dataset":
        return load_hf_dataset_prompts(args, tokenizer)
    raise ValueError(f"Unsupported prompt source: {args.prompt_source}")

# coding=utf-8
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from course_project.prompt_utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate continuation perplexity for generated rows.")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--summary_md", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-125m")
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--load_fp16", type=str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=None)
    return parser.parse_args()


def read_rows(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def load_model_and_tokenizer(args):
    model_kwargs = {}
    if args.load_fp16 and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, tokenizer, torch.device(device)


def build_scoring_example(row, tokenizer, max_length: int | None):
    prompt = row["prompt"]
    generated_text = row["generated_text"]

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt + generated_text, add_special_tokens=True)["input_ids"]
    prompt_len = len(prompt_ids)

    if max_length is not None and len(full_ids) > max_length:
        overflow = len(full_ids) - max_length
        full_ids = full_ids[overflow:]
        prompt_len = max(0, prompt_len - overflow)

    scored_tokens = max(0, len(full_ids) - prompt_len)
    labels = list(full_ids)
    labels[:prompt_len] = [-100] * prompt_len

    if scored_tokens == 0:
        labels = [-100] * len(full_ids)

    return {
        "input_ids": full_ids,
        "labels": labels,
        "scored_tokens": scored_tokens,
    }


def pad_batch(examples, tokenizer, device):
    max_len = max(len(example["input_ids"]) for example in examples)
    pad_id = tokenizer.pad_token_id

    input_ids = []
    attention_mask = []
    labels = []
    for example in examples:
        length = len(example["input_ids"])
        pad_len = max_len - length
        input_ids.append(example["input_ids"] + [pad_id] * pad_len)
        attention_mask.append([1] * length + [0] * pad_len)
        labels.append(example["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def score_batch(model, batch):
    with torch.no_grad():
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

    logits = outputs.logits[:, :-1, :].contiguous()
    labels = batch["labels"][:, 1:].contiguous()
    loss_flat = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(labels.shape)
    valid_mask = labels.ne(-100)
    nll_sums = (loss_flat * valid_mask).sum(dim=1)
    token_counts = valid_mask.sum(dim=1)
    return nll_sums.detach().cpu().tolist(), token_counts.detach().cpu().tolist()


def evaluate_rows(rows, args, model, tokenizer, device):
    examples = [build_scoring_example(row, tokenizer, args.max_length) for row in rows]
    evaluated_rows = []

    for start in range(0, len(rows), args.batch_size):
        end = min(start + args.batch_size, len(rows))
        batch_examples = examples[start:end]
        batch = pad_batch(batch_examples, tokenizer, device)
        nll_sums, token_counts = score_batch(model, batch)

        for row, example, nll_sum, token_count in zip(rows[start:end], batch_examples, nll_sums, token_counts):
            scored_tokens = int(token_count)
            avg_nll = (float(nll_sum) / scored_tokens) if scored_tokens else None
            ppl = math.exp(avg_nll) if avg_nll is not None and avg_nll < 100 else None
            evaluated = dict(row)
            evaluated["ppl_num_scored_tokens"] = scored_tokens
            evaluated["ppl_nll_sum"] = f"{float(nll_sum):.8f}" if scored_tokens else ""
            evaluated["ppl_avg_nll"] = f"{avg_nll:.8f}" if avg_nll is not None else ""
            evaluated["ppl"] = f"{ppl:.8f}" if ppl is not None else ""
            evaluated_rows.append(evaluated)

    return evaluated_rows


def write_rows(rows, output_csv: str):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return sorted_values[int(index)]
    weight = index - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    summary_rows = []
    for method, method_rows in grouped.items():
        token_total = sum(int(row["ppl_num_scored_tokens"]) for row in method_rows)
        nll_total = sum(float(row["ppl_nll_sum"]) for row in method_rows if row["ppl_nll_sum"])
        ppls = [float(row["ppl"]) for row in method_rows if row["ppl"]]
        avg_nll = nll_total / token_total if token_total else 0.0
        summary_rows.append(
            {
                "method": method,
                "samples": len(method_rows),
                "tokens": token_total,
                "mean_token_nll": avg_nll,
                "corpus_ppl": math.exp(avg_nll) if avg_nll < 100 else float("inf"),
                "mean_row_ppl": sum(ppls) / len(ppls) if ppls else 0.0,
                "median_row_ppl": percentile(ppls, 0.5),
                "p90_row_ppl": percentile(ppls, 0.9),
            }
        )
    return summary_rows


def sort_method_key(row):
    method = row["method"]
    if method == "No Watermark":
        return (0, 0.0)
    if method.startswith("Fixed Delta "):
        return (1, float(method.split()[-1]))
    if method in {"Adaptive Delta", "Current Adaptive Delta"}:
        return (2, 0.0)
    if method == "CA-KL":
        return (3, 0.0)
    if method == "CA-KL + Weighted Detector":
        return (4, 0.0)
    if method == "CA-KL + Candidate Greenlist":
        return (5, 0.0)
    if method == "CA-KL + Candidate Greenlist + Weighted/WinMax":
        return (6, 0.0)
    return (7, 0.0)


def write_summary(summary_rows, summary_md: str):
    output_path = Path(summary_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Continuation Perplexity Summary",
        "",
        "| Method | Samples | Scored Tokens | Mean Token NLL | Corpus PPL | Mean Row PPL | Median Row PPL | P90 Row PPL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary_rows, key=sort_method_key):
        lines.append(
            "| {method} | {samples} | {tokens} | {mean_token_nll:.4f} | {corpus_ppl:.4f} | {mean_row_ppl:.4f} | {median_row_ppl:.4f} | {p90_row_ppl:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- PPL is computed only on the generated continuation, conditioned on the prompt.",
            "- `Corpus PPL = exp(total continuation NLL / total scored continuation tokens)` is the most stable aggregate.",
            "- The scoring model is the unwatermarked model specified by `--model_name_or_path`.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    rows = read_rows(args.input_csv)
    model, tokenizer, device = load_model_and_tokenizer(args)
    evaluated_rows = evaluate_rows(rows, args, model, tokenizer, device)
    write_rows(evaluated_rows, args.output_csv)
    summary_rows = summarize(evaluated_rows)
    write_summary(summary_rows, args.summary_md)
    print(f"Saved PPL rows to {args.output_csv}")
    print(f"Saved PPL summary to {args.summary_md}")


if __name__ == "__main__":
    main()

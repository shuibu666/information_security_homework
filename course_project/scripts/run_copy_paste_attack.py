#!/usr/bin/env python
"""Construct token-level copy-paste attacks and score three watermark detectors.

Existing generated text is reused.  Legacy CSVs without ``token_ids`` are
re-tokenized once with the configured OPT tokenizer; newly generated CSVs keep
the exact ids emitted during generation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from course_project.processors.cakl_watermark_processor import CAKLModelAssistedDetector
from watermark_processor import WatermarkDetector


SCORE_FIELDS = ("z_score", "weighted_z_score", "winmax_weighted_z_score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a token-level copy-paste attack over stored generations.")
    parser.add_argument("--manifest", required=True, help="JSON file mapping watermark methods to result CSVs.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--total_tokens", type=int, default=200)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--positions", nargs="+", default=["beginning", "middle", "end"])
    parser.add_argument("--limit_samples", type=int, default=500)
    parser.add_argument("--window_sizes", default="20,40,80,max", help="Matches the current detector default; override only as an ablation.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--load_fp16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_csv(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> float | None:
    value = str(value or "").strip()
    return float(value) if value else None


def load_token_ids(row: dict[str, str], tokenizer) -> tuple[list[int], str]:
    stored = str(row.get("token_ids", "")).strip()
    if stored:
        try:
            token_ids = json.loads(stored)
            if isinstance(token_ids, list) and all(isinstance(value, int) for value in token_ids):
                return token_ids, "stored_generation_ids"
        except json.JSONDecodeError:
            pass
    return tokenizer(row["generated_text"], add_special_tokens=False)["input_ids"], "reencoded_generated_text"


def take_tokens(rows: list[dict[str, object]], start: int, count: int) -> tuple[list[int], list[str]]:
    """Take a deterministic token stream, crossing rows when a 100-token completion is too short."""
    tokens: list[int] = []
    source_ids: list[str] = []
    offset = start % len(rows)
    while len(tokens) < count:
        row = rows[offset]
        row_tokens = row["token_ids"]
        if not row_tokens:
            raise ValueError(f"Empty generated text for prompt_id={row['prompt_id']}")
        needed = count - len(tokens)
        tokens.extend(row_tokens[:needed])
        source_ids.append(str(row["prompt_id"]))
        offset = (offset + 1) % len(rows)
    return tokens, source_ids


def attack_layout(total_tokens: int, ratio: float, position: str) -> tuple[int, int, int]:
    watermarked = int(round(total_tokens * ratio))
    remaining = total_tokens - watermarked
    if position == "beginning":
        return 0, watermarked, remaining
    if position == "middle":
        return remaining // 2, watermarked, remaining - remaining // 2
    if position == "end":
        return remaining, watermarked, 0
    raise ValueError(f"Unsupported position: {position}")


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = 0.0
    for positive in pos:
        for negative in neg:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(pos) * len(neg))


def tpr_at_fpr(pos: list[float], neg: list[float], fpr: float) -> float | None:
    if not pos or not neg:
        return None
    ordered = sorted(neg)
    threshold = ordered[max(0, min(len(ordered) - 1, math.ceil((1.0 - fpr) * len(ordered)) - 1))]
    return sum(score >= threshold for score in pos) / len(pos)


def build_detectors(source: dict[str, object], model, tokenizer, device: torch.device, window_sizes: str):
    # The detector must use the model's full output vocabulary.  For OPT-1.3B
    # ``len(tokenizer)`` is smaller than the LM head vocabulary, so deriving
    # ids from the tokenizer would make greenlist construction inconsistent
    # with generation and can index outside the tokenizer-derived range.
    vocab = list(range(model.get_output_embeddings().weight.shape[0]))
    common = {
        "vocab": vocab,
        "gamma": float(source.get("gamma", 0.25)),
        "seeding_scheme": str(source.get("seeding_scheme", "simple_1")),
        "select_green_tokens": bool(source.get("select_green_tokens", True)),
    }
    standard = WatermarkDetector(tokenizer=tokenizer, device=device, z_threshold=4.0, normalizers=[], ignore_repeated_bigrams=False, **common)
    assisted = CAKLModelAssistedDetector(
        delta=float(source.get("delta_max", 3.0)),
        model=model,
        tokenizer=tokenizer,
        device=device,
        candidate_top_p=float(source.get("candidate_top_p", 0.95)),
        use_candidate_greenlist=bool(source.get("use_candidate_greenlist", False)),
        use_confidence_gate=bool(source.get("use_confidence_gate", False)),
        entropy_threshold=float(source.get("entropy_threshold", 0.10)),
        top1_threshold=float(source.get("top1_threshold", 0.95)),
        window_sizes=window_sizes,
        z_threshold=4.0,
        **common,
    )
    return standard, assisted


def load_source_rows(source: dict[str, object], tokenizer, limit: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_rows = read_csv(str(source["results_csv"]))
    watermarked = [dict(row) for row in all_rows if row.get("method") == source["method"]]
    no_watermark = [dict(row) for row in all_rows if row.get("method") == "No Watermark"]
    if len(watermarked) < limit or len(no_watermark) < limit:
        raise ValueError(f"{source['name']} requires {limit} watermarked and No Watermark rows in {source['results_csv']}.")
    for row in watermarked + no_watermark:
        row["token_ids"], row["tokenization_source"] = load_token_ids(row, tokenizer)
    return watermarked[:limit], no_watermark[:limit]


def draw_figures(summary_rows: list[dict[str, object]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    def ratio_lines(score_field: str, filename: str, ylabel: str) -> None:
        figure, axis = plt.subplots(figsize=(8, 5))
        for method in sorted({str(row["method"]) for row in summary_rows}):
            rows = [row for row in summary_rows if row["method"] == method and row["position"] == "pooled"]
            rows.sort(key=lambda row: float(row["watermarked_ratio"]))
            axis.plot([row["watermarked_ratio"] for row in rows], [row[score_field] for row in rows], marker="o", label=method)
        axis.set_xlabel("Watermarked-token ratio")
        axis.set_ylabel(ylabel)
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / filename)
        plt.close(figure)

    ratio_lines("winmax_tpr_at_1pct_fpr_in_sample", "ratio_vs_tpr1.pdf", "WinMax TPR at 1% FPR (in-sample)")
    ratio_lines("avg_weighted_z_score", "ratio_vs_weighted_score.pdf", "Average weighted z-score")
    ratio_lines("avg_winmax_weighted_z_score", "ratio_vs_winmax_score.pdf", "Average WinMax weighted z-score")

    figure, axis = plt.subplots(figsize=(9, 5))
    selected = [row for row in summary_rows if row["watermarked_ratio"] == 0.25 and row["position"] != "pooled"]
    labels = [f"{row['method']}\n{row['position']}" for row in selected]
    axis.bar(labels, [row["winmax_tpr_at_1pct_fpr_in_sample"] for row in selected], color="#2ca02c")
    axis.set_ylabel("WinMax TPR at 1% FPR (in-sample)")
    axis.tick_params(axis="x", rotation=35)
    axis.set_ylim(0, 1.05)
    figure.tight_layout()
    figure.savefig(output_dir / "position_comparison.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_kwargs = {"torch_dtype": torch.float16} if args.load_fp16 and torch.cuda.is_available() else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    score_path = output_dir / "per_sample_scores.csv"
    existing: dict[tuple[str, float, str, int], dict[str, str]] = {}
    if args.resume and score_path.exists():
        for row in read_csv(str(score_path)):
            existing[(row["method"], float(row["watermarked_ratio"]), row["position"], int(row["sample_id"]))] = row
    scores: list[dict[str, object]] = list(existing.values())
    attacked: list[dict[str, object]] = []

    for source in manifest["sources"]:
        standard, assisted = build_detectors(source, model, tokenizer, device, args.window_sizes)
        watermarked_rows, no_watermark_rows = load_source_rows(source, tokenizer, args.limit_samples)
        for ratio in args.ratios:
            for position in args.positions:
                prefix_count, watermarked_count, suffix_count = attack_layout(args.total_tokens, ratio, position)
                for sample_id in range(args.limit_samples):
                    key = (str(source["name"]), ratio, position, sample_id)
                    prompt_row = watermarked_rows[sample_id]
                    no_prefix, no_prefix_ids = take_tokens(no_watermark_rows, sample_id * 3, prefix_count)
                    watermark, watermark_ids = take_tokens(watermarked_rows, sample_id * 5 + 1, watermarked_count)
                    no_suffix, no_suffix_ids = take_tokens(no_watermark_rows, sample_id * 3 + 1, suffix_count)
                    attacked_ids = no_prefix + watermark + no_suffix
                    prompt_ids = tokenizer(str(prompt_row["prompt"]), add_special_tokens=True, return_tensors="pt")["input_ids"][0].to(device)
                    attacked.append(
                        {
                            "method": source["name"],
                            "watermarked_ratio": ratio,
                            "position": position,
                            "sample_id": sample_id,
                            "prompt_id": prompt_row["prompt_id"],
                            "total_tokens": len(attacked_ids),
                            "watermarked_tokens": watermarked_count,
                            "token_ids": attacked_ids,
                            "no_watermark_source_prompt_ids": no_prefix_ids + no_suffix_ids,
                            "watermark_source_prompt_ids": watermark_ids,
                            "tokenization_source": "stored_generation_ids" if all(row["tokenization_source"] == "stored_generation_ids" for row in watermarked_rows + no_watermark_rows) else "reencoded_generated_text",
                        }
                    )
                    if key in existing:
                        continue
                    generated = torch.tensor(attacked_ids, dtype=torch.long, device=device)
                    standard_result = standard.detect(tokenized_text=generated)
                    assisted_result = assisted.detect(prompt_token_ids=prompt_ids, generated_token_ids=generated)
                    scores.append(
                        {
                            "method": source["name"],
                            "watermarked_ratio": ratio,
                            "position": position,
                            "sample_id": sample_id,
                            "prompt_id": prompt_row["prompt_id"],
                            "z_score": standard_result["z_score"],
                            "weighted_z_score": assisted_result["weighted_z_score"],
                            "winmax_weighted_z_score": assisted_result["winmax_weighted_z_score"],
                            "standard_prediction_at_4": standard_result["z_score"] > 4.0,
                            "weighted_prediction_at_4": assisted_result["weighted_z_score"] > 4.0,
                            "winmax_prediction_at_4": assisted_result["winmax_weighted_z_score"] > 4.0,
                        }
                    )
                    if len(scores) % 25 == 0:
                        print(f"Scored {len(scores)} attack samples")

    fields = ["method", "watermarked_ratio", "position", "sample_id", "prompt_id", *SCORE_FIELDS, "standard_prediction_at_4", "weighted_prediction_at_4", "winmax_prediction_at_4"]
    scores.sort(key=lambda row: (str(row["method"]), float(row["watermarked_ratio"]), str(row["position"]), int(row["sample_id"])))
    write_csv(score_path, scores, fields)
    with (output_dir / "attacked_samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in attacked:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    score_groups: dict[tuple[str, float, str], list[dict[str, object]]] = defaultdict(list)
    for row in scores:
        score_groups[(str(row["method"]), float(row["watermarked_ratio"]), str(row["position"]))].append(row)
    summary: list[dict[str, object]] = []
    for method in sorted({str(row["method"]) for row in scores}):
        negatives = [row for row in scores if row["method"] == method and float(row["watermarked_ratio"]) == 0.0]
        for ratio in args.ratios:
            for position in [*args.positions, "pooled"]:
                group = [row for row in scores if row["method"] == method and float(row["watermarked_ratio"]) == ratio and (row["position"] == position if position != "pooled" else True)]
                negative_group = [row for row in negatives if row["position"] == position] if position != "pooled" else negatives
                if not group or not negative_group:
                    continue
                row: dict[str, object] = {"method": method, "watermarked_ratio": ratio, "position": position, "samples": len(group)}
                for score_field, short in (("z_score", "standard"), ("weighted_z_score", "weighted"), ("winmax_weighted_z_score", "winmax")):
                    pos_scores = [float(item[score_field]) for item in group]
                    neg_scores = [float(item[score_field]) for item in negative_group]
                    row[f"avg_{score_field}"] = mean(pos_scores)
                    row[f"{short}_auc_in_sample"] = auc(pos_scores, neg_scores)
                    row[f"{short}_tpr_at_1pct_fpr_in_sample"] = tpr_at_fpr(pos_scores, neg_scores, 0.01)
                    row[f"{short}_tpr_at_5pct_fpr_in_sample"] = tpr_at_fpr(pos_scores, neg_scores, 0.05)
                    row[f"{short}_fixed_threshold_detection_rate"] = sum(value > 4.0 for value in pos_scores) / len(pos_scores)
                    row[f"{short}_negative_rate_at_fixed_threshold"] = sum(value > 4.0 for value in neg_scores) / len(neg_scores)
                summary.append(row)
    summary_fields = list(dict.fromkeys(key for row in summary for key in row))
    write_csv(output_dir / "copy_paste_summary.csv", summary, summary_fields)
    (output_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "manifest": str(Path(args.manifest).resolve()),
                "model_name_or_path": args.model_name_or_path,
                "total_tokens": args.total_tokens,
                "ratios": args.ratios,
                "positions": args.positions,
                "limit_samples": args.limit_samples,
                "seed": args.seed,
                "window_sizes": args.window_sizes,
                "detector_note": "TPR@FPR columns in copy_paste_summary.csv are in-sample diagnostics. Run calibrate_detection.py on condition-specific score tables before reporting strict independent-calibration values.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    draw_figures(summary, output_dir)
    print(f"Saved copy-paste attack outputs to {output_dir}")


if __name__ == "__main__":
    main()

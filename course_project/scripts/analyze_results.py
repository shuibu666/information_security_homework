# coding=utf-8
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize watermark experiment results.")
    parser.add_argument("--input_csv", type=str, default="course_project/outputs/results.csv")
    parser.add_argument("--summary_md", type=str, default="course_project/outputs/summary.md")
    parser.add_argument("--zscore_chart", type=str, default="course_project/outputs/zscore_comparison.png")
    parser.add_argument("--green_chart", type=str, default="course_project/outputs/green_fraction_comparison.png")
    parser.add_argument("--tradeoff_chart", type=str, default="course_project/outputs/detection_quality_tradeoff.png")
    parser.add_argument("--weighted_auc_chart", type=str, default="course_project/outputs/weighted_auc_comparison.png")
    parser.add_argument("--kl_delta_chart", type=str, default="course_project/outputs/kl_delta_comparison.png")
    return parser.parse_args()


def read_rows(input_csv: str) -> list[dict[str, str]]:
    with Path(input_csv).open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def to_float(value: str):
    if value is None:
        return None
    stripped = str(value).strip()
    if stripped == "":
        return None
    return float(stripped)


def sort_method_key(method: str):
    if method == "No Watermark":
        return (0, 0.0)
    if method.startswith("Fixed Delta "):
        return (1, float(method.split()[-1]))
    if method in {"Adaptive Delta", "Current Adaptive Delta"}:
        return (2, 0.0)
    if method == "CA-KL":
        return (3, 0.0)
    if method == "CA-KL + Candidate Greenlist":
        return (4, 0.0)
    if method == "CA-KL + Weighted Detector":
        return (5, 0.0)
    if method == "CA-KL + Candidate Greenlist + Weighted/WinMax":
        return (6, 0.0)
    return (7, 0.0)


def summarize_rows(rows: list[dict[str, str]]):
    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[row["method"]].append(row)

    summaries = []
    for method in sorted(grouped_rows.keys(), key=sort_method_key):
        method_rows = grouped_rows[method]
        z_scores = [to_float(row["z_score"]) for row in method_rows if to_float(row["z_score"]) is not None]
        green_fractions = [to_float(row["green_fraction"]) for row in method_rows if to_float(row["green_fraction"]) is not None]
        tokens_counted = [to_float(row["tokens_counted"]) for row in method_rows if to_float(row["tokens_counted"]) is not None]
        word_counts = [to_float(row["word_count"]) for row in method_rows if to_float(row["word_count"]) is not None]
        distinct_1 = [to_float(row["distinct_1"]) for row in method_rows if to_float(row["distinct_1"]) is not None]
        distinct_2 = [to_float(row["distinct_2"]) for row in method_rows if to_float(row["distinct_2"]) is not None]
        repetition_rates = [to_float(row["repetition_rate"]) for row in method_rows if to_float(row["repetition_rate"]) is not None]
        weighted_z_scores = [to_float(row.get("weighted_z_score")) for row in method_rows if to_float(row.get("weighted_z_score")) is not None]
        winmax_scores = [to_float(row.get("winmax_weighted_z_score")) for row in method_rows if to_float(row.get("winmax_weighted_z_score")) is not None]
        avg_kls = [to_float(row.get("avg_kl")) for row in method_rows if to_float(row.get("avg_kl")) is not None]
        avg_deltas = [to_float(row.get("avg_delta")) for row in method_rows if to_float(row.get("avg_delta")) is not None]
        gate_pass_rates = [to_float(row.get("gate_pass_rate")) for row in method_rows if to_float(row.get("gate_pass_rate")) is not None]
        success_rate = sum(row["prediction"] == "Watermarked" for row in method_rows) / len(method_rows)

        summaries.append(
            {
                "method": method,
                "samples": len(method_rows),
                "avg_z_score": mean(z_scores) if z_scores else 0.0,
                "avg_green_fraction": mean(green_fractions) if green_fractions else 0.0,
                "watermark_success_rate": success_rate,
                "avg_tokens_counted": mean(tokens_counted) if tokens_counted else 0.0,
                "avg_word_count": mean(word_counts) if word_counts else 0.0,
                "avg_distinct_1": mean(distinct_1) if distinct_1 else 0.0,
                "avg_distinct_2": mean(distinct_2) if distinct_2 else 0.0,
                "avg_repetition_rate": mean(repetition_rates) if repetition_rates else 0.0,
                "avg_weighted_z_score": mean(weighted_z_scores) if weighted_z_scores else None,
                "avg_winmax_weighted_z_score": mean(winmax_scores) if winmax_scores else None,
                "avg_kl": mean(avg_kls) if avg_kls else None,
                "avg_delta": mean(avg_deltas) if avg_deltas else None,
                "avg_gate_pass_rate": mean(gate_pass_rates) if gate_pass_rates else None,
            }
        )
    return summaries


def compute_auc(pos_scores: list[float], neg_scores: list[float]) -> float | None:
    if not pos_scores or not neg_scores:
        return None
    wins = 0.0
    total = len(pos_scores) * len(neg_scores)
    for pos_score in pos_scores:
        for neg_score in neg_scores:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return wins / total


def compute_tpr_at_fpr(pos_scores: list[float], neg_scores: list[float], fpr: float) -> float | None:
    if not pos_scores or not neg_scores:
        return None
    sorted_neg = sorted(neg_scores, reverse=True)
    threshold_index = max(0, min(len(sorted_neg) - 1, math.ceil(fpr * len(sorted_neg)) - 1))
    threshold = sorted_neg[threshold_index]
    return sum(score >= threshold for score in pos_scores) / len(pos_scores)


def summarize_detection_quality(rows: list[dict[str, str]], score_field: str):
    neg_scores = [to_float(row.get(score_field)) for row in rows if row["method"] == "No Watermark" and to_float(row.get(score_field)) is not None]
    methods = sorted({row["method"] for row in rows if row["method"] != "No Watermark"}, key=sort_method_key)
    detection_rows = []
    for method in methods:
        pos_scores = [to_float(row.get(score_field)) for row in rows if row["method"] == method and to_float(row.get(score_field)) is not None]
        if not pos_scores or not neg_scores:
            continue
        detection_rows.append(
            {
                "method": method,
                "score_field": score_field,
                "auc": compute_auc(pos_scores, neg_scores),
                "tpr_at_1_fpr": compute_tpr_at_fpr(pos_scores, neg_scores, 0.01),
                "tpr_at_5_fpr": compute_tpr_at_fpr(pos_scores, neg_scores, 0.05),
            }
        )
    return detection_rows


def fmt_optional(value, digits=4):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def write_summary(summary_rows, detection_quality_rows, summary_md: str):
    output_path = Path(summary_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Watermark Experiment Summary",
        "",
        "| Method | Samples | Avg z-score | Avg Green Fraction | Detection Success Rate | Avg Tokens Counted | Avg Word Count | Avg Distinct-1 | Avg Distinct-2 | Avg Repetition Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary_rows:
        lines.append(
            "| {method} | {samples} | {avg_z_score:.4f} | {avg_green_fraction:.4f} | {watermark_success_rate:.2%} | {avg_tokens_counted:.2f} | {avg_word_count:.2f} | {avg_distinct_1:.4f} | {avg_distinct_2:.4f} | {avg_repetition_rate:.4f} |".format(
                **row
            )
        )

    if any(row["avg_weighted_z_score"] is not None or row["avg_kl"] is not None for row in summary_rows):
        lines.extend(
            [
                "",
                "## CA-KL-CG Diagnostics",
                "",
                "| Method | Avg Weighted z | Avg WinMax Weighted z | Avg KL | Avg Adaptive Delta | Avg Gate Pass Rate |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary_rows:
            lines.append(
                "| {method} | {weighted} | {winmax} | {kl} | {delta} | {gate} |".format(
                    method=row["method"],
                    weighted=fmt_optional(row["avg_weighted_z_score"]),
                    winmax=fmt_optional(row["avg_winmax_weighted_z_score"]),
                    kl=fmt_optional(row["avg_kl"]),
                    delta=fmt_optional(row["avg_delta"]),
                    gate=fmt_optional(row["avg_gate_pass_rate"]),
                )
            )

    if detection_quality_rows:
        lines.extend(
            [
                "",
                "## Calibrated Detection Quality",
                "",
                "| Score | Method | AUC | TPR@1%FPR | TPR@5%FPR |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in detection_quality_rows:
            lines.append(
                "| {score_field} | {method} | {auc:.4f} | {tpr_at_1_fpr:.2%} | {tpr_at_5_fpr:.2%} |".format(
                    **row
                )
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Detection Success Rate` counts the share of outputs predicted as `Watermarked`.",
            "- `Distinct-1` and `Distinct-2` provide a lightweight diversity estimate for generated text.",
            "- `Avg Repetition Rate` is `1 - unique_words / total_words`; lower values generally indicate less repetition.",
            "- Calibrated detection quality uses `No Watermark` scores as the negative class.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def maybe_draw_charts(summary_rows, detection_quality_rows, zscore_chart: str, green_chart: str, tradeoff_chart: str, weighted_auc_chart: str, kl_delta_chart: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping chart generation.")
        return

    methods = [row["method"] for row in summary_rows]
    z_scores = [row["avg_z_score"] for row in summary_rows]
    green_fractions = [row["avg_green_fraction"] for row in summary_rows]

    zscore_path = Path(zscore_chart)
    green_path = Path(green_chart)
    tradeoff_path = Path(tradeoff_chart)
    weighted_auc_path = Path(weighted_auc_chart)
    kl_delta_path = Path(kl_delta_chart)
    zscore_path.parent.mkdir(parents=True, exist_ok=True)
    green_path.parent.mkdir(parents=True, exist_ok=True)
    tradeoff_path.parent.mkdir(parents=True, exist_ok=True)
    weighted_auc_path.parent.mkdir(parents=True, exist_ok=True)
    kl_delta_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.bar(methods, z_scores, color="#3b82f6")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Average z-score")
    plt.title("Watermark z-score Comparison")
    plt.tight_layout()
    plt.savefig(zscore_path, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(methods, green_fractions, color="#10b981")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Average green fraction")
    plt.title("Green Fraction Comparison")
    plt.tight_layout()
    plt.savefig(green_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    success_rates = [row["watermark_success_rate"] for row in summary_rows]
    repetition_rates = [row["avg_repetition_rate"] for row in summary_rows]
    plt.scatter(repetition_rates, success_rates, color="#8b5cf6")
    for method, x_value, y_value in zip(methods, repetition_rates, success_rates):
        plt.annotate(method, (x_value, y_value), fontsize=8)
    plt.xlabel("Average repetition rate")
    plt.ylabel("Detection success rate")
    plt.title("Detection-Quality Tradeoff")
    plt.tight_layout()
    plt.savefig(tradeoff_path, dpi=200)
    plt.close()

    if detection_quality_rows:
        winmax_rows = [row for row in detection_quality_rows if row["score_field"] == "winmax_weighted_z_score"]
        if winmax_rows:
            plt.figure(figsize=(10, 5))
            plt.bar([row["method"] for row in winmax_rows], [row["auc"] for row in winmax_rows], color="#14b8a6")
            plt.xticks(rotation=20, ha="right")
            plt.ylabel("AUC")
            plt.title("WinMax Weighted AUC Comparison")
            plt.tight_layout()
            plt.savefig(weighted_auc_path, dpi=200)
            plt.close()

    kl_rows = [row for row in summary_rows if row["avg_kl"] is not None or row["avg_delta"] is not None]
    if kl_rows:
        x_positions = range(len(kl_rows))
        plt.figure(figsize=(10, 5))
        plt.bar([x - 0.2 for x in x_positions], [row["avg_kl"] or 0.0 for row in kl_rows], width=0.4, label="Avg KL", color="#f59e0b")
        plt.bar([x + 0.2 for x in x_positions], [row["avg_delta"] or 0.0 for row in kl_rows], width=0.4, label="Avg Delta", color="#2563eb")
        plt.xticks(list(x_positions), [row["method"] for row in kl_rows], rotation=20, ha="right")
        plt.title("CA-KL Distortion and Bias")
        plt.legend()
        plt.tight_layout()
        plt.savefig(kl_delta_path, dpi=200)
        plt.close()


def main():
    args = parse_args()
    rows = read_rows(args.input_csv)
    summary_rows = summarize_rows(rows)
    detection_quality_rows = []
    detection_quality_rows.extend(summarize_detection_quality(rows, "weighted_z_score"))
    detection_quality_rows.extend(summarize_detection_quality(rows, "winmax_weighted_z_score"))
    write_summary(summary_rows, detection_quality_rows, args.summary_md)
    maybe_draw_charts(
        summary_rows,
        detection_quality_rows,
        args.zscore_chart,
        args.green_chart,
        args.tradeoff_chart,
        args.weighted_auc_chart,
        args.kl_delta_chart,
    )
    print(f"Saved summary to {args.summary_md}")


if __name__ == "__main__":
    main()

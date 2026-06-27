# coding=utf-8
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize watermark experiment results.")
    parser.add_argument("--input_csv", type=str, default="course_project/outputs/results.csv")
    parser.add_argument("--summary_md", type=str, default="course_project/outputs/summary.md")
    parser.add_argument("--zscore_chart", type=str, default="course_project/outputs/zscore_comparison.png")
    parser.add_argument("--green_chart", type=str, default="course_project/outputs/green_fraction_comparison.png")
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
    if method == "Adaptive Delta":
        return (2, 0.0)
    return (3, 0.0)


def summarize_rows(rows: list[dict[str, str]]):
    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[row["method"]].append(row)

    summaries = []
    for method in sorted(grouped_rows.keys(), key=sort_method_key):
        method_rows = grouped_rows[method]
        z_scores = [to_float(row["z_score"]) for row in method_rows if to_float(row["z_score"]) is not None]
        green_fractions = [to_float(row["green_fraction"]) for row in method_rows if to_float(row["green_fraction"]) is not None]
        word_counts = [to_float(row["word_count"]) for row in method_rows if to_float(row["word_count"]) is not None]
        distinct_1 = [to_float(row["distinct_1"]) for row in method_rows if to_float(row["distinct_1"]) is not None]
        distinct_2 = [to_float(row["distinct_2"]) for row in method_rows if to_float(row["distinct_2"]) is not None]
        repetition_rates = [to_float(row["repetition_rate"]) for row in method_rows if to_float(row["repetition_rate"]) is not None]
        success_rate = sum(row["prediction"] == "Watermarked" for row in method_rows) / len(method_rows)

        summaries.append(
            {
                "method": method,
                "samples": len(method_rows),
                "avg_z_score": mean(z_scores) if z_scores else 0.0,
                "avg_green_fraction": mean(green_fractions) if green_fractions else 0.0,
                "watermark_success_rate": success_rate,
                "avg_word_count": mean(word_counts) if word_counts else 0.0,
                "avg_distinct_1": mean(distinct_1) if distinct_1 else 0.0,
                "avg_distinct_2": mean(distinct_2) if distinct_2 else 0.0,
                "avg_repetition_rate": mean(repetition_rates) if repetition_rates else 0.0,
            }
        )
    return summaries


def write_summary(summary_rows, summary_md: str):
    output_path = Path(summary_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Watermark Experiment Summary",
        "",
        "| Method | Samples | Avg z-score | Avg Green Fraction | Detection Success Rate | Avg Word Count | Avg Distinct-1 | Avg Distinct-2 | Avg Repetition Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary_rows:
        lines.append(
            "| {method} | {samples} | {avg_z_score:.4f} | {avg_green_fraction:.4f} | {watermark_success_rate:.2%} | {avg_word_count:.2f} | {avg_distinct_1:.4f} | {avg_distinct_2:.4f} | {avg_repetition_rate:.4f} |".format(
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
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def maybe_draw_charts(summary_rows, zscore_chart: str, green_chart: str):
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
    zscore_path.parent.mkdir(parents=True, exist_ok=True)
    green_path.parent.mkdir(parents=True, exist_ok=True)

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


def main():
    args = parse_args()
    rows = read_rows(args.input_csv)
    summary_rows = summarize_rows(rows)
    write_summary(summary_rows, args.summary_md)
    maybe_draw_charts(summary_rows, args.zscore_chart, args.green_chart)
    print(f"Saved summary to {args.summary_md}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build the KL-budget quality/detection Pareto tables and figures.

Each source in the manifest names exactly one method from one experiment CSV.
This prevents accidentally pooling values produced with different prompts or
detector configurations.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate a fixed-delta / CA-KL Pareto experiment.")
    parser.add_argument("--manifest", required=True, help="JSON source manifest.")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def read_csv(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: object) -> float | None:
    value = str(value or "").strip()
    return float(value) if value else None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    location = (len(ordered) - 1) * quantile
    low, high = math.floor(location), math.ceil(location)
    return ordered[low] if low == high else ordered[low] * (high - location) + ordered[high] * (location - low)


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    ranked = sorted([(value, 1) for value in pos] + [(value, 0) for value in neg], key=lambda item: item[0])
    rank_sum, start = 0.0, 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][0] == ranked[start][0]:
            end += 1
        rank_sum += ((start + 1 + end) / 2.0) * sum(label for _, label in ranked[start:end])
        start = end
    return (rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def threshold_tpr(pos: list[float], neg: list[float], target_fpr: float) -> float | None:
    if not pos or not neg:
        return None
    ordered = sorted(neg)
    threshold = ordered[max(0, min(len(ordered) - 1, math.ceil((1.0 - target_fpr) * len(ordered)) - 1))]
    return sum(value >= threshold for value in pos) / len(pos)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    result = {key: rows[0].get(key, "") for key in ("name", "family", "parameter", "epsilon", "source_id", "method")}
    result["samples"] = len(rows)
    numeric_fields = (
        "z_score", "green_fraction", "distinct_1", "distinct_2", "repetition_rate", "weighted_z_score",
        "winmax_weighted_z_score", "avg_kl", "avg_delta", "delta_std", "delta_p25", "delta_p50", "delta_p75", "gate_pass_rate",
    )
    output_names = {"avg_kl": "avg_actual_kl", "avg_delta": "avg_delta"}
    for field in numeric_fields:
        values = [as_float(row.get(field)) for row in rows]
        values = [value for value in values if value is not None]
        result[output_names.get(field, f"avg_{field}")] = mean(values) if values else None
    result["kgw_detection_rate"] = sum(row.get("prediction") == "Watermarked" for row in rows) / len(rows)
    result["fixed_threshold_weighted_rate"] = sum(as_float(row.get("weighted_z_score")) is not None and float(row["weighted_z_score"]) > 4.0 for row in rows) / len(rows)
    nll = [as_float(row.get("ppl_nll_sum")) for row in rows]
    tokens = [as_float(row.get("ppl_num_scored_tokens")) for row in rows]
    pairs = [(item_nll, item_tokens) for item_nll, item_tokens in zip(nll, tokens) if item_nll is not None and item_tokens and item_tokens > 0]
    result["corpus_ppl"] = math.exp(sum(item[0] for item in pairs) / sum(item[1] for item in pairs)) if pairs else None
    return result


def draw_figures(summary_rows: list[dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to create Pareto PDF figures.") from exc

    def plot(x_field: str, y_field: str, filename: str, y_label: str) -> None:
        figure, axis = plt.subplots(figsize=(8, 5))
        for family, marker, color in (("Fixed Delta", "o", "#1f77b4"), ("CA-KL", "s", "#d62728"), ("CA-KL-CG", "D", "#2ca02c")):
            group = [row for row in summary_rows if row.get("family") == family and row.get(x_field) is not None and row.get(y_field) is not None]
            if not group:
                continue
            group.sort(key=lambda row: float(row[x_field]))
            axis.plot([row[x_field] for row in group], [row[y_field] for row in group], marker=marker, color=color, label=family)
            for row in group:
                axis.annotate(str(row.get("parameter", row.get("method", ""))), (row[x_field], row[y_field]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        axis.set_xlabel("Corpus PPL (lower is better)")
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename)
        plt.close(figure)

    plot("corpus_ppl", "kgw_detection_rate", "ppl_vs_kgw_detection.pdf", "Standard KGW detection rate")
    plot("corpus_ppl", "tpr_at_1pct_fpr", "ppl_vs_tpr1.pdf", "TPR at 1% FPR")

    cakl_rows = sorted([row for row in summary_rows if row.get("family") == "CA-KL" and row.get("epsilon") is not None], key=lambda row: float(row["epsilon"]))
    for y_field, filename, label in (("avg_delta", "epsilon_vs_avg_delta.pdf", "Average delta"), ("avg_actual_kl", "epsilon_vs_avg_kl.pdf", "Average actual KL")):
        figure, axis = plt.subplots(figsize=(7, 4.5))
        valid = [row for row in cakl_rows if row.get(y_field) is not None]
        axis.plot([row["epsilon"] for row in valid], [row[y_field] for row in valid], marker="o", color="#d62728")
        axis.set_xlabel("KL budget epsilon")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / filename)
        plt.close(figure)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    warnings: list[str] = []
    for source in manifest["sources"]:
        result_rows = [row for row in read_csv(source["results_csv"]) if row.get("method") == source["method"]]
        if not result_rows:
            warnings.append(f"No rows for {source['name']} in {source['results_csv']}")
            continue
        ppl_by_id: dict[str, dict[str, str]] = {}
        if source.get("ppl_csv"):
            ppl_by_id = {row["prompt_id"]: row for row in read_csv(source["ppl_csv"]) if row.get("method") == source["method"]}
        for row in result_rows:
            combined = dict(row)
            combined.update({key: source.get(key, "") for key in ("name", "family", "parameter", "epsilon", "source_id")})
            combined.update({key: value for key, value in ppl_by_id.get(row["prompt_id"], {}).items() if key.startswith("ppl_")})
            all_rows.append(combined)
    if not all_rows:
        raise ValueError("The manifest did not yield any per-sample rows.")

    per_sample_fields = list(dict.fromkeys([key for row in all_rows for key in row]))
    write_csv(output_dir / "per_sample_results.csv", all_rows, per_sample_fields)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        grouped[str(row["name"])].append(row)
    summary_rows = []
    for name, method_rows in grouped.items():
        summary = aggregate(method_rows)
        summary["name"] = name
        summary_rows.append(summary)

    # These values are explicitly marked as in-sample. The separate calibration
    # script supplies the reportable cross-validated AUC/TPR values.
    for summary in summary_rows:
        source_id = summary.get("source_id")
        source_rows = [row for row in all_rows if row.get("source_id") == source_id]
        source = next(item for item in manifest["sources"] if item.get("source_id") == source_id)
        negatives = [row for row in read_csv(source["results_csv"]) if row.get("method") == "No Watermark"]
        for score_field, prefix in (("weighted_z_score", "weighted"), ("winmax_weighted_z_score", "winmax")):
            pos = [as_float(row.get(score_field)) for row in source_rows]
            neg = [as_float(row.get(score_field)) for row in negatives]
            pos = [value for value in pos if value is not None]
            neg = [value for value in neg if value is not None]
            summary[f"{prefix}_auc_in_sample"] = auc(pos, neg)
            summary[f"{prefix}_tpr_at_1pct_fpr_in_sample"] = threshold_tpr(pos, neg, 0.01)
            summary[f"{prefix}_tpr_at_5pct_fpr_in_sample"] = threshold_tpr(pos, neg, 0.05)
        summary["tpr_at_1pct_fpr"] = summary["weighted_tpr_at_1pct_fpr_in_sample"]
    summary_rows.sort(key=lambda row: (str(row.get("family")), float(row.get("epsilon") or -1), str(row.get("parameter"))))
    summary_fields = list(dict.fromkeys([key for row in summary_rows for key in row]))
    write_csv(output_dir / "pareto_summary.csv", summary_rows, summary_fields)

    config = dict(manifest)
    observed_epsilons = {float(row["epsilon"]) for row in summary_rows if row.get("family") == "CA-KL" and row.get("epsilon") not in ("", None)}
    expected_epsilons = {float(value) for value in manifest.get("required_epsilon_values", [])}
    missing_epsilons = sorted(expected_epsilons - observed_epsilons)
    if missing_epsilons:
        warnings.append("Missing full 500-prompt CA-KL sources for epsilon=" + ", ".join(f"{value:.2f}" for value in missing_epsilons))
    config.update(
        {
            "manifest_path": str(manifest_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "warnings": warnings,
            "metric_note": "AUC/TPR columns in pareto_summary.csv are in-sample diagnostics only. Use calibrate_detection.py outputs for reportable independent-calibration metrics.",
        }
    )
    (output_dir / "experiment_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_figures(summary_rows, output_dir)
    print(f"Saved Pareto rows, summary, config, and figures to {output_dir}")


if __name__ == "__main__":
    main()

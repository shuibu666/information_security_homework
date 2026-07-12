#!/usr/bin/env python
"""Five-fold threshold calibration and bootstrap intervals for detector scores.

The script is intentionally dependency-free so that calibration can be rerun on
an experiment CSV without loading the language model again.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


SCORE_FIELDS = ("z_score", "weighted_z_score", "winmax_weighted_z_score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently calibrate watermark detector thresholds.")
    parser.add_argument("--input_csv", required=True, help="Per-sample generation/detection CSV.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ppl_csv", default=None, help="Optional PPL CSV aligned by prompt_id and method.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bootstrap_repetitions", type=int, default=2000)
    parser.add_argument("--confidence_level", type=float, default=0.95)
    parser.add_argument("--methods", nargs="*", default=None, help="Methods to evaluate; defaults to every non-negative method.")
    return parser.parse_args()


def read_csv(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None) -> float | None:
    value = str(value or "").strip()
    return float(value) if value else None


def empirical_quantile(values: list[float], quantile: float) -> float:
    """Upper order statistic; conservative for a finite negative calibration set."""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def auc(pos_scores: list[float], neg_scores: list[float]) -> float:
    """Mann-Whitney AUC computed in O(n log n), including tie correction."""
    labelled = [(score, 1) for score in pos_scores] + [(score, 0) for score in neg_scores]
    labelled.sort(key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(labelled):
        end = index + 1
        while end < len(labelled) and labelled[end][0] == labelled[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in labelled[index:end])
        index = end
    return (positive_rank_sum - len(pos_scores) * (len(pos_scores) + 1) / 2.0) / (len(pos_scores) * len(neg_scores))


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    location = (len(ordered) - 1) * quantile
    low, high = math.floor(location), math.ceil(location)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - location) + ordered[high] * (location - low)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_score_maps(rows: list[dict[str, str]], methods: list[str]):
    maps: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        method = row.get("method", "")
        if method != "No Watermark" and method not in methods:
            continue
        prompt_id = row.get("prompt_id", "")
        for field in SCORE_FIELDS:
            score = as_float(row.get(field))
            if prompt_id and score is not None:
                maps[field][method][prompt_id] = score
    return maps


def bootstrap_intervals(
    records_by_method: dict[tuple[str, str], list[dict[str, object]]],
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    alpha = (1.0 - confidence_level) / 2.0
    output: list[dict[str, object]] = []
    for (method, score_field), rows in sorted(records_by_method.items()):
        if not rows:
            continue
        values: dict[str, list[float]] = defaultdict(list)
        for _ in range(repetitions):
            sample = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
            pos = [float(row["score"]) for row in sample]
            neg = [float(row["negative_score"]) for row in sample]
            values["mean_score"].append(mean(pos))
            values["fixed_threshold_detection_rate"].append(sum(score > 4.0 for score in pos) / len(pos))
            values["auc"].append(auc(pos, neg))
            values["tpr_at_1pct_fpr"].append(sum(bool(row["predicted_at_1pct"]) for row in sample) / len(sample))
            values["actual_fpr_at_1pct"].append(sum(bool(row["negative_predicted_at_1pct"]) for row in sample) / len(sample))
            values["tpr_at_5pct_fpr"].append(sum(bool(row["predicted_at_5pct"]) for row in sample) / len(sample))
            values["actual_fpr_at_5pct"].append(sum(bool(row["negative_predicted_at_5pct"]) for row in sample) / len(sample))

        for metric, samples in values.items():
            estimate = mean(samples)
            output.append(
                {
                    "method": method,
                    "score_field": score_field,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower": percentile(samples, alpha),
                    "ci_upper": percentile(samples, 1.0 - alpha),
                    "bootstrap_repetitions": repetitions,
                    "confidence_level": confidence_level,
                }
            )
    return output


def add_ppl_bootstrap(
    output: list[dict[str, object]],
    ppl_csv: str | None,
    methods: list[str],
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> None:
    if not ppl_csv:
        return
    rows = read_csv(ppl_csv)
    by_method: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row in rows:
        method = row.get("method", "")
        if method not in methods and method != "No Watermark":
            continue
        nll = as_float(row.get("ppl_nll_sum"))
        tokens = as_float(row.get("ppl_num_scored_tokens"))
        if nll is not None and tokens and tokens > 0:
            by_method[method].append((nll, int(tokens)))
    rng = random.Random(seed + 7919)
    alpha = (1.0 - confidence_level) / 2.0
    for method, rows_for_method in sorted(by_method.items()):
        samples = []
        for _ in range(repetitions):
            selected = [rows_for_method[rng.randrange(len(rows_for_method))] for _ in range(len(rows_for_method))]
            nll = sum(item[0] for item in selected)
            tokens = sum(item[1] for item in selected)
            samples.append(math.exp(nll / tokens))
        output.append(
            {
                "method": method,
                "score_field": "",
                "metric": "corpus_ppl",
                "estimate": mean(samples),
                "ci_lower": percentile(samples, alpha),
                "ci_upper": percentile(samples, 1.0 - alpha),
                "bootstrap_repetitions": repetitions,
                "confidence_level": confidence_level,
            }
        )


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    rows = read_csv(args.input_csv)
    available_methods = sorted({row.get("method", "") for row in rows if row.get("method") and row.get("method") != "No Watermark"})
    methods = args.methods or available_methods
    score_maps = build_score_maps(rows, methods)
    negative_ids = set(score_maps["z_score"].get("No Watermark", {}))
    if not negative_ids:
        raise ValueError("No Watermark z_score rows are required for calibration.")

    rng = random.Random(args.seed)
    shuffled_ids = sorted(negative_ids)
    rng.shuffle(shuffled_ids)
    fold_for_id = {prompt_id: index % args.folds + 1 for index, prompt_id in enumerate(shuffled_ids)}
    output_dir = Path(args.output_dir)
    write_csv(
        output_dir / "fold_assignments.csv",
        [{"prompt_id": prompt_id, "fold": fold_for_id[prompt_id]} for prompt_id in sorted(fold_for_id, key=lambda item: int(item) if item.isdigit() else item)],
        ["prompt_id", "fold"],
    )

    thresholds: dict[tuple[str, int, float], float] = {}
    threshold_rows: list[dict[str, object]] = []
    for score_field in SCORE_FIELDS:
        negative_scores = score_maps[score_field].get("No Watermark", {})
        for fold in range(1, args.folds + 1):
            calibration_ids = [prompt_id for prompt_id in negative_scores if fold_for_id.get(prompt_id) != fold]
            test_ids = [prompt_id for prompt_id in negative_scores if fold_for_id.get(prompt_id) == fold]
            if not calibration_ids or not test_ids:
                continue
            calibration_scores = [negative_scores[prompt_id] for prompt_id in calibration_ids]
            test_scores = [negative_scores[prompt_id] for prompt_id in test_ids]
            for target_fpr in (0.01, 0.05):
                threshold = empirical_quantile(calibration_scores, 1.0 - target_fpr)
                thresholds[(score_field, fold, target_fpr)] = threshold
                threshold_rows.append(
                    {
                        "score_field": score_field,
                        "fold": fold,
                        "target_fpr": target_fpr,
                        "threshold": threshold,
                        "calibration_negative_samples": len(calibration_scores),
                        "test_negative_samples": len(test_scores),
                        "test_actual_fpr": sum(score >= threshold for score in test_scores) / len(test_scores),
                    }
                )
    write_csv(
        output_dir / "fold_thresholds.csv",
        threshold_rows,
        ["score_field", "fold", "target_fpr", "threshold", "calibration_negative_samples", "test_negative_samples", "test_actual_fpr"],
    )

    oof_rows: list[dict[str, object]] = []
    paired_records: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    metric_rows: list[dict[str, object]] = []
    for score_field in SCORE_FIELDS:
        negatives = score_maps[score_field].get("No Watermark", {})
        for prompt_id, score in negatives.items():
            fold = fold_for_id[prompt_id]
            oof_rows.append(
                {
                    "fold": fold,
                    "prompt_id": prompt_id,
                    "method": "No Watermark",
                    "label": 0,
                    "score_field": score_field,
                    "score": score,
                    "threshold_at_1pct": thresholds[(score_field, fold, 0.01)],
                    "predicted_at_1pct": score >= thresholds[(score_field, fold, 0.01)],
                    "threshold_at_5pct": thresholds[(score_field, fold, 0.05)],
                    "predicted_at_5pct": score >= thresholds[(score_field, fold, 0.05)],
                }
            )
        for method in methods:
            positives = score_maps[score_field].get(method, {})
            common_ids = sorted(set(positives) & set(negatives))
            if not common_ids:
                continue
            method_oof = []
            for prompt_id in common_ids:
                fold = fold_for_id[prompt_id]
                row = {
                    "fold": fold,
                    "prompt_id": prompt_id,
                    "method": method,
                    "label": 1,
                    "score_field": score_field,
                    "score": positives[prompt_id],
                    "threshold_at_1pct": thresholds[(score_field, fold, 0.01)],
                    "predicted_at_1pct": positives[prompt_id] >= thresholds[(score_field, fold, 0.01)],
                    "threshold_at_5pct": thresholds[(score_field, fold, 0.05)],
                    "predicted_at_5pct": positives[prompt_id] >= thresholds[(score_field, fold, 0.05)],
                }
                method_oof.append(row)
                oof_rows.append(row)
                paired_records[(method, score_field)].append(
                    {
                        "score": positives[prompt_id],
                        "negative_score": negatives[prompt_id],
                        "predicted_at_1pct": row["predicted_at_1pct"],
                        "negative_predicted_at_1pct": negatives[prompt_id] >= thresholds[(score_field, fold, 0.01)],
                        "predicted_at_5pct": row["predicted_at_5pct"],
                        "negative_predicted_at_5pct": negatives[prompt_id] >= thresholds[(score_field, fold, 0.05)],
                    }
                )
            negative_oof = [row for row in oof_rows if row["method"] == "No Watermark" and row["score_field"] == score_field]
            metric_rows.append(
                {
                    "method": method,
                    "score_field": score_field,
                    "samples": len(method_oof),
                    "mean_score": mean(float(row["score"]) for row in method_oof),
                    "fixed_threshold_detection_rate": sum(float(row["score"]) > 4.0 for row in method_oof) / len(method_oof),
                    "auc": auc([float(row["score"]) for row in method_oof], [float(row["score"]) for row in negative_oof]),
                    "tpr_at_1pct_fpr": sum(bool(row["predicted_at_1pct"]) for row in method_oof) / len(method_oof),
                    "actual_fpr_at_1pct": sum(bool(row["predicted_at_1pct"]) for row in negative_oof) / len(negative_oof),
                    "tpr_at_5pct_fpr": sum(bool(row["predicted_at_5pct"]) for row in method_oof) / len(method_oof),
                    "actual_fpr_at_5pct": sum(bool(row["predicted_at_5pct"]) for row in negative_oof) / len(negative_oof),
                }
            )
    write_csv(
        output_dir / "out_of_fold_scores.csv",
        oof_rows,
        ["fold", "prompt_id", "method", "label", "score_field", "score", "threshold_at_1pct", "predicted_at_1pct", "threshold_at_5pct", "predicted_at_5pct"],
    )

    bootstrap_rows = bootstrap_intervals(paired_records, args.bootstrap_repetitions, args.confidence_level, args.seed)
    add_ppl_bootstrap(bootstrap_rows, args.ppl_csv, methods, args.bootstrap_repetitions, args.confidence_level, args.seed)
    write_csv(
        output_dir / "bootstrap_metrics.csv",
        bootstrap_rows,
        ["method", "score_field", "metric", "estimate", "ci_lower", "ci_upper", "bootstrap_repetitions", "confidence_level"],
    )

    ci_lookup = {(row["method"], row["score_field"], row["metric"]): row for row in bootstrap_rows}
    for row in metric_rows:
        for metric in ("mean_score", "fixed_threshold_detection_rate", "auc", "tpr_at_1pct_fpr", "tpr_at_5pct_fpr"):
            ci = ci_lookup.get((row["method"], row["score_field"], metric))
            if ci:
                row[f"{metric}_ci_lower"] = ci["ci_lower"]
                row[f"{metric}_ci_upper"] = ci["ci_upper"]
    metric_fields = [
        "method", "score_field", "samples", "mean_score", "mean_score_ci_lower", "mean_score_ci_upper",
        "fixed_threshold_detection_rate", "fixed_threshold_detection_rate_ci_lower", "fixed_threshold_detection_rate_ci_upper",
        "auc", "auc_ci_lower", "auc_ci_upper", "tpr_at_1pct_fpr", "tpr_at_1pct_fpr_ci_lower", "tpr_at_1pct_fpr_ci_upper",
        "actual_fpr_at_1pct", "tpr_at_5pct_fpr", "tpr_at_5pct_fpr_ci_lower", "tpr_at_5pct_fpr_ci_upper", "actual_fpr_at_5pct",
    ]
    write_csv(output_dir / "calibrated_metrics.csv", metric_rows, metric_fields)
    (output_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "input_csv": str(Path(args.input_csv).resolve()),
                "ppl_csv": str(Path(args.ppl_csv).resolve()) if args.ppl_csv else None,
                "methods": methods,
                "folds": args.folds,
                "seed": args.seed,
                "thresholds": {"tpr_at_1pct_fpr": "99th empirical percentile of each fold's 400 negative calibration scores", "tpr_at_5pct_fpr": "95th empirical percentile of each fold's 400 negative calibration scores"},
                "bootstrap_repetitions": args.bootstrap_repetitions,
                "confidence_level": args.confidence_level,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved independently calibrated metrics to {output_dir}")


if __name__ == "__main__":
    main()

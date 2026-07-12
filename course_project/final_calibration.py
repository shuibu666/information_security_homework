"""Detector-matched conservative calibration for the final-v1 protocol."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping


def conservative_threshold(scores: Iterable[float], target_fpr: float) -> float:
    """Return a threshold for the strict rule ``score > threshold``.

    It uses an order statistic that permits at most floor(n * target_fpr)
    validation false positives. Equal scores are intentionally never split.
    """
    values = sorted(float(score) for score in scores)
    if not values:
        raise ValueError("cannot calibrate a threshold from zero negative scores")
    if not 0.0 <= target_fpr < 1.0:
        raise ValueError("target_fpr must be in [0, 1)")
    allowed = math.floor(len(values) * target_fpr)
    return values[len(values) - allowed - 1]


def group_scores(rows: Iterable[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        detector_id = str(row.get("detector_config_id", ""))
        generation_id = str(row.get("generation_id", ""))
        prompt_id = str(row.get("prompt_id", ""))
        if not detector_id or not generation_id or not prompt_id:
            raise ValueError("each detection row requires detector_config_id, generation_id, and prompt_id")
        key = (detector_id, generation_id)
        if key in seen:
            raise ValueError(f"duplicate detection score for {key}")
        seen.add(key)
        if row.get("detector_config_hash") in (None, ""):
            raise ValueError(f"{key} lacks detector_config_hash")
        try:
            score = float(row["score"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{key} has no finite numeric score") from error
        if not math.isfinite(score):
            raise ValueError(f"{key} has a non-finite score")
        grouped[detector_id].append(row)
    return grouped


def calibrate_human_thresholds(validation_rows: Iterable[Mapping[str, object]], target_fprs: Iterable[float]) -> list[dict[str, object]]:
    grouped = group_scores(validation_rows)
    output: list[dict[str, object]] = []
    for detector_id, rows in sorted(grouped.items()):
        hashes = {str(row["detector_config_hash"]) for row in rows}
        if len(hashes) != 1:
            raise ValueError(f"detector_config_id {detector_id} maps to multiple hashes")
        detector_hash = next(iter(hashes))
        human = [float(row["score"]) for row in rows if row.get("role") == "human_completion"]
        if not human:
            raise ValueError(f"{detector_id} has no validation human_completion negatives")
        for target in target_fprs:
            threshold = conservative_threshold(human, target)
            output.append({
                "detector_config_id": detector_id,
                "detector_config_hash": detector_hash,
                "target_fpr": float(target),
                "threshold": threshold,
                "calibration_role": "human_completion",
                "calibration_samples": len(human),
                "calibration_empirical_fpr": sum(score > threshold for score in human) / len(human),
                "comparison": ">",
            })
    return output


def evaluate_test_scores(test_rows: Iterable[Mapping[str, object]], threshold_rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped = group_scores(test_rows)
    thresholds = {(str(row["detector_config_id"]), float(row["target_fpr"])): row for row in threshold_rows}
    output: list[dict[str, object]] = []
    for detector_id, rows in sorted(grouped.items()):
        hashes = {str(row["detector_config_hash"]) for row in rows}
        if len(hashes) != 1:
            raise ValueError(f"detector_config_id {detector_id} maps to multiple hashes")
        actual_hash = next(iter(hashes))
        human_ids = {str(row["prompt_id"]) for row in rows if row.get("role") == "human_completion"}
        if not human_ids:
            raise ValueError(f"{detector_id} has no held-out human_completion scores")
        grouped_roles: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            role = str(row.get("role", ""))
            key = (role, str(row.get("generator_id", "")), str(row.get("parameter_id", "")))
            grouped_roles[key].append(row)
        for (role, generator_id, parameter_id), role_rows in grouped_roles.items():
            if role == "human_completion":
                continue
            role_ids = {str(row["prompt_id"]) for row in role_rows}
            if role_ids != human_ids:
                raise ValueError(
                    f"{detector_id} {role}/{generator_id}/{parameter_id} does not share exactly the human prompt IDs"
                )
        for (threshold_detector_id, target), threshold_row in thresholds.items():
            if threshold_detector_id != detector_id:
                continue
            if str(threshold_row["detector_config_hash"]) != actual_hash:
                raise ValueError(f"test scores for {detector_id} do not match calibration detector hash")
            threshold = float(threshold_row["threshold"])
            for (role, generator_id, parameter_id), role_rows in sorted(grouped_roles.items()):
                output.append({
                    "detector_config_id": detector_id,
                    "detector_config_hash": actual_hash,
                    "target_fpr": target,
                    "threshold": threshold,
                    "comparison": ">",
                    "role": role,
                    "generator_id": generator_id,
                    "parameter_id": parameter_id,
                    "samples": len(role_rows),
                    "positive_rate": sum(float(row["score"]) > threshold for row in role_rows) / len(role_rows),
                })
    return output

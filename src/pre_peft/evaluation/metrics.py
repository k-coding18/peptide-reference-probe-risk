from __future__ import annotations

from math import ceil
from typing import Iterable, Sequence

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def binary_auroc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    valid = np.isfinite(values)
    labels, values = labels[valid], values[valid]
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _average_ranks(values)
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def average_precision(y_true: Sequence[int], scores: Sequence[float]) -> float:
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    order = np.argsort(-values, kind="mergesort")
    labels = labels[order]
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    cumulative = np.cumsum(labels)
    precision = cumulative / np.arange(1, len(labels) + 1)
    return float(precision[labels == 1].sum() / positives)


def brier_score(y_true: Sequence[int], probabilities: Sequence[float]) -> float:

    labels = np.asarray(y_true, dtype=float)
    values = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(labels) & np.isfinite(values)
    if not valid.any():
        return float("nan")
    return float(np.mean((values[valid] - labels[valid]) ** 2))


def expected_calibration_error(y_true: Sequence[int], probabilities: Sequence[float], bin_count: int = 10) -> float:

    if int(bin_count) < 1:
        raise ValueError("bin_count must be at least one")
    labels = np.asarray(y_true, dtype=float)
    values = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(labels) & np.isfinite(values)
    labels, values = labels[valid], values[valid]
    if len(values) == 0:
        return float("nan")
    if (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("probabilities must lie in [0, 1]")
    bins = np.minimum((values * int(bin_count)).astype(int), int(bin_count) - 1)
    ece = 0.0
    for index in range(int(bin_count)):
        selected = bins == index
        if selected.any():
            ece += float(selected.mean()) * abs(float(values[selected].mean()) - float(labels[selected].mean()))
    return float(ece)


def top_fraction_lift(y_true: Sequence[int], scores: Sequence[float], fraction: float = 0.20) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    n_top = max(1, int(ceil(len(labels) * fraction)))
    rate = float(labels[np.argsort(-values, kind="mergesort")[:n_top]].mean())
    baseline = float(labels.mean())
    return {"top_rate": rate, "base_rate": baseline, "lift": rate / baseline if baseline else float("nan")}


def risk_coverage_curve(
    errors: Sequence[int], scores: Sequence[float], coverages: Iterable[float], sample_labels: Sequence[int] | None = None
) -> list[dict[str, float]]:
    error_values = np.asarray(errors, dtype=int)
    score_values = np.asarray(scores, dtype=float)
    label_values = error_values if sample_labels is None else np.asarray(sample_labels, dtype=int)
    if len(label_values) != len(error_values):
        raise ValueError("sample_labels and errors must have identical lengths")
    order = np.argsort(score_values, kind="mergesort")
    rows: list[dict[str, float]] = []
    for coverage in coverages:
        retained = max(1, int(np.floor(len(error_values) * float(coverage))))
        rows.append(
            {
                "coverage": float(coverage),
                "n_retained": retained,
                "error_rate": float(error_values[order[:retained]].mean()),
                "positive_rate": float(label_values[order[:retained]].mean()),
            }
        )
    return rows


def high_loss_endpoint(losses: Sequence[float], fraction: float) -> np.ndarray:
    values = np.asarray(losses, dtype=float)
    cutoff = float(np.quantile(values, 1.0 - fraction))
    return (values >= cutoff).astype(int)


def metric_record(target: Sequence[int], score: Sequence[float], fraction: float = 0.20) -> dict[str, float]:
    lift = top_fraction_lift(target, score, fraction)
    return {"auroc": binary_auroc(target, score), "auprc": average_precision(target, score), "top_fraction_lift": lift["lift"], "top_fraction_rate": lift["top_rate"], "target_rate": lift["base_rate"]}

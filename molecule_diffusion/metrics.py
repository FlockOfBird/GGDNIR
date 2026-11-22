from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


def binary_classification_metrics(
    labels: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    num_calibration_bins: int = 10,
) -> dict:
    """Return discrimination and probability-calibration metrics.

    Calibration bins use fixed-width probability intervals, which makes the
    resulting reliability table directly plottable and comparable between runs.
    """
    labels_array = np.asarray(labels, dtype=np.float64).reshape(-1)
    probabilities_array = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels_array.size == 0:
        raise ValueError("At least one label is required.")
    if labels_array.shape != probabilities_array.shape:
        raise ValueError("Labels and probabilities must have the same shape.")
    if num_calibration_bins < 2:
        raise ValueError("num_calibration_bins must be at least 2.")
    if not np.all(np.isin(labels_array, [0.0, 1.0])):
        raise ValueError("Binary classification metrics require labels in {0, 1}.")

    epsilon = 1e-7
    clipped = np.clip(probabilities_array, epsilon, 1.0 - epsilon)
    predictions = (probabilities_array >= 0.5).astype(np.float64)
    prevalence = float(labels_array.mean())
    brier_score = float(np.mean((probabilities_array - labels_array) ** 2))
    negative_log_likelihood = float(
        -np.mean(labels_array * np.log(clipped) + (1.0 - labels_array) * np.log(1.0 - clipped))
    )

    null_brier = prevalence * (1.0 - prevalence)
    null_probability = float(np.clip(prevalence, epsilon, 1.0 - epsilon))
    null_negative_log_likelihood = float(
        -(prevalence * math.log(null_probability) + (1.0 - prevalence) * math.log(1.0 - null_probability))
    )

    bin_edges = np.linspace(0.0, 1.0, num_calibration_bins + 1)
    bin_indices = np.minimum(
        np.digitize(probabilities_array, bin_edges[1:-1], right=False),
        num_calibration_bins - 1,
    )
    reliability_bins = []
    expected_calibration_error = 0.0
    maximum_calibration_error = 0.0
    for bin_index in range(num_calibration_bins):
        selected = bin_indices == bin_index
        count = int(selected.sum())
        mean_probability = float(probabilities_array[selected].mean()) if count else None
        observed_frequency = float(labels_array[selected].mean()) if count else None
        absolute_gap = (
            abs(mean_probability - observed_frequency)
            if mean_probability is not None and observed_frequency is not None
            else None
        )
        if absolute_gap is not None:
            expected_calibration_error += (count / labels_array.size) * absolute_gap
            maximum_calibration_error = max(maximum_calibration_error, absolute_gap)
        reliability_bins.append(
            {
                "lower": float(bin_edges[bin_index]),
                "upper": float(bin_edges[bin_index + 1]),
                "count": count,
                "mean_probability": mean_probability,
                "observed_frequency": observed_frequency,
                "absolute_gap": absolute_gap,
            }
        )

    roc_auc = None
    average_precision = None
    balanced_accuracy = None
    if np.unique(labels_array).size > 1:
        roc_auc = float(roc_auc_score(labels_array, probabilities_array))
        average_precision = float(average_precision_score(labels_array, probabilities_array))
        balanced_accuracy = float(balanced_accuracy_score(labels_array, predictions))

    return {
        "num_examples": int(labels_array.size),
        "prevalence": prevalence,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "accuracy": float(np.mean(predictions == labels_array)),
        "balanced_accuracy": balanced_accuracy,
        "negative_log_likelihood": negative_log_likelihood,
        "null_negative_log_likelihood": null_negative_log_likelihood,
        "brier_score": brier_score,
        "null_brier_score": null_brier,
        "brier_skill_score": (
            float(1.0 - brier_score / null_brier) if null_brier > 0.0 else None
        ),
        "expected_calibration_error": float(expected_calibration_error),
        "maximum_calibration_error": float(maximum_calibration_error),
        "num_calibration_bins": num_calibration_bins,
        "reliability_bins": reliability_bins,
    }


def regression_metrics(
    targets: Sequence[float] | np.ndarray,
    predictions: Sequence[float] | np.ndarray,
) -> dict[str, float | int | None]:
    targets_array = np.asarray(targets, dtype=np.float64).reshape(-1)
    predictions_array = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if targets_array.size == 0:
        raise ValueError("At least one regression target is required.")
    if targets_array.shape != predictions_array.shape:
        raise ValueError("Targets and predictions must have the same shape.")

    errors = predictions_array - targets_array
    squared_error = float(np.sum(errors**2))
    centered_sum_of_squares = float(
        np.sum((targets_array - targets_array.mean()) ** 2)
    )
    target_std = float(targets_array.std())
    prediction_std = float(predictions_array.std())
    pearson = None
    spearman = None
    if target_std > 0.0 and prediction_std > 0.0 and targets_array.size > 1:
        pearson = float(np.corrcoef(targets_array, predictions_array)[0, 1])
        spearman_result = spearmanr(targets_array, predictions_array)
        if not np.isnan(spearman_result.statistic):
            spearman = float(spearman_result.statistic)

    return {
        "num_examples": int(targets_array.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "r2": (
            float(1.0 - squared_error / centered_sum_of_squares)
            if centered_sum_of_squares > 0.0
            else None
        ),
        "pearson": pearson,
        "spearman": spearman,
        "target_mean": float(targets_array.mean()),
        "prediction_mean": float(predictions_array.mean()),
    }


def summarize_scalar_metrics(metric_sets: Sequence[dict]) -> dict[str, dict[str, float]]:
    """Calculate mean/std for scalar metrics shared by multiple experiment runs."""
    if not metric_sets:
        return {}
    summaries: dict[str, dict[str, float]] = {}
    common_keys = set.intersection(*(set(metrics) for metrics in metric_sets))
    for key in sorted(common_keys):
        values = [metrics[key] for metrics in metric_sets]
        if any(value is None or isinstance(value, (bool, str, list, dict)) for value in values):
            continue
        if not all(isinstance(value, (int, float)) for value in values):
            continue
        array = np.asarray(values, dtype=np.float64)
        summaries[key] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        }
    return summaries

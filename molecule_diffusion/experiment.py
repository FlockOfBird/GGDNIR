from __future__ import annotations

from collections.abc import Sequence

import torch

from .diffusion import DiffusionSchedule
from .metrics import binary_classification_metrics, summarize_scalar_metrics
from .property_head import NoisyLatentPropertyClassifier


def noise_fractions_to_timesteps(
    noise_fractions: Sequence[float],
    num_steps: int,
) -> list[int]:
    if num_steps < 1:
        raise ValueError("num_steps must be positive.")
    if not noise_fractions:
        raise ValueError("At least one experiment noise fraction is required.")
    invalid = [value for value in noise_fractions if value < 0.0 or value > 1.0]
    if invalid:
        raise ValueError(f"Noise fractions must be in [0, 1], received {invalid}.")
    return sorted({int(round(value * (num_steps - 1))) for value in noise_fractions})


def evaluate_guidance_at_timesteps(
    model: NoisyLatentPropertyClassifier,
    schedule: DiffusionSchedule,
    dataloader,
    device: torch.device,
    *,
    timesteps: Sequence[int],
    noise_seed: int,
    num_calibration_bins: int = 10,
) -> dict:
    """Evaluate a guidance model against hard labels using identical fixed noise."""
    if not timesteps:
        raise ValueError("At least one evaluation timestep is required.")
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(noise_seed)
    aggregate_labels: list[float] = []
    aggregate_probabilities: list[float] = []
    per_timestep = {}

    with torch.no_grad():
        for timestep in timesteps:
            labels_at_t: list[float] = []
            probabilities_at_t: list[float] = []
            for batch in dataloader:
                sequence_embeddings = batch["sequence_embeddings"].to(device).float()
                attention_mask = batch["attention_mask"].to(device=device, dtype=torch.bool)
                labels = batch["labels"].to(device)
                valid_tokens = attention_mask.unsqueeze(-1)
                clean_latents = schedule.standardize_latents(sequence_embeddings)
                clean_latents = clean_latents * valid_tokens.to(dtype=clean_latents.dtype)
                timestep_batch = torch.full(
                    (sequence_embeddings.size(0),),
                    timestep,
                    device=device,
                    dtype=torch.long,
                )
                noise = torch.randn(
                    clean_latents.shape,
                    generator=generator,
                    device=device,
                    dtype=clean_latents.dtype,
                )
                noise = noise * valid_tokens.to(dtype=noise.dtype)
                noisy_latents, _ = schedule.q_sample(clean_latents, timestep_batch, noise=noise)
                noisy_latents = noisy_latents * valid_tokens.to(dtype=noisy_latents.dtype)
                probabilities = model(
                    noisy_latents,
                    timestep_batch,
                    attention_mask,
                )["probabilities"]
                labels_at_t.extend(labels.cpu().tolist())
                probabilities_at_t.extend(probabilities.cpu().tolist())

            timestep_metrics = binary_classification_metrics(
                labels_at_t,
                probabilities_at_t,
                num_calibration_bins=num_calibration_bins,
            )
            per_timestep[str(timestep)] = timestep_metrics
            aggregate_labels.extend(labels_at_t)
            aggregate_probabilities.extend(probabilities_at_t)

    return {
        "timesteps": list(timesteps),
        "aggregate": binary_classification_metrics(
            aggregate_labels,
            aggregate_probabilities,
            num_calibration_bins=num_calibration_bins,
        ),
        "per_timestep": per_timestep,
    }


def build_distillation_decision(
    *,
    clean_validation_metrics: dict,
    runs: list[dict],
    min_clean_validation_auc: float,
    max_clean_validation_ece: float,
    min_distillation_nll_improvement: float,
    max_guidance_auc_drop: float,
    max_guidance_ece_increase: float,
) -> tuple[dict, dict]:
    """Summarize matched runs and decide from validation data only."""
    summaries = {}
    for arm in ("supervised", "distilled"):
        summaries[arm] = {}
        for split in ("validation", "test"):
            metric_sets = [run[arm][split]["aggregate"] for run in runs]
            summaries[arm][split] = summarize_scalar_metrics(metric_sets)

    validation_deltas = {}
    for metric_name in (
        "negative_log_likelihood",
        "brier_score",
        "expected_calibration_error",
        "roc_auc",
        "average_precision",
    ):
        supervised_metric = summaries["supervised"]["validation"].get(metric_name)
        distilled_metric = summaries["distilled"]["validation"].get(metric_name)
        if supervised_metric is not None and distilled_metric is not None:
            validation_deltas[metric_name] = (
                distilled_metric["mean"] - supervised_metric["mean"]
            )

    clean_auc = clean_validation_metrics.get("roc_auc")
    clean_ece = clean_validation_metrics.get("expected_calibration_error")
    clean_brier_skill = clean_validation_metrics.get("brier_skill_score")
    clean_nll = clean_validation_metrics.get("negative_log_likelihood")
    clean_null_nll = clean_validation_metrics.get("null_negative_log_likelihood")
    strong_performance = clean_auc is not None and clean_auc >= min_clean_validation_auc
    reliable_calibration = (
        clean_ece is not None
        and clean_ece <= max_clean_validation_ece
        and clean_brier_skill is not None
        and clean_brier_skill > 0.0
        and clean_nll is not None
        and clean_null_nll is not None
        and clean_nll < clean_null_nll
    )

    nll_delta = validation_deltas.get("negative_log_likelihood")
    brier_delta = validation_deltas.get("brier_score")
    ece_delta = validation_deltas.get("expected_calibration_error")
    auc_delta = validation_deltas.get("roc_auc")
    guidance_supports_distillation = (
        nll_delta is not None
        and nll_delta < -min_distillation_nll_improvement
        and brier_delta is not None
        and brier_delta <= 0.0
        and ece_delta is not None
        and ece_delta <= max_guidance_ece_increase
        and auc_delta is not None
        and auc_delta >= -max_guidance_auc_drop
    )
    recommend_distillation = (
        strong_performance and reliable_calibration and guidance_supports_distillation
    )

    failed_checks = []
    if not strong_performance:
        failed_checks.append("clean validation ROC-AUC is below the required threshold")
    if not reliable_calibration:
        failed_checks.append("clean validation probabilities do not meet calibration criteria")
    if not guidance_supports_distillation:
        failed_checks.append("distillation does not improve matched validation guidance metrics")
    recommendation = (
        "Set DISTILL_GUIDANCE_CLASSIFIER = True."
        if recommend_distillation
        else "Keep DISTILL_GUIDANCE_CLASSIFIER = False: " + "; ".join(failed_checks) + "."
    )

    return summaries | {"distilled_minus_supervised_validation": validation_deltas}, {
        "clean_classifier_has_strong_validation_performance": strong_performance,
        "clean_classifier_has_reliable_calibration": reliable_calibration,
        "guidance_validation_supports_distillation": guidance_supports_distillation,
        "recommended_distill_guidance_classifier": recommend_distillation,
        "recommendation": recommendation,
        "decision_uses_test_split": False,
    }

from __future__ import annotations

import torch
from torch.nn import functional as F

from .property_head import NoisyLatentPropertyClassifier
from .property_head import NoisyLatentPropertyRegressor


def guidance_strength(
    step: int,
    total_steps: int,
    guidance_max: float,
    guidance_gamma: float,
) -> float:
    progress = 1.0 - (step / max(total_steps - 1, 1))
    return guidance_max * (progress**guidance_gamma)


def noisy_latent_property_gradient(
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    guidance_regressor: NoisyLatentPropertyRegressor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Gradient of a standardized property prediction with respect to the latent.

    The signed guidance scale is applied by the sampler. Using a standardized
    prediction keeps a unit of guidance comparable across differently scaled
    properties such as QED and molecular weight.
    """

    with torch.enable_grad():
        differentiable_latents = latents.detach().requires_grad_(True)
        standardized_prediction = guidance_regressor(
            differentiable_latents,
            timesteps,
            attention_mask,
        )["standardized_prediction"]
        gradient = torch.autograd.grad(
            standardized_prediction.sum(), differentiable_latents
        )[0]
    valid_tokens = attention_mask.unsqueeze(-1).to(dtype=gradient.dtype)
    return (gradient * valid_tokens).detach()


def noisy_latent_guidance_gradient(
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    guidance_classifier: NoisyLatentPropertyClassifier,
    attention_mask: torch.Tensor,
    target_label: int = 1,
) -> torch.Tensor:
    with torch.enable_grad():
        differentiable_latents = latents.detach().requires_grad_(True)
        logits = guidance_classifier(
            differentiable_latents,
            timesteps,
            attention_mask,
        )["logits"]
        if target_label == 1:
            log_probs = F.logsigmoid(logits)
        elif target_label == 0:
            log_probs = F.logsigmoid(-logits)
        else:
            raise ValueError(f"target_label must be 0 or 1, got {target_label}")
        gradient = torch.autograd.grad(log_probs.sum(), differentiable_latents)[0]
    valid_tokens = attention_mask.unsqueeze(-1).to(dtype=gradient.dtype)
    return (gradient * valid_tokens).detach()

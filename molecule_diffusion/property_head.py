from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import GuidanceClassifierConfig
from .diffusion import timestep_embedding
from .utils import masked_mean


class NoisyLatentPropertyClassifier(nn.Module):
    def __init__(self, latent_dim: int, config: GuidanceClassifierConfig) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.config = config
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        input_dim = latent_dim + config.hidden_dim
        classifier_layers: list[nn.Module] = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        ]
        for _ in range(max(config.num_hidden_layers - 1, 0)):
            classifier_layers.extend([
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
        classifier_layers.append(nn.Linear(config.hidden_dim, 1))
        self.model = nn.Sequential(*classifier_layers)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if noisy_latents.ndim == 3:
            if attention_mask is None:
                raise ValueError("Sequence guidance requires a latent attention mask.")
            noisy_latents = masked_mean(noisy_latents, attention_mask)
        elif noisy_latents.ndim != 2:
            raise ValueError("Guidance classifier expects vector or sequence latents.")
        time_features = timestep_embedding(timesteps, self.config.time_embedding_dim)
        time_features = self.time_mlp(time_features)
        inputs = torch.cat([noisy_latents, time_features], dim=-1)
        logits = self.model(inputs).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        return {
            "logits": logits,
            "probabilities": probabilities,
        }

    def supervised_loss(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self(noisy_latents, timesteps, attention_mask)
        loss = F.binary_cross_entropy_with_logits(outputs["logits"], labels)
        return {
            "loss": loss,
            "probabilities": outputs["probabilities"],
            "logits": outputs["logits"],
        }

    def distillation_loss(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
        target_probabilities: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self(noisy_latents, timesteps, attention_mask)
        loss = F.binary_cross_entropy_with_logits(outputs["logits"], target_probabilities)
        return {
            "loss": loss,
            "probabilities": outputs["probabilities"],
            "logits": outputs["logits"],
        }


# Backward-compatible import name for older local scripts. New checkpoints should be
# trained and loaded as NoisyLatentPropertyClassifier guidance classifiers.
LatentPropertyHead = NoisyLatentPropertyClassifier


class NoisyLatentPropertyRegressor(nn.Module):
    """Predict a normalized continuous property from a noised latent and timestep."""

    def __init__(
        self,
        latent_dim: int,
        config: GuidanceClassifierConfig,
        *,
        target_mean: float,
        target_scale: float,
    ) -> None:
        super().__init__()
        if target_scale <= 0.0:
            raise ValueError("target_scale must be positive.")
        self.latent_dim = latent_dim
        self.config = config
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_scale", torch.tensor(float(target_scale)))
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        input_dim = latent_dim + config.hidden_dim
        layers: list[nn.Module] = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        ]
        for _ in range(max(config.num_hidden_layers - 1, 0)):
            layers.extend(
                [
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
        layers.append(nn.Linear(config.hidden_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if noisy_latents.ndim == 3:
            if attention_mask is None:
                raise ValueError("Sequence guidance requires a latent attention mask.")
            noisy_latents = masked_mean(noisy_latents, attention_mask)
        elif noisy_latents.ndim != 2:
            raise ValueError("Guidance regressor expects vector or sequence latents.")
        time_features = timestep_embedding(timesteps, self.config.time_embedding_dim)
        time_features = self.time_mlp(time_features)
        inputs = torch.cat([noisy_latents, time_features], dim=-1)
        standardized_prediction = self.model(inputs).squeeze(-1)
        prediction = standardized_prediction * self.target_scale + self.target_mean
        return {
            "prediction": prediction,
            "standardized_prediction": standardized_prediction,
        }

    def supervised_loss(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self(noisy_latents, timesteps, attention_mask)
        standardized_targets = (targets - self.target_mean) / self.target_scale
        loss = F.smooth_l1_loss(
            outputs["standardized_prediction"], standardized_targets
        )
        return {"loss": loss, **outputs}

    def distillation_loss(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
        target_predictions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.supervised_loss(
            noisy_latents,
            timesteps,
            attention_mask,
            target_predictions,
        )

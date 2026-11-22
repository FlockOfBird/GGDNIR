from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import ClassifierConfig


class BBBPClassifier(nn.Module):
    def __init__(self, latent_dim: int, config: ClassifierConfig) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.config = config
        self.model = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.model(embeddings).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        return {
            "logits": logits,
            "probabilities": probabilities,
        }

    def loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self(embeddings)
        loss = F.binary_cross_entropy_with_logits(outputs["logits"], labels)
        return {
            "loss": loss,
            "probabilities": outputs["probabilities"],
            "logits": outputs["logits"],
        }


class PropertyRegressor(nn.Module):
    """Predict a continuous molecular property from a clean pooled latent."""

    def __init__(
        self,
        latent_dim: int,
        config: ClassifierConfig,
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
        self.model = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        standardized_prediction = self.model(embeddings).squeeze(-1)
        prediction = standardized_prediction * self.target_scale + self.target_mean
        return {
            "prediction": prediction,
            "standardized_prediction": standardized_prediction,
        }

    def loss(self, embeddings: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self(embeddings)
        standardized_targets = (targets - self.target_mean) / self.target_scale
        loss = F.smooth_l1_loss(
            outputs["standardized_prediction"], standardized_targets
        )
        return {"loss": loss, **outputs}

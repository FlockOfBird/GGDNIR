from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class LatentStandardizer(nn.Module):
    """Per-feature affine transform between raw encoder and diffusion coordinates."""

    VERSION = 1
    STANDARD_KIND = "per_feature_standardization"
    IDENTITY_KIND = "identity"

    def __init__(
        self,
        mean: torch.Tensor,
        scale: torch.Tensor,
        *,
        enabled: bool,
        epsilon: float = 1e-6,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32).detach().clone()
        scale = torch.as_tensor(scale, dtype=torch.float32).detach().clone()
        if mean.ndim != 1 or scale.ndim != 1 or mean.shape != scale.shape:
            raise ValueError("Latent standardization mean and scale must be equally sized 1D tensors.")
        if mean.numel() == 0:
            raise ValueError("Latent standardization statistics cannot be empty.")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("Latent standardization statistics must be finite.")
        if (scale <= 0).any():
            raise ValueError("Latent standardization scale must be strictly positive.")
        if epsilon <= 0:
            raise ValueError("Latent standardization epsilon must be strictly positive.")

        # These values live in checkpoint metadata, not the denoiser state dict. Keeping
        # the buffers non-persistent lets legacy denoiser checkpoints load strictly.
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.enabled = bool(enabled)
        self.epsilon = float(epsilon)
        self.provenance = dict(provenance or {})

    @property
    def latent_dim(self) -> int:
        return int(self.mean.numel())

    @classmethod
    def identity(cls, latent_dim: int) -> "LatentStandardizer":
        if latent_dim <= 0:
            raise ValueError("latent_dim must be strictly positive.")
        return cls(
            mean=torch.zeros(latent_dim),
            scale=torch.ones(latent_dim),
            enabled=False,
        )

    @classmethod
    def fit(
        cls,
        embeddings: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        epsilon: float = 1e-6,
        provenance: Mapping[str, Any] | None = None,
    ) -> "LatentStandardizer":
        values = torch.as_tensor(embeddings).detach().to(device="cpu")
        if values.ndim not in (2, 3) or values.size(0) == 0 or values.size(-1) == 0:
            raise ValueError(
                "Latent standardization requires non-empty [samples, features] or "
                "[samples, sequence, features] embeddings."
            )
        if epsilon <= 0:
            raise ValueError("Latent standardization epsilon must be strictly positive.")

        if values.ndim == 2:
            if attention_mask is not None:
                raise ValueError("attention_mask is only valid for sequence embeddings.")
            values = values.unsqueeze(1)
            mask = torch.ones(values.shape[:2], dtype=torch.bool)
        else:
            if attention_mask is None:
                mask = torch.ones(values.shape[:2], dtype=torch.bool)
            else:
                mask = torch.as_tensor(attention_mask, dtype=torch.bool, device="cpu")
                if mask.shape != values.shape[:2]:
                    raise ValueError(
                        f"Attention-mask shape {tuple(mask.shape)} does not match sequence "
                        f"embedding shape {tuple(values.shape[:2])}."
                    )

        feature_sum = torch.zeros(values.size(-1), dtype=torch.float64)
        feature_square_sum = torch.zeros_like(feature_sum)
        valid_count = 0
        # Avoid materializing the full half-precision sequence cache in float32.
        for start in range(0, values.size(0), 64):
            end = min(start + 64, values.size(0))
            chunk = values[start:end].to(dtype=torch.float32)
            chunk_mask = mask[start:end].unsqueeze(-1)
            if not torch.isfinite(chunk).all():
                raise ValueError("Cannot fit latent standardization to non-finite embeddings.")
            feature_sum += (chunk * chunk_mask).sum(dim=(0, 1), dtype=torch.float64)
            feature_square_sum += (
                chunk.square() * chunk_mask
            ).sum(dim=(0, 1), dtype=torch.float64)
            valid_count += int(chunk_mask.sum().item())

        if valid_count == 0:
            raise ValueError("Latent standardization received no unmasked tokens.")
        mean64 = feature_sum / valid_count
        variance64 = (feature_square_sum / valid_count - mean64.square()).clamp_min(0.0)
        mean = mean64.to(dtype=torch.float32)
        scale = variance64.sqrt().to(dtype=torch.float32).clamp_min(epsilon)
        return cls(
            mean=mean,
            scale=scale,
            enabled=True,
            epsilon=epsilon,
            provenance=provenance,
        )

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any] | None,
        *,
        latent_dim: int,
    ) -> "LatentStandardizer":
        # Checkpoints created before latent standardization intentionally retain their
        # original raw-coordinate behavior.
        if not metadata:
            return cls.identity(latent_dim)

        version = int(metadata.get("version", cls.VERSION))
        if version != cls.VERSION:
            raise ValueError(f"Unsupported latent standardization version: {version}.")

        stored_dim = int(metadata.get("latent_dim", latent_dim))
        if stored_dim != latent_dim:
            raise ValueError(
                f"Latent standardization dimension {stored_dim} does not match model dimension {latent_dim}."
            )

        kind = metadata.get("kind", cls.IDENTITY_KIND)
        if kind == cls.IDENTITY_KIND:
            return cls.identity(latent_dim)
        if kind != cls.STANDARD_KIND:
            raise ValueError(f"Unsupported latent standardization kind: {kind!r}.")

        if "mean" not in metadata or "scale" not in metadata:
            raise ValueError("Standardized latent metadata must contain both mean and scale.")
        standardizer = cls(
            mean=metadata["mean"],
            scale=metadata["scale"],
            enabled=True,
            epsilon=float(metadata.get("epsilon", 1e-6)),
            provenance=metadata.get("provenance"),
        )
        if standardizer.latent_dim != latent_dim:
            raise ValueError(
                f"Latent standardization dimension {standardizer.latent_dim} "
                f"does not match model dimension {latent_dim}."
            )
        return standardizer

    def to_metadata(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "version": self.VERSION,
                "kind": self.IDENTITY_KIND,
                "latent_dim": self.latent_dim,
            }
        return {
            "version": self.VERSION,
            "kind": self.STANDARD_KIND,
            "latent_dim": self.latent_dim,
            "epsilon": self.epsilon,
            "mean": self.mean.detach().cpu().clone(),
            "scale": self.scale.detach().cpu().clone(),
            "provenance": dict(self.provenance),
        }

    def _validate_latents(self, latents: torch.Tensor) -> None:
        if not latents.is_floating_point():
            raise TypeError("Latent standardization requires a floating-point tensor.")
        if latents.size(-1) != self.latent_dim:
            raise ValueError(
                f"Latent feature dimension {latents.size(-1)} does not match "
                f"standardization dimension {self.latent_dim}."
            )

    def standardize(self, latents: torch.Tensor) -> torch.Tensor:
        self._validate_latents(latents)
        if not self.enabled:
            return latents
        mean = self.mean.to(device=latents.device, dtype=latents.dtype)
        scale = self.scale.to(device=latents.device, dtype=latents.dtype)
        return (latents - mean) / scale

    def destandardize(self, latents: torch.Tensor) -> torch.Tensor:
        self._validate_latents(latents)
        if not self.enabled:
            return latents
        mean = self.mean.to(device=latents.device, dtype=latents.dtype)
        scale = self.scale.to(device=latents.device, dtype=latents.dtype)
        return latents * scale + mean

    def is_compatible_with(
        self,
        other: "LatentStandardizer",
        *,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> bool:
        if self.latent_dim != other.latent_dim or self.enabled != other.enabled:
            return False
        if not self.enabled:
            return True
        return bool(
            torch.allclose(self.mean.detach().cpu(), other.mean.detach().cpu(), atol=atol, rtol=rtol)
            and torch.allclose(self.scale.detach().cpu(), other.scale.detach().cpu(), atol=atol, rtol=rtol)
        )

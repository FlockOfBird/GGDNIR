from __future__ import annotations

from collections.abc import Mapping
import hashlib
from typing import Any

import torch
from torch import nn


class LatentPCAProjector(nn.Module):
    """Frozen PCA map from encoder coordinates to a smaller token space."""

    VERSION = 1
    KIND = "pca"

    def __init__(
        self,
        input_mean: torch.Tensor,
        components: torch.Tensor,
        *,
        explained_variance_ratio: torch.Tensor | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        input_mean = torch.as_tensor(input_mean, dtype=torch.float32).detach().clone()
        components = torch.as_tensor(components, dtype=torch.float32).detach().clone()
        if input_mean.ndim != 1:
            raise ValueError("PCA input mean must be one-dimensional.")
        if components.ndim != 2 or components.size(1) != input_mean.numel():
            raise ValueError("PCA components must have shape [output_dim, input_dim].")
        if components.size(0) <= 0 or components.size(0) >= components.size(1):
            raise ValueError("PCA output_dim must be positive and smaller than input_dim.")
        if not torch.isfinite(input_mean).all() or not torch.isfinite(components).all():
            raise ValueError("PCA projection parameters must be finite.")
        if explained_variance_ratio is None:
            explained_variance_ratio = torch.empty(0, dtype=torch.float32)
        self.register_buffer("input_mean", input_mean, persistent=False)
        self.register_buffer("components", components, persistent=False)
        self.register_buffer(
            "explained_variance_ratio",
            torch.as_tensor(explained_variance_ratio, dtype=torch.float32)
            .detach()
            .clone(),
            persistent=False,
        )
        self.provenance = dict(provenance or {})

    @property
    def input_dim(self) -> int:
        return int(self.components.size(1))

    @property
    def output_dim(self) -> int:
        return int(self.components.size(0))

    def project(self, latents: torch.Tensor) -> torch.Tensor:
        if not latents.is_floating_point():
            raise TypeError("PCA projection requires a floating-point tensor.")
        if latents.size(-1) != self.input_dim:
            raise ValueError(
                f"Projection input dimension {latents.size(-1)} does not match "
                f"the fitted dimension {self.input_dim}."
            )
        mean = self.input_mean.to(device=latents.device, dtype=latents.dtype)
        components = self.components.to(device=latents.device, dtype=latents.dtype)
        return torch.matmul(latents - mean, components.transpose(0, 1))

    @classmethod
    def fit(
        cls,
        embeddings: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        output_dim: int,
        max_fit_tokens: int,
        seed: int,
        device: torch.device,
        provenance: Mapping[str, Any] | None = None,
    ) -> "LatentPCAProjector":
        values = torch.as_tensor(embeddings).detach().cpu()
        mask = torch.as_tensor(attention_mask, dtype=torch.bool).detach().cpu()
        if values.ndim != 3 or mask.shape != values.shape[:2]:
            raise ValueError(
                "PCA fitting requires [samples, sequence, features] embeddings "
                "and a matching attention mask."
            )
        input_dim = int(values.size(-1))
        if output_dim <= 0 or output_dim >= input_dim:
            raise ValueError(
                f"Projection dimension must be between 1 and {input_dim - 1}."
            )
        if max_fit_tokens <= output_dim:
            raise ValueError("PCA fit token count must exceed the output dimension.")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        row_order = torch.randperm(values.size(0), generator=generator)
        sampled_chunks: list[torch.Tensor] = []
        sampled_count = 0
        for start in range(0, row_order.numel(), 256):
            indices = row_order[start : start + 256]
            chunk_tokens = values[indices].float()[mask[indices]]
            remaining = max_fit_tokens - sampled_count
            if chunk_tokens.size(0) > remaining:
                token_order = torch.randperm(
                    chunk_tokens.size(0), generator=generator
                )[:remaining]
                chunk_tokens = chunk_tokens[token_order]
            sampled_chunks.append(chunk_tokens)
            sampled_count += int(chunk_tokens.size(0))
            if sampled_count >= max_fit_tokens:
                break
        sample = torch.cat(sampled_chunks, dim=0)
        if sample.size(0) <= output_dim:
            raise ValueError("Too few valid training tokens were available for PCA.")

        input_mean = sample.mean(dim=0)
        centered = (sample - input_mean).to(device)
        q = min(output_dim + 16, input_dim, sample.size(0) - 1)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        _, singular_values, right_vectors = torch.pca_lowrank(
            centered, q=q, center=False, niter=3
        )
        components = right_vectors[:, :output_dim].transpose(0, 1).cpu()
        explained = singular_values[:output_dim].square().cpu()
        total_variance = centered.square().sum().detach().cpu().clamp_min(1e-12)
        return cls(
            input_mean=input_mean,
            components=components,
            explained_variance_ratio=explained / total_variance,
            provenance=provenance,
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "kind": self.KIND,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "input_mean": self.input_mean.detach().cpu().clone(),
            "components": self.components.detach().cpu().clone(),
            "explained_variance_ratio": self.explained_variance_ratio.detach()
            .cpu()
            .clone(),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "LatentPCAProjector":
        if int(metadata.get("version", cls.VERSION)) != cls.VERSION:
            raise ValueError("Unsupported latent PCA projection version.")
        if metadata.get("kind") != cls.KIND:
            raise ValueError(f"Unsupported latent projection: {metadata.get('kind')!r}.")
        projector = cls(
            input_mean=metadata["input_mean"],
            components=metadata["components"],
            explained_variance_ratio=metadata.get("explained_variance_ratio"),
            provenance=metadata.get("provenance"),
        )
        if projector.input_dim != int(metadata["input_dim"]):
            raise ValueError("Stored PCA input dimension is inconsistent.")
        if projector.output_dim != int(metadata["output_dim"]):
            raise ValueError("Stored PCA output dimension is inconsistent.")
        return projector

    def is_compatible_with(
        self,
        other: "LatentPCAProjector",
        *,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> bool:
        return bool(
            self.input_dim == other.input_dim
            and self.output_dim == other.output_dim
            and torch.allclose(
                self.input_mean.cpu(), other.input_mean.cpu(), atol=atol, rtol=rtol
            )
            and torch.allclose(
                self.components.cpu(), other.components.cpu(), atol=atol, rtol=rtol
            )
        )



class LearnedTokenProjector(nn.Module):
    """Trainable token-wise map into the diffusion coordinate space."""

    VERSION = 1
    KIND = "learned_token_linear"

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Projection dimensions must be positive.")
        if output_dim >= input_dim:
            raise ValueError("Learned projection output_dim must be smaller than input_dim.")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_norm = nn.LayerNorm(self.input_dim, elementwise_affine=False)
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(
        self,
        latents: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents.size(-1) != self.input_dim:
            raise ValueError(
                f"Projection input dimension {latents.size(-1)} does not match "
                f"the configured dimension {self.input_dim}."
            )
        projected = self.output_norm(self.linear(self.input_norm(latents)))
        if attention_mask is not None:
            if attention_mask.shape != latents.shape[:-1]:
                raise ValueError("Projection attention mask does not match the latent shape.")
            projected = projected * attention_mask.unsqueeze(-1).to(projected.dtype)
        return projected

    def project_pooled(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.forward(embeddings)

    def state_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def to_metadata(self, provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "kind": self.KIND,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "architecture": "layer_norm-linear-layer_norm",
            "state_fingerprint": self.state_fingerprint(),
            "provenance": dict(provenance or {}),
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        *,
        device: torch.device | None = None,
    ) -> "LearnedTokenProjector":
        metadata = checkpoint.get("metadata", checkpoint)
        if metadata.get("kind") != cls.KIND:
            raise ValueError(f"Unsupported learned projection: {metadata.get('kind')!r}.")
        model = cls(int(metadata["input_dim"]), int(metadata["output_dim"]))
        state = checkpoint.get("state_dict")
        if state is None:
            raise ValueError("Learned projector checkpoint is missing state_dict.")
        model.load_state_dict(state)
        if metadata.get("state_fingerprint") != model.state_fingerprint():
            raise ValueError("Learned projector checkpoint fingerprint does not match its weights.")
        if device is not None:
            model = model.to(device)
        return model


def projected_coordinate_space(output_dim: int) -> str:
    return f"standardized_chemberta_pca{int(output_dim)}_per_feature_v1"


def learned_coordinate_space(output_dim: int) -> str:
    return f"learned_encoder_projection_{int(output_dim)}_layernorm_v1"


def projection_metadata_compatible(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if left.get("kind") != right.get("kind"):
        return False
    if left.get("kind") == LatentPCAProjector.KIND:
        return LatentPCAProjector.from_metadata(left).is_compatible_with(
            LatentPCAProjector.from_metadata(right)
        )
    if left.get("kind") == LearnedTokenProjector.KIND:
        fields = ("version", "kind", "input_dim", "output_dim", "architecture")
        if any(left.get(field) != right.get(field) for field in fields):
            return False
        left_fingerprint = left.get("state_fingerprint")
        right_fingerprint = right.get("state_fingerprint")
        return bool(left_fingerprint and left_fingerprint == right_fingerprint)
    raise ValueError(f"Unsupported latent projection: {left.get('kind')!r}.")


def projection_json_summary(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return JSON-safe projection provenance without large tensors."""
    if metadata is None:
        return None
    if metadata.get("kind") == LearnedTokenProjector.KIND:
        return {
            key: metadata.get(key)
            for key in (
                "version",
                "kind",
                "input_dim",
                "output_dim",
                "architecture",
                "state_fingerprint",
                "provenance",
            )
        }
    projector = LatentPCAProjector.from_metadata(metadata)
    return {
        "version": projector.VERSION,
        "kind": projector.KIND,
        "input_dim": projector.input_dim,
        "output_dim": projector.output_dim,
        "explained_variance_ratio_sum": float(
            projector.explained_variance_ratio.sum().item()
        ),
        "provenance": dict(projector.provenance),
    }


def project_embedding_payload(
    payload: dict[str, Any],
    projector: LatentPCAProjector,
    *,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Project all cached token and pooled embeddings without changing sample order."""
    if int(payload["latent_dim"]) != projector.input_dim:
        raise ValueError(
            f"Embedding-cache dimension {payload['latent_dim']} does not match "
            f"PCA input dimension {projector.input_dim}."
        )
    projected = dict(payload)
    projected_splits: dict[str, dict[str, Any]] = {}
    for split_name, split_payload in payload["splits"].items():
        split = dict(split_payload)
        sequence = split["sequence_embeddings"]
        mask = split["attention_mask"]
        sequence_chunks: list[torch.Tensor] = []
        pooled_chunks: list[torch.Tensor] = []
        for start in range(0, sequence.size(0), batch_size):
            stop = start + batch_size
            projected_sequence = projector.project(sequence[start:stop].float())
            projected_sequence = projected_sequence * mask[start:stop].unsqueeze(-1)
            sequence_chunks.append(projected_sequence.to(sequence.dtype))
            pooled = split["embeddings"][start:stop]
            pooled_chunks.append(projector.project(pooled.float()).to(pooled.dtype))
        split["sequence_embeddings"] = torch.cat(sequence_chunks, dim=0)
        split["embeddings"] = torch.cat(pooled_chunks, dim=0)
        projected_splits[split_name] = split
    projected["splits"] = projected_splits
    projected["encoder_latent_dim"] = projector.input_dim
    projected["latent_dim"] = projector.output_dim
    projected["latent_projection"] = projector.to_metadata()
    projected.pop("latent_standardization", None)
    projected.pop("latent_coordinate_space", None)
    return projected

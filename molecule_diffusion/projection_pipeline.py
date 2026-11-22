from __future__ import annotations

from pathlib import Path

import torch

from .config import EncoderConfig, LatentProjectionConfig
from .projection import (
    LatentPCAProjector,
    project_embedding_payload,
    projection_metadata_compatible,
)
from .utils import ensure_dir


def _load_projector(path: Path) -> LatentPCAProjector:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("latent_projection", payload)
    return LatentPCAProjector.from_metadata(metadata)


def _validate_projector(
    projector: LatentPCAProjector,
    config: LatentProjectionConfig,
    encoder_config: EncoderConfig,
) -> None:
    if projector.output_dim != config.output_dim:
        raise ValueError(
            f"Stored PCA dimension is {projector.output_dim}, but "
            f"--latent-projection-dim is {config.output_dim}. Use a different "
            "--cache-dir or select the stored dimension."
        )
    stored_encoder = projector.provenance.get("encoder_config")
    if isinstance(stored_encoder, dict) and stored_encoder != encoder_config.to_dict():
        raise ValueError(
            "The stored PCA projector was fitted with a different encoder "
            "configuration. Use a separate --cache-dir."
        )


def prepare_projection(
    payload: dict,
    *,
    config: LatentProjectionConfig,
    encoder_config: EncoderConfig,
    projector_path: Path,
    device: torch.device,
    allow_fit: bool,
) -> tuple[dict, bool]:
    """Apply or validate the requested projection and report whether data changed."""
    existing_metadata = payload.get("latent_projection")
    if not config.enabled:
        if existing_metadata is not None:
            raise ValueError(
                "This cache contains projected latents, but projection is disabled. "
                "Pass --enable-latent-projection with the matching options, or use "
                "the original 768-D cache directory."
            )
        return payload, False

    if existing_metadata is not None:
        cached_projector = LatentPCAProjector.from_metadata(existing_metadata)
        _validate_projector(cached_projector, config, encoder_config)
        if projector_path.exists():
            stored_projector = _load_projector(projector_path)
            if not projection_metadata_compatible(
                cached_projector.to_metadata(), stored_projector.to_metadata()
            ):
                raise ValueError(
                    "The embedding cache and latent_projector.pt use different PCA bases."
                )
        else:
            ensure_dir(projector_path.parent)
            torch.save(
                {"latent_projection": cached_projector.to_metadata()}, projector_path
            )
        return payload, False

    if projector_path.exists():
        projector = _load_projector(projector_path)
        _validate_projector(projector, config, encoder_config)
    elif allow_fit:
        train = payload["splits"]["train"]
        projector = LatentPCAProjector.fit(
            train["sequence_embeddings"],
            attention_mask=train["attention_mask"],
            output_dim=config.output_dim,
            max_fit_tokens=config.fit_max_tokens,
            seed=config.seed,
            device=device,
            provenance={
                "fit_dataset": "generative",
                "fit_split": "train",
                "encoder_config": encoder_config.to_dict(),
                "latent_representation": payload.get("latent_representation"),
                "fit_max_tokens": config.fit_max_tokens,
                "seed": config.seed,
            },
        )
        ensure_dir(projector_path.parent)
        torch.save({"latent_projection": projector.to_metadata()}, projector_path)
    else:
        raise FileNotFoundError(
            f"PCA projector not found at {projector_path}. Prepare the generative "
            "embeddings first with the same projection flags."
        )

    return project_embedding_payload(payload, projector), True

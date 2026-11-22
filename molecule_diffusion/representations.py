"""Latent-representation contracts shared by caches and checkpoints."""
from __future__ import annotations

from typing import Any

import torch


TOKEN_SEQUENCE = "chemberta_token_sequence_v1"
MEAN_POOLED = "molecule_mean_pooled_v1"
SUPPORTED_REPRESENTATIONS = frozenset({TOKEN_SEQUENCE, MEAN_POOLED})

CLI_TO_REPRESENTATION = {
    "token_sequence": TOKEN_SEQUENCE,
    "mean_pooled": MEAN_POOLED,
}


def representation_from_cli(value: str) -> str:
    try:
        return CLI_TO_REPRESENTATION[value]
    except KeyError as error:
        raise ValueError(f"Unsupported latent representation: {value!r}") from error


def validate_representation(value: object, *, context: str) -> str:
    if value not in SUPPORTED_REPRESENTATIONS:
        raise ValueError(f"{context} uses unsupported latent representation {value!r}.")
    return str(value)


def mean_pool_embedding_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert an unprojected token cache into one vector per molecule.

    The source cache already stores the encoder's mask-aware pooled embedding in
    ``embeddings``. The converted diffusion view represents that vector as a
    length-one sequence so existing denoisers, decoders, and samplers can retain
    their batch/sequence/feature interface without preserving token structure.
    """
    source_representation = validate_representation(
        payload.get("latent_representation"), context="Source embedding cache"
    )
    if source_representation != TOKEN_SEQUENCE:
        raise ValueError("Mean pooling requires an unpooled token-sequence cache.")
    encoder_config = payload.get("encoder_config", {})
    if encoder_config.get("pooling", "mean") != "mean":
        raise ValueError(
            "The source cache must use attention-mask-aware mean pooling, not CLS pooling."
        )
    if payload.get("latent_projection") is not None:
        raise ValueError(
            "Mean pooling must happen before PCA or learned projection; use the raw cache."
        )

    converted = dict(payload)
    converted_splits: dict[str, dict[str, Any]] = {}
    latent_dim = int(payload["latent_dim"])
    for split_name, split_payload in payload["splits"].items():
        split = dict(split_payload)
        pooled = torch.as_tensor(split["embeddings"], dtype=torch.float32)
        if pooled.ndim != 2 or pooled.size(1) != latent_dim:
            raise ValueError(
                f"Split {split_name!r} pooled embeddings must have shape [N, {latent_dim}]."
            )
        split["sequence_embeddings"] = pooled.unsqueeze(1)
        split["attention_mask"] = torch.ones(
            pooled.size(0), 1, dtype=torch.bool
        )
        converted_splits[split_name] = split

    converted["splits"] = converted_splits
    converted["sequence_length"] = 1
    converted["latent_representation"] = MEAN_POOLED
    converted["representation_transform"] = {
        "kind": "attention_masked_mean_pool",
        "source_latent_representation": TOKEN_SEQUENCE,
        "pool_before_projection": True,
    }
    converted.pop("latent_standardization", None)
    converted.pop("latent_coordinate_space", None)
    return converted


def coordinate_space_name(
    latent_representation: str,
    latent_dim: int,
    *,
    projection_kind: str | None,
) -> str:
    validate_representation(latent_representation, context="Coordinate space")
    if latent_representation == TOKEN_SEQUENCE:
        if projection_kind == "pca":
            return f"standardized_chemberta_pca{latent_dim}_per_feature_v1"
        if projection_kind == "learned":
            return f"learned_encoder_projection_{latent_dim}_layernorm_v1"
        return "standardized_chemberta_per_feature_v1"
    if projection_kind == "pca":
        return f"standardized_molecule_mean_pooled_pca{latent_dim}_per_feature_v1"
    if projection_kind == "learned":
        return f"learned_molecule_mean_pooled_projection_{latent_dim}_layernorm_v1"
    return f"standardized_molecule_mean_pooled_{latent_dim}_per_feature_v1"

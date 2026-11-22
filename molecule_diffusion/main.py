from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from .benchmark import command_estimate_diffusion_time
from .classifier import BBBPClassifier, PropertyRegressor
from .config import (
    ClassifierConfig,
    DataConfig,
    DecoderConfig,
    DiffusionConfig,
    EncoderConfig,
    GuidanceClassifierConfig,
    LatentProjectionConfig,
    SamplingConfig,
    TrainingConfig,
)
from .data import build_data_bundle, create_tensor_loader, data_split_schema_version
from .decoder import EmbeddingToSelfiesDecoder
from .diffusion import DenoiserModel, DiffusionSchedule, build_denoiser
from .encoder import PretrainedMoleculeEncoder, encode_splits
from .evaluation import generation_metrics, save_property_distribution_plots
from .latent import LatentStandardizer
from .projection import (
    LearnedTokenProjector,
    projection_json_summary,
    projection_metadata_compatible,
)
from .projection_pipeline import prepare_projection
from .representations import (
    MEAN_POOLED,
    TOKEN_SEQUENCE,
    coordinate_space_name,
    representation_from_cli,
    validate_representation,
)
from .property_head import NoisyLatentPropertyClassifier
from .property_head import NoisyLatentPropertyRegressor
from .properties import SUPPORTED_PROPERTIES, normalize_property_name, property_column
from .sampling import sample_smiles
from .sampling_experiment import (
    create_sampling_experiment,
    latest_sampling_experiment,
    record_latest_sampling_experiment,
    sampling_experiment_paths,
    update_experiment_manifest,
    utc_timestamp,
)
from .tokenizer import (
    SelfiesVocabulary,
    encode_selfies_batch,
    load_encoder_tokenizer,
    split_selfies_tokens,
    smiles_to_selfies,
)
from .trainer import (
    load_checkpoint,
    save_model_checkpoint,
    train_classifier,
    train_decoder,
    train_diffusion,
    train_guidance_classifier,
    train_guidance_regressor,
    train_learned_projection_diffusion,
    train_property_regressor,
)
from .utils import ensure_dir, resolve_device, save_json, set_seed


def artifact_paths(
    artifact_dir: str,
    cache_dir: str | None = None,
    latent_projection_config: LatentProjectionConfig | None = None,
) -> dict[str, Path]:
    root = Path(artifact_dir)
    cache_root = Path(cache_dir) if cache_dir else root
    checkpoints = root / "checkpoints"
    return {
        "root": root,
        "cache_root": cache_root,
        "latent_projection_config": latent_projection_config or LatentProjectionConfig(),
        "checkpoints": checkpoints,
        "config": root / "run_config.json",
        "embedding_cache": cache_root / "embedding_cache.pt",
        "generative_embedding_cache": cache_root / "generative_embedding_cache.pt",
        "latent_projector": cache_root / "latent_projector.pt",
        "learned_projector": checkpoints / "learned_projector.pt",
        "decoder": checkpoints / "decoder.pt",
        "diffusion": checkpoints / "diffusion.pt",
        "classifier": checkpoints / "classifier.pt",
        "guidance_classifier": checkpoints / "guidance_classifier.pt",
        "property_regressor": checkpoints / "property_regressor.pt",
        "guidance_regressor": checkpoints / "guidance_regressor.pt",
        "sampling_experiments": root / "experiments" / "sampling",
        "metrics": root / "metrics.json",
    }


def write_metrics(paths: dict[str, Path], section: str, payload: dict) -> None:
    existing = {}
    if paths["metrics"].exists():
        try:
            existing = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing[section] = payload
    save_json(paths["metrics"], existing)


def _format_report_value(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_training_report(stage: str, metrics: dict) -> None:
    preferred_order = [
        "epoch",
        "train_loss",
        "train_rmse",
        "train_mae",
        "train_auc",
        "train_clean_loss",
        "train_robust_loss",
        "train_token_accuracy",
        "val_loss",
        "val_rmse",
        "val_mae",
        "val_auc",
        "test_loss",
        "test_rmse",
        "test_mae",
        "test_auc",
        "val_clean_loss",
        "val_robust_loss",
        "val_token_accuracy",
    ]
    scalar_metrics = {
        key for key, value in metrics.items() if not isinstance(value, (dict, list))
    }
    metric_keys = [key for key in preferred_order if key in scalar_metrics]
    metric_keys.extend(key for key in metrics if key not in metric_keys and key in scalar_metrics)

    print(f"\n{stage} training report")
    print("-" * (len(stage) + len(" training report")))
    for key in metric_keys:
        label = "epochs" if key == "epoch" else key
        print(f"{label}: {_format_report_value(metrics[key])}")
    print()


def _split_size(split_payload: dict) -> int:
    for key in ["embeddings", "input_ids", "smiles", "labels", "selfies_ids"]:
        if key in split_payload:
            return len(split_payload[key])
    return 0


def print_data_report(stage: str, payload: dict) -> None:
    splits = payload.get("splits", {})
    split_names = [name for name in ["train", "val", "test"] if name in splits]
    split_names.extend(name for name in splits if name not in split_names)
    split_sizes = {name: _split_size(splits[name]) for name in split_names}
    total_size = sum(split_sizes.values())

    print(f"\n{stage} data report")
    print("-" * (len(stage) + len(" data report")))
    for name in split_names:
        print(f"{name}_size: {split_sizes[name]}")
    print(f"total_size: {total_size}")
    print()


def make_configs(
    args,
) -> tuple[DataConfig, EncoderConfig, DecoderConfig, DiffusionConfig, ClassifierConfig, GuidanceClassifierConfig, TrainingConfig]:
    data_config = DataConfig(
        dataset_path=args.dataset_path,
        smiles_column=args.property_smiles_column,
        label_column=property_column(args.property_name),
        property_name=normalize_property_name(args.property_name),
        task_type="regression",
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
        split_strategy=args.split_strategy,
        canonicalize_smiles=not args.disable_canonicalize,
    )
    encoder_config = EncoderConfig(
        model_name=args.encoder_name,
        pooling=args.pooling,
        freeze_encoder=not args.unfreeze_encoder,
        local_files_only=args.local_files_only,
    )
    decoder_config = DecoderConfig(
        d_model=args.decoder_d_model,
        num_layers=args.decoder_layers,
        num_heads=args.decoder_heads,
        dim_feedforward=args.decoder_ffn_dim,
        dropout=args.decoder_dropout,
        latent_noise_std=args.decoder_latent_noise_std,
        robust_loss_weight=args.decoder_robust_loss_weight,
        max_decode_length=args.max_length,
    )
    diffusion_config = DiffusionConfig(
        denoiser_architecture="mlp",
        hidden_dim=args.diffusion_hidden_dim,
        time_embedding_dim=args.time_embedding_dim,
        max_sequence_length=args.max_length,
        num_steps=args.diffusion_steps,
        beta_schedule=args.beta_schedule,
        dropout=args.diffusion_dropout,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_token_hidden_dim=args.mlp_token_hidden_dim,
        mlp_num_layers=8,
    )
    classifier_config = ClassifierConfig(
        hidden_dim=args.classifier_hidden_dim,
        dropout=args.classifier_dropout,
    )
    guidance_classifier_config = GuidanceClassifierConfig(
        hidden_dim=args.guidance_classifier_hidden_dim,
        dropout=args.guidance_classifier_dropout,
        time_embedding_dim=args.guidance_classifier_time_embedding_dim,
        num_hidden_layers=args.guidance_classifier_layers,
        distill_from_classifier=args.distill_guidance_classifier,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        diffusion_learning_rate=args.diffusion_learning_rate,
        decoder_learning_rate=args.decoder_learning_rate,
        projection_learning_rate=args.projection_learning_rate,
        projection_reconstruction_weight=args.projection_reconstruction_weight,
        projection_variance_weight=args.projection_variance_weight,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        device=args.device,
        artifact_dir=args.artifact_dir,
        checkpoint_dir=str(Path(args.artifact_dir) / "checkpoints"),
    )
    return (
        data_config,
        encoder_config,
        decoder_config,
        diffusion_config,
        classifier_config,
        guidance_classifier_config,
        training_config,
    )


def make_latent_projection_config(args) -> LatentProjectionConfig:
    requested_mode = str(args.projection_mode)
    if args.enable_latent_projection:
        if requested_mode not in ("none", "pca"):
            raise ValueError(
                "--enable-latent-projection selects PCA and cannot be combined with "
                "--projection-mode learned."
            )
        requested_mode = "pca"
    config = LatentProjectionConfig(
        mode=requested_mode,
        enabled=requested_mode == "pca",
        output_dim=int(args.latent_projection_dim),
        fit_max_tokens=int(args.latent_projection_fit_tokens),
        seed=int(args.seed),
    )
    if config.enabled and not args.cache_dir:
        raise ValueError(
            "PCA projection requires a separate --cache-dir so the projected cache "
            "cannot overwrite the full-dimensional cache."
        )
    if config.output_dim <= 0:
        raise ValueError("--latent-projection-dim must be positive.")
    if config.enabled and config.fit_max_tokens <= config.output_dim:
        raise ValueError(
            "--latent-projection-fit-tokens must exceed --latent-projection-dim."
        )
    return config

def make_generative_data_config(args) -> DataConfig:
    return DataConfig(
        dataset_path=args.generative_dataset_path,
        smiles_column=args.generative_smiles_column,
        split_column=args.generative_split_column,
        label_column=None,
        property_name=None,
        task_type="regression",
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
        split_strategy=args.generative_split_strategy,
        canonicalize_smiles=not args.disable_canonicalize,
    )


def requested_latent_representation(args) -> str:
    return representation_from_cli(args.latent_representation)


def save_run_config(paths: dict[str, Path], payload: dict) -> None:
    ensure_dir(paths["root"])
    ensure_dir(paths["checkpoints"])
    save_json(paths["config"], payload)


def _embedding_data_config_matches(cached: object, expected: dict) -> bool:
    if not isinstance(cached, dict):
        return False
    ignored_runtime_fields = {"batch_size", "num_workers"}
    cached_semantics = {
        key: value for key, value in cached.items() if key not in ignored_runtime_fields
    }
    cached_semantics.setdefault("split_column", None)
    expected_semantics = {
        key: value for key, value in expected.items() if key not in ignored_runtime_fields
    }
    return cached_semantics == expected_semantics



def prepare_embeddings(
    cache_path: Path,
    data_config: DataConfig,
    encoder_config: EncoderConfig,
    device: torch.device,
    latent_representation: str = TOKEN_SEQUENCE,
) -> dict:
    dataset_sha256 = None
    if data_config.split_column is not None:
        digest = hashlib.sha256()
        with Path(data_config.dataset_path).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        dataset_sha256 = digest.hexdigest()
    validate_representation(latent_representation, context="Requested embedding cache")
    if latent_representation == MEAN_POOLED and encoder_config.pooling != "mean":
        raise ValueError(
            "The mean-pooled molecule representation requires --pooling mean."
        )
    expected_split_schema_version = data_split_schema_version(data_config)
    if cache_path.exists():
        cached_payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            cached_payload.get("decoder_target") == "SELFIES"
            and cached_payload.get("latent_representation") == latent_representation
            and all("sequence_embeddings" in split for split in cached_payload.get("splits", {}).values())
            and "selfies_vocabulary" in cached_payload
            and all("selfies_ids" in split_payload for split_payload in cached_payload.get("splits", {}).values())
            and _embedding_data_config_matches(cached_payload.get("data_config"), data_config.to_dict())
            and (dataset_sha256 is None or cached_payload.get("dataset_sha256") == dataset_sha256)
            and cached_payload.get("encoder_config") == encoder_config.to_dict()
            and cached_payload.get(
                "data_split_schema_version",
                1,
            ) == expected_split_schema_version
        ):
            return cached_payload

    splits = build_data_bundle(data_config)
    split_frames = {}
    preprocessing_counts = {}
    for split_name in ["train", "val", "test"]:
        frame = getattr(splits, split_name).copy()
        input_count = len(frame)
        frame["SELFIES"] = frame[data_config.smiles_column].map(smiles_to_selfies)
        frame = frame.dropna(subset=["SELFIES"]).reset_index(drop=True)
        split_frames[split_name] = frame
        preprocessing_counts[split_name] = {
            "input": input_count,
            "selfies_converted": len(frame),
            "selfies_failed": input_count - len(frame),
        }

    if data_config.split_column is not None:
        for split_name, frame in split_frames.items():
            overlength = sum(
                len(split_selfies_tokens(value)) + 2 > data_config.max_length
                for value in frame["SELFIES"]
            )
            if overlength:
                raise ValueError(
                    f"{overlength} benchmark {split_name} SELFIES exceed "
                    f"--max-length={data_config.max_length}; increase --max-length."
                )

    selfies_vocabulary = SelfiesVocabulary.build(
        split_frames["train"]["SELFIES"].astype(str).tolist(),
        max_length=data_config.max_length,
    )

    splits.train = split_frames["train"]
    splits.val = split_frames["val"]
    splits.test = split_frames["test"]
    tokenizer = load_encoder_tokenizer(encoder_config)
    encoder = PretrainedMoleculeEncoder(encoder_config).to(device)
    encoded_splits = encode_splits(
        splits, tokenizer, encoder, data_config, device,
        pooled_only=latent_representation == MEAN_POOLED,
    )
    latent_dim = encoder.latent_dim
    if latent_representation == MEAN_POOLED:
        sequence_length = 1
        representation_transform = {
            "kind": "attention_masked_mean_pool",
            "source_latent_representation": TOKEN_SEQUENCE,
            "pool_before_projection": True,
        }
    else:
        sequence_length = data_config.max_length
        representation_transform = None

    payload = {
        "encoder_config": encoder_config.to_dict(),
        "data_config": data_config.to_dict(),
        "dataset_sha256": dataset_sha256,
        "preprocessing_counts": preprocessing_counts,
        "data_split_schema_version": expected_split_schema_version,
        "latent_dim": latent_dim,
        "latent_representation": latent_representation,
        "sequence_length": sequence_length,
        "decoder_target": "SELFIES",
        "selfies_vocabulary": {
            "tokens": selfies_vocabulary.ordered_tokens,
            "max_length": selfies_vocabulary.max_length,
        },
        "splits": {
            name: (
                {
                    "smiles": split.smiles,
                    "selfies": split_frames[name]["SELFIES"].astype(str).tolist(),
                    "input_ids": split.input_ids,
                    "attention_mask": (
                        torch.ones(len(split.smiles), 1, dtype=torch.bool)
                        if latent_representation == MEAN_POOLED
                        else split.attention_mask
                    ),
                    "embeddings": split.embeddings,
                    "sequence_embeddings": (
                        split.embeddings.unsqueeze(1)
                        if latent_representation == MEAN_POOLED
                        else split.sequence_embeddings
                    ),
                    "selfies_ids": encode_selfies_batch(
                        split_frames[name]["SELFIES"].astype(str).tolist(),
                        selfies_vocabulary,
                    ),
                }
                | ({"labels": split.labels} if split.labels is not None else {})
            )
            for name, split in encoded_splits.items()
        },
    }
    if representation_transform is not None:
        payload["representation_transform"] = representation_transform
    torch.save(payload, cache_path)
    return payload


def _same_embedding_input(data_config: DataConfig, generative_data_config: DataConfig) -> bool:
    """Whether a property run can safely create the generative embedding source."""
    return (
        Path(data_config.dataset_path).resolve()
        == Path(generative_data_config.dataset_path).resolve()
        and data_config.smiles_column == generative_data_config.smiles_column
        and data_config.max_length == generative_data_config.max_length
        and data_config.canonicalize_smiles == generative_data_config.canonicalize_smiles
        and data_config.drop_invalid_smiles == generative_data_config.drop_invalid_smiles
    )


def _bind_property_to_shared_embeddings(
    shared_payload: dict,
    data_config: DataConfig,
    encoder_config: EncoderConfig,
) -> dict | None:
    """Attach labels to stored embeddings without writing another embedding tensor."""
    shared_representation = shared_payload.get("latent_representation", TOKEN_SEQUENCE)
    expected_sequence_length = (
        1 if shared_representation == MEAN_POOLED else data_config.max_length
    )
    cached_max_length = shared_payload.get("data_config", {}).get("max_length")
    if (
        data_config.label_column is None
        or shared_payload.get("encoder_config") != encoder_config.to_dict()
        or shared_payload.get("sequence_length") != expected_sequence_length
        or cached_max_length not in (None, data_config.max_length)
    ):
        return None

    source_splits = shared_payload["splits"]
    source_by_smiles: dict[str, tuple[str, int]] = {}
    for split_name, split_payload in source_splits.items():
        for index, smiles in enumerate(split_payload["smiles"]):
            if smiles in source_by_smiles:
                return None
            source_by_smiles[smiles] = (split_name, index)

    property_frames = build_data_bundle(data_config)
    bound_splits = {}
    tensor_keys = (
        "input_ids",
        "attention_mask",
        "embeddings",
        "sequence_embeddings",
        "selfies_ids",
    )
    for split_name in ("train", "val", "test"):
        frame = getattr(property_frames, split_name).copy()
        frame["SELFIES"] = frame[data_config.smiles_column].map(smiles_to_selfies)
        frame = frame.dropna(subset=["SELFIES"]).reset_index(drop=True)
        smiles = frame[data_config.smiles_column].astype(str).tolist()
        source_positions = [source_by_smiles.get(value) for value in smiles]
        if any(position is None for position in source_positions):
            return None
        labels = torch.tensor(
            frame[data_config.label_column].astype(float).tolist(), dtype=torch.float32
        )
        source_split = source_splits[split_name]
        if smiles == source_split["smiles"]:
            bound_split = dict(source_split)
        else:
            # Different property filtering or splitting can still reuse every matching
            # molecule. Only this in-memory training view needs gathered tensors.
            bound_split = {
                "smiles": smiles,
                "selfies": frame["SELFIES"].astype(str).tolist(),
            }
            for key in tensor_keys:
                template = source_split[key]
                gathered = torch.empty(
                    (len(smiles), *template.shape[1:]), dtype=template.dtype
                )
                for source_name, original in source_splits.items():
                    destinations = [
                        position
                        for position, location in enumerate(source_positions)
                        if location[0] == source_name
                    ]
                    if destinations:
                        original_rows = [source_positions[position][1] for position in destinations]
                        gathered[torch.tensor(destinations)] = original[key].index_select(
                            0, torch.tensor(original_rows)
                        )
                bound_split[key] = gathered
        bound_split["labels"] = labels
        bound_splits[split_name] = bound_split

    return {
        **shared_payload,
        "data_config": data_config.to_dict(),
        "data_split_schema_version": data_split_schema_version(data_config),
        "embedding_source": "generative_embedding_cache.pt",
        "splits": bound_splits,
    }


def prepare_property_embeddings(
    paths: dict[str, Path],
    data_config: DataConfig,
    encoder_config: EncoderConfig,
    device: torch.device,
    *,
    generative_data_config: DataConfig | None = None,
    generative_payload: dict | None = None,
    latent_representation: str = TOKEN_SEQUENCE,
) -> dict:
    ensure_dir(paths["root"])
    ensure_dir(paths["cache_root"])
    ensure_dir(paths["checkpoints"])

    if generative_payload is None and generative_data_config is not None:
        if paths["generative_embedding_cache"].exists() or _same_embedding_input(
            data_config, generative_data_config
        ):
            generative_payload = prepare_generative_embeddings(
                paths,
                generative_data_config,
                encoder_config,
                device,
                latent_representation=latent_representation,
            )

    payload = None
    if generative_payload is not None:
        payload = _bind_property_to_shared_embeddings(
            generative_payload, data_config, encoder_config
        )
    shared_embeddings = payload is not None
    projection_was_applied = False
    if payload is None:
        # A genuinely different property dataset may contain molecules absent from
        # the generative cache; retain the standalone cache for that case.
        payload = prepare_embeddings(
            paths["embedding_cache"],
            data_config,
            encoder_config,
            device,
            latent_representation=latent_representation,
        )
        payload, projection_was_applied = prepare_projection(
            payload,
            config=paths["latent_projection_config"],
            encoder_config=encoder_config,
            projector_path=paths["latent_projector"],
            device=device,
            allow_fit=False,
        )

    train_targets = payload["splits"]["train"]["labels"].float()
    target_mean = float(train_targets.mean())
    target_scale = float(train_targets.std(unbiased=False))
    if target_scale <= 1e-8:
        raise ValueError(
            f"Training targets for {data_config.property_name} have near-zero variance."
        )
    target_standardization = {
        "mean": target_mean,
        "scale": target_scale,
        "fit_split": "train",
        "property_name": data_config.property_name,
        "label_column": data_config.label_column,
    }
    if shared_embeddings:
        payload["target_standardization"] = target_standardization
    elif projection_was_applied or payload.get("target_standardization") != target_standardization:
        payload["target_standardization"] = target_standardization
        torch.save(payload, paths["embedding_cache"])
    return payload


def prepare_generative_embeddings(
    paths: dict[str, Path],
    generative_data_config: DataConfig,
    encoder_config: EncoderConfig,
    device: torch.device,
    latent_representation: str = TOKEN_SEQUENCE,
) -> dict:
    ensure_dir(paths["root"])
    ensure_dir(paths["cache_root"])
    ensure_dir(paths["checkpoints"])
    cache_path = paths["generative_embedding_cache"]
    payload = prepare_embeddings(
        cache_path,
        generative_data_config,
        encoder_config,
        device,
        latent_representation=latent_representation,
    )
    payload, _ = prepare_projection(
        payload,
        config=paths["latent_projection_config"],
        encoder_config=encoder_config,
        projector_path=paths["latent_projector"],
        device=device,
        allow_fit=True,
    )
    latent_dim = int(payload["latent_dim"])
    expected_coordinate_space = coordinate_space_name(
        latent_representation,
        latent_dim,
        projection_kind=(
            "pca" if payload.get("latent_projection") is not None else None
        ),
    )
    standardization_metadata = payload.get("latent_standardization")
    if standardization_metadata is None:
        train_payload = payload["splits"]["train"]
        latent_standardizer = LatentStandardizer.fit(
            train_payload["sequence_embeddings"],
            attention_mask=train_payload["attention_mask"],
            provenance={
                "fit_dataset": "generative",
                "latent_projection": payload.get("latent_projection"),
                "fit_split": "train",
                "data_config": generative_data_config.to_dict(),
                "encoder_config": encoder_config.to_dict(),
                "latent_representation": latent_representation,
            },
        )
        payload["latent_standardization"] = latent_standardizer.to_metadata()
        payload["latent_coordinate_space"] = expected_coordinate_space
        torch.save(payload, cache_path)
    else:
        latent_standardizer = LatentStandardizer.from_metadata(
            standardization_metadata,
            latent_dim=latent_dim,
        )
        if not latent_standardizer.enabled:
            raise ValueError(
                "The generative embedding cache does not contain enabled per-feature "
                "standardization. Delete the cache and prepare embeddings again."
            )
        coordinate_space = payload.get("latent_coordinate_space")
        if coordinate_space != expected_coordinate_space:
            payload["latent_coordinate_space"] = expected_coordinate_space
            torch.save(payload, cache_path)
    return payload


def _decoder_encoder_config(metadata: dict) -> dict:
    encoder_config = metadata.get("encoder_config")
    if isinstance(encoder_config, dict):
        return encoder_config
    # Decoder checkpoints created before coordinate provenance used the project
    # default mean pooling and recorded only the encoder model name.
    return {
        "model_name": metadata.get("encoder_model_name"),
        "pooling": metadata.get("encoder_pooling", "mean"),
    }


def _validate_decoder_coordinates(
    metadata: dict,
    *,
    expected_latent_dim: int | None = None,
    expected_encoder_config: dict | None = None,
    expected_standardizer: LatentStandardizer | None = None,
    expected_projection: dict | None = None,
    expected_latent_representation: str | None = None,
) -> None:
    checkpoint_representation = validate_representation(
        metadata.get("latent_representation"), context="Decoder checkpoint"
    )
    if (
        expected_latent_representation is not None
        and checkpoint_representation != expected_latent_representation
    ):
        raise ValueError(
            "Decoder checkpoint latent representation does not match the selected "
            "embedding cache. Retrain the decoder in this representation."
        )
    decoder_latent_dim = int(metadata["latent_dim"])
    if expected_latent_dim is not None and decoder_latent_dim != expected_latent_dim:
        raise ValueError(
            f"Decoder latent dimension {decoder_latent_dim} does not match expected dimension "
            f"{expected_latent_dim}."
        )
    coordinate_space = str(metadata.get("latent_coordinate_space", ""))
    learned_projection = (
        isinstance(metadata.get("latent_projection"), dict)
        and metadata["latent_projection"].get("kind") == LearnedTokenProjector.KIND
    )
    if not (
        coordinate_space.startswith("standardized_chemberta")
        or coordinate_space.startswith("learned_encoder_projection_")
        or coordinate_space.startswith("standardized_molecule_mean_pooled_")
        or coordinate_space.startswith("learned_molecule_mean_pooled_projection_")
    ):
        raise ValueError("Decoder checkpoint uses an unsupported latent coordinate space.")
    if not projection_metadata_compatible(
        metadata.get("latent_projection"), expected_projection
    ):
        raise ValueError(
            "Decoder latent projection does not match the selected embedding cache. "
            "Retrain the decoder in this latent space."
        )
    decoder_standardizer = LatentStandardizer.from_metadata(
        metadata.get("latent_standardization"),
        latent_dim=decoder_latent_dim,
    )
    if not decoder_standardizer.enabled and not learned_projection:
        raise ValueError(
            "Decoder checkpoint is missing enabled per-feature latent standardization. "
            "Retrain the decoder."
        )
    if expected_standardizer is not None and not decoder_standardizer.is_compatible_with(
        expected_standardizer
    ):
        raise ValueError(
            "Decoder latent standardization does not match the expected diffusion/cache "
            "standardization. Retrain the decoder with the current generative embedding cache."
        )
    if expected_encoder_config is not None:
        decoder_encoder_config = _decoder_encoder_config(metadata)
        encoder_fields = ("model_name", "pooling")
        mismatched_fields = [
            field
            for field in encoder_fields
            if decoder_encoder_config.get(field) != expected_encoder_config.get(field)
        ]
        if mismatched_fields:
            raise ValueError(
                "Decoder latent coordinates do not match the expected encoder configuration "
                f"({', '.join(mismatched_fields)}). Retrain the decoder or use a matching checkpoint."
            )


def instantiate_decoder(
    checkpoint_path: Path,
    device: torch.device,
    expected_latent_dim: int | None = None,
    expected_encoder_config: dict | None = None,
    expected_standardizer: LatentStandardizer | None = None,
    expected_projection: dict | None = None,
    expected_latent_representation: str | None = None,
) -> EmbeddingToSelfiesDecoder:
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = checkpoint["metadata"]
    _validate_decoder_coordinates(
        metadata,
        expected_latent_dim=expected_latent_dim,
        expected_projection=expected_projection,
        expected_encoder_config=expected_encoder_config,
        expected_standardizer=expected_standardizer,
        expected_latent_representation=expected_latent_representation,
    )
    vocabulary = SelfiesVocabulary.from_tokens(
        metadata["vocabulary_tokens"],
        max_length=metadata["max_decode_length"],
    )
    model = EmbeddingToSelfiesDecoder(
        latent_dim=metadata["latent_dim"],
        vocabulary=vocabulary,
        config=DecoderConfig(**metadata["model_config"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def instantiate_diffusion(checkpoint_path: Path, device: torch.device) -> tuple[DenoiserModel, DiffusionSchedule]:
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = checkpoint["metadata"]
    latent_representation = validate_representation(
        metadata.get("latent_representation"), context="Diffusion checkpoint"
    )
    config = DiffusionConfig(**metadata["model_config"])
    model = build_denoiser(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    latent_standardizer = LatentStandardizer.from_metadata(
        metadata.get("latent_standardization"),
        latent_dim=config.latent_dim,
    )
    schedule = DiffusionSchedule(
        config,
        latent_standardizer=latent_standardizer,
        sequence_length_counts=metadata.get("sequence_length_counts"),
    ).to(device)
    schedule.latent_projection = metadata.get("latent_projection")
    schedule.latent_coordinate_space = metadata.get("latent_coordinate_space")
    schedule.latent_representation = latent_representation
    return model, schedule



def instantiate_learned_projector(
    checkpoint_path: Path,
    device: torch.device,
    expected_projection: dict | None = None,
) -> LearnedTokenProjector:
    checkpoint = load_checkpoint(checkpoint_path)
    projector = LearnedTokenProjector.from_checkpoint(checkpoint, device=device)
    metadata = checkpoint["metadata"]
    if expected_projection is not None and not projection_metadata_compatible(
        metadata, expected_projection
    ):
        raise ValueError("Learned projector weights do not match the model checkpoint.")
    projector.eval()
    for parameter in projector.parameters():
        parameter.requires_grad_(False)
    return projector

def instantiate_classifier(checkpoint_path: Path, device: torch.device) -> BBBPClassifier:
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = checkpoint["metadata"]
    model = BBBPClassifier(
        latent_dim=metadata["latent_dim"],
        config=ClassifierConfig(**metadata["model_config"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def instantiate_property_regressor(
    checkpoint_path: Path,
    device: torch.device,
    expected_property_name: str | None = None,
    expected_projection: dict | None = None,
) -> PropertyRegressor:
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = checkpoint["metadata"]
    if metadata.get("task_type") != "regression":
        raise ValueError("Property checkpoint is not a continuous regression model.")
    property_name = normalize_property_name(metadata["property_name"])
    if (
        expected_property_name is not None
        and property_name != normalize_property_name(expected_property_name)
    ):
        raise ValueError(
            f"Property checkpoint predicts {property_name}, not {expected_property_name}."
        )
    if not projection_metadata_compatible(
        metadata.get("latent_projection"), expected_projection
    ):
        raise ValueError(
            "Property-regressor latent projection does not match the selected cache."
        )
    target = metadata["target_standardization"]
    model = PropertyRegressor(
        latent_dim=int(metadata["latent_dim"]),
        config=ClassifierConfig(**metadata["model_config"]),
        target_mean=float(target["mean"]),
        target_scale=float(target["scale"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def instantiate_guidance_classifier(
    checkpoint_path: Path,
    device: torch.device,
    expected_standardizer: LatentStandardizer | None = None,
    expected_projection: dict | None = None,
    expected_diffusion_config: DiffusionConfig | None = None,
) -> NoisyLatentPropertyClassifier:
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = checkpoint["metadata"]
    validate_representation(
        metadata.get("latent_representation"), context="Guidance checkpoint"
    )
    latent_dim = int(metadata["latent_dim"])
    if not projection_metadata_compatible(
        metadata.get("latent_projection"), expected_projection
    ):
        raise ValueError(
            "Guidance-classifier latent projection does not match diffusion."
        )
    if expected_standardizer is not None:
        guidance_standardizer = LatentStandardizer.from_metadata(
            metadata.get("latent_standardization"),
            latent_dim=latent_dim,
        )
        if not guidance_standardizer.is_compatible_with(expected_standardizer):
            raise ValueError(
                "Guidance-classifier latent standardization does not match the diffusion checkpoint. "
                "Retrain the guidance classifier with the current diffusion checkpoint."
            )
    if expected_diffusion_config is not None:
        guidance_schedule_payload = metadata.get("diffusion_schedule_config")
        if guidance_schedule_payload is None:
            # Legacy raw-space guidance checkpoints did not record their schedule.
            # Keep old identity-transform pairs loadable, but standardized pairs
            # must be fully attributable and verifiable.
            if expected_standardizer is not None and expected_standardizer.enabled:
                raise ValueError(
                    "Guidance-classifier checkpoint does not record its diffusion schedule. "
                    "Retrain the guidance classifier with the current diffusion checkpoint."
                )
        else:
            guidance_schedule_config = DiffusionConfig(**guidance_schedule_payload)
            schedule_fields = ("num_steps", "beta_schedule", "beta_start", "beta_end")
            mismatched_fields = [
                field
                for field in schedule_fields
                if getattr(guidance_schedule_config, field) != getattr(expected_diffusion_config, field)
            ]
            if mismatched_fields:
                raise ValueError(
                    "Guidance-classifier diffusion schedule does not match the diffusion checkpoint "
                    f"({', '.join(mismatched_fields)}). Retrain the guidance classifier with the "
                    "current diffusion checkpoint."
                )
    model_config = dict(metadata["model_config"])
    model_config.setdefault("num_hidden_layers", 1)
    model = NoisyLatentPropertyClassifier(
        latent_dim=latent_dim,
        config=GuidanceClassifierConfig(**model_config),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def instantiate_guidance_regressor(
    checkpoint_path: Path,
    device: torch.device,
    *,
    expected_property_name: str,
    expected_standardizer: LatentStandardizer | None = None,
    expected_projection: dict | None = None,
    expected_diffusion_config: DiffusionConfig | None = None,
) -> NoisyLatentPropertyRegressor:
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = checkpoint["metadata"]
    if metadata.get("task_type") != "regression":
        raise ValueError("Guidance checkpoint is not a continuous regression model.")
    property_name = normalize_property_name(metadata["property_name"])
    if property_name != normalize_property_name(expected_property_name):
        raise ValueError(
            f"Guidance checkpoint predicts {property_name}, not {expected_property_name}."
        )
    validate_representation(
        metadata.get("latent_representation"), context="Guidance regressor checkpoint"
    )
    latent_dim = int(metadata["latent_dim"])
    if not projection_metadata_compatible(
        metadata.get("latent_projection"), expected_projection
    ):
        raise ValueError(
            "Guidance-regressor latent projection does not match diffusion."
        )
    if expected_standardizer is not None:
        guidance_standardizer = LatentStandardizer.from_metadata(
            metadata.get("latent_standardization"), latent_dim=latent_dim
        )
        if not guidance_standardizer.is_compatible_with(expected_standardizer):
            raise ValueError(
                "Guidance-regressor latent standardization does not match diffusion."
            )
    if expected_diffusion_config is not None:
        schedule_payload = metadata.get("diffusion_schedule_config")
        if schedule_payload is None:
            raise ValueError("Guidance regressor is missing diffusion schedule metadata.")
        guidance_schedule = DiffusionConfig(**schedule_payload)
        schedule_fields = ("num_steps", "beta_schedule", "beta_start", "beta_end")
        mismatches = [
            field
            for field in schedule_fields
            if getattr(guidance_schedule, field)
            != getattr(expected_diffusion_config, field)
        ]
        if mismatches:
            raise ValueError(
                "Guidance-regressor diffusion schedule does not match diffusion "
                f"({', '.join(mismatches)})."
            )
    model_config = dict(metadata["model_config"])
    model_config.setdefault("num_hidden_layers", 1)
    target = metadata["target_standardization"]
    model = NoisyLatentPropertyRegressor(
        latent_dim=latent_dim,
        config=GuidanceClassifierConfig(**model_config),
        target_mean=float(target["mean"]),
        target_scale=float(target["scale"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def split_payload_to_loader(split_payload: dict, batch_size: int, shuffle: bool, num_workers: int = 0):
    return create_tensor_loader(split_payload, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def command_prepare_embeddings(args) -> None:
    configs = make_configs(args)
    data_config, encoder_config, decoder_config, diffusion_config, classifier_config, guidance_classifier_config, training_config = configs
    del decoder_config, diffusion_config, classifier_config, guidance_classifier_config
    generative_data_config = make_generative_data_config(args)
    device = resolve_device(training_config.device)
    set_seed(data_config.seed)
    paths = artifact_paths(training_config.artifact_dir, args.cache_dir, make_latent_projection_config(args))
    latent_representation = requested_latent_representation(args)
    generative_payload = prepare_generative_embeddings(
        paths,
        generative_data_config,
        encoder_config,
        device,
        latent_representation=latent_representation,
    )
    property_payload = prepare_property_embeddings(
        paths,
        data_config,
        encoder_config,
        device,
        generative_data_config=generative_data_config,
        generative_payload=generative_payload,
        latent_representation=latent_representation,
    )
    save_run_config(
        paths,
        {
            "property_data_config": data_config.to_dict(),
            "generative_data_config": generative_data_config.to_dict(),
            "encoder_config": encoder_config.to_dict(),
            "training_config": training_config.to_dict(),
            "latent_dim": generative_payload["latent_dim"],
            "latent_representation": generative_payload["latent_representation"],
            "latent_coordinate_space": generative_payload["latent_coordinate_space"],
            "latent_projection": projection_json_summary(
                generative_payload.get("latent_projection")
            ),
            "latent_projection_config": paths["latent_projection_config"].to_dict(),
            "diffusion_latent_dim": (
                paths["latent_projection_config"].output_dim
                if paths["latent_projection_config"].learned
                else generative_payload["latent_dim"]
            ),
            "sequence_length": generative_payload["sequence_length"],
            "decoder_target": generative_payload["decoder_target"],
        },
    )


def command_prepare_generative_embeddings(args) -> None:
    """Build only the unlabeled generative cache for benchmark corpora."""
    generative_data_config = make_generative_data_config(args)
    encoder_config = EncoderConfig(
        model_name=args.encoder_name,
        pooling=args.pooling,
        freeze_encoder=not args.unfreeze_encoder,
        local_files_only=args.local_files_only,
    )
    device = resolve_device(args.device)
    paths = artifact_paths(args.artifact_dir, args.cache_dir, make_latent_projection_config(args))
    payload = prepare_generative_embeddings(
        paths, generative_data_config, encoder_config, device,
        latent_representation=requested_latent_representation(args),
    )
    print_data_report("Generative embedding cache", payload)
    print(f"Cache: {paths['generative_embedding_cache']}")


def command_train_decoder(args) -> None:
    configs = make_configs(args)
    data_config, encoder_config, decoder_config, diffusion_config, classifier_config, guidance_classifier_config, training_config = configs
    del data_config, diffusion_config, classifier_config, guidance_classifier_config
    generative_data_config = make_generative_data_config(args)
    device = resolve_device(training_config.device)
    set_seed(generative_data_config.seed)
    projection_config = make_latent_projection_config(args)
    paths = artifact_paths(training_config.artifact_dir, args.cache_dir, projection_config)
    if projection_config.learned:
        raise ValueError(
            "With --projection-mode learned, run train-diffusion; it jointly trains "
            "the projector, decoder, and diffusion model."
        )
    payload = prepare_generative_embeddings(
        paths,
        generative_data_config,
        encoder_config,
        device,
        latent_representation=requested_latent_representation(args),
    )
    latent_dim = int(payload["latent_dim"])
    latent_standardizer = LatentStandardizer.from_metadata(
        payload["latent_standardization"],
        latent_dim=latent_dim,
    ).to(device)
    train_loader = split_payload_to_loader(
        payload["splits"]["train"],
        generative_data_config.batch_size,
        True,
        generative_data_config.num_workers,
    )
    val_loader = split_payload_to_loader(
        payload["splits"]["val"],
        generative_data_config.batch_size,
        False,
        generative_data_config.num_workers,
    )
    selfies_vocabulary = SelfiesVocabulary.from_tokens(
        payload["selfies_vocabulary"]["tokens"],
        max_length=payload["selfies_vocabulary"]["max_length"],
    )

    set_seed(args.seed if args.training_seed is None else args.training_seed)
    model = EmbeddingToSelfiesDecoder(latent_dim, selfies_vocabulary, decoder_config).to(device)
    print_data_report("Decoder", payload)
    metrics = train_decoder(
        model,
        latent_standardizer,
        train_loader,
        val_loader,
        device,
        training_config,
    )
    save_model_checkpoint(
        paths["decoder"],
        model,
        metadata={
            "latent_dim": latent_dim,
            "model_config": decoder_config.to_dict(),
            "encoder_model_name": encoder_config.model_name,
            "encoder_config": encoder_config.to_dict(),
            "latent_coordinate_schema_version": 3,
            "latent_coordinate_space": payload["latent_coordinate_space"],
            "latent_projection": payload.get("latent_projection"),
            "latent_standardization": latent_standardizer.to_metadata(),
            "latent_representation": payload["latent_representation"],
            "decoder_target": "SELFIES",
            "vocabulary_tokens": selfies_vocabulary.ordered_tokens,
            "max_decode_length": selfies_vocabulary.max_length,
        },
    )
    write_metrics(paths, "decoder", metrics)
    print_training_report("Decoder", metrics)



def _command_train_learned_projection_diffusion(
    args,
    configs,
    projection_config: LatentProjectionConfig,
) -> None:
    (
        _data_config,
        encoder_config,
        decoder_config,
        diffusion_config,
        _classifier_config,
        _guidance_config,
        training_config,
    ) = configs
    if args.joint_decoder_diffusion:
        raise ValueError("Learned projection already trains decoder and diffusion jointly.")
    if args.resume_diffusion or args.resume_diffusion_checkpoint:
        raise ValueError(
            "Resume is not available for learned projection runs because the projector, "
            "decoder, and denoiser must be resumed from one synchronized checkpoint set."
        )
    learning_rates = (
        training_config.diffusion_learning_rate,
        training_config.decoder_learning_rate,
        training_config.projection_learning_rate,
    )
    if any(rate <= 0 for rate in learning_rates):
        raise ValueError("All learned projection training rates must be positive.")
    if training_config.projection_reconstruction_weight < 0:
        raise ValueError("--projection-reconstruction-weight cannot be negative.")
    if training_config.projection_variance_weight < 0:
        raise ValueError("--projection-variance-weight cannot be negative.")

    generative_data_config = make_generative_data_config(args)
    device = resolve_device(training_config.device)
    set_seed(generative_data_config.seed)
    paths = artifact_paths(
        training_config.artifact_dir, args.cache_dir, projection_config
    )
    payload = prepare_generative_embeddings(
        paths,
        generative_data_config,
        encoder_config,
        device,
        latent_representation=requested_latent_representation(args),
    )
    if payload.get("latent_projection") is not None:
        raise ValueError(
            "Learned projection requires a raw full-dimensional embedding cache. "
            "Choose the original cache rather than a PCA cache."
        )
    input_dim = int(payload["latent_dim"])
    output_dim = int(projection_config.output_dim)
    if output_dim >= input_dim:
        raise ValueError(
            f"Learned projection dimension {output_dim} must be smaller than the "
            f"encoder dimension {input_dim}."
        )

    diffusion_config.latent_dim = output_dim
    diffusion_config.max_sequence_length = int(payload["sequence_length"])
    latent_standardizer = LatentStandardizer.identity(output_dim).to(device)
    train_loader = split_payload_to_loader(
        payload["splits"]["train"],
        generative_data_config.batch_size,
        True,
        generative_data_config.num_workers,
    )
    val_loader = split_payload_to_loader(
        payload["splits"]["val"],
        generative_data_config.batch_size,
        False,
        generative_data_config.num_workers,
    )
    train_attention_mask = payload["splits"]["train"]["attention_mask"]
    sequence_lengths = train_attention_mask.sum(dim=1).long()
    sequence_length_counts = torch.bincount(
        sequence_lengths,
        minlength=diffusion_config.max_sequence_length + 1,
    ).tolist()
    vocabulary = SelfiesVocabulary.from_tokens(
        payload["selfies_vocabulary"]["tokens"],
        max_length=payload["selfies_vocabulary"]["max_length"],
    )
    set_seed(args.seed if args.training_seed is None else args.training_seed)
    projector = LearnedTokenProjector(input_dim, output_dim).to(device)
    model = build_denoiser(diffusion_config).to(device)
    decoder = EmbeddingToSelfiesDecoder(
        output_dim, vocabulary, decoder_config
    ).to(device)
    schedule = DiffusionSchedule(
        diffusion_config,
        latent_standardizer=latent_standardizer,
        sequence_length_counts=sequence_length_counts,
    ).to(device)

    print_data_report("Learned projection diffusion", payload)
    print(
        "Learning rates: "
        f"diffusion={training_config.diffusion_learning_rate:g}, "
        f"decoder={training_config.decoder_learning_rate:g}, "
        f"projection={training_config.projection_learning_rate:g}",
        flush=True,
    )
    metrics = train_learned_projection_diffusion(
        projector,
        model,
        decoder,
        schedule,
        train_loader,
        val_loader,
        device,
        training_config,
    )
    projection_metadata = projector.to_metadata(
        provenance={
            "training": "joint_diffusion_and_selfies_reconstruction",
            "encoder_config": encoder_config.to_dict(),
            "seed": args.seed if args.training_seed is None else args.training_seed,
            "data_seed": generative_data_config.seed,
            "learning_rate": training_config.projection_learning_rate,
        }
    )
    coordinate_space = coordinate_space_name(
        payload["latent_representation"], output_dim, projection_kind="learned"
    )
    training_state = {
        "epoch": int(metrics["best_epoch"]),
        "completed_epoch": int(metrics["completed_epoch"]),
        "best_epoch": int(metrics["best_epoch"]),
        "best_val_loss": float(metrics["best_val_loss"]),
        "history": [dict(row) for row in metrics["history"]],
        "model_state": "best",
    }
    save_model_checkpoint(
        paths["learned_projector"],
        projector,
        metadata=projection_metadata,
        training_state=training_state,
    )
    save_model_checkpoint(
        paths["decoder"],
        decoder,
        metadata={
            "latent_dim": output_dim,
            "model_config": decoder_config.to_dict(),
            "encoder_model_name": encoder_config.model_name,
            "encoder_config": encoder_config.to_dict(),
            "latent_coordinate_schema_version": 4,
            "latent_coordinate_space": coordinate_space,
            "latent_projection": projection_metadata,
            "latent_standardization": latent_standardizer.to_metadata(),
            "latent_representation": payload["latent_representation"],
            "decoder_target": "SELFIES",
            "vocabulary_tokens": vocabulary.ordered_tokens,
            "max_decode_length": vocabulary.max_length,
            "joint_training": True,
            "training_seed": args.seed if args.training_seed is None else args.training_seed,
        },
        training_state=training_state,
    )
    save_model_checkpoint(
        paths["diffusion"],
        model,
        metadata={
            "model_config": diffusion_config.to_dict(),
            "denoiser_architecture": diffusion_config.denoiser_architecture,
            "training_objective": {
                "prediction_type": "epsilon",
                "reconstruction_weight": training_config.projection_reconstruction_weight,
                "variance_weight": training_config.projection_variance_weight,
            },
            "encoder_config": encoder_config.to_dict(),
            "latent_standardization": latent_standardizer.to_metadata(),
            "latent_coordinate_space": coordinate_space,
            "latent_projection": projection_metadata,
            "latent_representation": payload["latent_representation"],
            "sequence_length_counts": sequence_length_counts,
            "joint_training": True,
            "training_seed": args.seed if args.training_seed is None else args.training_seed,
            "learning_rates": metrics["learning_rates"],
        },
        training_state=training_state,
    )
    metrics["training_seed"] = args.seed if args.training_seed is None else args.training_seed
    write_metrics(paths, "diffusion", metrics)
    print_training_report("Learned projection diffusion", metrics)

def _command_train_joint_fixed_diffusion(
    args,
    configs,
    projection_config: LatentProjectionConfig,
) -> None:
    """Use the same joint decoder/denoiser objective as learned projection runs."""
    if projection_config.mode != "pca":
        raise ValueError("Matched fixed-latent joint training requires --projection-mode pca.")
    if args.resume_diffusion or args.resume_diffusion_checkpoint:
        raise ValueError("Resume is not supported for synchronized joint checkpoints.")
    (
        _data_config, encoder_config, decoder_config, diffusion_config,
        _classifier_config, _guidance_config, training_config,
    ) = configs
    generative_data_config = make_generative_data_config(args)
    device = resolve_device(training_config.device)
    paths = artifact_paths(training_config.artifact_dir, args.cache_dir, projection_config)
    payload = prepare_generative_embeddings(
        paths,
        generative_data_config,
        encoder_config,
        device,
        latent_representation=requested_latent_representation(args),
    )
    if payload.get("latent_projection") is None:
        raise ValueError("The selected cache does not contain PCA latents.")
    latent_dim = int(payload["latent_dim"])
    standardizer = LatentStandardizer.from_metadata(
        payload["latent_standardization"], latent_dim=latent_dim
    ).to(device)
    set_seed(args.seed if args.training_seed is None else args.training_seed)
    train_loader = split_payload_to_loader(
        payload["splits"]["train"], generative_data_config.batch_size, True,
        generative_data_config.num_workers,
    )
    val_loader = split_payload_to_loader(
        payload["splits"]["val"], generative_data_config.batch_size, False,
        generative_data_config.num_workers,
    )
    diffusion_config.latent_dim = latent_dim
    diffusion_config.max_sequence_length = int(payload["sequence_length"])
    lengths = payload["splits"]["train"]["attention_mask"].sum(dim=1).long()
    length_counts = torch.bincount(
        lengths, minlength=diffusion_config.max_sequence_length + 1
    ).tolist()
    vocabulary = SelfiesVocabulary.from_tokens(
        payload["selfies_vocabulary"]["tokens"],
        max_length=payload["selfies_vocabulary"]["max_length"],
    )
    denoiser = build_denoiser(diffusion_config).to(device)
    decoder = EmbeddingToSelfiesDecoder(latent_dim, vocabulary, decoder_config).to(device)
    schedule = DiffusionSchedule(
        diffusion_config, latent_standardizer=standardizer,
        sequence_length_counts=length_counts,
    ).to(device)
    print_data_report("Joint PCA diffusion", payload)
    metrics = train_learned_projection_diffusion(
        None, denoiser, decoder, schedule, train_loader, val_loader,
        device, training_config,
    )
    training_state = {
        "epoch": int(metrics["best_epoch"]),
        "completed_epoch": int(metrics["completed_epoch"]),
        "best_epoch": int(metrics["best_epoch"]),
        "best_val_loss": float(metrics["best_val_loss"]),
        "history": [dict(row) for row in metrics["history"]],
        "model_state": "best",
    }
    shared_metadata = {
        "encoder_config": encoder_config.to_dict(),
        "latent_standardization": standardizer.to_metadata(),
        "latent_coordinate_space": payload["latent_coordinate_space"],
        "latent_projection": payload["latent_projection"],
        "latent_representation": payload["latent_representation"],
        "joint_training": True,
        "training_seed": args.seed if args.training_seed is None else args.training_seed,
    }
    save_model_checkpoint(
        paths["decoder"], decoder,
        metadata={
            **shared_metadata,
            "latent_dim": latent_dim,
            "model_config": decoder_config.to_dict(),
            "encoder_model_name": encoder_config.model_name,
            "latent_coordinate_schema_version": 4,
            "decoder_target": "SELFIES",
            "vocabulary_tokens": vocabulary.ordered_tokens,
            "max_decode_length": vocabulary.max_length,
        },
        training_state=training_state,
    )
    save_model_checkpoint(
        paths["diffusion"], denoiser,
        metadata={
            **shared_metadata,
            "model_config": diffusion_config.to_dict(),
            "denoiser_architecture": diffusion_config.denoiser_architecture,
            "training_objective": {
                "prediction_type": "epsilon",
                "reconstruction_weight": training_config.projection_reconstruction_weight,
                "variance_weight": training_config.projection_variance_weight,
            },
            "sequence_length_counts": length_counts,
            "learning_rates": metrics["learning_rates"],
        },
        training_state=training_state,
    )
    metrics["training_seed"] = shared_metadata["training_seed"]
    write_metrics(paths, "diffusion", metrics)
    print_training_report("Joint PCA diffusion", metrics)


def command_train_diffusion(args) -> None:
    configs = make_configs(args)
    projection_config = make_latent_projection_config(args)
    if projection_config.learned:
        _command_train_learned_projection_diffusion(args, configs, projection_config)
        return
    if args.joint_decoder_diffusion:
        _command_train_joint_fixed_diffusion(args, configs, projection_config)
        return
    (
        data_config,
        encoder_config,
        decoder_config,
        diffusion_config,
        classifier_config,
        guidance_classifier_config,
        training_config,
    ) = configs
    del data_config, decoder_config, classifier_config, guidance_classifier_config
    generative_data_config = make_generative_data_config(args)
    device = resolve_device(training_config.device)
    set_seed(generative_data_config.seed)
    paths = artifact_paths(training_config.artifact_dir, args.cache_dir, make_latent_projection_config(args))
    payload = prepare_generative_embeddings(
        paths,
        generative_data_config,
        encoder_config,
        device,
        latent_representation=requested_latent_representation(args),
    )
    set_seed(args.seed if args.training_seed is None else args.training_seed)
    if not paths["decoder"].exists():
        raise FileNotFoundError("Decoder checkpoint is required before training diffusion.")

    latent_dim = int(payload["latent_dim"])
    latent_standardizer = LatentStandardizer.from_metadata(
        payload["latent_standardization"],
        latent_dim=latent_dim,
    ).to(device)
    decoder_metadata = load_checkpoint(paths["decoder"])["metadata"]
    _validate_decoder_coordinates(
        decoder_metadata,
        expected_latent_dim=latent_dim,
        expected_encoder_config=encoder_config.to_dict(),
        expected_standardizer=latent_standardizer,
        expected_projection=payload.get("latent_projection"),
        expected_latent_representation=payload["latent_representation"],
    )
    diffusion_config.latent_dim = latent_dim
    diffusion_config.max_sequence_length = int(payload["sequence_length"])

    train_loader = split_payload_to_loader(
        payload["splits"]["train"],
        generative_data_config.batch_size,
        True,
        generative_data_config.num_workers,
    )
    val_loader = split_payload_to_loader(
        payload["splits"]["val"],
        generative_data_config.batch_size,
        False,
        generative_data_config.num_workers,
    )
    train_attention_mask = payload["splits"]["train"]["attention_mask"]
    sequence_lengths = train_attention_mask.sum(dim=1).long()
    sequence_length_counts = torch.bincount(
        sequence_lengths,
        minlength=diffusion_config.max_sequence_length + 1,
    ).tolist()

    resume_requested = bool(
        args.resume_diffusion or args.resume_diffusion_checkpoint
    )
    resume_path = (
        Path(args.resume_diffusion_checkpoint)
        if args.resume_diffusion_checkpoint
        else paths["diffusion"]
    )
    resume_checkpoint = None
    resume_metadata: dict = {}
    initial_history: list[dict[str, float | int]] = []
    initial_best_metrics: dict[str, float | int] | None = None
    optimizer_state_dict = None
    start_epoch = 0

    if resume_requested:
        if not resume_path.exists():
            raise FileNotFoundError(
                f"Diffusion resume checkpoint does not exist: {resume_path}"
            )
        resume_checkpoint = load_checkpoint(resume_path)
        resume_metadata = dict(resume_checkpoint.get("metadata", {}))
        if resume_metadata.get("latent_representation") != payload["latent_representation"]:
            raise ValueError(
                "The resume checkpoint latent representation does not match the cache."
            )
        if "model_config" not in resume_metadata:
            raise ValueError("The resume checkpoint is missing diffusion model_config.")
        checkpoint_config = DiffusionConfig(**resume_metadata["model_config"])
        if checkpoint_config.latent_dim != latent_dim:
            raise ValueError(
                "Resume checkpoint latent dimension does not match the embedding cache."
            )
        if checkpoint_config.max_sequence_length != int(payload["sequence_length"]):
            raise ValueError(
                "Resume checkpoint sequence length does not match the embedding cache."
            )
        checkpoint_standardizer = LatentStandardizer.from_metadata(
            resume_metadata.get("latent_standardization"),
            latent_dim=latent_dim,
        ).to(device)
        if not checkpoint_standardizer.is_compatible_with(latent_standardizer):
            raise ValueError(
                "Resume checkpoint latent standardization does not match the cache."
            )
        if not projection_metadata_compatible(
            resume_metadata.get("latent_projection"), payload.get("latent_projection")
        ):
            raise ValueError(
                "Resume checkpoint latent projection does not match the cache."
            )

        objective = resume_metadata.get("training_objective", {})
        expected_objective = {"prediction_type": "epsilon"}
        unexpected_objective_fields = set(objective) - set(expected_objective)
        if unexpected_objective_fields:
            raise ValueError(
                "Resume checkpoint uses unsupported legacy diffusion-objective "
                "metadata. Start a new training run instead."
            )
        mismatched_objective = [
            key
            for key, expected in expected_objective.items()
            if key in objective and objective[key] != expected
        ]
        if mismatched_objective:
            details = ", ".join(
                f"{key}: checkpoint={objective[key]!r}, requested={expected_objective[key]!r}"
                for key in mismatched_objective
            )
            raise ValueError(
                "Resume diffusion objective does not match the checkpoint ("
                f"{details})."
            )

        saved_length_counts = resume_metadata.get("sequence_length_counts")
        if saved_length_counts is not None and list(saved_length_counts) != list(
            sequence_length_counts
        ):
            raise ValueError(
                "Resume checkpoint sequence-length distribution does not match the cache."
            )

        diffusion_config = checkpoint_config
        model = build_denoiser(diffusion_config).to(device)
        model.load_state_dict(resume_checkpoint["state_dict"])
        optimizer_state_dict = resume_checkpoint.get("optimizer_state_dict")

        previous_metrics: dict = {}
        if paths["metrics"].exists():
            try:
                previous_metrics = json.loads(
                    paths["metrics"].read_text(encoding="utf-8")
                ).get("diffusion", {})
            except Exception:
                previous_metrics = {}
        training_state = resume_checkpoint.get("training_state", {})
        initial_history = [
            dict(row)
            for row in (
                training_state.get("history")
                or previous_metrics.get("history")
                or []
            )
        ]
        if initial_history:
            start_epoch = max(int(row["epoch"]) for row in initial_history)
        else:
            start_epoch = int(
                training_state.get(
                    "completed_epoch",
                    previous_metrics.get(
                        "completed_epoch",
                        previous_metrics.get("epoch", 0),
                    ),
                )
            )

        model_state_kind = training_state.get("model_state", "best")
        if model_state_kind == "current_epoch":
            selected_epoch = int(training_state.get("epoch", start_epoch))
        else:
            selected_epoch = int(
                training_state.get(
                    "best_epoch",
                    previous_metrics.get("best_epoch", previous_metrics.get("epoch", 0)),
                )
            )
        initial_best_metrics = next(
            (
                dict(row)
                for row in initial_history
                if int(row["epoch"]) == selected_epoch
            ),
            None,
        )
        if initial_best_metrics is None and previous_metrics:
            required = ("train_loss", "val_loss")
            if all(key in previous_metrics for key in required):
                initial_best_metrics = {
                    "epoch": selected_epoch,
                    **{key: previous_metrics[key] for key in required},
                }
        if initial_best_metrics is None:
            raise ValueError(
                "Resume checkpoint has no recoverable selected-epoch metrics. "
                "Keep its metrics.json beside the checkpoint or resume from a new "
                "periodic checkpoint."
            )
        print(
            "Resuming diffusion from "
            f"{resume_path} (completed epoch {start_epoch}, loaded model epoch "
            f"{selected_epoch}, optimizer state "
            f"{'restored' if optimizer_state_dict is not None else 'unavailable; using a fresh AdamW state'}).",
            flush=True,
        )
    else:
        model = build_denoiser(diffusion_config).to(device)

    schedule = DiffusionSchedule(
        diffusion_config,
        latent_standardizer=latent_standardizer,
        sequence_length_counts=sequence_length_counts,
    ).to(device)

    checkpoint_metadata = dict(resume_metadata)
    checkpoint_metadata.update(
        {
            "model_config": diffusion_config.to_dict(),
            "denoiser_architecture": diffusion_config.denoiser_architecture,
            "training_objective": {"prediction_type": "epsilon"},
            "latent_standardization": schedule.latent_standardizer.to_metadata(),
            "latent_coordinate_space": payload["latent_coordinate_space"],
            "latent_projection": payload.get("latent_projection"),
            "latent_representation": payload["latent_representation"],
            "sequence_length_counts": sequence_length_counts,
        }
    )

    print_data_report("Diffusion", payload)
    metrics = train_diffusion(
        model,
        schedule,
        train_loader,
        val_loader,
        device,
        training_config,
        start_epoch=start_epoch,
        initial_history=initial_history,
        initial_best_metrics=initial_best_metrics,
        optimizer_state_dict=optimizer_state_dict,
    )
    final_training_state = {
        "epoch": int(metrics["best_epoch"]),
        "completed_epoch": int(metrics["completed_epoch"]),
        "best_epoch": int(metrics["best_epoch"]),
        "best_val_loss": float(metrics["best_val_loss"]),
        "history": [dict(row) for row in metrics["history"]],
        "model_state": "best",
        "resume_source": str(resume_path) if resume_requested else None,
    }
    save_model_checkpoint(
        paths["diffusion"],
        model,
        metadata=checkpoint_metadata,
        training_state=final_training_state,
    )
    write_metrics(paths, "diffusion", metrics)
    print_training_report("Diffusion", metrics)
def command_train_classifier(args) -> None:
    configs = make_configs(args)
    data_config, encoder_config, decoder_config, diffusion_config, classifier_config, guidance_classifier_config, training_config = configs
    del decoder_config, diffusion_config, guidance_classifier_config
    device = resolve_device(training_config.device)
    set_seed(data_config.seed)
    paths = artifact_paths(training_config.artifact_dir, args.cache_dir, make_latent_projection_config(args))
    payload = prepare_property_embeddings(
        paths,
        data_config,
        encoder_config,
        device,
        generative_data_config=make_generative_data_config(args),
        latent_representation=requested_latent_representation(args),
    )
    latent_dim = int(payload["latent_dim"])
    train_loader = split_payload_to_loader(payload["splits"]["train"], data_config.batch_size, True, data_config.num_workers)
    val_loader = split_payload_to_loader(payload["splits"]["val"], data_config.batch_size, False, data_config.num_workers)
    test_loader = None
    if "test" in payload["splits"]:
        test_loader = split_payload_to_loader(
            payload["splits"]["test"],
            data_config.batch_size,
            False,
            data_config.num_workers,
        )

    model = BBBPClassifier(latent_dim=latent_dim, config=classifier_config).to(device)
    print_data_report("Classifier", payload)
    metrics = train_classifier(model, train_loader, val_loader, device, training_config, test_loader=test_loader)
    save_model_checkpoint(
        paths["classifier"],
        model,
        metadata={
            "latent_dim": latent_dim,
            "model_config": classifier_config.to_dict(),
        },
    )
    write_metrics(paths, "classifier", metrics)
    print_training_report("Classifier", metrics)


def command_train_guidance_classifier(args) -> None:
    configs = make_configs(args)
    data_config, encoder_config, decoder_config, diffusion_config, classifier_config, guidance_classifier_config, training_config = configs
    del decoder_config, diffusion_config, classifier_config
    device = resolve_device(training_config.device)
    set_seed(data_config.seed)
    paths = artifact_paths(training_config.artifact_dir, args.cache_dir, make_latent_projection_config(args))
    payload = prepare_property_embeddings(
        paths,
        data_config,
        encoder_config,
        device,
        generative_data_config=make_generative_data_config(args),
        latent_representation=requested_latent_representation(args),
    )
    latent_dim = int(payload["latent_dim"])
    if not paths["diffusion"].exists():
        raise FileNotFoundError(
            "Diffusion checkpoint is required before training the guidance classifier "
            "so both models use the same latent standardization."
        )
    diffusion_checkpoint = load_checkpoint(paths["diffusion"])
    diffusion_metadata = diffusion_checkpoint["metadata"]
    if diffusion_metadata.get("latent_representation") != payload["latent_representation"]:
        raise ValueError(
            "Guidance training requires the property cache and diffusion checkpoint "
            "to use the same latent representation."
        )
    schedule_config = DiffusionConfig(**diffusion_metadata["model_config"])
    latent_standardizer = LatentStandardizer.from_metadata(
        diffusion_metadata.get("latent_standardization"),
        latent_dim=schedule_config.latent_dim,
    )
    if schedule_config.latent_dim != latent_dim:
        raise ValueError(
            f"Property embedding dimension {latent_dim} does not match diffusion latent "
            f"dimension {schedule_config.latent_dim}."
        )
    if not projection_metadata_compatible(
        diffusion_metadata.get("latent_projection"), payload.get("latent_projection")
    ):
        raise ValueError(
            "Property cache and diffusion checkpoint use different latent projections."
        )
    if schedule_config.max_sequence_length != int(payload["sequence_length"]):
        raise ValueError(
            "Property and diffusion embedding caches use different sequence lengths. "
            "Regenerate both caches with the same --max-length."
        )
    if latent_standardizer.enabled:
        diffusion_encoder_config = latent_standardizer.provenance.get("encoder_config")
        if not isinstance(diffusion_encoder_config, dict):
            raise ValueError(
                "Standardized diffusion checkpoint is missing encoder provenance. "
                "Retrain diffusion before training the guidance classifier."
            )
        encoder_fields = ("model_name",)
        mismatched_fields = [
            field
            for field in encoder_fields
            if diffusion_encoder_config.get(field) != getattr(encoder_config, field)
        ]
        if mismatched_fields:
            raise ValueError(
                "Property embeddings do not use the diffusion checkpoint's encoder configuration "
                f"({', '.join(mismatched_fields)}). Train guidance with the same encoder model "
                "as diffusion."
            )
    schedule = DiffusionSchedule(
        schedule_config,
        latent_standardizer=latent_standardizer,
        sequence_length_counts=diffusion_metadata.get("sequence_length_counts"),
    ).to(device)

    if guidance_classifier_config.distill_from_classifier:
        if not paths["classifier"].exists():
            raise FileNotFoundError("Classifier checkpoint is required before guidance-classifier distillation.")
        classifier = instantiate_classifier(paths["classifier"], device)
        for split_name in ["train", "val"]:
            embeddings = payload["splits"][split_name]["embeddings"].to(device)
            with torch.no_grad():
                probs = classifier(embeddings)["probabilities"].cpu()
            payload["splits"][split_name]["target_probabilities"] = probs

    train_loader = split_payload_to_loader(payload["splits"]["train"], data_config.batch_size, True, data_config.num_workers)
    val_loader = split_payload_to_loader(payload["splits"]["val"], data_config.batch_size, False, data_config.num_workers)
    model = NoisyLatentPropertyClassifier(latent_dim=latent_dim, config=guidance_classifier_config).to(device)
    print_data_report("Guidance classifier", payload)
    metrics = train_guidance_classifier(
        model,
        schedule,
        train_loader,
        val_loader,
        device,
        training_config,
        distill=guidance_classifier_config.distill_from_classifier,
    )
    save_model_checkpoint(
        paths["guidance_classifier"],
        model,
        metadata={
            "latent_dim": latent_dim,
            "model_config": guidance_classifier_config.to_dict(),
            "diffusion_schedule_config": schedule_config.to_dict(),
            "latent_standardization": schedule.latent_standardizer.to_metadata(),
            "latent_coordinate_space": diffusion_metadata.get("latent_coordinate_space"),
            "latent_projection": payload.get("latent_projection"),
            "latent_representation": payload["latent_representation"],
            "guidance_formula": "mean + guidance_strength(t) * posterior_variance(t) * grad_log_p_target(noisy_latent, t)",
        },
    )
    write_metrics(paths, "guidance_classifier", metrics)
    print_training_report("Guidance classifier", metrics)


def command_train_property_regressor(args) -> None:
    configs = make_configs(args)
    (
        data_config,
        encoder_config,
        _decoder_config,
        _diffusion_config,
        regressor_config,
        _guidance_config,
        training_config,
    ) = configs
    device = resolve_device(training_config.device)
    set_seed(data_config.seed)
    paths = artifact_paths(training_config.artifact_dir, args.cache_dir, make_latent_projection_config(args))
    payload = prepare_property_embeddings(
        paths,
        data_config,
        encoder_config,
        device,
        generative_data_config=make_generative_data_config(args),
        latent_representation=requested_latent_representation(args),
    )
    target = payload["target_standardization"]
    latent_dim = int(payload["latent_dim"])
    latent_projection = payload.get("latent_projection")
    projector = None
    if make_latent_projection_config(args).learned:
        if not paths["learned_projector"].exists():
            raise FileNotFoundError(
                "Train diffusion with --projection-mode learned before training the "
                "property regressor."
            )
        projector_checkpoint = load_checkpoint(paths["learned_projector"])
        latent_projection = projector_checkpoint["metadata"]
        projector = instantiate_learned_projector(
            paths["learned_projector"], device, latent_projection
        )
        if projector.input_dim != latent_dim:
            raise ValueError("Property embeddings do not match the learned projector input.")
        latent_dim = projector.output_dim
    train_loader = split_payload_to_loader(
        payload["splits"]["train"],
        data_config.batch_size,
        True,
        data_config.num_workers,
    )
    val_loader = split_payload_to_loader(
        payload["splits"]["val"],
        data_config.batch_size,
        False,
        data_config.num_workers,
    )
    test_loader = split_payload_to_loader(
        payload["splits"]["test"],
        data_config.batch_size,
        False,
        data_config.num_workers,
    )
    model = PropertyRegressor(
        latent_dim,
        regressor_config,
        target_mean=float(target["mean"]),
        target_scale=float(target["scale"]),
    ).to(device)
    print_data_report(f"{data_config.property_name} property regressor", payload)
    metrics = train_property_regressor(
        model,
        train_loader,
        val_loader,
        device,
        training_config,
        test_loader=test_loader,
        projector=projector,
    )
    save_model_checkpoint(
        paths["property_regressor"],
        model,
        metadata={
            "task_type": "regression",
            "property_name": data_config.property_name,
            "label_column": data_config.label_column,
            "latent_dim": latent_dim,
            "model_config": regressor_config.to_dict(),
            "target_standardization": target,
            "latent_projection": latent_projection,
            "latent_representation": payload["latent_representation"],
            "encoder_config": encoder_config.to_dict(),
        },
    )
    write_metrics(paths, "property_regressor", metrics)
    print_training_report("Property regressor", metrics)


def command_train_guidance_regressor(args) -> None:
    configs = make_configs(args)
    (
        data_config,
        encoder_config,
        _decoder_config,
        _diffusion_config,
        _regressor_config,
        guidance_config,
        training_config,
    ) = configs
    device = resolve_device(training_config.device)
    set_seed(data_config.seed)
    paths = artifact_paths(training_config.artifact_dir, args.cache_dir, make_latent_projection_config(args))
    payload = prepare_property_embeddings(
        paths,
        data_config,
        encoder_config,
        device,
        generative_data_config=make_generative_data_config(args),
        latent_representation=requested_latent_representation(args),
    )
    target = payload["target_standardization"]
    encoder_latent_dim = int(payload["latent_dim"])
    latent_dim = encoder_latent_dim
    projector = None
    if not paths["diffusion"].exists():
        raise FileNotFoundError(
            "Train the diffusion model before training the guidance regressor."
        )
    diffusion_metadata = load_checkpoint(paths["diffusion"])["metadata"]
    if diffusion_metadata.get("latent_representation") != payload["latent_representation"]:
        raise ValueError(
            "Guidance regression requires the property cache and diffusion checkpoint "
            "to use the same latent representation."
        )
    schedule_config = DiffusionConfig(**diffusion_metadata["model_config"])
    latent_standardizer = LatentStandardizer.from_metadata(
        diffusion_metadata.get("latent_standardization"),
        latent_dim=schedule_config.latent_dim,
    )
    latent_projection = diffusion_metadata.get("latent_projection")
    if (
        isinstance(latent_projection, dict)
        and latent_projection.get("kind") == LearnedTokenProjector.KIND
    ):
        if payload.get("latent_projection") is not None:
            raise ValueError("Learned guidance training requires the raw embedding cache.")
        projector = instantiate_learned_projector(
            paths["learned_projector"], device, latent_projection
        )
        if projector.input_dim != encoder_latent_dim:
            raise ValueError("Property embeddings do not match the learned projector input.")
        latent_dim = projector.output_dim
    elif not projection_metadata_compatible(
        latent_projection, payload.get("latent_projection")
    ):
        raise ValueError(
            "Property cache and diffusion checkpoint use different latent projections."
        )
    if schedule_config.latent_dim != latent_dim:
        raise ValueError("Property and diffusion latent dimensions do not match.")
    if schedule_config.max_sequence_length != int(payload["sequence_length"]):
        raise ValueError("Property and diffusion sequence lengths do not match.")
    if latent_standardizer.enabled:
        diffusion_encoder = latent_standardizer.provenance.get("encoder_config")
        if not isinstance(diffusion_encoder, dict):
            raise ValueError("Diffusion checkpoint is missing encoder provenance.")
        if diffusion_encoder.get("model_name") != encoder_config.model_name:
            raise ValueError("Property and diffusion embeddings use different encoders.")
    schedule = DiffusionSchedule(
        schedule_config,
        latent_standardizer=latent_standardizer,
        sequence_length_counts=diffusion_metadata.get("sequence_length_counts"),
    ).to(device)

    distill = guidance_config.distill_from_classifier
    if distill:
        if not paths["property_regressor"].exists():
            raise FileNotFoundError(
                "Train the clean property regressor before distilling guidance."
            )
        teacher = instantiate_property_regressor(
            paths["property_regressor"],
            device,
            expected_property_name=data_config.property_name,
        expected_projection=latent_projection,
        )
        for split_name in ("train", "val"):
            predictions = []
            split = payload["splits"][split_name]
            embeddings = split["embeddings"]
            sequence_embeddings = split["sequence_embeddings"]
            attention_masks = split["attention_mask"]
            with torch.no_grad():
                for start_index in range(0, embeddings.size(0), 1024):
                    end_index = start_index + 1024
                    if projector is None:
                        teacher_embeddings = embeddings[start_index:end_index].to(device)
                    else:
                        mask = attention_masks[start_index:end_index].to(
                            device=device, dtype=torch.bool
                        )
                        tokens = projector(
                            sequence_embeddings[start_index:end_index].to(device).float(),
                            mask,
                        )
                        weights = mask.unsqueeze(-1).to(tokens.dtype)
                        teacher_embeddings = tokens.sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
                    predictions.append(teacher(teacher_embeddings)["prediction"].cpu())
            split["target_predictions"] = torch.cat(predictions)

    train_loader = split_payload_to_loader(
        payload["splits"]["train"],
        data_config.batch_size,
        True,
        data_config.num_workers,
    )
    val_loader = split_payload_to_loader(
        payload["splits"]["val"],
        data_config.batch_size,
        False,
        data_config.num_workers,
    )
    model = NoisyLatentPropertyRegressor(
        latent_dim,
        guidance_config,
        target_mean=float(target["mean"]),
        target_scale=float(target["scale"]),
    ).to(device)
    print_data_report(f"{data_config.property_name} guidance regressor", payload)
    metrics = train_guidance_regressor(
        model,
        schedule,
        train_loader,
        val_loader,
        device,
        training_config,
        distill=distill,
        projector=projector,
    )
    save_model_checkpoint(
        paths["guidance_regressor"],
        model,
        metadata={
            "task_type": "regression",
            "property_name": data_config.property_name,
            "label_column": data_config.label_column,
            "latent_dim": latent_dim,
            "model_config": guidance_config.to_dict(),
            "target_standardization": target,
            "diffusion_schedule_config": schedule_config.to_dict(),
            "latent_standardization": schedule.latent_standardizer.to_metadata(),
            "latent_coordinate_space": diffusion_metadata.get("latent_coordinate_space"),
            "latent_projection": latent_projection,
            "latent_representation": payload["latent_representation"],
            "guidance_formula": (
                "mean + signed_guidance_scale(t) * posterior_variance(t) * "
                "grad_standardized_property(noisy_latent, t)"
            ),
        },
    )
    write_metrics(paths, "guidance_regressor", metrics)
    print_training_report("Guidance regressor", metrics)


def command_train_all(args) -> None:
    command_prepare_embeddings(args)
    if not make_latent_projection_config(args).learned:
        command_train_decoder(args)
    command_train_diffusion(args)
    command_train_property_regressor(args)
    command_train_guidance_regressor(args)


def command_sample(args) -> None:
    configs = make_configs(args)
    (
        data_config,
        encoder_config,
        decoder_config,
        diffusion_config,
        classifier_config,
        guidance_classifier_config,
        training_config,
    ) = configs
    del decoder_config, diffusion_config, classifier_config
    del guidance_classifier_config
    generative_data_config = make_generative_data_config(args)
    device = resolve_device(training_config.device)
    sampling_seed = (
        data_config.seed if args.sampling_seed is None else int(args.sampling_seed)
    )
    if sampling_seed < 0:
        raise ValueError("--sampling-seed must be non-negative.")
    set_seed(sampling_seed)
    paths = artifact_paths(training_config.artifact_dir, args.cache_dir, make_latent_projection_config(args))
    payload = prepare_generative_embeddings(
        paths,
        generative_data_config,
        encoder_config,
        device,
        latent_representation=requested_latent_representation(args),
    )
    denoiser, schedule = instantiate_diffusion(paths["diffusion"], device)
    if schedule.latent_representation != payload["latent_representation"]:
        raise ValueError(
            "Diffusion checkpoint and embedding cache use different latent representations."
        )
    expected_projection = payload.get("latent_projection")
    if (
        isinstance(schedule.latent_projection, dict)
        and schedule.latent_projection.get("kind") == LearnedTokenProjector.KIND
    ):
        if expected_projection is not None:
            raise ValueError("Learned projection sampling requires the raw embedding cache.")
        if not paths["learned_projector"].exists():
            raise FileNotFoundError(
                "The diffusion checkpoint requires checkpoints/learned_projector.pt."
            )
        instantiate_learned_projector(
            paths["learned_projector"], device, schedule.latent_projection
        )
        expected_projection = schedule.latent_projection
    elif not projection_metadata_compatible(
        schedule.latent_projection, expected_projection
    ):
        raise ValueError(
            "Diffusion checkpoint and embedding cache use different latent projections."
        )
    diffusion_encoder_config = schedule.latent_standardizer.provenance.get(
        "encoder_config"
    )
    if schedule.latent_standardizer.enabled and not isinstance(
        diffusion_encoder_config, dict
    ):
        raise ValueError(
            "Standardized diffusion checkpoint is missing encoder provenance. "
            "Retrain diffusion before sampling."
        )
    decoder = instantiate_decoder(
        paths["decoder"],
        device,
        expected_latent_dim=denoiser.config.latent_dim,
        expected_encoder_config=(
            diffusion_encoder_config
            if isinstance(diffusion_encoder_config, dict)
            else encoder_config.to_dict()
        ),
        expected_standardizer=schedule.latent_standardizer,
        expected_projection=expected_projection,
        expected_latent_representation=payload["latent_representation"],
    )
    guidance_regressor = None
    if args.guidance_scale != 0.0:
        if not paths["guidance_regressor"].exists():
            raise FileNotFoundError(
                "Guided sampling requires checkpoints/guidance_regressor.pt. "
                "Run train-guidance-regressor first or set --guidance-scale 0."
            )
        guidance_regressor = instantiate_guidance_regressor(
            paths["guidance_regressor"],
            device,
            expected_property_name=data_config.property_name,
        expected_projection=expected_projection,
            expected_standardizer=schedule.latent_standardizer,
            expected_diffusion_config=denoiser.config,
        )

    experiment_paths = create_sampling_experiment(
        training_config.artifact_dir,
        property_name=data_config.property_name,
        guidance_scale=args.guidance_scale,
        seed=sampling_seed,
        experiment_name=args.experiment_name,
    )
    sampling_config = SamplingConfig(
        num_samples=args.num_samples,
        batch_size=args.sample_batch_size,
        guidance_scale=args.guidance_scale,
        guidance_gamma=args.guidance_gamma,
        temperature=args.temperature,
        output_path=str(experiment_paths["samples"]),
    )
    direction = (
        "increase"
        if args.guidance_scale > 0.0
        else "decrease"
        if args.guidance_scale < 0.0
        else "unguided"
    )
    update_experiment_manifest(
        experiment_paths,
        {
            "schema_version": 1,
            "status": "running",
            "command": "sample",
            "created_at": utc_timestamp(),
            "experiment_name": experiment_paths["root"].name,
            "property_name": data_config.property_name,
            "guidance_direction": direction,
            "seed": sampling_seed,
            "data_seed": data_config.seed,
            "requested_device": training_config.device,
            "resolved_device": str(device),
            "sampling_config": sampling_config.to_dict(),
            "data_config": generative_data_config.to_dict(),
            "encoder_config": encoder_config.to_dict(),
            "diffusion_config": denoiser.config.to_dict(),
            "latent_representation": payload["latent_representation"],
            "shared_artifacts": {
                "artifact_dir": str(paths["root"]),
                "generative_embedding_cache": str(
                    paths["generative_embedding_cache"]
                ),
                "decoder_checkpoint": str(paths["decoder"]),
                "diffusion_checkpoint": str(paths["diffusion"]),
                "guidance_regressor_checkpoint": (
                    str(paths["guidance_regressor"])
                    if guidance_regressor is not None
                    else None
                ),
            },
        },
    )

    # Model construction can consume framework RNG state. Reset immediately before
    # generation so --sampling-seed controls the sample stream itself.
    set_seed(sampling_seed)
    try:
        samples = sample_smiles(
            denoiser=denoiser,
            schedule=schedule,
            decoder=decoder,
            num_samples=sampling_config.num_samples,
            batch_size=sampling_config.batch_size,
            guidance_regressor=guidance_regressor,
            guidance_scale=sampling_config.guidance_scale,
            guidance_gamma=sampling_config.guidance_gamma,
            temperature=sampling_config.temperature,
            device=device,
        )
        pd.DataFrame(samples).to_csv(experiment_paths["samples"], index=False)

        if args.skip_evaluation:
            update_experiment_manifest(
                experiment_paths,
                {
                    "status": "complete",
                    "completed_at": utc_timestamp(),
                    "outputs": {"samples": "samples.csv"},
                    "evaluation": "deferred_to_latent_space_study",
                },
            )
        else:
            smiles_list = [sample["SMILES"] for sample in samples]
            metrics = generation_metrics(
                smiles_list=smiles_list,
                training_smiles=payload["splits"]["train"]["smiles"],
                property_name=data_config.property_name,
                guidance_scale=sampling_config.guidance_scale,
                device=device,
            )
            write_metrics(experiment_paths, "evaluation", metrics)
            plot_paths = save_property_distribution_plots(
                metrics["property_distributions"], experiment_paths["plots"]
            )
            relative_plot_paths = {
                name: str(Path(path).relative_to(experiment_paths["root"]))
                for name, path in plot_paths.items()
            }
            update_experiment_manifest(
                experiment_paths,
                {
                    "status": "complete",
                    "completed_at": utc_timestamp(),
                    "outputs": {
                        "samples": "samples.csv",
                        "metrics": "metrics.json",
                        "plots": relative_plot_paths,
                    },
                },
            )
        record_latest_sampling_experiment(
            training_config.artifact_dir, experiment_paths
        )
    except Exception as error:
        update_experiment_manifest(
            experiment_paths,
            {
                "status": "failed",
                "failed_at": utc_timestamp(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise

    print(f"\nSampling experiment: {experiment_paths['root']}\n")


def command_evaluate(args) -> None:
    configs = make_configs(args)
    (
        data_config,
        encoder_config,
        decoder_config,
        diffusion_config,
        classifier_config,
        guidance_classifier_config,
        training_config,
    ) = configs
    del decoder_config, diffusion_config, classifier_config
    del guidance_classifier_config
    generative_data_config = make_generative_data_config(args)
    device = resolve_device(training_config.device)
    paths = artifact_paths(training_config.artifact_dir, args.cache_dir, make_latent_projection_config(args))
    payload = prepare_generative_embeddings(
        paths,
        generative_data_config,
        encoder_config,
        device,
        latent_representation=requested_latent_representation(args),
    )

    if args.samples_path:
        sample_path = Path(args.samples_path)
        experiment_paths = sampling_experiment_paths(sample_path.parent)
    else:
        experiment_paths = latest_sampling_experiment(
            training_config.artifact_dir
        )
        sample_path = experiment_paths["samples"]

    sample_frame = pd.read_csv(sample_path)
    if "SMILES" not in sample_frame.columns:
        raise ValueError(f"Sample file does not contain a SMILES column: {sample_path}")
    smiles_list = sample_frame["SMILES"].fillna("").astype(str).tolist()
    metrics = generation_metrics(
        smiles_list=smiles_list,
        training_smiles=payload["splits"]["train"]["smiles"],
        property_name=data_config.property_name,
        guidance_scale=args.guidance_scale,
        device=device,
    )
    write_metrics(experiment_paths, "evaluation", metrics)
    plot_paths = save_property_distribution_plots(
        metrics["property_distributions"], experiment_paths["plots"]
    )
    if experiment_paths["manifest"].exists():
        relative_plot_paths = {
            name: str(Path(path).relative_to(experiment_paths["root"]))
            for name, path in plot_paths.items()
        }
        update_experiment_manifest(
            experiment_paths,
            {
                "last_evaluated_at": utc_timestamp(),
                "last_evaluation": {
                    "property_name": data_config.property_name,
                    "guidance_scale": args.guidance_scale,
                    "metrics": "metrics.json",
                    "plots": relative_plot_paths,
                },
            },
        )
    print(f"\nEvaluated sampling experiment: {experiment_paths['root']}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ChemBERTa latent diffusion with signed continuous-property guidance."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_arguments(
        subparser: argparse.ArgumentParser,
        *,
        epochs_default: int,
        batch_size_default: int,
    ) -> None:
        subparser.add_argument("--dataset-path", default="data/250k_zinc_clean.csv")
        subparser.add_argument("--generative-dataset-path", default="data/250k_zinc_clean.csv")
        subparser.add_argument(
            "--property-name", default="qed", choices=list(SUPPORTED_PROPERTIES)
        )
        subparser.add_argument("--property-smiles-column", default="smiles")
        subparser.add_argument("--artifact-dir", default="artifacts")
        subparser.add_argument(
            "--cache-dir",
            default="",
            help=(
                "Optional directory containing shared embedding caches. When omitted, "
                "caches are read from and written beneath --artifact-dir."
            ),
        )
        subparser.add_argument(
            "--projection-mode",
            choices=["none", "pca", "learned"],
            default="none",
            help=(
                "Latent reduction mode. 'learned' trains a token projection jointly "
                "with diffusion and keeps the embedding cache full-dimensional."
            ),
        )
        subparser.add_argument(
            "--latent-representation",
            choices=["token_sequence", "mean_pooled"],
            default="token_sequence",
            help=(
                "Diffusion sample unit: every encoder token (legacy default), or one "
                "attention-mask-aware mean-pooled vector per molecule."
            ),
        )
        subparser.add_argument(
            "--enable-latent-projection",
            action="store_true",
            help="Legacy alias for --projection-mode pca.",
        )
        subparser.add_argument(
            "--latent-projection-dim",
            type=int,
            default=256,
            help="Projected latent dimension when projection is enabled (default: 256).",
        )
        subparser.add_argument(
            "--latent-projection-fit-tokens",
            type=int,
            default=50000,
            help=(
                "Maximum training vectors used to fit PCA: tokens in token-sequence "
                "mode or molecules in mean-pooled mode (default: 50000)."
            ),
        )
        subparser.add_argument("--device", default="cuda")
        subparser.add_argument("--epochs", type=int, default=epochs_default)
        subparser.add_argument("--batch-size", type=int, default=batch_size_default)
        subparser.add_argument("--learning-rate", type=float, default=1e-4)
        subparser.add_argument(
            "--diffusion-learning-rate", type=float, default=1e-4,
            help="Learning rate for the diffusion denoiser (default: 1e-4).",
        )
        subparser.add_argument(
            "--decoder-learning-rate", type=float, default=1e-4,
            help="Learning rate for the jointly trained SELFIES decoder (default: 1e-4).",
        )
        subparser.add_argument(
            "--projection-learning-rate", type=float, default=1e-5,
            help="Lower learning rate for the learned encoder projection (default: 1e-5).",
        )
        subparser.add_argument(
            "--projection-reconstruction-weight", type=float, default=1.0,
        )
        subparser.add_argument(
            "--projection-variance-weight", type=float, default=1e-3,
        )
        subparser.add_argument("--weight-decay", type=float, default=1e-5)
        subparser.add_argument("--grad-clip-norm", type=float, default=1.0)
        subparser.add_argument("--max-length", type=int, default=160)
        subparser.add_argument("--seed", type=int, default=42)
        subparser.add_argument("--training-seed", type=int, default=None,
                               help="Model initialization and training RNG seed; --seed fixes the data split/cache.")
        subparser.add_argument("--joint-decoder-diffusion", action="store_true",
                               help="Jointly train decoder and denoiser for fixed PCA latents.")
        subparser.add_argument(
            "--split-strategy", default="scaffold", choices=["scaffold", "random"]
        )
        subparser.add_argument("--disable-canonicalize", action="store_true")
        subparser.add_argument("--generative-smiles-column", default="smiles")
        subparser.add_argument("--generative-split-column", default=None,
                               help="Column containing fixed train, val, and test assignments.")
        subparser.add_argument("--generative-split-strategy", default="scaffold", choices=["scaffold"])

        subparser.add_argument("--encoder-name", default="DeepChem/ChemBERTa-100M-MLM")
        subparser.add_argument("--pooling", default="mean", choices=["mean", "cls"])
        subparser.add_argument("--unfreeze-encoder", action="store_true")
        subparser.add_argument("--local-files-only", action="store_true")

        subparser.add_argument("--decoder-d-model", type=int, default=512)
        subparser.add_argument("--decoder-layers", type=int, default=4)
        subparser.add_argument("--decoder-heads", type=int, default=8)
        subparser.add_argument("--decoder-ffn-dim", type=int, default=2048)
        subparser.add_argument("--decoder-dropout", type=float, default=0.1)
        subparser.add_argument("--decoder-latent-noise-std", type=float, default=0.1)
        subparser.add_argument("--decoder-robust-loss-weight", type=float, default=0.5)

        subparser.add_argument("--diffusion-steps", type=int, default=500)
        subparser.add_argument("--diffusion-hidden-dim", type=int, default=768)
        subparser.add_argument("--diffusion-dropout", type=float, default=0.1)
        subparser.add_argument("--time-embedding-dim", type=int, default=256)
        subparser.add_argument("--mlp-hidden-dim", type=int, default=6144)
        subparser.add_argument("--mlp-token-hidden-dim", type=int, default=256)
        subparser.add_argument("--beta-schedule", default="linear", choices=["linear"])
        subparser.add_argument(
            "--resume-diffusion",
            action="store_true",
            help=(
                "Continue train-diffusion from <artifact-dir>/checkpoints/diffusion.pt. "
                "--epochs is the number of additional epochs."
            ),
        )
        subparser.add_argument(
            "--resume-diffusion-checkpoint",
            default="",
            help=(
                "Optional checkpoint path to resume instead of the default diffusion.pt; "
                "implies --resume-diffusion."
            ),
        )
        subparser.add_argument(
            "--property-regressor-hidden-dim",
            "--classifier-hidden-dim",
            dest="classifier_hidden_dim",
            type=int,
            default=256,
        )
        subparser.add_argument(
            "--property-regressor-dropout",
            "--classifier-dropout",
            dest="classifier_dropout",
            type=float,
            default=0.1,
        )
        subparser.add_argument(
            "--guidance-regressor-hidden-dim",
            "--guidance-classifier-hidden-dim",
            "--latent-head-hidden-dim",
            dest="guidance_classifier_hidden_dim",
            type=int,
            default=512,
        )
        subparser.add_argument(
            "--guidance-regressor-dropout",
            "--guidance-classifier-dropout",
            "--latent-head-dropout",
            dest="guidance_classifier_dropout",
            type=float,
            default=0.1,
        )
        subparser.add_argument(
            "--guidance-regressor-time-embedding-dim",
            "--guidance-classifier-time-embedding-dim",
            dest="guidance_classifier_time_embedding_dim",
            type=int,
            default=512,
        )
        subparser.add_argument(
            "--guidance-regressor-layers",
            "--guidance-classifier-layers",
            dest="guidance_classifier_layers",
            type=int,
            default=3,
        )
        subparser.add_argument(
            "--distill-guidance-regressor",
            "--distill-guidance-classifier",
            "--distill-latent-head",
            dest="distill_guidance_classifier",
            action="store_true",
        )

    command_defaults = {
        "prepare-embeddings": (10, 512),
        "prepare-generative-embeddings": (10, 32),
        "train-decoder": (10, 128),
        "train-diffusion": (20, 256),
        "train-property-regressor": (15, 512),
        "train-guidance-regressor": (20, 512),
        "train-classifier": (15, 512),
        "train-guidance-classifier": (20, 512),
        "train-latent-head": (20, 512),
        "train-all": (20, 128),
    }
    for name, (epochs_default, batch_size_default) in command_defaults.items():
        subparser = subparsers.add_parser(name)
        add_shared_arguments(
            subparser,
            epochs_default=epochs_default,
            batch_size_default=batch_size_default,
        )

    benchmark_parser = subparsers.add_parser("estimate-diffusion-time")
    add_shared_arguments(benchmark_parser, epochs_default=20, batch_size_default=256)
    benchmark_parser.add_argument("--benchmark-warmup-steps", type=int, default=2)
    benchmark_parser.add_argument("--benchmark-steps", type=int, default=5)

    sample_parser = subparsers.add_parser("sample")
    add_shared_arguments(sample_parser, epochs_default=10, batch_size_default=512)
    sample_parser.add_argument("--num-samples", type=int, default=1024)
    sample_parser.add_argument("--skip-evaluation", action="store_true",
                               help="Save molecules without the legacy per-run metrics; use the study evaluator later.")
    sample_parser.add_argument("--sample-batch-size", type=int, default=64)
    sample_parser.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help=(
            "Random seed used only for generation. When omitted, --seed is used. "
            "Keep --seed fixed to the embedding-cache split and vary this option "
            "for repeated sampling runs."
        ),
    )
    sample_parser.add_argument(
        "--experiment-name",
        default="",
        help="Optional unique run-folder name; default includes time, property, scale, and seed.",
    )
    sample_parser.add_argument(
        "--guidance-scale",
        "--guidance-max",
        dest="guidance_scale",
        type=float,
        default=0.0,
        help="Positive increases, negative decreases, and zero disables guidance.",
    )
    sample_parser.add_argument("--guidance-gamma", type=float, default=1.0)
    sample_parser.add_argument("--temperature", type=float, default=1.0)

    evaluate_parser = subparsers.add_parser("evaluate")
    add_shared_arguments(evaluate_parser, epochs_default=10, batch_size_default=512)
    evaluate_parser.add_argument("--samples-path", default="")
    evaluate_parser.add_argument("--guidance-scale", type=float, default=0.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    command_map = {
        "prepare-embeddings": command_prepare_embeddings,
        "prepare-generative-embeddings": command_prepare_generative_embeddings,
        "train-decoder": command_train_decoder,
        "estimate-diffusion-time": command_estimate_diffusion_time,
        "train-diffusion": command_train_diffusion,
        "train-property-regressor": command_train_property_regressor,
        "train-guidance-regressor": command_train_guidance_regressor,
        "train-classifier": command_train_property_regressor,
        "train-guidance-classifier": command_train_guidance_regressor,
        "train-latent-head": command_train_guidance_regressor,
        "train-all": command_train_all,
        "sample": command_sample,
        "evaluate": command_evaluate,
    }
    command_map[args.command](args)


if __name__ == "__main__":
    main()

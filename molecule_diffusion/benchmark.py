from __future__ import annotations

import math
import time

import torch
from torch import optim

from .diffusion import DenoiserModel, DiffusionSchedule, build_denoiser
from .latent import LatentStandardizer
from .utils import cycle, resolve_device, set_seed


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _move_diffusion_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_embeddings = batch["sequence_embeddings"].to(device).float()
    attention_mask = batch["attention_mask"].to(device)
    return sequence_embeddings, attention_mask


def _training_step(
    model: DenoiserModel,
    schedule: DiffusionSchedule,
    optimizer: optim.Optimizer,
    batch: dict,
    device: torch.device,
    grad_clip_norm: float,
) -> None:
    sequence_embeddings, attention_mask = _move_diffusion_batch(batch, device)
    standardized_latents = schedule.standardize_latents(sequence_embeddings)
    standardized_latents = standardized_latents * attention_mask.unsqueeze(-1)
    optimizer.zero_grad(set_to_none=True)
    loss = model.training_loss(standardized_latents, attention_mask, schedule)["loss"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()


def _validation_step(
    model: DenoiserModel,
    schedule: DiffusionSchedule,
    batch: dict,
    device: torch.device,
) -> None:
    sequence_embeddings, attention_mask = _move_diffusion_batch(batch, device)
    standardized_latents = schedule.standardize_latents(sequence_embeddings)
    standardized_latents = standardized_latents * attention_mask.unsqueeze(-1)
    model.training_loss(standardized_latents, attention_mask, schedule)


def _format_duration(seconds: float) -> str:
    rounded_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(rounded_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def command_estimate_diffusion_time(args) -> None:
    # Reuse the canonical configuration/cache helpers without creating a module
    # import cycle when the main CLI imports this command.
    from .main import (
        artifact_paths,
        make_configs,
        make_latent_projection_config,
        make_generative_data_config,
        prepare_generative_embeddings,
        requested_latent_representation,
        split_payload_to_loader,
    )

    if args.benchmark_steps < 1:
        raise ValueError("--benchmark-steps must be at least 1.")
    if args.benchmark_warmup_steps < 0:
        raise ValueError("--benchmark-warmup-steps cannot be negative.")

    configs = make_configs(args)
    (
        _data_config,
        encoder_config,
        _decoder_config,
        diffusion_config,
        _classifier_config,
        _guidance_config,
        training_config,
    ) = configs
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

    diffusion_config.latent_dim = int(payload["latent_dim"])
    diffusion_config.max_sequence_length = int(payload["sequence_length"])
    train_payload = payload["splits"]["train"]
    val_payload = payload["splits"]["val"]
    train_loader = split_payload_to_loader(
        train_payload,
        generative_data_config.batch_size,
        True,
        generative_data_config.num_workers,
    )
    val_loader = split_payload_to_loader(
        val_payload,
        generative_data_config.batch_size,
        False,
        generative_data_config.num_workers,
    )

    train_attention_mask = train_payload["attention_mask"]
    standardizer = LatentStandardizer.from_metadata(
        payload["latent_standardization"],
        latent_dim=diffusion_config.latent_dim,
    )
    sequence_lengths = train_attention_mask.sum(dim=1).long()
    sequence_length_counts = torch.bincount(
        sequence_lengths,
        minlength=diffusion_config.max_sequence_length + 1,
    ).tolist()
    model = build_denoiser(diffusion_config).to(device)
    schedule = DiffusionSchedule(
        diffusion_config,
        latent_standardizer=standardizer,
        sequence_length_counts=sequence_length_counts,
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    train_batches = cycle(train_loader)
    model.train()
    for _ in range(args.benchmark_warmup_steps):
        _training_step(
            model,
            schedule,
            optimizer,
            next(train_batches),
            device,
            training_config.grad_clip_norm,
        )
    _synchronize(device)
    training_started = time.perf_counter()
    for _ in range(args.benchmark_steps):
        _training_step(
            model,
            schedule,
            optimizer,
            next(train_batches),
            device,
            training_config.grad_clip_norm,
        )
    _synchronize(device)
    training_seconds_per_batch = (
        time.perf_counter() - training_started
    ) / args.benchmark_steps

    validation_steps = min(args.benchmark_steps, len(val_loader))
    validation_batches = cycle(val_loader)
    model.eval()
    _synchronize(device)
    validation_started = time.perf_counter()
    with torch.no_grad():
        for _ in range(validation_steps):
            _validation_step(model, schedule, next(validation_batches), device)
    _synchronize(device)
    validation_seconds_per_batch = (
        time.perf_counter() - validation_started
    ) / validation_steps

    train_batches_per_epoch = math.ceil(
        len(train_payload["sequence_embeddings"]) / generative_data_config.batch_size
    )
    validation_batches_per_epoch = math.ceil(
        len(val_payload["sequence_embeddings"]) / generative_data_config.batch_size
    )
    seconds_per_epoch = (
        train_batches_per_epoch * training_seconds_per_batch
        + validation_batches_per_epoch * validation_seconds_per_batch
    )
    total_seconds = seconds_per_epoch * training_config.epochs
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print("\nDiffusion training-time estimate")
    print("--------------------------------")
    print(f"device: {device}")
    print(f"model_parameters: {parameter_count:,}")
    print(f"train_examples: {len(train_payload['sequence_embeddings']):,}")
    print(f"validation_examples: {len(val_payload['sequence_embeddings']):,}")
    print(f"batch_size: {generative_data_config.batch_size}")
    print(f"epochs: {training_config.epochs}")
    print(f"measured_training_batches: {args.benchmark_steps}")
    print(f"standardizer_fit_examples: {len(train_payload['sequence_embeddings'])}")
    print(f"training_seconds_per_batch: {training_seconds_per_batch:.3f}")
    print(f"validation_seconds_per_batch: {validation_seconds_per_batch:.3f}")
    print(f"estimated_seconds_per_epoch: {seconds_per_epoch:.1f}")
    print(f"estimated_time_per_epoch: {_format_duration(seconds_per_epoch)}")
    print(f"estimated_total_time: {_format_duration(total_seconds)}")
    print(
        "planning_range_20_percent: "
        f"{_format_duration(total_seconds * 0.8)} to {_format_duration(total_seconds * 1.2)}"
    )
    print("checkpoint_written: False")
    print(
        "note: embedding-cache creation and one-time data preparation are not included; "
        "rerun the estimate after changing device, batch size, sequence length, or model size."
    )

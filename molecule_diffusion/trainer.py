from __future__ import annotations

from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from .classifier import BBBPClassifier, PropertyRegressor
from .config import TrainingConfig
from .decoder import EmbeddingToSelfiesDecoder
from .diffusion import DenoiserModel, DiffusionSchedule
from .latent import LatentStandardizer
from .metrics import binary_classification_metrics, regression_metrics
from .property_head import NoisyLatentPropertyClassifier, NoisyLatentPropertyRegressor
from .projection import LearnedTokenProjector


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if key == "sequence_embeddings":
                value = value.float()
            moved[key] = value
        else:
            moved[key] = value
    return moved


def _standardized_sequence_embeddings(
    batch: dict,
    latent_standardizer: LatentStandardizer,
) -> torch.Tensor:
    attention_mask = batch["attention_mask"]
    valid_tokens = attention_mask.unsqueeze(-1).to(
        dtype=batch["sequence_embeddings"].dtype
    )
    standardized_latents = latent_standardizer.standardize(
        batch["sequence_embeddings"]
    )
    return standardized_latents * valid_tokens


def _run_decoder_epoch(
    model: EmbeddingToSelfiesDecoder,
    latent_standardizer: LatentStandardizer,
    dataloader,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    grad_clip_norm: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_clean = 0.0
    total_robust = 0.0
    total_token_accuracy = 0.0

    for batch in tqdm(dataloader, desc="decoder", leave=False):
        batch = _move_batch_to_device(batch, device)
        attention_mask = batch["attention_mask"]
        standardized_latents = _standardized_sequence_embeddings(
            batch, latent_standardizer
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        outputs = model.training_loss(
            standardized_latents,
            attention_mask,
            batch["selfies_ids"],
        )
        loss = outputs["loss"]
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        total_loss += loss.item() * batch["embeddings"].size(0)
        total_clean += outputs["clean_loss"].item() * batch["embeddings"].size(0)
        total_robust += outputs["robust_loss"].item() * batch["embeddings"].size(0)
        total_token_accuracy += outputs["token_accuracy"].item() * batch["embeddings"].size(0)

    size = len(dataloader.dataset)
    return {
        "loss": total_loss / size,
        "clean_loss": total_clean / size,
        "robust_loss": total_robust / size,
        "token_accuracy": total_token_accuracy / size,
    }


def train_decoder(
    model: EmbeddingToSelfiesDecoder,
    latent_standardizer: LatentStandardizer,
    train_loader,
    val_loader,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, float]:
    optimizer = optim.AdamW(
        model.parameters(), lr=config.decoder_learning_rate, weight_decay=config.weight_decay
    )
    best_state = None
    best_val = float("inf")
    history = {}

    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}", flush=True)
        train_metrics = _run_decoder_epoch(
            model,
            latent_standardizer,
            train_loader,
            optimizer,
            device,
            config.grad_clip_norm,
        )
        with torch.no_grad():
            val_metrics = _run_decoder_epoch(
                model,
                latent_standardizer,
                val_loader,
                None,
                device,
                config.grad_clip_norm,
            )
        history = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_clean_loss": train_metrics["clean_loss"],
            "train_robust_loss": train_metrics["robust_loss"],
            "train_token_accuracy": train_metrics["token_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_clean_loss": val_metrics["clean_loss"],
            "val_robust_loss": val_metrics["robust_loss"],
            "val_token_accuracy": val_metrics["token_accuracy"],
        }
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _run_diffusion_epoch(
    model: DenoiserModel,
    schedule: DiffusionSchedule,
    dataloader,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="diffusion", leave=False):
        batch = _move_batch_to_device(batch, device)
        standardized_latents = _standardized_sequence_embeddings(
            batch, schedule.latent_standardizer
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        outputs = model.training_loss(
            standardized_latents,
            batch["attention_mask"],
            schedule,
        )
        loss = outputs["loss"]
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
        total_loss += loss.item() * batch["embeddings"].size(0)
    size = len(dataloader.dataset)
    return {"loss": total_loss / size}



def _run_learned_projection_epoch(
    projector: LearnedTokenProjector | None,
    model: DenoiserModel,
    decoder: EmbeddingToSelfiesDecoder,
    schedule: DiffusionSchedule,
    dataloader,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, float]:
    training = optimizer is not None
    if projector is not None:
        projector.train(training)
    model.train(training)
    decoder.train(training)
    totals = {"loss": 0.0, "diffusion_loss": 0.0, "reconstruction_loss": 0.0,
              "variance_loss": 0.0, "token_accuracy": 0.0}

    for batch in tqdm(dataloader, desc="learned-projection-diffusion", leave=False):
        batch = _move_batch_to_device(batch, device)
        attention_mask = batch["attention_mask"].to(dtype=torch.bool)
        if training:
            optimizer.zero_grad(set_to_none=True)
        projected = (
            projector(batch["sequence_embeddings"], attention_mask)
            if projector is not None
            else _standardized_sequence_embeddings(batch, schedule.latent_standardizer)
        )
        diffusion_outputs = model.training_loss(projected, attention_mask, schedule)
        decoder_outputs = decoder.training_loss(
            projected, attention_mask, batch["selfies_ids"]
        )
        valid = projected[attention_mask]
        if valid.size(0) > 1:
            feature_std = valid.float().std(dim=0, unbiased=False)
            variance_loss = torch.relu(1.0 - feature_std).mean()
        else:
            variance_loss = projected.new_zeros(())
        loss = (
            diffusion_outputs["loss"]
            + config.projection_reconstruction_weight * decoder_outputs["loss"]
            + config.projection_variance_weight * variance_loss
        )
        if training:
            loss.backward()
            parameters = list(model.parameters()) + list(decoder.parameters())
            if projector is not None:
                parameters += list(projector.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip_norm)
            optimizer.step()
        batch_size = batch["sequence_embeddings"].size(0)
        totals["loss"] += float(loss.detach()) * batch_size
        totals["diffusion_loss"] += float(diffusion_outputs["loss"].detach()) * batch_size
        totals["reconstruction_loss"] += float(decoder_outputs["loss"].detach()) * batch_size
        totals["variance_loss"] += float(variance_loss.detach()) * batch_size
        totals["token_accuracy"] += float(decoder_outputs["token_accuracy"].detach()) * batch_size

    size = len(dataloader.dataset)
    return {key: value / size for key, value in totals.items()}


def train_learned_projection_diffusion(
    projector: LearnedTokenProjector | None,
    model: DenoiserModel,
    decoder: EmbeddingToSelfiesDecoder,
    schedule: DiffusionSchedule,
    train_loader,
    val_loader,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, object]:
    """Jointly train denoiser and decoder, with an optional learned projection."""
    if config.epochs <= 0:
        raise ValueError("Diffusion epochs must be positive.")
    parameter_groups = [
        {"params": model.parameters(), "lr": config.diffusion_learning_rate},
        {"params": decoder.parameters(), "lr": config.decoder_learning_rate},
    ]
    if projector is not None:
        parameter_groups.append(
            {"params": projector.parameters(), "lr": config.projection_learning_rate}
        )
    optimizer = optim.AdamW(parameter_groups, weight_decay=config.weight_decay)
    best_val = float("inf")
    best_states = None
    best_metrics = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}", flush=True)
        train_metrics = _run_learned_projection_epoch(
            projector, model, decoder, schedule, train_loader, optimizer, device, config
        )
        with torch.no_grad():
            val_metrics = _run_learned_projection_epoch(
                projector, model, decoder, schedule, val_loader, None, device, config
            )
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_metrics = dict(row)
            modules = (model, decoder) if projector is None else (projector, model, decoder)
            best_states = tuple(
                {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
                for module in modules
            )
    if best_states is None or best_metrics is None:
        raise RuntimeError("Joint projection training did not produce a checkpoint.")
    modules = (model, decoder) if projector is None else (projector, model, decoder)
    for module, state in zip(modules, best_states):
        module.load_state_dict(state)
    return {
        **best_metrics,
        "best_epoch": int(best_metrics["epoch"]),
        "best_val_loss": float(best_metrics["val_loss"]),
        "completed_epoch": config.epochs,
        "history": history,
        "learning_rates": {
            "diffusion": config.diffusion_learning_rate,
            "decoder": config.decoder_learning_rate,
            "projection": config.projection_learning_rate if projector is not None else None,
        },
    }

def train_diffusion(
    model: DenoiserModel,
    schedule: DiffusionSchedule,
    train_loader,
    val_loader,
    device: torch.device,
    config: TrainingConfig,
    *,
    start_epoch: int = 0,
    initial_history: list[dict[str, float | int]] | None = None,
    initial_best_metrics: dict[str, float | int] | None = None,
    optimizer_state_dict: dict | None = None,
) -> dict[str, object]:
    if start_epoch < 0:
        raise ValueError("start_epoch must be non-negative.")
    if config.epochs <= 0:
        raise ValueError("Diffusion epochs must be positive.")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.diffusion_learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer_restored = optimizer_state_dict is not None
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        # The continuation command controls the new learning rate and weight decay,
        # even when momentum/variance buffers are available in a resume checkpoint.
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = config.diffusion_learning_rate
            parameter_group["weight_decay"] = config.weight_decay

    epoch_history: list[dict[str, float | int]] = [
        dict(row) for row in (initial_history or [])
    ]
    best_state = None
    best_val = float("inf")
    best_metrics: dict[str, float | int] | None = None
    if initial_best_metrics is not None:
        best_metrics = dict(initial_best_metrics)
        best_val = float(best_metrics["val_loss"])
        # The standard diffusion.pt checkpoint stores the previously selected best
        # model, so its state is the correct initial best state for continuation.
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    final_epoch = start_epoch + config.epochs
    for epoch in range(start_epoch + 1, final_epoch + 1):
        print(f"\nEpoch {epoch}/{final_epoch}", flush=True)
        train_metrics = _run_diffusion_epoch(
            model, schedule, train_loader, optimizer, device, config
        )
        with torch.no_grad():
            val_metrics = _run_diffusion_epoch(
                model, schedule, val_loader, None, device, config
            )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
        }
        epoch_history.append(epoch_metrics)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_metrics = dict(epoch_metrics)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None or best_metrics is None:
        raise RuntimeError("Diffusion training completed without producing a checkpoint.")
    model.load_state_dict(best_state)
    return {
        **best_metrics,
        "best_epoch": int(best_metrics["epoch"]),
        "best_val_loss": float(best_metrics["val_loss"]),
        "completed_epoch": final_epoch,
        "start_epoch": start_epoch,
        "epochs_added": config.epochs,
        "resumed": start_epoch > 0,
        "optimizer_state_restored": optimizer_restored,
        "history": epoch_history,
    }


def _run_classifier_epoch(
    model: BBBPClassifier,
    dataloader,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    grad_clip_norm: float,
    num_calibration_bins: int = 10,
) -> dict[str, float | None]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels_all = []
    probabilities_all = []

    for batch in tqdm(dataloader, desc="classifier", leave=False):
        batch = _move_batch_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        outputs = model.loss(batch["embeddings"], batch["labels"])
        loss = outputs["loss"]
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        total_loss += loss.item() * batch["embeddings"].size(0)
        labels_all.extend(batch["labels"].detach().cpu().tolist())
        probabilities_all.extend(outputs["probabilities"].detach().cpu().tolist())

    prediction_metrics = binary_classification_metrics(
        labels_all,
        probabilities_all,
        num_calibration_bins=num_calibration_bins,
    )
    return {
        "loss": total_loss / len(dataloader.dataset),
        "auc": prediction_metrics["roc_auc"],
        **prediction_metrics,
    }


def evaluate_classifier(
    model: BBBPClassifier,
    dataloader,
    device: torch.device,
    grad_clip_norm: float = 1.0,
    num_calibration_bins: int = 10,
) -> dict:
    with torch.no_grad():
        return _run_classifier_epoch(
            model, dataloader, None, device, grad_clip_norm, num_calibration_bins
        )


def _prefixed_metrics(prefix: str, metrics: dict) -> dict:
    prefixed = {f"{prefix}_{key}": value for key, value in metrics.items() if key != "auc"}
    if "auc" in metrics:
        prefixed[f"{prefix}_auc"] = metrics["auc"]
    return prefixed


def train_classifier(
    model: BBBPClassifier,
    train_loader,
    val_loader,
    device: torch.device,
    config: TrainingConfig,
    test_loader=None,
) -> dict[str, float | None]:
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = None
    best_score = float("-inf")
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}", flush=True)
        train_metrics = _run_classifier_epoch(model, train_loader, optimizer, device, config.grad_clip_norm)
        with torch.no_grad():
            val_metrics = _run_classifier_epoch(model, val_loader, None, device, config.grad_clip_norm)
        score = val_metrics["auc"] if val_metrics["auc"] is not None else -float(val_metrics["loss"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    train_metrics = evaluate_classifier(model, train_loader, device, config.grad_clip_norm)
    val_metrics = evaluate_classifier(model, val_loader, device, config.grad_clip_norm)
    history = {
        "epoch": best_epoch,
        **_prefixed_metrics("train", train_metrics),
        **_prefixed_metrics("val", val_metrics),
    }
    if test_loader is not None:
        test_metrics = evaluate_classifier(model, test_loader, device, config.grad_clip_norm)
        history.update(_prefixed_metrics("test", test_metrics))
    return history


def _run_guidance_classifier_epoch(
    model: NoisyLatentPropertyClassifier,
    schedule: DiffusionSchedule,
    dataloader,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    grad_clip_norm: float,
    distill: bool,
) -> dict[str, float | None]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels_all = []
    probabilities_all = []

    for batch in tqdm(dataloader, desc="guidance-classifier", leave=False):
        batch = _move_batch_to_device(batch, device)
        # The clean property classifier still consumes pooled embeddings, while
        # guidance follows diffusion in token-sequence coordinates.
        attention_mask = batch["attention_mask"].to(dtype=torch.bool)
        valid_tokens = attention_mask.unsqueeze(-1)
        clean_latents = schedule.standardize_latents(batch["sequence_embeddings"])
        clean_latents = clean_latents * valid_tokens.to(dtype=clean_latents.dtype)
        batch_size = clean_latents.size(0)
        timesteps = torch.randint(
            low=0,
            high=schedule.num_steps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn_like(clean_latents) * valid_tokens.to(dtype=clean_latents.dtype)
        noisy_latents, _ = schedule.q_sample(clean_latents, timesteps, noise=noise)
        noisy_latents = noisy_latents * valid_tokens.to(dtype=noisy_latents.dtype)

        if training:
            optimizer.zero_grad(set_to_none=True)
        if distill:
            outputs = model.distillation_loss(
                noisy_latents,
                timesteps,
                attention_mask,
                batch["target_probabilities"],
            )
        else:
            outputs = model.supervised_loss(
                noisy_latents,
                timesteps,
                attention_mask,
                batch["labels"],
            )
        loss = outputs["loss"]
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        total_loss += loss.item() * batch_size
        if "labels" in batch:
            labels_all.extend(batch["labels"].detach().cpu().tolist())
            probabilities_all.extend(outputs["probabilities"].detach().cpu().tolist())

    prediction_metrics = binary_classification_metrics(labels_all, probabilities_all)
    return {
        "loss": total_loss / len(dataloader.dataset),
        "auc": prediction_metrics["roc_auc"],
        **prediction_metrics,
    }


def train_guidance_classifier(
    model: NoisyLatentPropertyClassifier,
    schedule: DiffusionSchedule,
    train_loader,
    val_loader,
    device: torch.device,
    config: TrainingConfig,
    distill: bool,
) -> dict[str, float | None]:
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = None
    best_score = float("-inf")
    history = {}

    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}", flush=True)
        train_metrics = _run_guidance_classifier_epoch(
            model,
            schedule,
            train_loader,
            optimizer,
            device,
            config.grad_clip_norm,
            distill,
        )
        with torch.no_grad():
            val_metrics = _run_guidance_classifier_epoch(
                model,
                schedule,
                val_loader,
                None,
                device,
                config.grad_clip_norm,
                distill,
            )
        score = val_metrics["auc"] if val_metrics["auc"] is not None else -float(val_metrics["loss"])
        history = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
        }
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _run_property_regressor_epoch(
    model: PropertyRegressor,
    dataloader,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    grad_clip_norm: float,
    projector: LearnedTokenProjector | None = None,
) -> dict[str, float | int | None]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets_all: list[float] = []
    predictions_all: list[float] = []

    for batch in tqdm(dataloader, desc="property-regressor", leave=False):
        batch = _move_batch_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        embeddings = batch["embeddings"]
        if projector is not None:
            with torch.no_grad():
                attention_mask = batch["attention_mask"].to(dtype=torch.bool)
                projected_tokens = projector(
                    batch["sequence_embeddings"], attention_mask
                )
                weights = attention_mask.unsqueeze(-1).to(projected_tokens.dtype)
                embeddings = projected_tokens.sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        outputs = model.loss(embeddings, batch["labels"])
        loss = outputs["loss"]
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        total_loss += loss.item() * batch["embeddings"].size(0)
        targets_all.extend(batch["labels"].detach().cpu().tolist())
        predictions_all.extend(outputs["prediction"].detach().cpu().tolist())

    return {
        "loss": total_loss / len(dataloader.dataset),
        **regression_metrics(targets_all, predictions_all),
    }


def evaluate_property_regressor(
    model: PropertyRegressor,
    dataloader,
    device: torch.device,
    grad_clip_norm: float = 1.0,
    projector: LearnedTokenProjector | None = None,
) -> dict[str, float | int | None]:
    with torch.no_grad():
        return _run_property_regressor_epoch(
            model, dataloader, None, device, grad_clip_norm, projector
        )


def train_property_regressor(
    model: PropertyRegressor,
    train_loader,
    val_loader,
    device: torch.device,
    config: TrainingConfig,
    test_loader=None,
    projector: LearnedTokenProjector | None = None,
) -> dict[str, float | int | None]:
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state = None
    best_rmse = float("inf")
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}", flush=True)
        _run_property_regressor_epoch(
            model, train_loader, optimizer, device, config.grad_clip_norm, projector
        )
        with torch.no_grad():
            val_metrics = _run_property_regressor_epoch(
                model, val_loader, None, device, config.grad_clip_norm, projector
            )
        if float(val_metrics["rmse"]) < best_rmse:
            best_rmse = float(val_metrics["rmse"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    train_metrics = evaluate_property_regressor(model, train_loader, device, projector=projector)
    val_metrics = evaluate_property_regressor(model, val_loader, device, projector=projector)
    history = {
        "epoch": best_epoch,
        **_prefixed_metrics("train", train_metrics),
        **_prefixed_metrics("val", val_metrics),
    }
    if test_loader is not None:
        history.update(
            _prefixed_metrics(
                "test", evaluate_property_regressor(model, test_loader, device, projector=projector)
            )
        )
    return history


def _run_guidance_regressor_epoch(
    model: NoisyLatentPropertyRegressor,
    schedule: DiffusionSchedule,
    dataloader,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    grad_clip_norm: float,
    distill: bool,
    projector: LearnedTokenProjector | None = None,
) -> dict[str, float | int | None]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets_all: list[float] = []
    predictions_all: list[float] = []

    for batch in tqdm(dataloader, desc="guidance-regressor", leave=False):
        batch = _move_batch_to_device(batch, device)
        attention_mask = batch["attention_mask"].to(dtype=torch.bool)
        valid_tokens = attention_mask.unsqueeze(-1)
        clean_latents = batch["sequence_embeddings"]
        if projector is not None:
            with torch.no_grad():
                clean_latents = projector(clean_latents, attention_mask)
        clean_latents = schedule.standardize_latents(clean_latents)
        clean_latents = clean_latents * valid_tokens.to(dtype=clean_latents.dtype)
        batch_size = clean_latents.size(0)
        timesteps = torch.randint(
            0, schedule.num_steps, (batch_size,), device=device, dtype=torch.long
        )
        noise = torch.randn_like(clean_latents) * valid_tokens.to(
            dtype=clean_latents.dtype
        )
        noisy_latents, _ = schedule.q_sample(clean_latents, timesteps, noise=noise)
        noisy_latents = noisy_latents * valid_tokens.to(dtype=noisy_latents.dtype)

        if training:
            optimizer.zero_grad(set_to_none=True)
        targets = batch["target_predictions"] if distill else batch["labels"]
        outputs = model.supervised_loss(
            noisy_latents, timesteps, attention_mask, targets
        )
        loss = outputs["loss"]
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        total_loss += loss.item() * batch_size
        targets_all.extend(batch["labels"].detach().cpu().tolist())
        predictions_all.extend(outputs["prediction"].detach().cpu().tolist())

    return {
        "loss": total_loss / len(dataloader.dataset),
        **regression_metrics(targets_all, predictions_all),
    }


def train_guidance_regressor(
    model: NoisyLatentPropertyRegressor,
    schedule: DiffusionSchedule,
    train_loader,
    val_loader,
    device: torch.device,
    config: TrainingConfig,
    distill: bool = False,
    projector: LearnedTokenProjector | None = None,
) -> dict[str, float | int | None]:
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state = None
    best_rmse = float("inf")
    history: dict[str, float | int | None] = {}

    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}", flush=True)
        train_metrics = _run_guidance_regressor_epoch(
            model,
            schedule,
            train_loader,
            optimizer,
            device,
            config.grad_clip_norm,
            distill,
            projector,
        )
        with torch.no_grad():
            val_metrics = _run_guidance_regressor_epoch(
                model,
                schedule,
                val_loader,
                None,
                device,
                config.grad_clip_norm,
                distill,
                projector,
            )
        history = {
            "epoch": epoch,
            **_prefixed_metrics("train", train_metrics),
            **_prefixed_metrics("val", val_metrics),
        }
        if float(val_metrics["rmse"]) < best_rmse:
            best_rmse = float(val_metrics["rmse"])
            best_state = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def save_model_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    metadata: dict,
    *,
    optimizer: optim.Optimizer | None = None,
    training_state: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "metadata": metadata}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if training_state is not None:
        payload["training_state"] = training_state
    torch.save(payload, path)

def load_checkpoint(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)

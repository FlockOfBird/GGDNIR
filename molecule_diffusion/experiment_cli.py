from __future__ import annotations

from pathlib import Path

import torch

from .config import DiffusionConfig, GuidanceClassifierConfig
from .diffusion import DiffusionSchedule
from .experiment import (
    build_distillation_decision,
    evaluate_guidance_at_timesteps,
    noise_fractions_to_timesteps,
)
from .latent import LatentStandardizer
from .property_head import NoisyLatentPropertyClassifier
from .trainer import (
    evaluate_classifier,
    load_checkpoint,
    save_model_checkpoint,
    train_guidance_classifier,
)
from .utils import ensure_dir, resolve_device, save_json, set_seed


def _load_matching_schedule(
    diffusion_path: Path,
    payload: dict,
    encoder_config,
    device: torch.device,
) -> tuple[DiffusionConfig, DiffusionSchedule]:
    if not diffusion_path.exists():
        raise FileNotFoundError(
            "The distillation experiment requires checkpoints/diffusion.pt. "
            "Train the diffusion model first."
        )
    diffusion_metadata = load_checkpoint(diffusion_path)["metadata"]
    if diffusion_metadata.get("latent_representation") != "chemberta_token_sequence_v1":
        raise ValueError("The experiment requires a token-sequence diffusion checkpoint.")
    schedule_config = DiffusionConfig(**diffusion_metadata["model_config"])
    latent_dim = int(payload["latent_dim"])
    if schedule_config.latent_dim != latent_dim:
        raise ValueError("Property embeddings and diffusion use different latent dimensions.")
    if schedule_config.max_sequence_length != int(payload["sequence_length"]):
        raise ValueError(
            "Property and diffusion caches use different sequence lengths; regenerate them "
            "with the same --max-length."
        )
    latent_standardizer = LatentStandardizer.from_metadata(
        diffusion_metadata.get("latent_standardization"),
        latent_dim=latent_dim,
    )
    if latent_standardizer.enabled:
        diffusion_encoder_config = latent_standardizer.provenance.get("encoder_config")
        if not isinstance(diffusion_encoder_config, dict):
            raise ValueError("The diffusion checkpoint is missing encoder provenance.")
        if diffusion_encoder_config.get("model_name") != encoder_config.model_name:
            raise ValueError("Property embeddings and diffusion use different ChemBERTa models.")
    schedule = DiffusionSchedule(
        schedule_config,
        latent_standardizer=latent_standardizer,
        sequence_length_counts=diffusion_metadata.get("sequence_length_counts"),
    ).to(device)
    return schedule_config, schedule


def _attach_teacher_probabilities(classifier, payload: dict, device: torch.device) -> None:
    classifier.eval()
    for split_payload in payload["splits"].values():
        probabilities = []
        embeddings = split_payload["embeddings"]
        with torch.no_grad():
            for batch in embeddings.split(1024):
                probabilities.append(classifier(batch.to(device))["probabilities"].cpu())
        split_payload["target_probabilities"] = torch.cat(probabilities)


def command_compare_guidance_training(args) -> None:
    # Imported lazily to avoid a module cycle while reusing the canonical cache
    # and checkpoint constructors from the CLI entry point.
    from .main import (
        artifact_paths,
        instantiate_classifier,
        make_configs,
        make_generative_data_config,
        prepare_property_embeddings,
        print_data_report,
        split_payload_to_loader,
    )

    configs = make_configs(args)
    (
        data_config,
        encoder_config,
        _decoder_config,
        _diffusion_config,
        _classifier_config,
        guidance_config,
        training_config,
    ) = configs
    device = resolve_device(training_config.device)
    paths = artifact_paths(training_config.artifact_dir, args.cache_dir)
    payload = prepare_property_embeddings(
        paths,
        data_config,
        encoder_config,
        device,
        generative_data_config=make_generative_data_config(args),
    )
    if "test" not in payload["splits"]:
        raise ValueError("The experiment requires train, validation, and test splits.")
    if not paths["classifier"].exists():
        raise FileNotFoundError(
            "The experiment requires checkpoints/classifier.pt. Train the clean classifier first."
        )

    schedule_config, schedule = _load_matching_schedule(
        paths["diffusion"], payload, encoder_config, device
    )
    classifier = instantiate_classifier(paths["classifier"], device)
    _attach_teacher_probabilities(classifier, payload, device)
    val_loader = split_payload_to_loader(
        payload["splits"]["val"], data_config.batch_size, False, data_config.num_workers
    )
    test_loader = split_payload_to_loader(
        payload["splits"]["test"], data_config.batch_size, False, data_config.num_workers
    )
    clean_validation = evaluate_classifier(
        classifier, val_loader, device, num_calibration_bins=args.calibration_bins
    )
    clean_test = evaluate_classifier(
        classifier, test_loader, device, num_calibration_bins=args.calibration_bins
    )
    del classifier

    experiment_seeds = list(dict.fromkeys(args.experiment_seeds))
    if not experiment_seeds:
        raise ValueError("At least one experiment seed is required.")
    timesteps = noise_fractions_to_timesteps(
        args.experiment_noise_fractions,
        schedule.num_steps,
    )
    experiment_root = ensure_dir(
        paths["root"] / "experiments" / "guidance_distillation"
    )
    runs = []
    latent_dim = int(payload["latent_dim"])
    print_data_report("Guidance distillation experiment", payload)

    for seed in experiment_seeds:
        run = {"seed": seed}
        for arm, distill in (("supervised", False), ("distilled", True)):
            # Reset immediately before each arm. Architecture initialization,
            # shuffle order, sampled timesteps, and training noise therefore
            # match between the two arms within a seed.
            set_seed(seed)
            arm_config = GuidanceClassifierConfig(
                hidden_dim=guidance_config.hidden_dim,
                dropout=guidance_config.dropout,
                time_embedding_dim=guidance_config.time_embedding_dim,
                num_hidden_layers=guidance_config.num_hidden_layers,
                distill_from_classifier=distill,
            )
            train_loader = split_payload_to_loader(
                payload["splits"]["train"],
                data_config.batch_size,
                True,
                data_config.num_workers,
            )
            arm_val_loader = split_payload_to_loader(
                payload["splits"]["val"],
                data_config.batch_size,
                False,
                data_config.num_workers,
            )
            model = NoisyLatentPropertyClassifier(latent_dim, arm_config).to(device)
            training_metrics = train_guidance_classifier(
                model,
                schedule,
                train_loader,
                arm_val_loader,
                device,
                training_config,
                distill=distill,
            )
            evaluation_seed = seed + 1_000_000
            validation_metrics = evaluate_guidance_at_timesteps(
                model,
                schedule,
                val_loader,
                device,
                timesteps=timesteps,
                noise_seed=evaluation_seed,
                num_calibration_bins=args.calibration_bins,
            )
            test_metrics = evaluate_guidance_at_timesteps(
                model,
                schedule,
                test_loader,
                device,
                timesteps=timesteps,
                noise_seed=evaluation_seed + 1,
                num_calibration_bins=args.calibration_bins,
            )
            checkpoint_path = None
            if args.save_experiment_checkpoints:
                checkpoint_path = experiment_root / f"seed_{seed}_{arm}.pt"
                save_model_checkpoint(
                    checkpoint_path,
                    model,
                    metadata={
                        "experiment_arm": arm,
                        "seed": seed,
                        "latent_dim": latent_dim,
                        "model_config": arm_config.to_dict(),
                        "diffusion_schedule_config": schedule_config.to_dict(),
                        "latent_standardization": schedule.latent_standardizer.to_metadata(),
                        "latent_representation": "chemberta_token_sequence_v1",
                    },
                )
            run[arm] = {
                "training": training_metrics,
                "validation": validation_metrics,
                "test": test_metrics,
                "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
            }
            del model
        runs.append(run)

    criteria = {
        "min_clean_validation_auc": args.min_clean_validation_auc,
        "max_clean_validation_ece": args.max_clean_validation_ece,
        "clean_brier_skill_must_be_positive": True,
        "clean_nll_must_beat_prevalence_null": True,
        "min_distillation_nll_improvement": args.min_distillation_nll_improvement,
        "distillation_brier_must_not_increase": True,
        "max_guidance_auc_drop": args.max_guidance_auc_drop,
        "max_guidance_ece_increase": args.max_guidance_ece_increase,
    }
    summary, decision = build_distillation_decision(
        clean_validation_metrics=clean_validation,
        runs=runs,
        min_clean_validation_auc=args.min_clean_validation_auc,
        max_clean_validation_ece=args.max_clean_validation_ece,
        min_distillation_nll_improvement=args.min_distillation_nll_improvement,
        max_guidance_auc_drop=args.max_guidance_auc_drop,
        max_guidance_ece_increase=args.max_guidance_ece_increase,
    )
    report = {
        "schema_version": 1,
        "experiment": "matched_guidance_supervision_vs_clean_classifier_distillation",
        "seeds": experiment_seeds,
        "noise_fractions": args.experiment_noise_fractions,
        "timesteps": timesteps,
        "calibration_bins": args.calibration_bins,
        "criteria": criteria,
        "clean_classifier": {
            "validation": clean_validation,
            "test_confirmation": clean_test,
        },
        "runs": runs,
        "summary": summary,
        "decision": decision,
    }
    report_path = experiment_root / "report.json"
    save_json(report_path, report)
    print(f"\n{decision['recommendation']}")
    print(f"Experiment report: {report_path}\n")

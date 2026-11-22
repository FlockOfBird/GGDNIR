"""Self-supervised molecular adaptation of a pretrained or scratch encoder.

The output is a regular Hugging Face directory. The diffusion pipeline loads it
through --encoder-name and freezes it while preparing the embedding cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    RobertaPreLayerNormConfig,
    get_scheduler,
)

from .config import DataConfig
from .data import build_data_bundle
from .smiles_tokenizer import build_atom_tokenizer
from .tokenizer import smiles_to_selfies
from .utils import set_seed


class TokenDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
        }


def create_lm_batch(
    batch: dict[str, torch.Tensor],
    *,
    objective: str,
    tokenizer,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    input_ids = batch["input_ids"].clone()
    attention_mask = batch["attention_mask"]
    labels = input_ids.clone()
    if objective == "clm":
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    if tokenizer.mask_token_id is None:
        raise ValueError("Masked language modeling requires a tokenizer mask token.")
    eligible = attention_mask.bool()
    for special_id in tokenizer.all_special_ids:
        eligible &= input_ids != special_id
    selected = (torch.rand(input_ids.shape, generator=generator) < 0.15) & eligible
    if not selected.any() and eligible.any():
        first = eligible.nonzero()[0]
        selected[first[0], first[1]] = True
    labels[~selected] = -100
    replacement = torch.rand(input_ids.shape, generator=generator)
    input_ids[selected & (replacement < 0.8)] = tokenizer.mask_token_id
    random_positions = selected & (replacement >= 0.8) & (replacement < 0.9)
    random_tokens = torch.randint(
        len(tokenizer), input_ids.shape, generator=generator, dtype=input_ids.dtype
    )
    input_ids[random_positions] = random_tokens[random_positions]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _dataset_digest(smiles: list[str]) -> str:
    digest = hashlib.sha256()
    for item in smiles:
        digest.update(item.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _encode_smiles(smiles: list[str], tokenizer, max_length: int) -> TokenDataset:
    encoded = tokenizer(
        smiles, padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    return TokenDataset(encoded["input_ids"], encoded["attention_mask"])


def resolve_training_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply preset defaults while retaining explicit command-line overrides."""
    small = args.preset == "small-smiles-mlm"
    defaults = {
        "max_length": 128 if small else 160,
        "batch_size": 64 if small else 8,
        "accumulation_steps": 4,
        "epochs": 100 if small else 3,
        "learning_rate": 3e-4 if small else 2e-5,
        "adam_beta2": 0.98 if small else 0.999,
        "adam_eps": 1e-6 if small else 1e-8,
        "warmup_ratio": 0.06 if small else 0.0,
        "lr_schedule": "linear" if small else "constant",
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if small:
        if args.objective == "clm" or args.source_model is not None:
            raise ValueError("small-smiles-mlm builds its own MLM model and tokenizer; omit --source-model.")
        args.scratch = True
        args.objective = "mlm"
    elif not args.source_model:
        raise ValueError("--source-model is required for the existing adaptation/scratch procedure.")
    if args.epochs < 1 or args.batch_size < 1 or args.accumulation_steps < 1:
        raise ValueError("Epochs, batch size, and accumulation steps must be positive.")
    if args.learning_rate <= 0 or args.max_length < 3:
        raise ValueError("Learning rate must be positive and maximum length at least three.")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.adam_beta2 < 1:
        raise ValueError("Warmup ratio and Adam beta2 must be in [0, 1).")
    if args.adam_eps <= 0:
        raise ValueError("Adam epsilon must be positive.")
    if small and (args.hidden_size < 1 or args.num_heads < 1 or args.num_layers < 1
                  or args.hidden_size % args.num_heads):
        raise ValueError("Layer/head counts must be positive and hidden size divisible by head count.")
    return args


def build_small_mlm(args: argparse.Namespace, train_smiles: list[str]):
    tokenizer = build_atom_tokenizer(train_smiles, args.max_length)
    config = RobertaPreLayerNormConfig(
        vocab_size=len(tokenizer), hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers, num_attention_heads=args.num_heads,
        intermediate_size=4 * args.hidden_size, hidden_act="gelu",
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-5, type_vocab_size=1,
        # RoBERTa position IDs start after the padding ID.
        max_position_embeddings=args.max_length + tokenizer.pad_token_id + 1,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer, AutoModelForMaskedLM.from_config(config)


def tokenization_report(
    smiles: list[str], tokenizer, max_length: int, *, description: str = "Tokenizer checks",
) -> dict:
    lengths, unknown, tokens = [], 0, 0
    unknown_id = tokenizer.unk_token_id
    for start in tqdm(
        range(0, len(smiles), 1024), desc=description,
        unit="batch", dynamic_ncols=True, mininterval=1.0,
    ):
        encoded = tokenizer(smiles[start:start + 1024], truncation=False, padding=False)
        for ids in encoded["input_ids"]:
            lengths.append(len(ids))
            unknown += ids.count(unknown_id)
            tokens += len(ids)
    return {
        "max_tokens_with_specials": max(lengths),
        "mean_tokens_with_specials": sum(lengths) / len(lengths),
        "truncated_molecules": sum(length > max_length for length in lengths),
        "unknown_tokens": unknown,
        "unknown_token_fraction": unknown / max(tokens, 1),
    }


def adapt_encoder(args: argparse.Namespace) -> dict:
    args = resolve_training_args(args)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Encoder output directory is already populated: {output_dir}")
    data_config = DataConfig(
        dataset_path=args.dataset_path,
        smiles_column=args.smiles_column,
        split_column=args.split_column,
        label_column=None,
        property_name=None,
        canonicalize_smiles=not args.disable_canonicalize,
        max_length=args.max_length,
        seed=args.data_seed,
        split_strategy="scaffold",
    )
    print("Preparing data: loading, canonicalizing, and splitting molecules...", flush=True)
    splits = build_data_bundle(data_config)
    train_smiles = [
        value for value in tqdm(
            splits.train[args.smiles_column].astype(str), desc="Checking training molecules",
            unit="mol", dynamic_ncols=True, mininterval=1.0,
        )
        if smiles_to_selfies(value) is not None
    ]
    val_smiles = [
        value for value in tqdm(
            splits.val[args.smiles_column].astype(str), desc="Checking validation molecules",
            unit="mol", dynamic_ncols=True, mininterval=1.0,
        )
        if smiles_to_selfies(value) is not None
    ]
    if not train_smiles or not val_smiles:
        raise ValueError("The molecular training and validation splits must be nonempty.")
    set_seed(args.training_seed)
    print("Initializing encoder model and tokenizer...", flush=True)
    if args.preset == "small-smiles-mlm":
        tokenizer, model = build_small_mlm(args, train_smiles)
        config = model.config
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.source_model, use_fast=True)
        config = AutoConfig.from_pretrained(args.source_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.cls_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer has no usable padding token.")
    objective = args.objective
    if objective == "auto":
        objective = "clm" if config.model_type == "llama" else "mlm"
    if args.scratch and objective != "mlm":
        raise ValueError("The scratch control uses the ChemBERTa-style MLM objective.")
    model_class = AutoModelForCausalLM if objective == "clm" else AutoModelForMaskedLM
    if args.preset != "small-smiles-mlm":
        model = (model_class.from_config(config) if args.scratch
                 else model_class.from_pretrained(args.source_model))
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "use_cache", False):
        model.config.use_cache = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    model.to(device)
    print(f"Tokenizing {len(train_smiles):,} training molecules...", flush=True)
    train_data = _encode_smiles(train_smiles, tokenizer, args.max_length)
    print(f"Tokenizing {len(val_smiles):,} validation molecules...", flush=True)
    val_data = _encode_smiles(val_smiles, tokenizer, args.max_length)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, args.adam_beta2),
        eps=args.adam_eps, weight_decay=0.01,
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        args.lr_schedule, optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    history = []
    best_val = float("inf")
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenization = {
        "train": tokenization_report(
            train_smiles, tokenizer, args.max_length, description="Training tokenizer checks",
        ),
        "val": tokenization_report(
            val_smiles, tokenizer, args.max_length, description="Validation tokenizer checks",
        ),
    }
    print(json.dumps({"preset": args.preset, "trainable_parameters": trainable,
                      "vocab_size": len(tokenizer), "tokenization": tokenization}), flush=True)
    print(
        f"Encoder {objective.upper()} training: {args.epochs} epochs | "
        f"{len(train_data):,} train / {len(val_data):,} validation molecules | "
        f"effective batch {args.batch_size * args.accumulation_steps} | "
        f"{total_steps:,} optimizer steps | {'bf16' if use_bf16 else 'fp32'}",
        flush=True,
    )
    optimizer_steps = 0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total = 0.0
        train_tokens = 0
        window_tokens = 0
        train_progress = tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs} train",
            unit="batch", dynamic_ncols=True, mininterval=1.0,
        )
        for index, source in enumerate(train_progress, 1):
            batch = create_lm_batch(source, objective=objective, tokenizer=tokenizer)
            count = int((batch["labels"][:, 1:] != -100).sum()) if objective == "clm" else int((batch["labels"] != -100).sum())
            if count == 0:
                raise ValueError("A training batch has no language-model targets.")
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Encoder adaptation produced a non-finite loss.")
            train_total += float(loss.detach()) * count
            train_tokens += count
            window_tokens += count
            (loss * count).backward()
            if index % args.accumulation_steps == 0 or index == len(train_loader):
                # Normalize by actual targets, including a short final accumulation window.
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(window_tokens)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
                window_tokens = 0
            train_progress.set_postfix(
                loss=f"{train_total / train_tokens:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                step=f"{optimizer_steps}/{total_steps}", refresh=False,
            )
        model.eval()
        val_total = 0.0
        val_tokens = 0
        val_generator = torch.Generator().manual_seed(args.data_seed + 100_003)
        with torch.no_grad():
            val_progress = tqdm(
                val_loader, desc=f"Epoch {epoch}/{args.epochs} validation",
                unit="batch", dynamic_ncols=True, mininterval=1.0,
            )
            for source in val_progress:
                batch = create_lm_batch(
                    source, objective=objective, tokenizer=tokenizer,
                    generator=val_generator,
                )
                count = int((batch["labels"][:, 1:] != -100).sum()) if objective == "clm" else int((batch["labels"] != -100).sum())
                if count == 0:
                    raise ValueError("A validation batch has no language-model targets.")
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Encoder validation produced a non-finite loss.")
                val_total += float(loss) * count
                val_tokens += count
                val_progress.set_postfix(loss=f"{val_total / val_tokens:.4f}", refresh=False)
        row = {
            "epoch": epoch,
            "train_loss": train_total / train_tokens,
            "val_loss": val_total / val_tokens,
            "learning_rate": scheduler.get_last_lr()[0],
            "optimizer_steps": optimizer_steps,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "elapsed_seconds": time.perf_counter() - started,
        }
        row["eta_seconds"] = row["elapsed_seconds"] / epoch * (args.epochs - epoch)
        history.append(row)
        print(json.dumps(row), flush=True)
        (output_dir / "training_history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_epoch = epoch
            model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)
        print(
            f"Epoch {epoch}/{args.epochs} complete | "
            f"train loss {row['train_loss']:.4f} | val loss {row['val_loss']:.4f} | "
            f"best val {best_val:.4f} (epoch {best_epoch}) | "
            f"elapsed {row['elapsed_seconds'] / 60:.1f} min | "
            f"ETA {row['eta_seconds'] / 60:.1f} min",
            flush=True,
        )
    metadata = {
        "schema_version": 2,
        "status": "complete",
        "source_model": args.source_model,
        "objective": objective,
        "scratch": args.scratch,
        "preset": args.preset,
        "model_config": model.config.to_dict(),
        "vocab_size": len(tokenizer),
        "tokenizer_type": "regex_atom_level" if args.preset == "small-smiles-mlm" else "source_model",
        "tokenization": tokenization,
        "data_seed": args.data_seed,
        "training_seed": args.training_seed,
        "dataset_path": args.dataset_path,
        "max_length": args.max_length,
        "train_molecules": len(train_smiles),
        "val_molecules": len(val_smiles),
        "train_sha256": _dataset_digest(train_smiles),
        "val_sha256": _dataset_digest(val_smiles),
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "effective_batch_size": args.batch_size * args.accumulation_steps,
        "epochs": args.epochs,
        "optimizer": {"name": "AdamW", "betas": [0.9, args.adam_beta2],
                      "eps": args.adam_eps, "weight_decay": 0.01},
        "lr_schedule": args.lr_schedule,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "optimizer_steps": optimizer_steps,
        "gradient_clip_norm": 1.0,
        "precision": "bf16" if use_bf16 else "fp32",
        "masking": {"probability": 0.15, "mask_fraction": 0.8,
                    "random_fraction": 0.1, "unchanged_fraction": 0.1,
                    "dynamic_train": True, "fixed_validation": True},
        "loss_aggregation": "target_token_mean",
        "trainable_parameters": trainable,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }
    (output_dir / "adaptation.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["existing", "small-smiles-mlm"], default="existing")
    parser.add_argument("--source-model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-path", default="data/250k_zinc_clean.csv")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--split-column", default=None)
    parser.add_argument("--disable-canonicalize", action="store_true")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=101)
    parser.add_argument("--objective", choices=["auto", "mlm", "clm"], default="auto")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--adam-beta2", type=float)
    parser.add_argument("--adam-eps", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--lr-schedule", choices=["constant", "linear", "cosine"])
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = adapt_encoder(args)
    print(f"Saved adapted encoder to {args.output_dir}; best val loss={metadata['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()

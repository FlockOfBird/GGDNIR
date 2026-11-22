"""Measure encoder geometry and chemical neighborhoods on fixed molecule subsets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors

from .config import DataConfig, EncoderConfig
from .data import build_data_bundle
from .encoder import PretrainedMoleculeEncoder
from .properties import molecule_property
from .tokenizer import smiles_to_selfies
from .study_evaluation import PROPERTIES
from .tokenizer import load_encoder_tokenizer


def _sample_smiles(frame, column: str, count: int, seed: int) -> list[str]:
    values = [value for value in frame[column].astype(str).tolist()
              if smiles_to_selfies(value) is not None]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(values))[:min(count, len(values))]
    return [values[index] for index in indices]


def _encode(smiles: list[str], encoder, tokenizer, device: torch.device,
            max_length: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    features = []
    lengths = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            chunk = smiles[start:start + batch_size]
            untruncated = tokenizer(chunk, padding=False, truncation=False)
            lengths.extend(len(ids) for ids in untruncated["input_ids"])
            tokens = tokenizer(chunk, padding="max_length", truncation=True,
                               max_length=max_length, return_tensors="pt")
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            hidden = encoder.token_embeddings(input_ids, attention_mask)
            pooled = encoder.pool_embeddings(hidden, attention_mask)
            features.append(pooled.float().cpu().numpy())
    return np.concatenate(features), np.asarray(lengths)


def _properties(smiles: list[str]) -> np.ndarray:
    values = []
    for text in smiles:
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            raise ValueError(f"Invalid molecule reached the latent diagnostic: {text}")
        values.append([molecule_property(molecule, name) for name in PROPERTIES])
    return np.asarray(values, dtype=np.float64)


def diagnose(args) -> dict:
    if args.train_count < 10 or args.val_count < 10:
        raise ValueError("Diagnostic subsets must contain at least ten molecules.")
    data_config = DataConfig(
        dataset_path=args.dataset_path, label_column=None, property_name=None,
        seed=args.data_seed, split_strategy="scaffold",
    )
    splits = build_data_bundle(data_config)
    train_smiles = _sample_smiles(splits.train, data_config.smiles_column,
                                  args.train_count, args.data_seed + 81)
    val_smiles = _sample_smiles(splits.val, data_config.smiles_column,
                                args.val_count, args.data_seed + 82)
    encoder_config = EncoderConfig(model_name=args.encoder_name, pooling="mean")
    tokenizer = load_encoder_tokenizer(encoder_config)
    device = torch.device(args.device)
    encoder = PretrainedMoleculeEncoder(encoder_config).to(device)
    train_x, train_lengths = _encode(train_smiles, encoder, tokenizer, device,
                                      args.max_length, args.batch_size)
    val_x, val_lengths = _encode(val_smiles, encoder, tokenizer, device,
                                  args.max_length, args.batch_size)
    train_y = _properties(train_smiles)
    val_y = _properties(val_smiles)
    scales = np.subtract(*np.quantile(train_y, [0.75, 0.25], axis=0))
    scales = np.maximum(scales, 1e-8)
    centered = train_x[:min(args.geometry_count, len(train_x))].astype(np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    singular = torch.linalg.svdvals(torch.from_numpy(centered).to(device))
    eigenvalues = singular.square().cpu().numpy()
    eigenvalues = eigenvalues[eigenvalues > eigenvalues.max() * 1e-12]
    weights = eigenvalues / eigenvalues.sum()
    effective_rank = float(np.exp(-np.sum(weights * np.log(weights))))
    top_10_fraction = float(weights[:10].sum())
    standardized_x = (train_x - train_x.mean(axis=0)) / np.maximum(train_x.std(axis=0), 1e-6)
    val_standardized = (val_x - train_x.mean(axis=0)) / np.maximum(train_x.std(axis=0), 1e-6)
    probe = Ridge(alpha=10.0, solver="lsqr")
    probe.fit(standardized_x, train_y)
    predictions = probe.predict(val_standardized)
    probe_r2 = {
        name: float(r2_score(val_y[:, index], predictions[:, index]))
        for index, name in enumerate(PROPERTIES)
    }
    neighbors = NearestNeighbors(n_neighbors=min(6, len(train_x)), metric="cosine")
    neighbors.fit(train_x)
    indices = neighbors.kneighbors(train_x, return_distance=False)[:, 1:]
    property_distances = np.abs(
        (train_y[:, None, :] - train_y[indices]) / scales
    ).mean(axis=2)
    rng = np.random.default_rng(args.data_seed + 83)
    random_indices = rng.integers(0, len(train_x), size=indices.shape)
    random_distances = np.abs(
        (train_y[:, None, :] - train_y[random_indices]) / scales
    ).mean(axis=2)
    result = {
        "schema_version": 1,
        "encoder_model": args.encoder_name,
        "data_seed": args.data_seed,
        "train_count": len(train_x),
        "val_count": len(val_x),
        "latent_dim": train_x.shape[1],
        "effective_rank": effective_rank,
        "top_10_variance_fraction": top_10_fraction,
        "pooled_feature_rms_variation": float(np.sqrt(np.mean(centered * centered))),
        "neighbor_property_distance": float(property_distances.mean()),
        "random_property_distance": float(random_distances.mean()),
        "scaffold_val_probe_r2": probe_r2,
        "train_truncation_rate": float(np.mean(train_lengths > args.max_length)),
        "val_truncation_rate": float(np.mean(val_lengths > args.max_length)),
        "train_mean_token_length": float(train_lengths.mean()),
        "val_mean_token_length": float(val_lengths.mean()),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-path", default="data/250k_zinc_clean.csv")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--geometry-count", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(diagnose(args), indent=2))


if __name__ == "__main__":
    main()

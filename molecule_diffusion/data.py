from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig


LEGACY_DATA_SPLIT_SCHEMA_VERSION = 1
BALANCED_SCAFFOLD_SPLIT_SCHEMA_VERSION = 2
REGRESSION_SCAFFOLD_SPLIT_SCHEMA_VERSION = 3
BALANCED_SCAFFOLD_CANDIDATES = 512


def data_split_schema_version(config: DataConfig) -> int:
    if (
        config.split_strategy == "scaffold"
        and config.label_column is not None
        and config.task_type == "classification"
    ):
        return BALANCED_SCAFFOLD_SPLIT_SCHEMA_VERSION
    if (
        config.split_strategy == "scaffold"
        and config.label_column is not None
        and config.task_type == "regression"
    ):
        return REGRESSION_SCAFFOLD_SPLIT_SCHEMA_VERSION
    return LEGACY_DATA_SPLIT_SCHEMA_VERSION


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _normalize_smiles(smiles: str, canonicalize: bool) -> str | None:
    smiles = str(smiles).strip()
    if not smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    if canonicalize:
        return Chem.MolToSmiles(molecule, canonical=True)
    return smiles


def _scaffold_key(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return smiles
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
    return scaffold or smiles


def load_smiles_dataframe(config: DataConfig) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(config.dataset_path))
    required_columns = [config.smiles_column]
    if config.split_column is not None:
        if config.split_column not in dataframe.columns:
            raise KeyError(f"Split column '{config.split_column}' not found in {config.dataset_path}")
        required_columns.append(config.split_column)
    if config.label_column is not None:
        if config.label_column not in dataframe.columns:
            raise KeyError(f"Label column '{config.label_column}' not found in {config.dataset_path}")
        required_columns.append(config.label_column)

    dataframe = dataframe[required_columns].copy()
    if config.label_column is not None:
        dataframe = dataframe.dropna(subset=[config.smiles_column, config.label_column])
        dataframe[config.label_column] = dataframe[config.label_column].astype(float)
    else:
        dataframe = dataframe.dropna(subset=[config.smiles_column])

    cleaned_smiles = [
        _normalize_smiles(smiles, canonicalize=config.canonicalize_smiles)
        for smiles in dataframe[config.smiles_column].tolist()
    ]
    dataframe[config.smiles_column] = cleaned_smiles
    if config.drop_invalid_smiles:
        dataframe = dataframe.dropna(subset=[config.smiles_column])
    dataframe = dataframe[dataframe[config.smiles_column].astype(str).str.len() > 0]
    if config.split_column is not None:
        conflicting = dataframe.groupby(config.smiles_column)[config.split_column].nunique()
        if (conflicting > 1).any():
            raise ValueError(
                "A canonical molecule occurs in multiple predefined splits; "
                "refusing to silently change benchmark partitions."
            )
    dataframe = dataframe.drop_duplicates(subset=[config.smiles_column]).reset_index(drop=True)
    return dataframe


def _predefined_split(dataframe: pd.DataFrame, config: DataConfig) -> DatasetSplit:
    if config.split_column is None:
        raise ValueError("A predefined split requires a split column.")
    values = set(dataframe[config.split_column].astype(str))
    if values != {"train", "val", "test"}:
        raise ValueError(
            f"Predefined split must contain train, val, and test only; found {sorted(values)}."
        )
    return DatasetSplit(**{
        name: dataframe.loc[dataframe[config.split_column] == name]
        .drop(columns=[config.split_column]).reset_index(drop=True)
        for name in ("train", "val", "test")
    })


def _stratified_split(dataframe: pd.DataFrame, config: DataConfig) -> DatasetSplit:
    if config.label_column is None:
        raise ValueError("Stratified splitting requires a label column.")
    train_df, temp_df = train_test_split(
        dataframe,
        test_size=1.0 - config.train_fraction,
        stratify=dataframe[config.label_column],
        random_state=config.seed,
    )
    relative_val_fraction = config.val_fraction / (1.0 - config.train_fraction)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - relative_val_fraction,
        stratify=temp_df[config.label_column],
        random_state=config.seed,
    )
    return DatasetSplit(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def _random_split(dataframe: pd.DataFrame, config: DataConfig) -> DatasetSplit:
    train_df, temp_df = train_test_split(
        dataframe,
        test_size=1.0 - config.train_fraction,
        random_state=config.seed,
    )
    relative_val_fraction = config.val_fraction / (1.0 - config.train_fraction)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - relative_val_fraction,
        random_state=config.seed,
    )
    return DatasetSplit(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def _greedy_scaffold_split(dataframe: pd.DataFrame, config: DataConfig) -> DatasetSplit:
    """Size-based scaffold split for generative data and continuous targets."""

    scaffold_groups: dict[str, list[int]] = defaultdict(list)
    for index, smiles in enumerate(dataframe[config.smiles_column].tolist()):
        scaffold_groups[_scaffold_key(smiles)].append(index)

    ordered_groups = sorted(scaffold_groups.values(), key=len, reverse=True)
    total = len(dataframe)
    train_target = int(total * config.train_fraction)
    val_target = int(total * config.val_fraction)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for group in ordered_groups:
        if len(train_indices) + len(group) <= train_target:
            train_indices.extend(group)
        elif len(val_indices) + len(group) <= val_target:
            val_indices.extend(group)
        else:
            test_indices.extend(group)

    unassigned = set(range(total)) - set(train_indices) - set(val_indices) - set(test_indices)
    test_indices.extend(sorted(unassigned))

    return DatasetSplit(
        train=dataframe.iloc[sorted(train_indices)].reset_index(drop=True),
        val=dataframe.iloc[sorted(val_indices)].reset_index(drop=True),
        test=dataframe.iloc[sorted(test_indices)].reset_index(drop=True),
    )


def _best_balanced_group_holdout(
    dataframe: pd.DataFrame,
    indices: np.ndarray,
    config: DataConfig,
    *,
    holdout_fraction: float,
    seed: int,
    minimum_remaining_class_scaffolds: int,
    minimum_holdout_class_scaffolds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a scaffold-disjoint holdout with size and label balance."""

    if config.label_column is None:
        raise ValueError("Balanced scaffold splitting requires a label column.")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("Scaffold holdout fraction must be in (0, 1).")

    subset = dataframe.iloc[indices]
    labels = subset[config.label_column].to_numpy()
    scaffolds = np.asarray(
        [_scaffold_key(smiles) for smiles in subset[config.smiles_column]],
        dtype=object,
    )
    classes = tuple(pd.unique(labels).tolist())
    splitter = GroupShuffleSplit(
        n_splits=BALANCED_SCAFFOLD_CANDIDATES,
        test_size=holdout_fraction,
        random_state=seed,
    )
    best: tuple[tuple[float, float, float], np.ndarray, np.ndarray] | None = None

    for remaining_positions, holdout_positions in splitter.split(
        np.zeros(len(indices)),
        labels,
        groups=scaffolds,
    ):
        remaining_labels = labels[remaining_positions]
        holdout_labels = labels[holdout_positions]
        feasible = True
        for class_value in classes:
            remaining_class_scaffolds = set(
                scaffolds[remaining_positions][remaining_labels == class_value]
            )
            holdout_class_scaffolds = set(
                scaffolds[holdout_positions][holdout_labels == class_value]
            )
            if (
                len(remaining_class_scaffolds) < minimum_remaining_class_scaffolds
                or len(holdout_class_scaffolds) < minimum_holdout_class_scaffolds
            ):
                feasible = False
                break
        if not feasible:
            continue

        size_error = abs(len(holdout_positions) / len(indices) - holdout_fraction)
        class_fraction_error = sum(
            abs(
                int((holdout_labels == class_value).sum())
                / int((labels == class_value).sum())
                - holdout_fraction
            )
            for class_value in classes
        ) / len(classes)
        prevalence_error = sum(
            abs(
                float((holdout_labels == class_value).mean())
                - float((labels == class_value).mean())
            )
            for class_value in classes
        ) / len(classes)
        score = (class_fraction_error, size_error, prevalence_error)
        if best is None or score < best[0]:
            best = (
                score,
                indices[remaining_positions],
                indices[holdout_positions],
            )

    if best is None:
        raise ValueError(
            "Could not construct a label-balanced scaffold holdout. Each class must "
            "occur in enough distinct scaffolds for every split. Use --split-strategy "
            "stratified if scaffold-disjoint evaluation is infeasible."
        )
    return best[1], best[2]


def _balanced_scaffold_split(dataframe: pd.DataFrame, config: DataConfig) -> DatasetSplit:
    if config.label_column is None:
        raise ValueError("Balanced scaffold splitting requires a label column.")
    test_fraction = 1.0 - config.train_fraction - config.val_fraction
    if config.train_fraction <= 0.0 or config.val_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("Train, validation, and test fractions must all be positive.")

    labels = dataframe[config.label_column].to_numpy()
    classes = tuple(pd.unique(labels).tolist())
    if len(classes) < 2:
        raise ValueError("Classifier splitting requires at least two label classes.")
    scaffolds = np.asarray(
        [_scaffold_key(smiles) for smiles in dataframe[config.smiles_column]],
        dtype=object,
    )
    for class_value in classes:
        class_scaffold_count = len(set(scaffolds[labels == class_value]))
        if class_scaffold_count < 3:
            raise ValueError(
                f"Class {class_value!r} occurs in only {class_scaffold_count} distinct "
                "scaffolds; at least three are required for scaffold-disjoint train, "
                "validation, and test splits."
            )

    all_indices = np.arange(len(dataframe), dtype=np.int64)
    temporary_fraction = config.val_fraction + test_fraction
    train_indices, temporary_indices = _best_balanced_group_holdout(
        dataframe,
        all_indices,
        config,
        holdout_fraction=temporary_fraction,
        seed=config.seed,
        minimum_remaining_class_scaffolds=1,
        minimum_holdout_class_scaffolds=2,
    )
    relative_test_fraction = test_fraction / temporary_fraction
    val_indices, test_indices = _best_balanced_group_holdout(
        dataframe,
        temporary_indices,
        config,
        holdout_fraction=relative_test_fraction,
        seed=config.seed + 1,
        minimum_remaining_class_scaffolds=1,
        minimum_holdout_class_scaffolds=1,
    )

    return DatasetSplit(
        train=dataframe.iloc[sorted(train_indices)].reset_index(drop=True),
        val=dataframe.iloc[sorted(val_indices)].reset_index(drop=True),
        test=dataframe.iloc[sorted(test_indices)].reset_index(drop=True),
    )


def _scaffold_split(dataframe: pd.DataFrame, config: DataConfig) -> DatasetSplit:
    if config.label_column is None or config.task_type == "regression":
        return _greedy_scaffold_split(dataframe, config)
    return _balanced_scaffold_split(dataframe, config)


def split_dataframe(dataframe: pd.DataFrame, config: DataConfig) -> DatasetSplit:
    if config.split_column is not None:
        return _predefined_split(dataframe, config)
    if config.split_strategy == "scaffold":
        return _scaffold_split(dataframe, config)
    if config.task_type == "classification":
        return _stratified_split(dataframe, config)
    return _random_split(dataframe, config)


class TensorDataset(Dataset):
    def __init__(self, payload: dict[str, torch.Tensor | list[str]]) -> None:
        self.payload = payload
        self.size = len(payload["smiles"])

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item = {}
        for key, value in self.payload.items():
            if isinstance(value, torch.Tensor):
                item[key] = value[index]
            else:
                item[key] = value[index]
        return item


def create_tensor_loader(
    payload: dict[str, torch.Tensor | list[str]],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        TensorDataset(payload),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_data_bundle(config: DataConfig) -> DatasetSplit:
    dataframe = load_smiles_dataframe(config)
    return split_dataframe(dataframe, config)

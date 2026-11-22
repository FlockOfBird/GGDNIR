"""Prepare the official GuacaMol and MOSES molecule splits for this pipeline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import six
import sys


GUACAMOL_FILES = {
    "train": ("guacamol_v1_train.smiles", "13612760", "05ad85d871958a05c02ab51a4fde8530"),
    "val": ("guacamol_v1_valid.smiles", "13612766", "e53db4bff7dc4784123ae6df72e3b1f0"),
    "test": ("guacamol_v1_test.smiles", "13612757", "677b757ccec4809febd83850b43e1616"),
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _official_guacamol_file(directory: Path, name: str, file_id: str, md5: str) -> Path:
    path = directory / name
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        try:
            with urlopen(f"https://ndownloader.figshare.com/files/{file_id}", timeout=120) as source:
                with temporary.open("wb") as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
            if _md5(temporary) != md5:
                raise ValueError(f"Downloaded GuacaMol file has an unexpected MD5: {name}")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    if _md5(path) != md5:
        raise ValueError(f"GuacaMol file has an unexpected MD5: {path}")
    return path


def _write_rows(path: Path, splits: dict[str, list[str]]) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    counts: dict[str, int] = {}
    try:
        with temporary.open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target)
            writer.writerow(["smiles", "benchmark_split"])
            for split, molecules in splits.items():
                counts[split] = 0
                for smiles in molecules:
                    item = str(smiles).strip()
                    if item:
                        writer.writerow([item, split])
                        counts[split] += 1
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"A benchmark split is empty: {counts}")
    return counts


def prepare_guacamol(output_dir: Path, source_dir: Path | None = None) -> dict:
    source_dir = source_dir or output_dir / "official"
    paths = {
        split: _official_guacamol_file(source_dir, name, file_id, md5)
        for split, (name, file_id, md5) in GUACAMOL_FILES.items()
    }
    splits = {
        split: path.read_text(encoding="utf-8").splitlines()
        for split, path in paths.items()
    }
    counts = _write_rows(output_dir / "pipeline.csv", splits)
    return {
        "benchmark": "guacamol", "pipeline_csv": str((output_dir / "pipeline.csv").resolve()),
        "split_column": "benchmark_split", "counts": counts,
        "official_train_file": str(paths["train"].resolve()),
        "official_files_md5": {split: _md5(path) for split, path in paths.items()},
    }


def prepare_moses(output_dir: Path, seed: int = 42, validation_fraction: float = 0.05) -> dict:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")
    # MOSES 0.3.1 imports APIs removed from modern RDKit and pandas.
    sys.modules.setdefault("rdkit.six", six)
    if not hasattr(pd.DataFrame, "append"):
        pd.DataFrame.append = lambda self, other, sort=False: pd.concat([self, other], sort=sort)
    import moses
    train = list(moses.get_dataset("train"))
    test = list(moses.get_dataset("test"))
    scaffold_test = list(moses.get_dataset("test_scaffolds"))
    import numpy as np
    rng = np.random.default_rng(seed)
    validation_indices = set(rng.choice(
        len(train), size=max(1, round(len(train) * validation_fraction)), replace=False
    ).tolist())
    fit = [item for index, item in enumerate(train) if index not in validation_indices]
    validation = [item for index, item in enumerate(train) if index in validation_indices]
    counts = _write_rows(output_dir / "pipeline.csv", {
        "train": fit, "val": validation, "test": test,
    })
    return {
        "benchmark": "moses", "pipeline_csv": str((output_dir / "pipeline.csv").resolve()),
        "split_column": "benchmark_split", "counts": counts,
        "official_train_count": len(train), "official_test_count": len(test),
        "official_scaffold_test_count": len(scaffold_test),
        "validation_from_official_train": True,
        "validation_seed": seed,
        "validation_fraction": validation_fraction,
        "moses_version": getattr(moses, "__version__", "unknown"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["guacamol", "moses"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--guacamol-source-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    args = parser.parse_args()
    manifest_path = args.output_dir / "dataset.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("benchmark") != args.benchmark or not Path(existing["pipeline_csv"]).exists():
            raise ValueError(f"Dataset manifest does not match the request: {manifest_path}")
        if args.benchmark == "moses" and (
            existing.get("validation_seed") != args.seed
            or existing.get("validation_fraction") != args.validation_fraction
        ):
            raise ValueError(
                f"Existing MOSES validation split uses different settings: {manifest_path}"
            )
        print(json.dumps(existing, indent=2))
        return
    result = (
        prepare_guacamol(args.output_dir, args.guacamol_source_dir)
        if args.benchmark == "guacamol"
        else prepare_moses(args.output_dir, args.seed, args.validation_fraction)
    )
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

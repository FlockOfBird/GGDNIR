from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from molecule_diffusion.tokenizer import smiles_to_selfies
from molecule_diffusion.validation import validate_smiles


REQUIRED_COLUMNS = ("smiles", "logP", "qed", "SAS")


@dataclass
class CleaningStats:
    input_rows: int
    after_required_fields: int
    after_rdkit_validation: int
    after_deduplication: int
    after_selfies_filter: int

    def as_dict(self) -> dict[str, int]:
        return {
            "input_rows": self.input_rows,
            "after_required_fields": self.after_required_fields,
            "after_rdkit_validation": self.after_rdkit_validation,
            "after_deduplication": self.after_deduplication,
            "after_selfies_filter": self.after_selfies_filter,
            "dropped_missing_or_empty": self.input_rows - self.after_required_fields,
            "dropped_rdkit_invalid": self.after_required_fields - self.after_rdkit_validation,
            "dropped_duplicates": self.after_rdkit_validation - self.after_deduplication,
            "dropped_selfies_invalid": self.after_deduplication - self.after_selfies_filter,
        }


def clean_zinc_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    dataframe = dataframe.rename(columns={column: str(column).strip() for column in dataframe.columns})
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    cleaned = dataframe[list(REQUIRED_COLUMNS)].copy()
    cleaned["smiles"] = cleaned["smiles"].astype(str).str.strip()
    for column in ["logP", "qed", "SAS"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    input_rows = len(cleaned)
    cleaned = cleaned.replace({"smiles": {"": pd.NA}})
    cleaned = cleaned.dropna(subset=["smiles", "logP", "qed", "SAS"]).reset_index(drop=True)
    after_required_fields = len(cleaned)

    validations = cleaned["smiles"].map(validate_smiles)
    cleaned["smiles"] = [
        validation.canonical_smiles if validation.is_valid else None
        for validation in validations
    ]
    cleaned = cleaned.dropna(subset=["smiles"]).reset_index(drop=True)
    after_rdkit_validation = len(cleaned)

    cleaned = cleaned.drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    after_deduplication = len(cleaned)

    cleaned["SELFIES"] = cleaned["smiles"].map(smiles_to_selfies)
    cleaned = cleaned.dropna(subset=["SELFIES"]).reset_index(drop=True)
    cleaned["SELFIES"] = cleaned["SELFIES"].astype(str).str.strip()
    cleaned["molecular_weight"] = cleaned["smiles"].map(
        lambda smiles: float(Descriptors.MolWt(Chem.MolFromSmiles(smiles)))
    )
    cleaned = cleaned[
        ["smiles", "SELFIES", "logP", "qed", "SAS", "molecular_weight"]
    ]
    after_selfies_filter = len(cleaned)

    stats = CleaningStats(
        input_rows=input_rows,
        after_required_fields=after_required_fields,
        after_rdkit_validation=after_rdkit_validation,
        after_deduplication=after_deduplication,
        after_selfies_filter=after_selfies_filter,
    )
    return cleaned, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean the ZINC 250k CSV for training.")
    parser.add_argument("--input-path", default="data/250k_rndm_zinc_drugs.csv")
    parser.add_argument("--output-path", default="data/250k_zinc_clean.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    dataframe = pd.read_csv(input_path)
    cleaned, stats = clean_zinc_dataframe(dataframe)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    print(f"Wrote cleaned ZINC data to {output_path}")
    for key, value in stats.as_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

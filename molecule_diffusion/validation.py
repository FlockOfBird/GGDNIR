from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


@dataclass
class SmilesValidationResult:
    input_smiles: str
    canonical_smiles: str | None
    parse_success: bool
    sanitization_success: bool
    chemistry_consistency_success: bool

    @property
    def is_valid(self) -> bool:
        return self.parse_success and self.sanitization_success and self.chemistry_consistency_success


def validate_smiles(smiles: str | None) -> SmilesValidationResult:
    if smiles is None:
        return SmilesValidationResult(
            input_smiles="",
            canonical_smiles=None,
            parse_success=False,
            sanitization_success=False,
            chemistry_consistency_success=False,
        )

    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    if molecule is None:
        return SmilesValidationResult(
            input_smiles=smiles,
            canonical_smiles=None,
            parse_success=False,
            sanitization_success=False,
            chemistry_consistency_success=False,
        )

    try:
        Chem.SanitizeMol(molecule)
        sanitization_success = True
    except Exception:
        return SmilesValidationResult(
            input_smiles=smiles,
            canonical_smiles=None,
            parse_success=True,
            sanitization_success=False,
            chemistry_consistency_success=False,
        )

    chemistry_consistency_success = True
    try:
        if Chem.DetectChemistryProblems(molecule):
            chemistry_consistency_success = False
        kekulized = Chem.Mol(molecule)
        Chem.Kekulize(kekulized, clearAromaticFlags=False)
    except Exception:
        chemistry_consistency_success = False

    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True) if chemistry_consistency_success else None
    return SmilesValidationResult(
        input_smiles=smiles,
        canonical_smiles=canonical_smiles,
        parse_success=True,
        sanitization_success=sanitization_success,
        chemistry_consistency_success=chemistry_consistency_success,
    )

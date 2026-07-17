#!/usr/bin/env python3
"""Generate compound-level classical molecular features for the PlantCyc
reaction-neighbor task.

This script intentionally does not rebuild the compound master or pair-index
tables. Those are created by:

    src/data_processing/make_plantcyc_reaction_neighbor_compound_master.py

Input
-----
data/features/plantcyc_reaction_neighbor/metadata.tsv

Outputs
-------
data/features/plantcyc_reaction_neighbor/morgan_r2_2048.npz
data/features/plantcyc_reaction_neighbor/morgan_r3_2048.npz
data/features/plantcyc_reaction_neighbor/rdkit_descriptors.tsv
data/features/plantcyc_reaction_neighbor/feature_manifest.json
qc/data_processing/plantcyc_reaction_neighbor_feature_qc.txt
qc/data_processing/plantcyc_reaction_neighbor_invalid_compounds.tsv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy import sparse

RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Morgan fingerprints and RDKit descriptors for the "
            "PlantCyc reaction-neighbor compound master."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "data/features/plantcyc_reaction_neighbor/metadata.tsv"
        ),
        help="Compound metadata created by the compound-master script.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/features/plantcyc_reaction_neighbor"),
        help="Directory for feature matrices and manifest.",
    )
    parser.add_argument(
        "--qc-dir",
        type=Path,
        default=Path("qc/data_processing"),
        help="Directory for QC reports.",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=2048,
        help="Morgan fingerprint length.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    metadata = pd.read_csv(path, sep="\t", low_memory=False)
    required = {
        "compound_index",
        "task_compound_id",
        "canonical_smiles",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise RuntimeError(
            f"Metadata is missing required columns: {missing}"
        )

    if metadata.empty:
        raise RuntimeError("Metadata table is empty.")

    if metadata["compound_index"].duplicated().any():
        raise RuntimeError("Duplicate compound_index values.")
    if metadata["task_compound_id"].duplicated().any():
        raise RuntimeError("Duplicate task_compound_id values.")
    if metadata["canonical_smiles"].duplicated().any():
        raise RuntimeError("Duplicate canonical_smiles values.")

    expected_index = list(range(len(metadata)))
    observed_index = metadata["compound_index"].astype(int).tolist()
    if observed_index != expected_index:
        raise RuntimeError(
            "compound_index must be contiguous, zero-based, and already in "
            "feature-row order."
        )

    missing_smiles = (
        metadata["canonical_smiles"].isna()
        | metadata["canonical_smiles"].astype(str).str.strip().eq("")
    )
    if missing_smiles.any():
        raise RuntimeError(
            f"Metadata contains {int(missing_smiles.sum())} empty SMILES."
        )

    return metadata


def parse_molecules(
    metadata: pd.DataFrame,
    invalid_file: Path,
) -> list[Chem.Mol]:
    molecules: list[Chem.Mol] = []
    invalid_rows: list[dict[str, object]] = []

    for row in metadata.itertuples(index=False):
        smiles = str(row.canonical_smiles)
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception:
            mol = None

        if mol is None:
            invalid_rows.append(
                {
                    "compound_index": int(row.compound_index),
                    "task_compound_id": row.task_compound_id,
                    "canonical_smiles": smiles,
                }
            )
        else:
            molecules.append(mol)

    if invalid_rows:
        invalid = pd.DataFrame(invalid_rows)
        invalid.to_csv(invalid_file, sep="\t", index=False)
        raise RuntimeError(
            f"Found {len(invalid_rows)} invalid SMILES. See {invalid_file}"
        )

    if invalid_file.exists():
        invalid_file.unlink()

    if len(molecules) != len(metadata):
        raise RuntimeError("Molecule count does not match metadata rows.")

    return molecules


def build_morgan_matrix(
    molecules: list[Chem.Mol],
    radius: int,
    n_bits: int,
) -> sparse.csr_matrix:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )

    row_indices: list[int] = []
    col_indices: list[int] = []

    for row_index, mol in enumerate(molecules):
        fingerprint = generator.GetFingerprint(mol)
        active_bits = list(fingerprint.GetOnBits())
        row_indices.extend([row_index] * len(active_bits))
        col_indices.extend(active_bits)

    values = np.ones(len(row_indices), dtype=np.uint8)
    matrix = sparse.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(molecules), n_bits),
        dtype=np.uint8,
    )
    matrix.sort_indices()
    return matrix


def build_descriptor_table(
    metadata: pd.DataFrame,
    molecules: list[Chem.Mol],
) -> tuple[pd.DataFrame, dict[str, int]]:
    descriptor_definitions = list(Descriptors._descList)
    descriptor_names = [name for name, _ in descriptor_definitions]

    values = np.full(
        (len(molecules), len(descriptor_definitions)),
        np.nan,
        dtype=np.float64,
    )

    for i, mol in enumerate(molecules):
        for j, (_, function) in enumerate(descriptor_definitions):
            try:
                value = float(function(mol))
                if math.isfinite(value):
                    values[i, j] = value
            except Exception:
                pass

    descriptor_frame = pd.DataFrame(values, columns=descriptor_names)
    descriptor_frame.insert(
        0,
        "canonical_smiles",
        metadata["canonical_smiles"].to_numpy(),
    )
    descriptor_frame.insert(
        0,
        "task_compound_id",
        metadata["task_compound_id"].to_numpy(),
    )
    descriptor_frame.insert(
        0,
        "compound_index",
        metadata["compound_index"].to_numpy(),
    )

    numeric = descriptor_frame[descriptor_names]
    qc = {
        "n_descriptor_columns": len(descriptor_names),
        "n_missing_descriptor_values": int(numeric.isna().sum().sum()),
        "n_compounds_with_any_missing_descriptor": int(
            numeric.isna().any(axis=1).sum()
        ),
        "n_descriptor_columns_with_any_missing": int(
            numeric.isna().any(axis=0).sum()
        ),
        "n_all_missing_descriptor_columns": int(
            numeric.isna().all(axis=0).sum()
        ),
    }
    return descriptor_frame, qc


def main() -> None:
    args = parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.qc_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata)
    invalid_file = (
        args.qc_dir
        / "plantcyc_reaction_neighbor_invalid_compounds.tsv"
    )
    molecules = parse_molecules(metadata, invalid_file)

    r2_path = args.out_dir / f"morgan_r2_{args.n_bits}.npz"
    r3_path = args.out_dir / f"morgan_r3_{args.n_bits}.npz"
    descriptor_path = args.out_dir / "rdkit_descriptors.tsv"
    manifest_path = args.out_dir / "feature_manifest.json"
    qc_path = (
        args.qc_dir
        / "plantcyc_reaction_neighbor_feature_qc.txt"
    )

    print(f"Metadata rows: {len(metadata):,}")
    print(f"Valid RDKit molecules: {len(molecules):,}")

    print(
        f"Computing Morgan radius 2, {args.n_bits} bits..."
    )
    morgan_r2 = build_morgan_matrix(
        molecules,
        radius=2,
        n_bits=args.n_bits,
    )
    sparse.save_npz(r2_path, morgan_r2, compressed=True)

    print(
        f"Computing Morgan radius 3, {args.n_bits} bits..."
    )
    morgan_r3 = build_morgan_matrix(
        molecules,
        radius=3,
        n_bits=args.n_bits,
    )
    sparse.save_npz(r3_path, morgan_r3, compressed=True)

    print("Computing RDKit descriptors...")
    descriptors, descriptor_qc = build_descriptor_table(
        metadata,
        molecules,
    )
    descriptors.to_csv(
        descriptor_path,
        sep="\t",
        index=False,
        na_rep="",
    )

    if morgan_r2.shape != (len(metadata), args.n_bits):
        raise RuntimeError("Unexpected Morgan radius-2 matrix shape.")
    if morgan_r3.shape != (len(metadata), args.n_bits):
        raise RuntimeError("Unexpected Morgan radius-3 matrix shape.")
    if len(descriptors) != len(metadata):
        raise RuntimeError("Descriptor row count does not match metadata.")

    if not np.array_equal(
        descriptors["compound_index"].to_numpy(),
        metadata["compound_index"].to_numpy(),
    ):
        raise RuntimeError("Descriptor rows are not aligned to metadata.")

    manifest = {
        "task": "plantcyc_reaction_neighbor",
        "metadata_file": str(args.metadata),
        "metadata_sha256": sha256_file(args.metadata),
        "n_compounds": int(len(metadata)),
        "rdkit_version": rdkit.__version__,
        "features": {
            "morgan_r2": {
                "path": str(r2_path),
                "shape": list(morgan_r2.shape),
                "dtype": str(morgan_r2.dtype),
                "nnz": int(morgan_r2.nnz),
                "radius": 2,
                "n_bits": args.n_bits,
            },
            "morgan_r3": {
                "path": str(r3_path),
                "shape": list(morgan_r3.shape),
                "dtype": str(morgan_r3.dtype),
                "nnz": int(morgan_r3.nnz),
                "radius": 3,
                "n_bits": args.n_bits,
            },
            "rdkit_descriptors": {
                "path": str(descriptor_path),
                **descriptor_qc,
            },
        },
        "note": (
            "Descriptor missing-value imputation must be fitted on each "
            "training split only, then applied to validation and test data."
        ),
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    qc_lines = [
        "PlantCyc reaction-neighbor feature QC",
        "====================================",
        f"metadata_file: {args.metadata}",
        f"metadata_sha256: {manifest['metadata_sha256']}",
        f"n_compounds: {len(metadata)}",
        f"rdkit_version: {rdkit.__version__}",
        "",
        f"morgan_r2_shape: {morgan_r2.shape}",
        f"morgan_r2_nnz: {morgan_r2.nnz}",
        f"morgan_r3_shape: {morgan_r3.shape}",
        f"morgan_r3_nnz: {morgan_r3.nnz}",
        "",
    ]
    qc_lines.extend(
        f"{key}: {value}"
        for key, value in descriptor_qc.items()
    )
    qc_lines.extend(
        [
            "",
            "Descriptor missing values were retained.",
            (
                "Imputation must be fitted on the training partition only "
                "during model training."
            ),
        ]
    )
    qc_path.write_text(
        "\n".join(qc_lines) + "\n",
        encoding="utf-8",
    )

    print("\nWrote:")
    print(r2_path)
    print(r3_path)
    print(descriptor_path)
    print(manifest_path)
    print(qc_path)
    print("\nFeature generation completed successfully.")


if __name__ == "__main__":
    main()

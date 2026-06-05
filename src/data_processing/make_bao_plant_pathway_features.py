#!/usr/bin/env python3

import re
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator


TASK_FILE = "data/tasks/bao_plant_pathway/bao_plant_pathway_task.tsv"
OUT_DIR = Path("data/features/bao_plant_pathway")


def get_label_columns(df):
    """
    Return only binary label columns.
    Excludes metadata columns such as:
        label_classes
    """
    pattern = re.compile(r"^label_\d+_")
    return [c for c in df.columns if pattern.match(c)]

def mol_from_smiles(smiles):
    if pd.isna(smiles):
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def make_morgan_matrix(mols, radius=2, n_bits=2048):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )

    rows = []
    cols = []
    data = []

    for i, mol in enumerate(mols):
        if mol is None:
            continue

        fp = generator.GetFingerprint(mol)
        on_bits = list(fp.GetOnBits())

        rows.extend([i] * len(on_bits))
        cols.extend(on_bits)
        data.extend([1] * len(on_bits))

    X = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(mols), n_bits),
        dtype=np.int8,
    )

    return X


def make_rdkit_descriptor_table(df, mols):
    desc_list = Descriptors._descList
    desc_names = [x[0] for x in desc_list]
    desc_funcs = [x[1] for x in desc_list]

    rows = []

    for idx, mol in enumerate(mols):
        row = {
            "task_compound_id": df.iloc[idx]["task_compound_id"],
        }

        if mol is None:
            for name in desc_names:
                row[name] = np.nan
            rows.append(row)
            continue

        for name, func in zip(desc_names, desc_funcs):
            try:
                val = func(mol)
            except Exception:
                val = np.nan
            row[name] = val

        rows.append(row)

    desc_df = pd.DataFrame(rows)

    # Replace infinite values with NaN.
    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)

    return desc_df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TASK_FILE, sep="\t", dtype=str)

    if "canonical_smiles" not in df.columns:
        raise ValueError("Missing canonical_smiles column.")

    label_cols = get_label_columns(df)
    if not label_cols:
        raise ValueError("No label columns found. Expected columns starting with label_.")

    # Keep row order fixed. This order must match all feature matrices and labels.
    df = df.reset_index(drop=True)

    mols = [mol_from_smiles(s) for s in df["canonical_smiles"]]
    valid_mol = [m is not None for m in mols]

    print(f"Input compounds: {len(df):,}")
    print(f"Valid RDKit molecules: {sum(valid_mol):,}")
    print(f"Invalid RDKit molecules: {len(df) - sum(valid_mol):,}")
    print(f"Label columns: {len(label_cols):,}")

    # Save metadata.
    metadata_cols = [
        "task_compound_id",
        "dataset_name",
        "split",
        "canonical_smiles",
        "inchikey",
        "dedup_key",
        "murcko_scaffold",
        "label_classes",
        "pathway_names",
    ]
    metadata_cols = [c for c in metadata_cols if c in df.columns]

    metadata = df[metadata_cols].copy()
    metadata["rdkit_valid"] = valid_mol
    metadata.to_csv(OUT_DIR / "metadata.tsv", sep="\t", index=False)

    # Save labels as dense numpy array.
    label_df = df[label_cols].copy()

    n_missing = int(label_df.isna().sum().sum())
    print(f"Missing label values filled as 0: {n_missing}")
    
    label_df = label_df.fillna(0)
    
    y = label_df.astype(int).values
    np.save(OUT_DIR / "labels.npy", y)

    with open(OUT_DIR / "label_columns.txt", "w") as f:
        for c in label_cols:
            f.write(c + "\n")

    with open(OUT_DIR / "feature_manifest.json", "w") as f:
        json.dump(
            {
                "task_file": TASK_FILE,
                "n_compounds": int(len(df)),
                "n_labels": int(len(label_cols)),
                "label_columns": label_cols,
                "row_order_key": "task_compound_id",
            },
            f,
            indent=2,
        )

    # Morgan fingerprints.
    print("Computing Morgan radius 2, 2048 bits...")
    X_morgan_r2 = make_morgan_matrix(mols, radius=2, n_bits=2048)
    sparse.save_npz(OUT_DIR / "morgan_r2_2048.npz", X_morgan_r2)

    print("Computing Morgan radius 3, 2048 bits...")
    X_morgan_r3 = make_morgan_matrix(mols, radius=3, n_bits=2048)
    sparse.save_npz(OUT_DIR / "morgan_r3_2048.npz", X_morgan_r3)

    # RDKit descriptors.
    print("Computing RDKit descriptors...")
    desc_df = make_rdkit_descriptor_table(df, mols)
    desc_df.to_csv(OUT_DIR / "rdkit_descriptors.tsv", sep="\t", index=False)

    print("Done.")
    print(f"Wrote: {OUT_DIR / 'metadata.tsv'}")
    print(f"Wrote: {OUT_DIR / 'labels.npy'}")
    print(f"Wrote: {OUT_DIR / 'label_columns.txt'}")
    print(f"Wrote: {OUT_DIR / 'morgan_r2_2048.npz'}")
    print(f"Wrote: {OUT_DIR / 'morgan_r3_2048.npz'}")
    print(f"Wrote: {OUT_DIR / 'rdkit_descriptors.tsv'}")


if __name__ == "__main__":
    main()
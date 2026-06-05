#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
from rdkit import Chem


A_FILE = "data/processed_bao/A_trainval/compounds_master.tsv"
B_FILE = "data/processed_bao/B_external_test/compounds_master.tsv"

OUT_DIR = Path("qc/data_processing")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TABLE = OUT_DIR / "bao_A_B_overlap_compounds.tsv"
OUT_SUMMARY = OUT_DIR / "bao_A_B_overlap_summary.txt"


def normalize_smiles_any(smiles):
    if pd.isna(smiles):
        return ""
    parts = str(smiles).split("|")
    normalized = []
    for s in parts:
        s = s.strip()
        if not s:
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        normalized.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return "|".join(sorted(set(normalized)))


def make_dedup_key(df):
    df = df.copy()

    df["inchikey"] = df["inchikey"].fillna("")
    df["canonical_smiles"] = df["canonical_smiles"].fillna("")

    df["dedup_key"] = df["inchikey"].where(
        df["inchikey"].ne(""),
        df["canonical_smiles"]
    )

    df["dedup_key_type"] = df["inchikey"].where(
        df["inchikey"].ne(""),
        ""
    )
    df["dedup_key_type"] = df["dedup_key_type"].apply(
        lambda x: "inchikey" if x else "canonical_smiles"
    )

    return df


def main():
    a = pd.read_csv(A_FILE, sep="\t", dtype=str)
    b = pd.read_csv(B_FILE, sep="\t", dtype=str)

    a = make_dedup_key(a)
    b = make_dedup_key(b)

    # Re-normalize original SMILES to independently verify canonicalization.
    a["original_smiles_renormalized"] = a["original_smiles"].apply(normalize_smiles_any)
    b["original_smiles_renormalized"] = b["original_smiles"].apply(normalize_smiles_any)

    overlap = a.merge(
        b,
        on="dedup_key",
        how="inner",
        suffixes=("_A", "_B")
    )

    # Comparison flags.
    overlap["canonical_smiles_identical"] = (
        overlap["canonical_smiles_A"] == overlap["canonical_smiles_B"]
    ).astype(int)

    overlap["inchikey_identical"] = (
        overlap["inchikey_A"] == overlap["inchikey_B"]
    ).astype(int)

    overlap["renormalized_original_smiles_identical"] = (
        overlap["original_smiles_renormalized_A"] ==
        overlap["original_smiles_renormalized_B"]
    ).astype(int)

    keep_cols = [
        "dedup_key",
        "dedup_key_type_A",
        "source_id_A",
        "source_id_B",
        "original_smiles_A",
        "original_smiles_B",
        "original_smiles_renormalized_A",
        "original_smiles_renormalized_B",
        "canonical_smiles_A",
        "canonical_smiles_B",
        "inchikey_A",
        "inchikey_B",
        "canonical_smiles_identical",
        "inchikey_identical",
        "renormalized_original_smiles_identical",
        "label_classes_A",
        "label_classes_B",
        "pathway_names_A",
        "pathway_names_B",
    ]

    keep_cols = [c for c in keep_cols if c in overlap.columns]
    overlap_out = overlap[keep_cols].copy()

    overlap_out.to_csv(OUT_TABLE, sep="\t", index=False)

    with open(OUT_SUMMARY, "w") as f:
        f.write("Bao A/B overlap check\n")
        f.write("=====================\n\n")
        f.write(f"A compounds: {len(a):,}\n")
        f.write(f"B compounds: {len(b):,}\n")
        f.write(f"Overlapping compounds by dedup_key: {len(overlap_out):,}\n\n")

        if len(overlap_out) > 0:
            f.write("Overlap key type counts:\n")
            f.write(str(overlap["dedup_key_type_A"].value_counts()) + "\n\n")

            f.write("Canonical SMILES identical:\n")
            f.write(str(overlap["canonical_smiles_identical"].value_counts()) + "\n\n")

            f.write("InChIKey identical:\n")
            f.write(str(overlap["inchikey_identical"].value_counts()) + "\n\n")

            f.write("Re-normalized original SMILES identical:\n")
            f.write(str(overlap["renormalized_original_smiles_identical"].value_counts()) + "\n\n")

    print("Done.")
    print(f"A compounds: {len(a):,}")
    print(f"B compounds: {len(b):,}")
    print(f"Overlapping compounds: {len(overlap_out):,}")
    print(f"Wrote: {OUT_TABLE}")
    print(f"Wrote: {OUT_SUMMARY}")



if __name__ == "__main__":
    main()
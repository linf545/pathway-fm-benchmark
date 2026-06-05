#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


#python src/process_bao_dataset.py   --bao_csv raw_data/bao/bao_dataset_A_trainval.csv   --pathway_label_csv raw_data/bao/pathway_label.csv   --out_dir data/processed_bao/A_trainval   --source_name Bao_Dataset_A_trainval

#python src/process_bao_dataset.py   --bao_csv raw_data/bao/bao_dataset_B_external_test.csv   --pathway_label_csv raw_data/bao/pathway_label.csv   --out_dir data/processed_bao/B_external_test   --source_name Bao_Dataset_B_external_test

# python src/process_bao_dataset.py  --bao_csv raw_data/bao/bao_dataset_C_kegg.csv   --pathway_label_csv raw_data/bao/pathway_label.csv   --out_dir data/processed_bao/C_kegg   --source_name Bao_Dataset_C_kegg

# python src/process_bao_dataset.py  --bao_csv raw_data/bao/bao_dataset_D_npclassifier.csv  --pathway_label_csv raw_data/bao/pathway_label.csv   --out_dir data/processed_bao/D_npclassifier   --source_name Bao_Dataset_D_npclassifier

def read_label_mapping(path: str) -> dict:
    """
    Read pathway_label.csv.

    Expected:
        col1 = numeric class label
        col2 = pathway string
        col3 = ignored

    Works with or without header.
    """
    df = pd.read_csv(path, dtype=str)

    # If header is strange or missing, use first two columns.
    if "class" in df.columns and "Pathway" in df.columns:
        label_col = "class"
        pathway_col = "Pathway"
    else:
        label_col = df.columns[0]
        pathway_col = df.columns[1]

    mapping = {}
    for _, row in df.iterrows():
        label = str(row[label_col]).strip()
        pathway = str(row[pathway_col]).strip()

        if label == "" or label.lower() == "nan":
            continue
        if pathway == "" or pathway.lower() == "nan":
            continue

        mapping[label] = pathway

    return mapping


def parse_label_string(x):
    """
    Convert label field to list of string labels.

    Handles:
        9
        2,3,11
        "2, 3, 11"
        [2,3,11] if present
    """
    if pd.isna(x):
        return []

    s = str(x).strip()
    s = s.replace("[", "").replace("]", "")
    s = s.replace(";", ",")
    s = s.replace("|", ",")

    labels = []
    for token in s.split(","):
        token = token.strip()
        if token == "":
            continue

        # Keep numeric-looking label as integer string.
        m = re.search(r"\d+", token)
        if m:
            labels.append(str(int(m.group(0))))

    return sorted(set(labels), key=lambda z: int(z))


def canonicalize_smiles(smiles: str):
    """
    Return:
        canonical_smiles, inchikey, mol

    If invalid, return:
        None, None, None
    """
    if pd.isna(smiles):
        return None, None, None

    smiles = str(smiles).strip()
    if smiles == "":
        return None, None, None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        inchikey = ""

    return canonical, inchikey, mol


def get_murcko_scaffold(mol):
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return ""
        return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def molecule_properties(mol):
    return {
        "molecular_weight": Descriptors.MolWt(mol),
        "num_atoms": mol.GetNumAtoms(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "num_rings": rdMolDescriptors.CalcNumRings(mol),
        "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "num_hbd": rdMolDescriptors.CalcNumHBD(mol),
        "num_hba": rdMolDescriptors.CalcNumHBA(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "logp": Descriptors.MolLogP(mol),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bao_csv",
        default="/PATH/TO/bao_dataset.csv",
        help="Bao-style CSV file with columns No., SMILES, Label",
    )
    parser.add_argument(
        "--pathway_label_csv",
        default="/PATH/TO/pathway_label.csv",
        help="CSV file where col1 is numeric class and col2 is pathway name",
    )
    parser.add_argument(
        "--out_dir",
        default="/PATH/TO/output_dir",
        help="Output directory",
    )
    parser.add_argument(
        "--source_name",
        default="Bao_PMNDataset",
        help="Source name stored in output files",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_mapping = read_label_mapping(args.pathway_label_csv)

    df = pd.read_csv(args.bao_csv, dtype=str)

    # Robust column detection.
    if "SMILES" not in df.columns:
        raise ValueError(f"Cannot find SMILES column. Columns are: {list(df.columns)}")

    if "Label" not in df.columns:
        raise ValueError(f"Cannot find Label column. Columns are: {list(df.columns)}")

    no_col = "No." if "No." in df.columns else df.columns[0]

    compound_rows = []
    label_long_rows = []
    invalid_rows = []

    for idx, row in df.iterrows():
        source_id = str(row.get(no_col, idx + 1)).strip()
        original_smiles = str(row["SMILES"]).strip()
        raw_label = row["Label"]

        canonical_smiles, inchikey, mol = canonicalize_smiles(original_smiles)

        if mol is None:
            invalid_rows.append({
                "source_id": source_id,
                "original_smiles": original_smiles,
                "raw_label": raw_label,
                "reason": "invalid_smiles",
            })
            continue

        label_classes = parse_label_string(raw_label)
        pathway_names = [label_mapping.get(x, "") for x in label_classes]
        pathway_names = [x for x in pathway_names if x]

        props = molecule_properties(mol)
        scaffold = get_murcko_scaffold(mol)

        compound_rows.append({
            "source_id": source_id,
            "source_database": args.source_name,
            "original_smiles": original_smiles,
            "canonical_smiles": canonical_smiles,
            "inchikey": inchikey,
            "label_classes": "|".join(label_classes),
            "pathway_names": "|".join(pathway_names),
            "murcko_scaffold": scaffold,
            **props,
        })

        for label_class in label_classes:
            label_long_rows.append({
                "source_id": source_id,
                "source_database": args.source_name,
                "canonical_smiles": canonical_smiles,
                "inchikey": inchikey,
                "label_class": label_class,
                "pathway_name": label_mapping.get(label_class, ""),
            })

    compounds = pd.DataFrame(compound_rows)

    if compounds.empty:
        raise RuntimeError("No valid compounds were parsed. Check SMILES column and file format.")

    # Deduplicate by InChIKey if available; otherwise canonical SMILES.
    compounds["dedup_key"] = compounds["inchikey"].where(
        compounds["inchikey"].astype(str).str.len() > 0,
        compounds["canonical_smiles"],
    )

    # Aggregate duplicate compounds.
    agg = {
        "source_id": lambda x: "|".join(sorted(set(map(str, x)))),
        "source_database": lambda x: "|".join(sorted(set(map(str, x)))),
        "original_smiles": lambda x: "|".join(sorted(set(map(str, x)))),
        "canonical_smiles": "first",
        "inchikey": "first",
        "label_classes": lambda x: "|".join(sorted(
            set(
                y
                for val in x.dropna().astype(str)
                for y in val.split("|")
                if y
            ),
            key=lambda z: int(z) if z.isdigit() else 999999,
        )),
        "pathway_names": lambda x: "|".join(sorted(
            set(
                y
                for val in x.dropna().astype(str)
                for y in val.split("|")
                if y
            )
        )),
        "murcko_scaffold": "first",
        "molecular_weight": "first",
        "num_atoms": "first",
        "num_heavy_atoms": "first",
        "num_rings": "first",
        "num_aromatic_rings": "first",
        "num_hbd": "first",
        "num_hba": "first",
        "tpsa": "first",
        "logp": "first",
    }

    compounds_dedup = compounds.groupby("dedup_key", as_index=False).agg(agg)

    # Assign internal compound ID.
    compounds_dedup = compounds_dedup.sort_values("dedup_key").reset_index(drop=True)
    compounds_dedup.insert(
        0,
        "internal_compound_id",
        [f"CMPD_{i:07d}" for i in range(1, len(compounds_dedup) + 1)],
    )

    # This is your first version of compounds_master.tsv.
    compounds_master = compounds_dedup.copy()
    compounds_master["is_bao"] = 1
    compounds_master["is_plantcyc"] = 0
    compounds_master["is_kegg"] = 0
    compounds_master["is_npclassifier"] = 0

    # Rebuild long label table with internal IDs.
    key_to_internal = dict(
        zip(compounds_master["dedup_key"], compounds_master["internal_compound_id"])
    )

    label_long = pd.DataFrame(label_long_rows)
    if not label_long.empty:
        label_long["dedup_key"] = label_long["inchikey"].where(
            label_long["inchikey"].astype(str).str.len() > 0,
            label_long["canonical_smiles"],
        )
        label_long["internal_compound_id"] = label_long["dedup_key"].map(key_to_internal)
        label_long = label_long[
            [
                "internal_compound_id",
                "source_id",
                "source_database",
                "canonical_smiles",
                "inchikey",
                "label_class",
                "pathway_name",
            ]
        ].drop_duplicates()

    # Multilabel matrix.
    if not label_long.empty:
        multilabel = (
            label_long.assign(value=1)
            .pivot_table(
                index="internal_compound_id",
                columns="label_class",
                values="value",
                fill_value=0,
                aggfunc="max",
            )
            .reset_index()
        )

        # Add pathway names to matrix column names if possible.
        rename_cols = {}
        for c in multilabel.columns:
            if c == "internal_compound_id":
                continue
            pathway = label_mapping.get(str(c), "")
            if pathway:
                safe_pathway = re.sub(r"[^A-Za-z0-9]+", "_", pathway).strip("_")
                rename_cols[c] = f"label_{c}_{safe_pathway}"
            else:
                rename_cols[c] = f"label_{c}"

        multilabel = multilabel.rename(columns=rename_cols)
    else:
        multilabel = pd.DataFrame()

    # Save outputs.
    compounds_master.to_csv(out_dir / "compounds_master.tsv", sep="\t", index=False)
    label_long.to_csv(out_dir / "bao_compound_labels_long.tsv", sep="\t", index=False)
    multilabel.to_csv(out_dir / "bao_compound_labels_multilabel.tsv", sep="\t", index=False)

    pd.DataFrame(
        [{"label_class": k, "pathway_name": v} for k, v in label_mapping.items()]
    ).sort_values("label_class").to_csv(
        out_dir / "bao_label_mapping.tsv", sep="\t", index=False
    )

    invalid_df = pd.DataFrame(
        invalid_rows,
        columns=["source_id", "original_smiles", "raw_label", "reason"]
    )
    invalid_df.to_csv(out_dir / "bao_invalid_smiles.tsv", sep="\t", index=False)

    # Print summary.
    print("Done.")
    print(f"Input rows: {len(df):,}")
    print(f"Valid compounds before dedup: {len(compounds):,}")
    print(f"Valid compounds after dedup: {len(compounds_master):,}")
    print(f"Invalid SMILES: {len(invalid_rows):,}")
    print(f"Labels in mapping: {len(label_mapping):,}")

    print("\nOutput files:")
    print(out_dir / "compounds_master.tsv")
    print(out_dir / "bao_compound_labels_long.tsv")
    print(out_dir / "bao_compound_labels_multilabel.tsv")
    print(out_dir / "bao_label_mapping.tsv")
    print(out_dir / "bao_invalid_smiles.tsv")

    print("\nLabel counts:")
    if not label_long.empty:
        print(label_long["label_class"].value_counts().sort_index())
    else:
        print("No labels parsed.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

from pathlib import Path
import re
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

RANDOM_SEED = 2

IN_FILE = Path("data/tasks/plantcyc_reaction_neighbor/plantcyc_positive_pairs_filtered.tsv")
OUT_DIR = Path("data/tasks/plantcyc_reaction_neighbor")
QC_DIR = Path("qc/data_processing")

OUT_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)


def mol_from_smiles(smiles):
    if pd.isna(smiles):
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def canonical_smiles(smiles):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def murcko_scaffold(smiles):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return canonical_smiles(smiles)
        return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)
    except Exception:
        return canonical_smiles(smiles)


def first_pathway_name(x):
    if pd.isna(x):
        return "UNKNOWN"
    vals = [v for v in str(x).split("|") if v]
    return vals[0] if vals else "UNKNOWN"


def make_random_split(df):
    """
    Random pair-level split.
    This is useful as a basic baseline split, but it can have compound/scaffold leakage.
    """
    df = df.copy()
    df["stratify_label"] = df["pathway_names"].apply(first_pathway_name)

    vc = df["stratify_label"].value_counts()
    rare = set(vc[vc < 3].index)
    df["stratify_label2"] = df["stratify_label"].where(
        ~df["stratify_label"].isin(rare),
        "RARE",
    )

    stratify = df["stratify_label2"]
    if stratify.value_counts().min() < 3:
        stratify = None

    train, tmp = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )

    tmp_stratify = tmp["stratify_label2"]
    if tmp_stratify.value_counts().min() < 2:
        tmp_stratify = None

    val, test = train_test_split(
        tmp,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=tmp_stratify,
    )

    df["random_split"] = "unassigned"
    df.loc[train.index, "random_split"] = "train"
    df.loc[val.index, "random_split"] = "val"
    df.loc[test.index, "random_split"] = "test"

    df = df.drop(columns=["stratify_label", "stratify_label2"], errors="ignore")
    return df


def greedy_group_split(df, group_col, train_frac=0.8, val_frac=0.1):
    """
    Split by scaffold-pair group so the same scaffold-pair group does not appear
    in multiple splits.

    This is stricter than random split.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    group_sizes = (
        df.groupby(group_col)
        .size()
        .reset_index(name="n")
        .sample(frac=1.0, random_state=RANDOM_SEED)
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )

    total = len(df)
    target_train = total * train_frac
    target_val = total * val_frac
    target_test = total - target_train - target_val

    split_counts = {"scaffold_train": 0, "scaffold_val": 0, "scaffold_test": 0}
    split_groups = {"scaffold_train": [], "scaffold_val": [], "scaffold_test": []}

    targets = {
        "scaffold_train": target_train,
        "scaffold_val": target_val,
        "scaffold_test": target_test,
    }

    for _, row in group_sizes.iterrows():
        group = row[group_col]
        n = int(row["n"])

        # Assign group to the split that is most below its target.
        deficits = {
            split: targets[split] - split_counts[split]
            for split in split_counts
        }

        # Prefer positive deficit. If all full, choose smallest relative overflow.
        best_split = max(deficits, key=deficits.get)

        split_groups[best_split].append(group)
        split_counts[best_split] += n

    group_to_split = {}
    for split, groups in split_groups.items():
        for g in groups:
            group_to_split[g] = split

    out = df[group_col].map(group_to_split)
    return out


def leakage_summary(df, split_col, key_col):
    rows = []
    splits = sorted(df[split_col].dropna().unique())

    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1:]:
            k1 = set(df.loc[df[split_col] == s1, key_col])
            k2 = set(df.loc[df[split_col] == s2, key_col])
            rows.append({
                "split_a": s1,
                "split_b": s2,
                "key_col": key_col,
                "overlap_count": len(k1 & k2),
            })

    return pd.DataFrame(rows)


def main():
    pairs = pd.read_csv(IN_FILE, sep="\t", dtype=str)

    print(f"Input filtered positive pairs: {len(pairs):,}")

    # Add pair IDs.
    pairs = pairs.reset_index(drop=True)
    pairs.insert(0, "positive_pair_id", [f"PCYC_POS_{i:07d}" for i in range(1, len(pairs) + 1)])

    # Canonicalize and scaffold.
    pairs["substrate_canonical_smiles"] = pairs["substrate_smiles"].apply(canonical_smiles)
    pairs["product_canonical_smiles"] = pairs["product_smiles"].apply(canonical_smiles)

    pairs["substrate_scaffold"] = pairs["substrate_smiles"].apply(murcko_scaffold)
    pairs["product_scaffold"] = pairs["product_smiles"].apply(murcko_scaffold)

    # Remove anything that failed canonicalization, just in case.
    before = len(pairs)

    pairs["substrate_valid_smiles"] = pairs["substrate_canonical_smiles"].ne("")
    pairs["product_valid_smiles"] = pairs["product_canonical_smiles"].ne("")
    
    invalid_pairs = pairs[
        ~pairs["substrate_valid_smiles"] | ~pairs["product_valid_smiles"]
    ].copy()
    
    invalid_pairs["invalid_reason"] = ""
    invalid_pairs.loc[~invalid_pairs["substrate_valid_smiles"], "invalid_reason"] += "invalid_substrate_smiles"
    invalid_pairs.loc[~invalid_pairs["product_valid_smiles"], "invalid_reason"] += "|invalid_product_smiles"
    invalid_pairs["invalid_reason"] = invalid_pairs["invalid_reason"].str.strip("|")
    
    invalid_pairs.to_csv(
        QC_DIR / "plantcyc_positive_pairs_invalid_smiles.tsv",
        sep="\t",
        index=False,
    )
    
    pairs = pairs[
        pairs["substrate_valid_smiles"] & pairs["product_valid_smiles"]
    ].copy()
    
    after = len(pairs)
    
    print(f"Pairs after canonical SMILES check: {after:,}")
    print(f"Removed due to failed canonicalization: {before - after:,}")
    print(f"Wrote invalid SMILES pairs: {QC_DIR / 'plantcyc_positive_pairs_invalid_smiles.tsv'}")

    # Define keys.
    pairs["directed_compound_pair_key"] = (
        pairs["substrate_id"].astype(str) + ">>" + pairs["product_id"].astype(str)
    )

    pairs["undirected_compound_pair_key"] = pairs.apply(
        lambda r: "||".join(sorted([str(r["substrate_id"]), str(r["product_id"])])),
        axis=1,
    )

    pairs["directed_scaffold_pair_key"] = (
        pairs["substrate_scaffold"].astype(str) + ">>" + pairs["product_scaffold"].astype(str)
    )

    pairs["undirected_scaffold_pair_key"] = pairs.apply(
        lambda r: "||".join(sorted([str(r["substrate_scaffold"]), str(r["product_scaffold"])])),
        axis=1,
    )

    # Random split.
    pairs = make_random_split(pairs)

    # Scaffold-pair split.
    pairs["scaffold_pair_split"] = greedy_group_split(
        pairs,
        group_col="undirected_scaffold_pair_key",
        train_frac=0.8,
        val_frac=0.1,
    )

    out_file = OUT_DIR / "plantcyc_positive_pairs_with_splits.tsv"
    pairs.to_csv(out_file, sep="\t", index=False)

    # QC.
    random_scaffold_leak = leakage_summary(
        pairs,
        split_col="random_split",
        key_col="undirected_scaffold_pair_key",
    )

    scaffold_scaffold_leak = leakage_summary(
        pairs,
        split_col="scaffold_pair_split",
        key_col="undirected_scaffold_pair_key",
    )

    scaffold_compound_leak = leakage_summary(
        pairs,
        split_col="scaffold_pair_split",
        key_col="undirected_compound_pair_key",
    )

    qc_file = QC_DIR / "plantcyc_positive_pair_split_qc.txt"
    with open(qc_file, "w") as f:
        f.write("PlantCyc positive reaction-neighbor split QC\n")
        f.write("============================================\n\n")

        f.write(f"Input filtered positive pairs: {len(pairs):,}\n")
        f.write(f"Unique directed compound pairs: {pairs['directed_compound_pair_key'].nunique():,}\n")
        f.write(f"Unique undirected compound pairs: {pairs['undirected_compound_pair_key'].nunique():,}\n")
        f.write(f"Unique undirected scaffold-pair groups: {pairs['undirected_scaffold_pair_key'].nunique():,}\n\n")

        f.write("Random split counts:\n")
        f.write(str(pairs["random_split"].value_counts()) + "\n\n")

        f.write("Scaffold-pair split counts:\n")
        f.write(str(pairs["scaffold_pair_split"].value_counts()) + "\n\n")

        f.write("Pathway counts by random split:\n")
        for split in ["train", "val", "test"]:
            sub = pairs[pairs["random_split"] == split]
            f.write(f"\n[{split}] n={len(sub):,}\n")
            f.write(str(sub["pathway_names"].fillna("UNKNOWN").str.split("|").explode().value_counts().head(30)) + "\n")

        f.write("\nPathway counts by scaffold-pair split:\n")
        for split in ["scaffold_train", "scaffold_val", "scaffold_test"]:
            sub = pairs[pairs["scaffold_pair_split"] == split]
            f.write(f"\n[{split}] n={len(sub):,}\n")
            f.write(str(sub["pathway_names"].fillna("UNKNOWN").str.split("|").explode().value_counts().head(30)) + "\n")

        f.write("\nRandom split scaffold-pair leakage:\n")
        f.write(str(random_scaffold_leak) + "\n\n")

        f.write("Scaffold-pair split scaffold-pair leakage:\n")
        f.write(str(scaffold_scaffold_leak) + "\n\n")

        f.write("Scaffold-pair split compound-pair leakage:\n")
        f.write(str(scaffold_compound_leak) + "\n")

    print("Done.")
    print("Random split counts:")
    print(pairs["random_split"].value_counts())
    print("\nScaffold-pair split counts:")
    print(pairs["scaffold_pair_split"].value_counts())
    print(f"\nWrote: {out_file}")
    print(f"Wrote: {qc_file}")


if __name__ == "__main__":
    main()
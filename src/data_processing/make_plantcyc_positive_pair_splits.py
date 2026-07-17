#!/usr/bin/env python3
"""Build deduplicated positive PlantCyc reaction-neighbor pairs and data splits.

Key design choices
------------------
1. The prediction target is UNDIRECTED reaction adjacency.
2. Positive observations are deduplicated by an undirected pair of canonical,
   isomeric SMILES before any split is assigned.
3. Reaction, pathway, compound-ID, and name provenance is aggregated across
   all PlantCyc rows that resolve to the same chemical pair.
4. The random split operates on unique chemical pairs.
5. The scaffold-pair split keeps each undirected scaffold-pair group entirely
   within one partition.
"""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.*")

RANDOM_SEED = 2
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10

IN_FILE = Path(
    "data/tasks/plantcyc_reaction_neighbor/plantcyc_positive_pairs_filtered.tsv"
)
OUT_DIR = Path("data/tasks/plantcyc_reaction_neighbor")
QC_DIR = Path("qc/data_processing")

OUT_FILE = OUT_DIR / "plantcyc_positive_pairs_with_splits.tsv"
INVALID_FILE = QC_DIR / "plantcyc_positive_pairs_invalid_smiles.tsv"
SELF_PAIR_FILE = QC_DIR / "plantcyc_positive_pairs_self_pairs_removed.tsv"
DUPLICATE_GROUP_FILE = QC_DIR / "plantcyc_positive_pair_duplicate_groups.tsv"
QC_FILE = QC_DIR / "plantcyc_positive_pair_split_qc.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)


def mol_from_smiles(smiles):
    if pd.isna(smiles):
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def canonical_smiles(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def murcko_scaffold_from_canonical(smiles: str) -> str:
    """Return a canonical Murcko scaffold; use the molecule itself if acyclic."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return smiles
        return Chem.MolToSmiles(
            scaffold,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception:
        return smiles


def undirected_key(a: str, b: str) -> str:
    return "||".join(sorted([str(a), str(b)]))


def directed_key(a: str, b: str) -> str:
    return f"{a}>>{b}"


def clean_tokens(values: Iterable, split_pipe: bool = True) -> list[str]:
    tokens: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        parts = text.split("|") if split_pipe else [text]
        for part in parts:
            part = part.strip()
            if part and part.lower() != "nan":
                tokens.add(part)
    return sorted(tokens)


def compact_values(values: Iterable, split_pipe: bool = True) -> str:
    return "|".join(clean_tokens(values, split_pipe=split_pipe))


def first_pathway_name(value) -> str:
    vals = clean_tokens([value], split_pipe=True)
    return vals[0] if vals else "UNKNOWN"


def safe_stratify(labels: pd.Series, test_size: float):
    """Return labels only when sklearn stratification is mathematically valid."""
    labels = labels.astype(str)
    counts = labels.value_counts()
    if counts.empty or counts.min() < 2:
        return None

    n = len(labels)
    n_test = int(np.ceil(n * test_size)) if isinstance(test_size, float) else int(test_size)
    n_train = n - n_test
    n_classes = counts.size
    if n_test < n_classes or n_train < n_classes:
        return None
    return labels


def make_random_split(df: pd.DataFrame) -> pd.DataFrame:
    """Create an 80/10/10 split at the unique undirected-pair level."""
    out = df.copy()
    out["stratify_label"] = out["pathway_names"].apply(first_pathway_name)

    # A class needs enough observations to reasonably contribute to both
    # validation and test after the two-stage split. Very small classes are
    # pooled into RARE; if stratification is still impossible, fall back to a
    # seeded unstratified split.
    counts = out["stratify_label"].value_counts()
    rare = set(counts[counts < 10].index)
    out["stratify_label2"] = out["stratify_label"].where(
        ~out["stratify_label"].isin(rare),
        "RARE",
    )

    stratify_1 = safe_stratify(out["stratify_label2"], test_size=0.20)
    train, tmp = train_test_split(
        out,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=stratify_1,
    )

    stratify_2 = safe_stratify(tmp["stratify_label2"], test_size=0.50)
    val, test = train_test_split(
        tmp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=stratify_2,
    )

    out["random_split"] = "unassigned"
    out.loc[train.index, "random_split"] = "train"
    out.loc[val.index, "random_split"] = "val"
    out.loc[test.index, "random_split"] = "test"
    out = out.drop(columns=["stratify_label", "stratify_label2"])

    assert not out["random_split"].eq("unassigned").any()
    return out


def greedy_group_split(
    df: pd.DataFrame,
    group_col: str,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> pd.Series:
    """Assign complete scaffold-pair groups to approximately 80/10/10 splits."""
    group_sizes = (
        df.groupby(group_col, dropna=False)
        .size()
        .reset_index(name="n")
        .sample(frac=1.0, random_state=RANDOM_SEED)
        .sort_values("n", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )

    total = len(df)
    targets = {
        "scaffold_train": total * train_frac,
        "scaffold_val": total * val_frac,
        "scaffold_test": total * (1.0 - train_frac - val_frac),
    }
    counts = {split: 0 for split in targets}
    group_to_split: dict[str, str] = {}

    for _, row in group_sizes.iterrows():
        group = row[group_col]
        n_group = int(row["n"])

        # Choose the split with the largest relative remaining capacity.
        relative_deficit = {
            split: (targets[split] - counts[split]) / max(targets[split], 1.0)
            for split in targets
        }
        chosen = max(relative_deficit, key=relative_deficit.get)
        group_to_split[group] = chosen
        counts[chosen] += n_group

    assigned = df[group_col].map(group_to_split)
    if assigned.isna().any():
        raise RuntimeError("Some scaffold-pair groups were not assigned to a split.")
    return assigned


def leakage_summary(df: pd.DataFrame, split_col: str, key_col: str) -> pd.DataFrame:
    rows = []
    splits = sorted(df[split_col].dropna().unique())
    for i, split_a in enumerate(splits):
        for split_b in splits[i + 1 :]:
            keys_a = set(df.loc[df[split_col] == split_a, key_col])
            keys_b = set(df.loc[df[split_col] == split_b, key_col])
            rows.append(
                {
                    "split_a": split_a,
                    "split_b": split_b,
                    "key_col": key_col,
                    "overlap_count": len(keys_a & keys_b),
                }
            )
    return pd.DataFrame(rows)


def build_compound_metadata(rows: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Aggregate IDs and names for each canonical structure from both sides."""
    meta: dict[str, dict[str, set[str] | str]] = {}

    for _, row in rows.iterrows():
        for side in ("substrate", "product"):
            smiles = row[f"{side}_canonical_smiles"]
            if not smiles:
                continue
            entry = meta.setdefault(
                smiles,
                {
                    "ids": set(),
                    "names": set(),
                    "scaffold": murcko_scaffold_from_canonical(smiles),
                },
            )
            entry["ids"].update(clean_tokens([row.get(f"{side}_id", "")]))
            entry["names"].update(clean_tokens([row.get(f"{side}_name", "")]))

    compact: dict[str, dict[str, str]] = {}
    for smiles, entry in meta.items():
        compact[smiles] = {
            "compound_ids": "|".join(sorted(entry["ids"])),
            "compound_names": "|".join(sorted(entry["names"])),
            "scaffold": str(entry["scaffold"]),
        }
    return compact


def collapse_positive_pairs(rows: pd.DataFrame) -> pd.DataFrame:
    compound_meta = build_compound_metadata(rows)
    collapsed = []

    for pair_key, group in rows.groupby("undirected_compound_pair_key", sort=True):
        side_a = group["pair_side_a_smiles"].iloc[0]
        side_b = group["pair_side_b_smiles"].iloc[0]
        meta_a = compound_meta[side_a]
        meta_b = compound_meta[side_b]

        collapsed.append(
            {
                "source_row_count": len(group),
                "source_filtered_pair_ids": compact_values(
                    group["source_filtered_pair_id"],
                    split_pipe=True,
                ),
                "reaction_id": compact_values(group.get("reaction_id", pd.Series(dtype=str))),
                "reaction_name": compact_values(group.get("reaction_name", pd.Series(dtype=str))),
                "substrate_id": meta_a["compound_ids"],
                "substrate_name": meta_a["compound_names"],
                "substrate_smiles": side_a,
                "substrate_canonical_smiles": side_a,
                "substrate_scaffold": meta_a["scaffold"],
                "product_id": meta_b["compound_ids"],
                "product_name": meta_b["compound_names"],
                "product_smiles": side_b,
                "product_canonical_smiles": side_b,
                "product_scaffold": meta_b["scaffold"],
                "pathway_names": compact_values(group.get("pathway_names", pd.Series(dtype=str))),
                "directed_compound_pair_key": directed_key(side_a, side_b),
                "undirected_compound_pair_key": pair_key,
                "directed_scaffold_pair_key": directed_key(
                    meta_a["scaffold"],
                    meta_b["scaffold"],
                ),
                "undirected_scaffold_pair_key": undirected_key(
                    meta_a["scaffold"],
                    meta_b["scaffold"],
                ),
            }
        )

    out = pd.DataFrame(collapsed).sort_values(
        "undirected_compound_pair_key",
        kind="mergesort",
    ).reset_index(drop=True)
    out.insert(
        0,
        "positive_pair_id",
        [f"PCYC_POS_{i:07d}" for i in range(1, len(out) + 1)],
    )
    return out


def assert_positive_integrity(pairs: pd.DataFrame) -> None:
    if pairs["undirected_compound_pair_key"].duplicated().any():
        raise AssertionError("Repeated undirected positive chemical pairs remain.")
    if (pairs["substrate_canonical_smiles"] == pairs["product_canonical_smiles"]).any():
        raise AssertionError("Self-pairs remain in the positive table.")
    if pairs.groupby("undirected_compound_pair_key")["random_split"].nunique().max() != 1:
        raise AssertionError("A positive chemical pair crosses random splits.")
    if pairs.groupby("undirected_compound_pair_key")["scaffold_pair_split"].nunique().max() != 1:
        raise AssertionError("A positive chemical pair crosses scaffold-pair splits.")
    if pairs.groupby("undirected_scaffold_pair_key")["scaffold_pair_split"].nunique().max() != 1:
        raise AssertionError("A scaffold-pair group crosses scaffold splits.")


def main() -> None:
    raw = pd.read_csv(IN_FILE, sep="\t", dtype=str)
    print(f"Input filtered positive rows: {len(raw):,}")

    required = {
        "substrate_smiles",
        "product_smiles",
        "substrate_id",
        "product_id",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    raw = raw.reset_index(drop=True)
    raw.insert(
        0,
        "source_filtered_pair_id",
        [f"PCYC_FILTERED_{i:07d}" for i in range(1, len(raw) + 1)],
    )

    raw["substrate_canonical_smiles"] = raw["substrate_smiles"].apply(canonical_smiles)
    raw["product_canonical_smiles"] = raw["product_smiles"].apply(canonical_smiles)

    valid_sub = raw["substrate_canonical_smiles"].ne("")
    valid_prod = raw["product_canonical_smiles"].ne("")
    invalid = raw.loc[~valid_sub | ~valid_prod].copy()
    invalid["invalid_reason"] = ""
    invalid.loc[~valid_sub, "invalid_reason"] += "invalid_substrate_smiles"
    invalid.loc[~valid_prod, "invalid_reason"] += "|invalid_product_smiles"
    invalid["invalid_reason"] = invalid["invalid_reason"].str.strip("|")
    invalid.to_csv(INVALID_FILE, sep="\t", index=False)

    valid = raw.loc[valid_sub & valid_prod].copy()
    valid["pair_side_a_smiles"] = valid.apply(
        lambda row: min(
            row["substrate_canonical_smiles"],
            row["product_canonical_smiles"],
        ),
        axis=1,
    )
    valid["pair_side_b_smiles"] = valid.apply(
        lambda row: max(
            row["substrate_canonical_smiles"],
            row["product_canonical_smiles"],
        ),
        axis=1,
    )
    valid["undirected_compound_pair_key"] = valid.apply(
        lambda row: undirected_key(
            row["pair_side_a_smiles"],
            row["pair_side_b_smiles"],
        ),
        axis=1,
    )

    self_pairs = valid.loc[
        valid["pair_side_a_smiles"] == valid["pair_side_b_smiles"]
    ].copy()
    self_pairs.to_csv(SELF_PAIR_FILE, sep="\t", index=False)
    valid = valid.loc[
        valid["pair_side_a_smiles"] != valid["pair_side_b_smiles"]
    ].copy()

    source_rows_after_qc = len(valid)
    pairs = collapse_positive_pairs(valid)

    duplicate_groups = pairs.loc[pairs["source_row_count"] > 1].copy()
    duplicate_groups.to_csv(DUPLICATE_GROUP_FILE, sep="\t", index=False)

    pairs = make_random_split(pairs)
    pairs["scaffold_pair_split"] = greedy_group_split(
        pairs,
        group_col="undirected_scaffold_pair_key",
    )

    assert_positive_integrity(pairs)
    pairs.to_csv(OUT_FILE, sep="\t", index=False)

    random_pair_leak = leakage_summary(
        pairs,
        "random_split",
        "undirected_compound_pair_key",
    )
    scaffold_pair_leak = leakage_summary(
        pairs,
        "scaffold_pair_split",
        "undirected_compound_pair_key",
    )
    scaffold_group_leak = leakage_summary(
        pairs,
        "scaffold_pair_split",
        "undirected_scaffold_pair_key",
    )

    with QC_FILE.open("w") as handle:
        handle.write("PlantCyc positive reaction-neighbor split QC\n")
        handle.write("============================================\n\n")
        handle.write(f"Input filtered rows: {len(raw):,}\n")
        handle.write(f"Invalid-SMILES rows removed: {len(invalid):,}\n")
        handle.write(f"Self-pair rows removed: {len(self_pairs):,}\n")
        handle.write(f"Valid source rows before deduplication: {source_rows_after_qc:,}\n")
        handle.write(f"Unique undirected positive pairs: {len(pairs):,}\n")
        handle.write(
            f"Source rows collapsed by pair deduplication: "
            f"{source_rows_after_qc - len(pairs):,}\n"
        )
        handle.write(
            f"Positive pairs with multiple source rows: "
            f"{(pairs['source_row_count'] > 1).sum():,}\n"
        )
        handle.write(
            f"Unique undirected scaffold-pair groups: "
            f"{pairs['undirected_scaffold_pair_key'].nunique():,}\n\n"
        )

        handle.write("Random split counts:\n")
        handle.write(pairs["random_split"].value_counts().to_string() + "\n\n")
        handle.write("Scaffold-pair split counts:\n")
        handle.write(pairs["scaffold_pair_split"].value_counts().to_string() + "\n\n")

        handle.write("Random split undirected-pair leakage:\n")
        handle.write(random_pair_leak.to_string(index=False) + "\n\n")
        handle.write("Scaffold split undirected-pair leakage:\n")
        handle.write(scaffold_pair_leak.to_string(index=False) + "\n\n")
        handle.write("Scaffold split scaffold-pair-group leakage:\n")
        handle.write(scaffold_group_leak.to_string(index=False) + "\n")

    print(f"Rows after valid-SMILES and self-pair filtering: {source_rows_after_qc:,}")
    print(f"Unique undirected positive pairs: {len(pairs):,}")
    print("\nRandom split counts:")
    print(pairs["random_split"].value_counts())
    print("\nScaffold-pair split counts:")
    print(pairs["scaffold_pair_split"].value_counts())
    print(f"\nWrote: {OUT_FILE}")
    print(f"Wrote: {QC_FILE}")
    print(f"Wrote: {DUPLICATE_GROUP_FILE}")


if __name__ == "__main__":
    main()

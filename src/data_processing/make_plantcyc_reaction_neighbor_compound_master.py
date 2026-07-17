#!/usr/bin/env python3
"""Build the compound master table and pair-index tables for the PlantCyc
reaction-neighbor task.

The script reads the six balanced 1:1 task tables (easy, hard, and mixed;
random and scaffold-pair splits), creates one deduplicated compound metadata
table keyed by canonical SMILES, and writes pair tables containing integer
indices into that metadata table.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


EXPECTED_TASK_FILES = {
    "plantcyc_reaction_neighbor_easy_random.tsv",
    "plantcyc_reaction_neighbor_hard_random.tsv",
    "plantcyc_reaction_neighbor_mixed_random.tsv",
    "plantcyc_reaction_neighbor_easy_scaffold_pair.tsv",
    "plantcyc_reaction_neighbor_hard_scaffold_pair.tsv",
    "plantcyc_reaction_neighbor_mixed_scaffold_pair.tsv",
}

REQUIRED_COLUMNS = [
    "pair_id",
    "split_scheme",
    "split",
    "label",
    "pair_type",
    "negative_type",
    "negative_subtype",
    "substrate_id",
    "substrate_name",
    "substrate_smiles",
    "substrate_scaffold",
    "product_id",
    "product_name",
    "product_smiles",
    "product_scaffold",
]

PAIR_INDEX_COLUMNS = [
    "pair_id",
    "split_scheme",
    "split",
    "label",
    "pair_type",
    "negative_type",
    "negative_subtype",
    "substrate_compound_index",
    "product_compound_index",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deduplicated compound master and pair-index tables for "
            "the PlantCyc reaction-neighbor task."
        )
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("data/tasks/plantcyc_reaction_neighbor"),
        help="Directory containing the six balanced task TSV files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/features/plantcyc_reaction_neighbor"),
        help="Output directory for metadata and pair-index tables.",
    )
    return parser.parse_args()


def join_unique(values: Iterable[object]) -> str:
    cleaned = {
        str(value).strip()
        for value in values
        if pd.notna(value)
        and str(value).strip()
        and str(value).strip().lower() != "nan"
    }
    return "|".join(sorted(cleaned))


def read_and_validate_task(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise RuntimeError(f"{path.name}: missing required columns: {missing}")

    for column in ["substrate_smiles", "product_smiles"]:
        if df[column].isna().any():
            raise RuntimeError(f"{path.name}: missing values in {column}")
        if df[column].astype(str).str.strip().eq("").any():
            raise RuntimeError(f"{path.name}: empty values in {column}")

    return df


def endpoint_rows(df: pd.DataFrame, side: str, source_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "canonical_smiles": df[f"{side}_smiles"].astype(str),
            "plantcyc_ids": df[f"{side}_id"].fillna("").astype(str),
            "compound_names": df[f"{side}_name"].fillna("").astype(str),
            "murcko_scaffold": df[f"{side}_scaffold"].fillna("").astype(str),
            "source_task_file": source_name,
        }
    )


def main() -> None:
    args = parse_args()
    task_dir = args.task_dir
    out_dir = args.out_dir
    pair_index_dir = out_dir / "pair_indices"

    out_dir.mkdir(parents=True, exist_ok=True)
    pair_index_dir.mkdir(parents=True, exist_ok=True)

    task_files = sorted(task_dir.glob("plantcyc_reaction_neighbor_*_*.tsv"))
    found = {path.name for path in task_files}

    missing_files = sorted(EXPECTED_TASK_FILES - found)
    unexpected_files = sorted(found - EXPECTED_TASK_FILES)
    if missing_files or unexpected_files:
        raise RuntimeError(
            "Expected exactly the six balanced task files.\n"
            f"Missing: {missing_files}\n"
            f"Unexpected: {unexpected_files}"
        )

    task_tables: dict[str, pd.DataFrame] = {}
    compound_frames: list[pd.DataFrame] = []

    for path in task_files:
        df = read_and_validate_task(path)
        task_tables[path.name] = df
        compound_frames.append(endpoint_rows(df, "substrate", path.name))
        compound_frames.append(endpoint_rows(df, "product", path.name))

    all_compounds = pd.concat(compound_frames, ignore_index=True)

    scaffold_counts = (
        all_compounds.loc[
            all_compounds["murcko_scaffold"].astype(str).str.strip().ne("")
        ]
        .groupby("canonical_smiles")["murcko_scaffold"]
        .nunique()
    )
    conflicts = scaffold_counts[scaffold_counts > 1]
    if not conflicts.empty:
        raise RuntimeError(
            f"{len(conflicts)} canonical structures have conflicting Murcko scaffolds."
        )

    metadata = (
        all_compounds.groupby("canonical_smiles", as_index=False)
        .agg(
            {
                "plantcyc_ids": join_unique,
                "compound_names": join_unique,
                "murcko_scaffold": join_unique,
                "source_task_file": join_unique,
            }
        )
        .sort_values("canonical_smiles")
        .reset_index(drop=True)
    )

    metadata.insert(0, "compound_index", range(len(metadata)))
    metadata.insert(
        1,
        "task_compound_id",
        [f"PCYC_NBR_{i:07d}" for i in range(1, len(metadata) + 1)],
    )

    metadata_path = out_dir / "metadata.tsv"
    metadata.to_csv(metadata_path, sep="\t", index=False)

    smiles_to_index = dict(
        zip(metadata["canonical_smiles"], metadata["compound_index"])
    )

    print(f"Task files: {len(task_files)}")
    print(f"Unique compounds: {len(metadata):,}")
    print(f"Wrote: {metadata_path}")

    for file_name, df in task_tables.items():
        indexed = df.copy()
        indexed["substrate_compound_index"] = indexed["substrate_smiles"].map(
            smiles_to_index
        )
        indexed["product_compound_index"] = indexed["product_smiles"].map(
            smiles_to_index
        )

        index_columns = [
            "substrate_compound_index",
            "product_compound_index",
        ]
        if indexed[index_columns].isna().any().any():
            raise RuntimeError(f"{file_name}: compound-index mapping failed")

        indexed[index_columns] = indexed[index_columns].astype(int)

        out_path = pair_index_dir / file_name
        indexed[PAIR_INDEX_COLUMNS].to_csv(out_path, sep="\t", index=False)

        unique_endpoint_count = len(
            set(df["substrate_smiles"]) | set(df["product_smiles"])
        )
        print(
            f"{file_name}: pairs={len(df):,}; "
            f"unique endpoint compounds={unique_endpoint_count:,}; "
            f"wrote={out_path}"
        )

    # Final integrity checks.
    reread = pd.read_csv(metadata_path, sep="\t")
    if reread["compound_index"].duplicated().any():
        raise RuntimeError("Duplicate compound_index values in metadata")
    if reread["canonical_smiles"].duplicated().any():
        raise RuntimeError("Duplicate canonical_smiles values in metadata")
    if reread["compound_index"].tolist() != list(range(len(reread))):
        raise RuntimeError("compound_index values are not contiguous from zero")

    print("\nIntegrity checks passed.")
    print(f"Pair-index directory: {pair_index_dir}")


if __name__ == "__main__":
    main()

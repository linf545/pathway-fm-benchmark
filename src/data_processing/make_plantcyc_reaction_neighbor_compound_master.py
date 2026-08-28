#!/usr/bin/env python3
"""Build the compound master table and pair-index tables for the PlantCyc
reaction-neighbor task.

The script reads balanced reaction-neighbor task tables, creates one
compound-level metadata table keyed by canonical SMILES, adds an InChIKey for
cross-model alignment, and writes pair tables containing integer indices into
that metadata table.

Supported task layouts
----------------------
Legacy unseeded files:
    plantcyc_reaction_neighbor_<variant>_<scheme>.tsv

Repeated benchmark files:
    plantcyc_reaction_neighbor_<variant>_<scheme>_seed<seed>.tsv

In ``auto`` mode, seeded files are preferred when present; otherwise the six
legacy files are used. The script does not mix seeded and unseeded tables in
auto mode.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np
from rdkit import Chem


VARIANTS = ("easy", "hard", "mixed")
SCHEMES = ("random", "scaffold_pair")

LEGACY_EXPECTED_TASK_FILES = {
    f"plantcyc_reaction_neighbor_{variant}_{scheme}.tsv"
    for variant in VARIANTS
    for scheme in SCHEMES
}

TASK_FILE_PATTERN = re.compile(
    r"^plantcyc_reaction_neighbor_"
    r"(?P<variant>easy|hard|mixed)_"
    r"(?P<scheme>random|scaffold_pair)"
    r"(?:_seed(?P<seed>\d+))?\.tsv$"
)

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
            "Create an InChIKey-annotated compound master and pair-index "
            "tables for the PlantCyc reaction-neighbor task."
        )
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("data/tasks/plantcyc_reaction_neighbor"),
        help="Directory containing balanced Task 3 TSV files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/features/plantcyc_reaction_neighbor"),
        help="Output directory for metadata and pair-index tables.",
    )
    parser.add_argument(
        "--task-file-mode",
        choices=["auto", "legacy", "seeded", "all"],
        default="auto",
        help=(
            "Which task tables to consume. auto prefers seeded files when "
            "present, otherwise the six legacy files."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing metadata and selected pair-index files.",
    )
    parser.add_argument(
        "--allow-reindex",
        action="store_true",
        help=(
            "Permit the canonical-SMILES row order to differ from an existing "
            "metadata.tsv. Do not use this when existing compound-level "
            "fingerprints or FM embeddings must remain aligned."
        ),
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


def parse_task_name(path: Path) -> dict[str, str | int | None]:
    match = TASK_FILE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unsupported task filename: {path.name}")
    seed_text = match.group("seed")
    return {
        "variant": match.group("variant"),
        "scheme": match.group("scheme"),
        "seed": int(seed_text) if seed_text is not None else None,
    }


def discover_task_files(task_dir: Path, mode: str) -> list[Path]:
    if not task_dir.exists():
        raise FileNotFoundError(task_dir)

    candidates = sorted(
        path
        for path in task_dir.glob("plantcyc_reaction_neighbor_*.tsv")
        if TASK_FILE_PATTERN.match(path.name)
    )
    legacy = [path for path in candidates if parse_task_name(path)["seed"] is None]
    seeded = [path for path in candidates if parse_task_name(path)["seed"] is not None]

    if mode == "auto":
        selected = seeded if seeded else legacy
        selected_mode = "seeded" if seeded else "legacy"
    elif mode == "legacy":
        selected = legacy
        selected_mode = "legacy"
    elif mode == "seeded":
        selected = seeded
        selected_mode = "seeded"
    else:
        selected = candidates
        selected_mode = "all"

    if not selected:
        raise FileNotFoundError(
            f"No supported Task 3 files found in {task_dir} for mode={mode}"
        )

    if selected_mode == "legacy":
        observed = {path.name for path in selected}
        missing = sorted(LEGACY_EXPECTED_TASK_FILES - observed)
        unexpected = sorted(observed - LEGACY_EXPECTED_TASK_FILES)
        if missing or unexpected:
            raise RuntimeError(
                "Legacy mode expects exactly six task files.\n"
                f"Missing: {missing}\nUnexpected: {unexpected}"
            )

    if selected_mode in {"seeded", "all"} and seeded:
        condition_to_seeds: dict[tuple[str, str], set[int]] = {
            (variant, scheme): set()
            for variant in VARIANTS
            for scheme in SCHEMES
        }
        for path in seeded:
            parsed = parse_task_name(path)
            condition_to_seeds[
                (str(parsed["variant"]), str(parsed["scheme"]))
            ].add(int(parsed["seed"]))

        seed_sets = list(condition_to_seeds.values())
        missing_conditions = [
            f"{variant}/{scheme}"
            for (variant, scheme), seeds in condition_to_seeds.items()
            if not seeds
        ]
        if missing_conditions:
            raise RuntimeError(
                "Seeded task files are missing conditions: "
                + ", ".join(missing_conditions)
            )
        if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
            details = {
                f"{variant}/{scheme}": sorted(seeds)
                for (variant, scheme), seeds in condition_to_seeds.items()
            }
            raise RuntimeError(
                "Seeded task conditions do not share the same seed set:\n"
                + "\n".join(f"  {key}: {value}" for key, value in details.items())
            )

    print(
        f"Task-file mode: {mode}; selected layout: {selected_mode}; "
        f"files: {len(selected)}"
    )
    return selected


def read_and_validate_task(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise RuntimeError(f"{path.name}: missing required columns: {missing}")

    for column in ["substrate_smiles", "product_smiles"]:
        if df[column].isna().any():
            raise RuntimeError(f"{path.name}: missing values in {column}")
        if df[column].astype(str).str.strip().eq("").any():
            raise RuntimeError(f"{path.name}: empty values in {column}")

    df["label"] = df["label"].astype(int)
    if set(df["label"].unique()) != {0, 1}:
        raise RuntimeError(f"{path.name}: labels are not exactly {{0, 1}}")
    if df["pair_id"].astype(str).duplicated().any():
        raise RuntimeError(f"{path.name}: duplicate pair_id values")

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


def smiles_to_inchikey(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise RuntimeError(f"Invalid canonical SMILES in compound master: {smiles}")

    inchikey = Chem.MolToInchiKey(mol)
    if not inchikey:
        raise RuntimeError(f"Could not generate InChIKey: {smiles}")
    return inchikey


def atomic_write_tsv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_csv(temporary, sep="\t", index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    task_dir = args.task_dir
    out_dir = args.out_dir
    pair_index_dir = out_dir / "pair_indices"

    task_files = discover_task_files(task_dir, args.task_file_mode)
    metadata_path = out_dir / "metadata.tsv"
    planned_pair_paths = [pair_index_dir / path.name for path in task_files]
    existing_outputs = [
        path for path in [metadata_path, *planned_pair_paths] if path.exists()
    ]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "Outputs already exist. Use --overwrite to replace them:\n"
            + "\n".join(str(path) for path in existing_outputs[:20])
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
    metadata.insert(
        metadata.columns.get_loc("canonical_smiles") + 1,
        "inchikey",
        metadata["canonical_smiles"].map(smiles_to_inchikey),
    )

    if metadata["inchikey"].isna().any() or metadata["inchikey"].eq("").any():
        raise RuntimeError("Compound metadata contains missing InChIKeys")

    # Standard InChI normalizes some tautomeric/prototropic forms. Distinct
    # canonical SMILES can therefore legitimately share an InChIKey. Task 3
    # compound identity remains task_compound_id/canonical_smiles; InChIKey is
    # retained as a cross-reference rather than enforced as a unique key.
    metadata.insert(
        metadata.columns.get_loc("inchikey") + 1,
        "inchikey_collision_size",
        metadata.groupby("inchikey")["inchikey"].transform("size").astype(int),
    )

    duplicate_inchikeys = metadata.loc[
        metadata["inchikey_collision_size"].gt(1),
        [
            "compound_index",
            "task_compound_id",
            "canonical_smiles",
            "inchikey",
            "inchikey_collision_size",
        ],
    ]
    if not duplicate_inchikeys.empty:
        print(
            "WARNING: distinct canonical SMILES share standard InChIKeys. "
            "They are retained as separate Task 3 compounds and must be aligned "
            "using task_compound_id, not inchikey."
        )
        print(duplicate_inchikeys.to_string(index=False))

    # Existing fingerprints and FM embeddings are aligned to metadata row order.
    # Adding InChIKey must not silently reindex those matrices.
    if metadata_path.exists():
        existing_metadata = pd.read_csv(
            metadata_path,
            sep="\t",
            low_memory=False,
        )
        required_existing = {"compound_index", "canonical_smiles"}
        missing_existing = required_existing - set(existing_metadata.columns)
        if missing_existing:
            raise RuntimeError(
                f"Existing {metadata_path} lacks columns: "
                f"{sorted(missing_existing)}"
            )

        existing_smiles = (
            existing_metadata.sort_values("compound_index")["canonical_smiles"]
            .astype(str)
            .tolist()
        )
        new_smiles = metadata["canonical_smiles"].astype(str).tolist()
        if existing_smiles != new_smiles and not args.allow_reindex:
            raise RuntimeError(
                "The regenerated compound order differs from the existing "
                "metadata.tsv. Existing Morgan/RDKit/FM matrices would become "
                "misaligned. Re-run with the same task-file layout, or use "
                "--allow-reindex only when all compound-level features will be "
                "regenerated."
            )
        if existing_smiles == new_smiles:
            print(
                "Existing compound row order matches regenerated metadata; "
                "current feature matrices remain aligned."
            )

    smiles_to_index = dict(
        zip(metadata["canonical_smiles"], metadata["compound_index"])
    )

    indexed_outputs: dict[Path, pd.DataFrame] = {}
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

        output_columns = PAIR_INDEX_COLUMNS.copy()
        if "split_seed" in indexed.columns:
            output_columns.insert(2, "split_seed")

        indexed_outputs[pair_index_dir / file_name] = indexed[output_columns]

    # Complete all generation and validation before replacing any file.
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_index_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_tsv(metadata, metadata_path)
    for path, frame in indexed_outputs.items():
        atomic_write_tsv(frame, path)

    # Final on-disk integrity checks.
    reread = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    required_metadata = {
        "compound_index",
        "task_compound_id",
        "canonical_smiles",
        "inchikey",
        "inchikey_collision_size",
    }
    missing_metadata = required_metadata - set(reread.columns)
    if missing_metadata:
        raise RuntimeError(
            f"Written metadata lacks columns: {sorted(missing_metadata)}"
        )
    if reread["compound_index"].duplicated().any():
        raise RuntimeError("Duplicate compound_index values in metadata")
    if reread["task_compound_id"].duplicated().any():
        raise RuntimeError("Duplicate task_compound_id values in metadata")
    if reread["canonical_smiles"].duplicated().any():
        raise RuntimeError("Duplicate canonical_smiles values in metadata")
    observed_collision_sizes = (
        reread.groupby("inchikey")["inchikey"].transform("size").astype(int)
    )
    if not np.array_equal(
        reread["inchikey_collision_size"].astype(int).to_numpy(),
        observed_collision_sizes.to_numpy(),
    ):
        raise RuntimeError(
            "Written inchikey_collision_size values do not match the "
            "observed standard-InChIKey collision groups"
        )
    if reread["compound_index"].tolist() != list(range(len(reread))):
        raise RuntimeError("compound_index values are not contiguous from zero")

    print(f"Task files: {len(task_files)}")
    print(f"Unique compounds: {len(metadata):,}")
    print(f"Unique standard InChIKeys: {metadata['inchikey'].nunique():,}")
    print(
        "Compounds in non-unique InChIKey groups:",
        int(metadata["inchikey_collision_size"].gt(1).sum()),
    )
    print(f"Wrote: {metadata_path}")
    print(f"Pair-index files written: {len(indexed_outputs)}")
    print(f"Pair-index directory: {pair_index_dir}")
    print("\nIntegrity checks passed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create seed-specific PlantCyc reaction-neighbor benchmark replicates.

This script reuses the accepted PlantCyc positive-pair and negative-sampling
logic without rerunning SMILES canonicalization. It starts from the existing
collapsed positive-pair table, creates a distinct random and scaffold-pair
split for every requested seed, resamples easy/hard/mixed negatives for that
seed, and writes matching pair-index files against the existing compound
metadata.

The six easy/hard/mixed x random/scaffold_pair datasets are experimental
conditions. The seed-specific files created here are the statistical
replicates used by the modeling benchmark.

Expected placement
------------------
Place this file under ``src/data_processing`` next to:

- make_plantcyc_positive_pair_splits.py
- make_plantcyc_reaction_neighbor_negatives.py

Output naming
-------------
Task tables:
    plantcyc_reaction_neighbor_<variant>_<scheme>_seed<seed>.tsv

Pair-index tables:
    data/features/plantcyc_reaction_neighbor/pair_indices/
    plantcyc_reaction_neighbor_<variant>_<scheme>_seed<seed>.tsv

The existing unseeded files are not modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import make_plantcyc_positive_pair_splits as positive_module  # noqa: E402
import make_plantcyc_reaction_neighbor_negatives as negative_module  # noqa: E402


SCRIPT_VERSION = "plantcyc_reaction_neighbor_replicates_v1"
DEFAULT_SEEDS = [2026, 2027, 2028, 2029, 2030]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate seed-specific PlantCyc reaction-neighbor task tables "
            "and pair-index files for repeated statistical evaluation."
        )
    )
    parser.add_argument(
        "--base-positive-file",
        type=Path,
        default=Path(
            "data/tasks/plantcyc_reaction_neighbor/"
            "plantcyc_positive_pairs_with_splits.tsv"
        ),
        help=(
            "Existing deduplicated positive-pair table. Its old split columns "
            "are overwritten in memory for each replicate."
        ),
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("data/tasks/plantcyc_reaction_neighbor"),
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=Path(
            "data/features/plantcyc_reaction_neighbor/metadata.tsv"
        ),
        help=(
            "Existing compound metadata. All replicate endpoints must already "
            "be represented in this table."
        ),
    )
    parser.add_argument(
        "--pair-index-dir",
        type=Path,
        default=Path(
            "data/features/plantcyc_reaction_neighbor/pair_indices"
        ),
    )
    parser.add_argument(
        "--qc-dir",
        type=Path,
        default=Path("qc/data_processing/plantcyc_replicates"),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing seed-specific outputs.",
    )
    return parser.parse_args()


def validate_base_positive(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    required = {
        "positive_pair_id",
        "substrate_canonical_smiles",
        "product_canonical_smiles",
        "substrate_scaffold",
        "product_scaffold",
        "pathway_names",
        "undirected_compound_pair_key",
        "undirected_scaffold_pair_key",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")

    frame = frame.copy()
    frame["positive_pair_id"] = frame["positive_pair_id"].astype(str)
    if frame["positive_pair_id"].duplicated().any():
        raise ValueError(f"{path}: duplicated positive_pair_id")
    if frame["undirected_compound_pair_key"].duplicated().any():
        raise ValueError(f"{path}: duplicated undirected positive pairs")
    if (
        frame["substrate_canonical_smiles"].astype(str)
        == frame["product_canonical_smiles"].astype(str)
    ).any():
        raise ValueError(f"{path}: self-pairs remain")

    return frame.sort_values("positive_pair_id").reset_index(drop=True)


def load_metadata(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    metadata = pd.read_csv(path, sep="\t", low_memory=False)
    required = {"compound_index", "canonical_smiles"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")

    metadata = metadata.copy()
    metadata["compound_index"] = metadata["compound_index"].astype(int)
    metadata["canonical_smiles"] = metadata["canonical_smiles"].astype(str)
    if metadata["canonical_smiles"].duplicated().any():
        raise ValueError(f"{path}: duplicated canonical_smiles")
    if not np.array_equal(
        metadata["compound_index"].to_numpy(),
        np.arange(len(metadata), dtype=int),
    ):
        raise ValueError(
            f"{path}: compound_index is not contiguous, zero-based, and "
            "aligned to row order"
        )

    return metadata, dict(
        zip(metadata["canonical_smiles"], metadata["compound_index"])
    )


def build_positive_replicate(
    base_positive: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    positive_module.RANDOM_SEED = int(seed)
    replicate = positive_module.make_random_split(base_positive.copy())
    replicate["scaffold_pair_split"] = (
        positive_module.greedy_group_split(
            replicate,
            group_col="undirected_scaffold_pair_key",
        )
    )
    replicate["split_seed"] = int(seed)
    positive_module.assert_positive_integrity(replicate)
    return replicate


def assignment_signature(
    frame: pd.DataFrame,
    split_column: str,
) -> tuple[str, ...]:
    ordered = frame.sort_values("positive_pair_id")
    return tuple(ordered[split_column].astype(str).tolist())


def validate_nonidentical_replicates(
    positives: dict[int, pd.DataFrame],
) -> None:
    for split_column in ["random_split", "scaffold_pair_split"]:
        seen: dict[tuple[str, ...], int] = {}
        for seed, frame in positives.items():
            signature = assignment_signature(frame, split_column)
            if signature in seen:
                raise RuntimeError(
                    f"Seeds {seen[signature]} and {seed} produced identical "
                    f"{split_column} assignments. Change the requested seeds "
                    "or revise the split generator before treating them as "
                    "independent replicates."
                )
            seen[signature] = seed


def task_file_name(variant: str, scheme: str, seed: int) -> str:
    return (
        f"plantcyc_reaction_neighbor_{variant}_{scheme}_seed{seed}.tsv"
    )


def index_task_table(
    task: pd.DataFrame,
    smiles_to_index: dict[str, int],
    seed: int,
) -> pd.DataFrame:
    indexed = task.copy()
    indexed["substrate_compound_index"] = (
        indexed["substrate_smiles"].astype(str).map(smiles_to_index)
    )
    indexed["product_compound_index"] = (
        indexed["product_smiles"].astype(str).map(smiles_to_index)
    )

    index_columns = [
        "substrate_compound_index",
        "product_compound_index",
    ]
    if indexed[index_columns].isna().any().any():
        missing_smiles = sorted(
            set(
                indexed.loc[
                    indexed["substrate_compound_index"].isna(),
                    "substrate_smiles",
                ].astype(str)
            )
            | set(
                indexed.loc[
                    indexed["product_compound_index"].isna(),
                    "product_smiles",
                ].astype(str)
            )
        )
        raise ValueError(
            "Replicate endpoints are absent from the existing compound "
            f"metadata; n_missing={len(missing_smiles)}. First examples: "
            f"{missing_smiles[:5]}"
        )

    indexed[index_columns] = indexed[index_columns].astype(int)
    indexed["split_seed"] = int(seed)
    return indexed


def qc_rows_for_task(
    task: pd.DataFrame,
    variant: str,
    scheme: str,
    seed: int,
    source_file: Path,
    audit: dict,
) -> list[dict]:
    rows = []
    for split_name, subset in task.groupby("split", sort=False):
        row = {
            "task": "plantcyc_reaction_neighbor",
            "negative_variant": variant,
            "split_scheme": scheme,
            "split_seed": int(seed),
            "split": str(split_name),
            "n_pairs": int(len(subset)),
            "n_negative": int((subset["label"] == 0).sum()),
            "n_positive": int((subset["label"] == 1).sum()),
            "positive_fraction": float(subset["label"].mean()),
            "n_easy_negative": int(
                (
                    (subset["label"] == 0)
                    & (subset["negative_type"] == "easy")
                ).sum()
            ),
            "n_hard_negative": int(
                (
                    (subset["label"] == 0)
                    & (subset["negative_type"] == "hard")
                ).sum()
            ),
            "source_positive_file": str(source_file),
        }
        row.update({f"audit_{key}": value for key, value in audit.items()})
        rows.append(row)
    return rows


def planned_output_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for seed in args.seeds:
        paths.append(
            args.task_dir
            / f"plantcyc_positive_pairs_with_splits_seed{seed}.tsv"
        )
        for scheme in negative_module.SPLIT_SCHEMES:
            for variant in negative_module.VARIANTS:
                name = task_file_name(variant, scheme, seed)
                paths.extend(
                    [
                        args.task_dir / name,
                        args.pair_index_dir / name,
                        args.qc_dir
                        / name.replace(".tsv", "_qc.tsv"),
                    ]
                )
    paths.extend(
        [
            args.qc_dir / "plantcyc_reaction_neighbor_replicate_manifest.tsv",
            args.qc_dir / "plantcyc_reaction_neighbor_replicate_config.json",
        ]
    )
    return paths


def main() -> None:
    args = parse_args()
    args.seeds = list(dict.fromkeys(args.seeds))
    if len(args.seeds) < 2:
        raise ValueError(
            "Use at least two seeds for a repeated benchmark; five are "
            "recommended."
        )

    if not args.base_positive_file.exists():
        raise FileNotFoundError(args.base_positive_file)

    existing = [path for path in planned_output_paths(args) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Seed-specific outputs already exist. Use --overwrite to replace "
            "them:\n" + "\n".join(str(path) for path in existing[:20])
        )

    base_positive = validate_base_positive(
        pd.read_csv(args.base_positive_file, sep="\t", low_memory=False),
        args.base_positive_file,
    )
    _, smiles_to_index = load_metadata(args.metadata_file)

    positive_replicates = {
        seed: build_positive_replicate(base_positive, seed)
        for seed in args.seeds
    }
    validate_nonidentical_replicates(positive_replicates)

    generated_files: dict[Path, pd.DataFrame | str] = {}
    manifest_rows: list[dict] = []

    for seed, positive in positive_replicates.items():
        positive_path = (
            args.task_dir
            / f"plantcyc_positive_pairs_with_splits_seed{seed}.tsv"
        )
        generated_files[positive_path] = positive

        negative_module.RANDOM_SEED = int(seed)

        for scheme, config in negative_module.SPLIT_SCHEMES.items():
            split_col = config["split_col"]
            split_order = negative_module.observed_split_order(
                positive,
                split_col,
                config["split_order"],
            )

            negative_pools, pool_qc = (
                negative_module.generate_negative_pools(
                    pos=positive,
                    split_col=split_col,
                    split_scheme=scheme,
                    split_order=split_order,
                    seed_offset=config["seed_offset"],
                )
            )

            for variant in negative_module.VARIANTS:
                task, _, mixed_targets = (
                    negative_module.build_variant_dataset(
                        pos=positive,
                        negative_pools=negative_pools,
                        split_col=split_col,
                        split_scheme=scheme,
                        split_order=split_order,
                        variant=variant,
                        seed_offset=config["seed_offset"],
                    )
                )
                audit = negative_module.assert_final_integrity(
                    out=task,
                    split_scheme=scheme,
                    variant=variant,
                    mixed_targets=mixed_targets,
                )
                task = negative_module.sort_output(task, split_order)
                task["split_seed"] = int(seed)

                name = task_file_name(variant, scheme, seed)
                task_path = args.task_dir / name
                pair_index_path = args.pair_index_dir / name
                qc_path = args.qc_dir / name.replace(".tsv", "_qc.tsv")

                indexed = index_task_table(task, smiles_to_index, seed)
                qc = pd.DataFrame(
                    qc_rows_for_task(
                        task=task,
                        variant=variant,
                        scheme=scheme,
                        seed=seed,
                        source_file=positive_path,
                        audit=audit,
                    )
                )

                generated_files[task_path] = task
                generated_files[pair_index_path] = indexed
                generated_files[qc_path] = qc

                manifest_rows.append(
                    {
                        "split_seed": int(seed),
                        "split_scheme": scheme,
                        "negative_variant": variant,
                        "task_file": str(task_path),
                        "pair_index_file": str(pair_index_path),
                        "n_pairs": int(len(task)),
                        "n_positive": int((task["label"] == 1).sum()),
                        "n_negative": int((task["label"] == 0).sum()),
                        "pool_qc_rows": int(len(pool_qc)),
                    }
                )

    args.task_dir.mkdir(parents=True, exist_ok=True)
    args.pair_index_dir.mkdir(parents=True, exist_ok=True)
    args.qc_dir.mkdir(parents=True, exist_ok=True)

    # All generation and integrity checks have completed before any output is
    # replaced. This avoids the destructive partial-overwrite behavior of the
    # original one-off negative generator.
    for path, content in generated_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, pd.DataFrame):
            content.to_csv(path, sep="\t", index=False)
        else:
            path.write_text(content)

    manifest_path = (
        args.qc_dir
        / "plantcyc_reaction_neighbor_replicate_manifest.tsv"
    )
    pd.DataFrame(manifest_rows).to_csv(
        manifest_path,
        sep="\t",
        index=False,
    )

    config_path = (
        args.qc_dir
        / "plantcyc_reaction_neighbor_replicate_config.json"
    )
    config_path.write_text(
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "base_positive_file": str(args.base_positive_file),
                "metadata_file": str(args.metadata_file),
                "seeds": args.seeds,
                "positive_random_split_method": (
                    "pathway-stratified pair-level 80/10/10"
                ),
                "positive_scaffold_split_method": (
                    "greedy undirected-scaffold-pair-group 80/10/10"
                ),
                "negative_variants": list(negative_module.VARIANTS),
                "negative_sampling_seed_equals_split_seed": True,
            },
            indent=2,
        )
    )

    print("\nGenerated PlantCyc reaction-neighbor replicates")
    print("Seeds:", args.seeds)
    print("Task files:", len(args.seeds) * 6)
    print("Pair-index files:", len(args.seeds) * 6)
    print("Wrote:", manifest_path)
    print("Wrote:", config_path)


if __name__ == "__main__":
    main()

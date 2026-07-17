#!/usr/bin/env python

"""Build a balanced COCONUT-vs-ZINC NP-likeness benchmark task."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

TASK_NAME = "np_likeness"
LABEL_COL = "is_np"
DEFAULT_SPLIT_SEEDS = [2026, 2027, 2028, 2029, 2030]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the balanced COCONUT-vs-ZINC NP-likeness task."
    )
    parser.add_argument(
        "--input-csv",
        default="raw_data/NPlikeliness/data/coconut_synthetic.csv",
    )
    parser.add_argument(
        "--keep-file",
        default="raw_data/NPlikeliness/data/to_keep_molecules.pkl",
    )
    parser.add_argument("--task-dir", default="data/tasks/np_likeness")
    parser.add_argument("--split-dir", default="data/splits/np_likeness")
    parser.add_argument("--balance-seed", type=int, default=2026)
    parser.add_argument(
        "--split-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SPLIT_SEEDS,
    )
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument(
        "--scaffold-candidates",
        type=int,
        default=10,
        help="Randomized greedy scaffold assignments evaluated per seed.",
    )
    parser.add_argument("--scaffold-size-weight", type=float, default=1.0)
    parser.add_argument("--scaffold-label-weight", type=float, default=1.0)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=max(1, min(8, mp.cpu_count())),
    )
    parser.add_argument("--chunksize", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    fractions = np.asarray(
        [args.train_fraction, args.validation_fraction, args.test_fraction],
        dtype=float,
    )
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Train/validation/test fractions must be positive and sum to 1.")
    if args.scaffold_candidates <= 0:
        raise ValueError("--scaffold-candidates must be positive.")
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be positive.")


def apply_keep_filter(df: pd.DataFrame, keep_path: Path) -> tuple[pd.DataFrame, str]:
    with keep_path.open("rb") as handle:
        keep = pickle.load(handle)

    arr = np.asarray(keep)
    if arr.dtype == bool:
        if len(arr) != len(df):
            raise ValueError("Boolean keep mask length does not match input rows.")
        filtered = df.loc[arr].copy()
        description = "boolean row mask"
    elif np.issubdtype(arr.dtype, np.integer):
        if len(arr) == 0 or arr.min() < 0 or arr.max() >= len(df):
            raise ValueError("Integer keep indices are empty or out of bounds.")
        filtered = df.iloc[arr].copy()
        description = "integer row-index list"
    elif len(arr) > 0 and isinstance(arr[0], str):
        filtered = df[df["raw_smiles"].isin(set(map(str, arr)))].copy()
        description = "SMILES membership list"
    else:
        raise TypeError(f"Unsupported keep-file format: dtype={arr.dtype}")

    return filtered.reset_index(drop=True), description


def process_smiles(smiles: str) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None, None, None, None, "rdkit_parse_failed"

        canonical_smiles = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )
        canonical_mol = Chem.MolFromSmiles(canonical_smiles)
        if canonical_mol is None:
            return None, None, None, None, "canonical_parse_failed"

        inchikey = Chem.MolToInchiKey(canonical_mol)
        if not inchikey:
            return canonical_smiles, None, None, None, "inchikey_failed"

        murcko_scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=canonical_mol,
            includeChirality=True,
        )
        scaffold_split_key = (
            murcko_scaffold
            if murcko_scaffold
            else f"ACYCLIC::{canonical_smiles}"
        )
        return canonical_smiles, inchikey, murcko_scaffold, scaffold_split_key, None
    except Exception as exc:  # defensive for unusual RDKit failures
        return None, None, None, None, f"{type(exc).__name__}: {exc}"


def process_all_smiles(
    smiles_values: Iterable[str],
    n_jobs: int,
    chunksize: int,
) -> list[tuple[str | None, str | None, str | None, str | None, str | None]]:
    values = list(smiles_values)
    total = len(values)

    if n_jobs == 1:
        results = []
        for i, smiles in enumerate(values, start=1):
            results.append(process_smiles(smiles))
            if i % 10000 == 0 or i == total:
                print(f"Processed molecules: {i:,}/{total:,}")
        return results

    results = []
    with mp.get_context("spawn").Pool(processes=n_jobs) as pool:
        iterator = pool.imap(process_smiles, values, chunksize=chunksize)
        for i, result in enumerate(iterator, start=1):
            results.append(result)
            if i % 10000 == 0 or i == total:
                print(f"Processed molecules: {i:,}/{total:,}")
    return results


def canonicalize_and_deduplicate(
    df: pd.DataFrame,
    n_jobs: int,
    chunksize: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    results = process_all_smiles(
        df["raw_smiles"].astype(str),
        n_jobs=n_jobs,
        chunksize=chunksize,
    )

    processed = df.copy()
    processed[[
        "canonical_smiles",
        "inchikey",
        "murcko_scaffold",
        "scaffold_split_key",
        "processing_error",
    ]] = pd.DataFrame(results, index=processed.index)

    invalid_rows = int(processed["processing_error"].notna().sum())
    processed = processed[processed["processing_error"].isna()].copy()

    per_key_labels = processed.groupby("inchikey")[LABEL_COL].nunique()
    conflict_keys = set(per_key_labels[per_key_labels > 1].index)
    conflict_rows = int(processed["inchikey"].isin(conflict_keys).sum())
    processed = processed[~processed["inchikey"].isin(conflict_keys)].copy()

    before_dedup = len(processed)
    processed = (
        processed.sort_values(
            [LABEL_COL, "inchikey", "raw_row_index"],
            kind="mergesort",
        )
        .drop_duplicates("inchikey", keep="first")
        .reset_index(drop=True)
    )

    qc = {
        "invalid_rows": invalid_rows,
        "cross_label_conflict_keys": len(conflict_keys),
        "cross_label_conflict_rows": conflict_rows,
        "same_class_duplicate_rows_removed": before_dedup - len(processed),
    }
    return processed, qc


def balance_dataset(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, int]]:
    negatives = df[df[LABEL_COL] == 0].copy()
    positives = df[df[LABEL_COL] == 1].copy()
    if negatives.empty or positives.empty:
        raise ValueError("Both classes must be present after processing.")

    target = min(len(negatives), len(positives))
    negatives = negatives.sample(n=target, random_state=seed)
    positives = positives.sample(n=target, random_state=seed)

    balanced = pd.concat([negatives, positives], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    balanced.insert(
        0,
        "compound_id",
        [f"NPLIKE_{i:07d}" for i in range(1, len(balanced) + 1)],
    )

    qc = {
        "valid_negatives_before_balancing": len(df[df[LABEL_COL] == 0]),
        "valid_positives_before_balancing": len(df[df[LABEL_COL] == 1]),
        "selected_per_class": target,
        "balanced_total": len(balanced),
    }
    return balanced, qc


def build_split_table(task: pd.DataFrame, split: np.ndarray) -> pd.DataFrame:
    result = task[[
        "compound_id",
        "inchikey",
        LABEL_COL,
        "source",
        "murcko_scaffold",
        "scaffold_split_key",
    ]].copy()
    result["split"] = split
    if result["split"].isna().any() or (result["split"] == "").any():
        raise RuntimeError("Some compounds were not assigned to a split.")
    return result


def make_random_split(
    task: pd.DataFrame,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> pd.DataFrame:
    indices = np.arange(len(task))
    labels = task[LABEL_COL].to_numpy(dtype=int)

    train_idx, remainder_idx = train_test_split(
        indices,
        test_size=validation_fraction + test_fraction,
        random_state=seed,
        stratify=labels,
    )
    relative_test_fraction = test_fraction / (validation_fraction + test_fraction)
    validation_idx, test_idx = train_test_split(
        remainder_idx,
        test_size=relative_test_fraction,
        random_state=seed,
        stratify=labels[remainder_idx],
    )

    split = np.full(len(task), "", dtype=object)
    split[train_idx] = "train"
    split[validation_idx] = "validation"
    split[test_idx] = "test"
    return build_split_table(task, split)


def make_scaffold_groups(task: pd.DataFrame) -> pd.DataFrame:
    groups = (
        task.groupby("scaffold_split_key", sort=False)[LABEL_COL]
        .agg(["size", "sum"])
        .rename(columns={"sum": "n_positive"})
        .reset_index()
    )
    groups["n_negative"] = groups["size"] - groups["n_positive"]
    return groups


def assignment_objective(
    counts: np.ndarray,
    targets: np.ndarray,
    size_weight: float,
    label_weight: float,
) -> float:
    normalized = (counts - targets) / np.maximum(targets, 1.0)
    size_error = float(np.square(normalized[:, 0]).sum())
    label_error = float(np.square(normalized[:, 1:]).sum())
    return size_weight * size_error + label_weight * label_error


def one_scaffold_assignment(
    groups: pd.DataFrame,
    seed: int,
    targets: np.ndarray,
    size_weight: float,
    label_weight: float,
) -> tuple[dict[str, str], float]:
    rng = np.random.default_rng(seed)
    work = groups.copy()
    work["random_tie_break"] = rng.random(len(work))
    work = work.sort_values(
        ["size", "random_tie_break"],
        ascending=[False, True],
        kind="mergesort",
    )

    split_names = ["train", "validation", "test"]
    counts = np.zeros((3, 3), dtype=float)
    assignment: dict[str, str] = {}

    for row in work.itertuples(index=False):
        group_vector = np.asarray(
            [row.size, row.n_positive, row.n_negative],
            dtype=float,
        )
        objectives = []
        for split_index in range(3):
            candidate = counts.copy()
            candidate[split_index] += group_vector
            objectives.append(
                assignment_objective(
                    candidate,
                    targets,
                    size_weight=size_weight,
                    label_weight=label_weight,
                )
            )

        best = min(objectives)
        tied = [i for i, value in enumerate(objectives) if np.isclose(value, best)]
        chosen = int(rng.choice(tied))
        counts[chosen] += group_vector
        assignment[str(row.scaffold_split_key)] = split_names[chosen]

    return assignment, assignment_objective(
        counts,
        targets,
        size_weight=size_weight,
        label_weight=label_weight,
    )


def make_scaffold_split(
    task: pd.DataFrame,
    seed: int,
    fractions: np.ndarray,
    n_candidates: int,
    size_weight: float,
    label_weight: float,
) -> pd.DataFrame:
    groups = make_scaffold_groups(task)
    total = float(len(task))
    positive = float(task[LABEL_COL].sum())
    negative = total - positive
    targets = np.column_stack(
        [fractions * total, fractions * positive, fractions * negative]
    )

    best_assignment = None
    best_objective = np.inf
    for candidate_index in range(n_candidates):
        assignment, objective = one_scaffold_assignment(
            groups=groups,
            seed=int(seed) * 100000 + candidate_index,
            targets=targets,
            size_weight=size_weight,
            label_weight=label_weight,
        )
        if objective < best_objective:
            best_assignment = assignment
            best_objective = objective

    if best_assignment is None:
        raise RuntimeError("Scaffold assignment failed.")

    split = task["scaffold_split_key"].map(best_assignment).to_numpy()
    return build_split_table(task, split)


def split_qc_lines(split_table: pd.DataFrame, mode: str, seed: int) -> list[str]:
    lines = [f"[{mode} seed={seed}]"]
    for split_name in ["train", "validation", "test"]:
        subset = split_table[split_table["split"] == split_name]
        counts = subset[LABEL_COL].value_counts().sort_index().to_dict()
        lines.append(
            f"{split_name}: n={len(subset):,}; "
            f"synthetic={counts.get(0, 0):,}; "
            f"natural_product={counts.get(1, 0):,}; "
            f"positive_fraction={subset[LABEL_COL].mean():.6f}"
        )

    if mode == "scaffold":
        for left, right in [
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ]:
            left_keys = set(
                split_table.loc[split_table["split"] == left, "scaffold_split_key"]
            )
            right_keys = set(
                split_table.loc[split_table["split"] == right, "scaffold_split_key"]
            )
            overlap = left_keys & right_keys
            lines.append(f"scaffold-key overlap {left}/{right}: {len(overlap)}")
            if overlap:
                raise RuntimeError(
                    f"Scaffold leakage detected for {left}/{right}: {len(overlap)}"
                )
    lines.append("")
    return lines


def check_output_safety(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        preview = "\n".join(f"  {path}" for path in existing[:20])
        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace them:\n" + preview
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    task_dir = Path(args.task_dir)
    split_dir = Path(args.split_dir)
    task_path = task_dir / "np_likeness_task.tsv"
    qc_path = task_dir / "np_likeness_qc.txt"
    split_paths = [
        split_dir / f"np_likeness_{mode}_split_seed{seed}.tsv"
        for mode in ["random", "scaffold"]
        for seed in args.split_seeds
    ]
    check_output_safety([task_path, qc_path, *split_paths], args.overwrite)

    task_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(
        args.input_csv,
        usecols=["smiles", "npl", "ZINC_ID", LABEL_COL],
    ).rename(
        columns={
            "smiles": "raw_smiles",
            "npl": "original_npl_score",
            "ZINC_ID": "zinc_id",
        }
    )
    raw.insert(0, "raw_row_index", np.arange(len(raw), dtype=int))

    if not set(raw[LABEL_COL].dropna().unique()).issubset({0, 1}):
        raise ValueError("is_np contains values outside {0, 1}.")

    filtered, keep_description = apply_keep_filter(raw, Path(args.keep_file))
    print(f"Rows after published filter: {len(filtered):,}")
    print(filtered[LABEL_COL].value_counts().sort_index().to_dict())

    processed, processing_qc = canonicalize_and_deduplicate(
        filtered,
        n_jobs=args.n_jobs,
        chunksize=args.chunksize,
    )
    processed["source"] = np.where(
        processed[LABEL_COL].eq(1),
        "COCONUT_natural_product",
        "ZINC_synthetic_decoy",
    )

    balanced, balance_qc = balance_dataset(processed, seed=args.balance_seed)
    task_columns = [
        "compound_id",
        "raw_smiles",
        "canonical_smiles",
        "inchikey",
        "murcko_scaffold",
        "scaffold_split_key",
        LABEL_COL,
        "source",
        "original_npl_score",
        "zinc_id",
        "raw_row_index",
    ]
    task = balanced[task_columns].copy()

    if task["inchikey"].duplicated().any():
        raise RuntimeError("Final task contains duplicated InChIKeys.")
    if task[LABEL_COL].value_counts().nunique() != 1:
        raise RuntimeError("Final task is not class-balanced.")

    task.to_csv(task_path, sep="\t", index=False)
    print("Wrote:", task_path)

    qc_lines = [
        f"task: {TASK_NAME}",
        f"raw_input: {args.input_csv}",
        f"published_keep_file: {args.keep_file}",
        f"published_keep_format: {keep_description}",
        f"raw_rows: {len(raw):,}",
        f"rows_after_published_filter: {len(filtered):,}",
        "published_filtered_class_counts: "
        + str(filtered[LABEL_COL].value_counts().sort_index().to_dict()),
        f"invalid_rows_removed: {processing_qc['invalid_rows']:,}",
        f"cross_label_conflict_keys_removed: {processing_qc['cross_label_conflict_keys']:,}",
        f"cross_label_conflict_rows_removed: {processing_qc['cross_label_conflict_rows']:,}",
        f"same_class_duplicate_rows_removed: {processing_qc['same_class_duplicate_rows_removed']:,}",
        f"valid_negatives_before_balancing: {balance_qc['valid_negatives_before_balancing']:,}",
        f"valid_positives_before_balancing: {balance_qc['valid_positives_before_balancing']:,}",
        f"balance_seed: {args.balance_seed}",
        f"selected_per_class: {balance_qc['selected_per_class']:,}",
        f"balanced_total: {balance_qc['balanced_total']:,}",
        "acyclic_scaffold_policy: empty Murcko scaffolds use a molecule-specific scaffold_split_key",
        "",
    ]

    fractions = np.asarray(
        [args.train_fraction, args.validation_fraction, args.test_fraction],
        dtype=float,
    )

    for seed in args.split_seeds:
        random_table = make_random_split(
            task,
            seed=seed,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
        )
        random_path = split_dir / f"np_likeness_random_split_seed{seed}.tsv"
        random_table.to_csv(random_path, sep="\t", index=False)
        qc_lines.extend(split_qc_lines(random_table, "random", seed))
        print("Wrote:", random_path)

        scaffold_table = make_scaffold_split(
            task,
            seed=seed,
            fractions=fractions,
            n_candidates=args.scaffold_candidates,
            size_weight=args.scaffold_size_weight,
            label_weight=args.scaffold_label_weight,
        )
        scaffold_path = split_dir / f"np_likeness_scaffold_split_seed{seed}.tsv"
        scaffold_table.to_csv(scaffold_path, sep="\t", index=False)
        qc_lines.extend(split_qc_lines(scaffold_table, "scaffold", seed))
        print("Wrote:", scaffold_path)

    qc_path.write_text("\n".join(qc_lines))
    print("Wrote:", qc_path)
    print("Final rows:", f"{len(task):,}")
    print("Final class counts:", task[LABEL_COL].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()

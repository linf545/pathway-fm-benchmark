#!/usr/bin/env python
"""
Final repeated NPClassifier benchmark.

This script replaces the separate fingerprint/descriptor and frozen-embedding
training scripts while preserving their single-label multiclass logistic-
regression design.

Efficiency changes:
1. Morgan fingerprints are CSR sparse matrices, not dense 78k x 2048 arrays.
2. Morgan features are cached; RDKit descriptors are cached only when explicitly requested and reused.
3. Independent split/seed runs can execute concurrently.
4. All retained representations share one loading, QC, metric, resume, and
   aggregation framework.
5. FM arrays are memory-mapped before the Task 1 rows are aligned.

The default C grid contains only 1.0, preserving the prior fixed-C experiment
without multiplying runtime. Supplying multiple C values selects C using
validation macro one-vs-rest AUROC, then macro AUPRC, then smaller C.

Primary metric: macro one-vs-rest AUROC.
Secondary metric: macro one-vs-rest AUPRC.
F1 and accuracy metrics are supplementary.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_VERSION = "npclassifier_full_repeated_v2_sparse_lbfgs"

DEFAULT_SPLIT_SEEDS = [2026, 2027, 2028, 2029, 2030]

DEFAULT_FEATURES = [
    "morgan_r2_2048",
    "morgan_r3_2048",
    "chemberta_77m_mlm",
    "chemberta_100m_mlm",
    "molformer_c3_1_1b",
    "nafm_graph_mean",
    "molclr_gcn",
    "molclr_gin",
    "graphmvp_gin",
    "molebert_gin",
]

FM_FEATURES = {
    "chemberta_77m_mlm",
    "chemberta_100m_mlm",
    "molformer_c3_1_1b",
    "nafm_graph_mean",
    "molclr_gcn",
    "molclr_gin",
    "graphmvp_gin",
    "molebert_gin",
}

OVERALL_METRICS = [
    "auroc_ovr_macro",
    "auprc_ovr_macro",
    "macro_f1",
    "weighted_f1",
    "accuracy",
    "balanced_accuracy",
]


@dataclass(frozen=True)
class SplitSpec:
    split_mode: str
    split_column: str
    train_value: str
    validation_value: str
    test_value: str
    split_seed: int


@dataclass(frozen=True)
class FeatureData:
    name: str
    kind: str
    matrix: Any
    source_path: str
    n_features: int
    sparse_matrix: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeated frozen-feature NPClassifier benchmark."
    )
    parser.add_argument(
        "--task",
        default="data/tasks/np_classification/npclassifier_label_task.tsv",
    )
    parser.add_argument(
        "--embedding-dir",
        default="data/features/bao_plant_pathway/full",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/features/np_classification/full",
    )
    parser.add_argument(
        "--output-dir",
        default="results/np_classification/full",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
    )
    parser.add_argument(
        "--split-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SPLIT_SEEDS,
    )
    parser.add_argument(
        "--split-modes",
        nargs="+",
        choices=["random", "scaffold"],
        default=["random", "scaffold"],
    )
    parser.add_argument(
        "--C-grid",
        nargs="+",
        type=float,
        default=[0.01, 0.1, 1.0],
        help=(
            "Regularization values evaluated on the validation set. "
            "Default: 0.01 0.1 1.0."
        ),
    )
    parser.add_argument(
        "--parallel-runs",
        type=int,
        default=2,
        help=(
            "Number of split/seed fits run concurrently for one feature. "
            "Use BLAS/OpenMP thread counts of 1."
        ),
    )
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument(
        "--dense-solver",
        default="lbfgs",
        help="Solver for FM embeddings and RDKit descriptors.",
    )
    parser.add_argument(
        "--sparse-solver",
        default="lbfgs",
        help=(
            "Solver for sparse Morgan fingerprints. L-BFGS accepts CSR input "
            "and preserves the multiclass logistic-regression objective while "
            "avoiding the slow SAGA convergence observed for this task."
        ),
    )
    parser.add_argument("--dense-max-iter", type=int, default=500)
    parser.add_argument("--sparse-max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument(
        "--skip-missing-features",
        action="store_true",
    )
    parser.add_argument(
        "--rebuild-feature-cache",
        action="store_true",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save validation/test class probabilities; off by default.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite-results", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    unknown = sorted(set(args.features) - set(DEFAULT_FEATURES))
    if unknown:
        raise ValueError(
            f"Unknown features: {unknown}. Allowed: {DEFAULT_FEATURES}"
        )
    if len(set(args.features)) != len(args.features):
        raise ValueError("--features contains duplicates")
    if len(set(args.split_seeds)) != len(args.split_seeds):
        raise ValueError("--split-seeds contains duplicates")
    if args.parallel_runs < 1:
        raise ValueError("--parallel-runs must be >= 1")
    if not args.C_grid:
        raise ValueError("--C-grid cannot be empty")
    if any((not np.isfinite(value)) or value <= 0 for value in args.C_grid):
        raise ValueError("All C values must be positive and finite")


def load_task(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    task = pd.read_csv(path, sep="\t", low_memory=False)
    required = {"inchikey", "canonical_smiles"}
    missing = required - set(task.columns)
    if missing:
        raise ValueError(f"Task is missing columns: {sorted(missing)}")
    if task["inchikey"].isna().any() or not task["inchikey"].is_unique:
        raise ValueError("Task InChIKeys must be nonmissing and unique")
    if task["canonical_smiles"].isna().any():
        raise ValueError("Task contains missing canonical SMILES")

    label_columns = sorted(
        [column for column in task.columns if column.startswith("label_")],
        key=lambda column: int(column.replace("label_", "")),
    )
    if len(label_columns) < 2:
        raise ValueError(
            f"Expected at least two label columns; found {label_columns}"
        )

    label_matrix = task[label_columns].to_numpy(dtype=np.int8)
    cardinality = label_matrix.sum(axis=1)
    if np.any(cardinality != 1):
        bad = np.flatnonzero(cardinality != 1)
        examples = task.iloc[bad[:10]][["inchikey", *label_columns]]
        raise ValueError(
            f"{len(bad)} rows are not single-label:\n"
            + examples.to_string(index=False)
        )

    y = label_matrix.argmax(axis=1).astype(np.int64)
    expected_classes = np.arange(len(label_columns))
    if not np.array_equal(np.unique(y), expected_classes):
        raise ValueError("Observed classes do not match label columns")

    label_names = [
        column.replace("label_", "") for column in label_columns
    ]
    return task, y, label_names


def split_spec(mode: str, seed: int) -> SplitSpec:
    if mode == "random":
        return SplitSpec(
            "random",
            f"random_split_{seed}",
            "train",
            "val",
            "test",
            seed,
        )
    if mode == "scaffold":
        return SplitSpec(
            "scaffold",
            f"scaffold_split_{seed}",
            "scaffold_train",
            "scaffold_val",
            "scaffold_test",
            seed,
        )
    raise ValueError(f"Unknown split mode: {mode}")


def build_splits(
    task: pd.DataFrame,
    y: np.ndarray,
    modes: Iterable[str],
    seeds: Iterable[int],
    n_classes: int,
) -> tuple[dict[tuple[str, int], dict[str, np.ndarray]], pd.DataFrame]:
    split_indices: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    qc_rows: list[dict[str, Any]] = []

    scaffold_column = next(
        (
            candidate
            for candidate in ["murcko_scaffold", "scaffold"]
            if candidate in task.columns
        ),
        None,
    )

    for mode in modes:
        for seed in seeds:
            spec = split_spec(mode, seed)
            if spec.split_column not in task.columns:
                raise ValueError(
                    f"Missing split column: {spec.split_column}"
                )

            values = task[spec.split_column].astype(str)
            allowed = {
                spec.train_value,
                spec.validation_value,
                spec.test_value,
            }
            unexpected = set(values.unique()) - allowed
            if unexpected:
                raise ValueError(
                    f"{spec.split_column} has unexpected values: "
                    f"{sorted(unexpected)}"
                )

            indices = {
                "train": np.flatnonzero(
                    values.eq(spec.train_value).to_numpy()
                ),
                "validation": np.flatnonzero(
                    values.eq(spec.validation_value).to_numpy()
                ),
                "test": np.flatnonzero(
                    values.eq(spec.test_value).to_numpy()
                ),
            }
            combined = np.concatenate(list(indices.values()))
            if len(combined) != len(task) or len(np.unique(combined)) != len(task):
                raise ValueError(
                    f"{spec.split_column} does not form a complete disjoint split"
                )

            for eval_set, row_indices in indices.items():
                counts = np.bincount(
                    y[row_indices],
                    minlength=n_classes,
                )
                if np.any(counts == 0):
                    missing_classes = np.flatnonzero(counts == 0).tolist()
                    raise ValueError(
                        f"{spec.split_column} {eval_set} lacks classes "
                        f"{missing_classes}"
                    )
                row = {
                    "task": "np_classification",
                    "split": mode,
                    "split_column": spec.split_column,
                    "split_seed": seed,
                    "eval_set": eval_set,
                    "n_rows": int(len(row_indices)),
                    "class_count_min": int(counts.min()),
                    "class_count_max": int(counts.max()),
                    "class_counts": "|".join(
                        str(int(value)) for value in counts
                    ),
                }
                if scaffold_column is not None:
                    row["n_scaffolds"] = int(
                        task.iloc[row_indices][scaffold_column]
                        .fillna("")
                        .astype(str)
                        .nunique()
                    )
                qc_rows.append(row)

            if mode == "scaffold" and scaffold_column is not None:
                sets = {
                    name: {
                        value
                        for value in task.iloc[row_indices][scaffold_column]
                        .fillna("")
                        .astype(str)
                        .tolist()
                        if value
                    }
                    for name, row_indices in indices.items()
                }
                for left, right in [
                    ("train", "validation"),
                    ("train", "test"),
                    ("validation", "test"),
                ]:
                    overlap = sets[left] & sets[right]
                    if overlap:
                        raise ValueError(
                            f"{spec.split_column} has {len(overlap)} scaffold "
                            f"values shared by {left} and {right}"
                        )

            split_indices[(mode, seed)] = indices

    return split_indices, pd.DataFrame(qc_rows)


def task_cache_metadata(task: pd.DataFrame) -> pd.DataFrame:
    return task[["inchikey", "canonical_smiles"]].reset_index(drop=True)


def cache_metadata_path(cache_dir: Path, feature: str) -> Path:
    return cache_dir / f"{feature}.metadata.tsv"


def write_cache_metadata(
    task: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    task_cache_metadata(task).to_csv(path, sep="\t", index=False)


def validate_cache_metadata(
    task: pd.DataFrame,
    path: Path,
    feature: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing cache metadata: {path}")
    observed = pd.read_csv(path, sep="\t", low_memory=False)
    expected = task_cache_metadata(task)
    if list(observed.columns) != list(expected.columns) or not observed.equals(
        expected
    ):
        raise ValueError(
            f"{feature} cache is not aligned to the current task. "
            "Use --rebuild-feature-cache."
        )


def molecule_iterator(smiles: list[str]) -> Iterable[Chem.Mol]:
    for row_index, value in enumerate(smiles):
        try:
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(str(value))
        except Exception:
            molecule = None
        if molecule is None:
            raise ValueError(
                f"Invalid canonical SMILES at task row {row_index}: {value}"
            )
        yield molecule


def build_sparse_morgan(
    smiles: list[str],
    radius: int,
    n_bits: int,
) -> sparse.csr_matrix:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )
    bit_indices: list[int] = []
    indptr = np.zeros(len(smiles) + 1, dtype=np.int64)

    for row_index, molecule in enumerate(molecule_iterator(smiles)):
        with rdBase.BlockLogs():
            fingerprint = generator.GetFingerprint(molecule)
        bit_indices.extend(fingerprint.GetOnBits())
        indptr[row_index + 1] = len(bit_indices)

        if (row_index + 1) % 10000 == 0:
            print(
                f"Computed Morgan r{radius}: "
                f"{row_index + 1:,}/{len(smiles):,}"
            )

    indices = np.asarray(bit_indices, dtype=np.int32)
    data = np.ones(len(indices), dtype=np.float32)
    matrix = sparse.csr_matrix(
        (data, indices, indptr),
        shape=(len(smiles), n_bits),
        dtype=np.float32,
    )
    matrix.sort_indices()
    return matrix


def load_or_build_morgan(
    task: pd.DataFrame,
    cache_dir: Path,
    feature: str,
    radius: int,
    rebuild: bool,
) -> FeatureData:
    matrix_path = cache_dir / f"{feature}.features.npz"
    metadata_path = cache_metadata_path(cache_dir, feature)

    if matrix_path.exists() and not rebuild:
        validate_cache_metadata(task, metadata_path, feature)
        matrix = sparse.load_npz(matrix_path).tocsr()
        if matrix.shape != (len(task), 2048):
            raise ValueError(
                f"{feature} cache has unexpected shape {matrix.shape}"
            )
        if matrix.data.size and not np.isfinite(matrix.data).all():
            raise ValueError(f"{feature} cache contains non-finite values")
        print(
            f"Loaded cached {feature}: shape={matrix.shape}, "
            f"nnz={matrix.nnz:,}"
        )
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        matrix = build_sparse_morgan(
            task["canonical_smiles"].astype(str).tolist(),
            radius,
            2048,
        )
        sparse.save_npz(matrix_path, matrix, compressed=True)
        write_cache_metadata(task, metadata_path)
        print(f"Wrote cached {feature}: {matrix_path}")

    return FeatureData(
        feature,
        "sparse_morgan",
        matrix,
        str(matrix_path),
        int(matrix.shape[1]),
        True,
    )


def build_rdkit_descriptors(
    smiles: list[str],
) -> tuple[np.ndarray, list[str]]:
    descriptor_names = [name for name, _ in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
        descriptor_names
    )
    matrix = np.full(
        (len(smiles), len(descriptor_names)),
        np.nan,
        dtype=np.float32,
    )

    for row_index, molecule in enumerate(molecule_iterator(smiles)):
        try:
            with rdBase.BlockLogs():
                values = calculator.CalcDescriptors(molecule)
            row = np.asarray(values, dtype=np.float64)
            row[~np.isfinite(row)] = np.nan
            matrix[row_index] = row.astype(np.float32)
        except Exception:
            pass

        if (row_index + 1) % 10000 == 0:
            print(
                "Computed RDKit descriptors: "
                f"{row_index + 1:,}/{len(smiles):,}"
            )

    all_missing = np.isnan(matrix).all(axis=0)
    if np.any(all_missing):
        dropped = [
            descriptor_names[index]
            for index in np.flatnonzero(all_missing)
        ]
        print("Dropping all-missing descriptors:", ", ".join(dropped))
        matrix = matrix[:, ~all_missing]
        descriptor_names = [
            name
            for index, name in enumerate(descriptor_names)
            if not all_missing[index]
        ]

    if matrix.shape[1] == 0:
        raise RuntimeError("No usable RDKit descriptors")
    return matrix, descriptor_names


def load_or_build_rdkit(
    task: pd.DataFrame,
    cache_dir: Path,
    rebuild: bool,
) -> FeatureData:
    feature = "rdkit_descriptors"
    matrix_path = cache_dir / f"{feature}.features.npy"
    names_path = cache_dir / f"{feature}.names.json"
    metadata_path = cache_metadata_path(cache_dir, feature)

    cache_ready = all(
        path.exists() for path in [matrix_path, names_path, metadata_path]
    )
    if cache_ready and not rebuild:
        validate_cache_metadata(task, metadata_path, feature)
        matrix = np.load(matrix_path, mmap_mode="r")
        names = json.loads(names_path.read_text())
        if matrix.ndim != 2 or matrix.shape[0] != len(task):
            raise ValueError(
                f"RDKit cache has unexpected shape {matrix.shape}"
            )
        if matrix.shape[1] != len(names):
            raise ValueError("RDKit cache descriptor names do not match width")
        print(f"Loaded cached RDKit descriptors: {matrix.shape}")
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        matrix, names = build_rdkit_descriptors(
            task["canonical_smiles"].astype(str).tolist()
        )
        np.save(matrix_path, matrix, allow_pickle=False)
        names_path.write_text(json.dumps(names, indent=2) + "\n")
        write_cache_metadata(task, metadata_path)
        print(f"Wrote cached RDKit descriptors: {matrix_path}")

    return FeatureData(
        feature,
        "rdkit",
        matrix,
        str(matrix_path),
        int(matrix.shape[1]),
        False,
    )


def find_fm_paths(
    feature: str,
    embedding_dir: Path,
) -> tuple[Path, Path]:
    exact = embedding_dir / f"{feature}.embeddings.npy"
    if exact.exists():
        embedding_path = exact
    else:
        matches = sorted(
            embedding_dir.glob(f"{feature}*.embeddings.npy")
        )
        if not matches:
            raise FileNotFoundError(
                f"No embedding file for {feature} in {embedding_dir}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"{feature} matched multiple embedding files: {matches}"
            )
        embedding_path = matches[0]

    stem = embedding_path.name.removesuffix(".embeddings.npy")
    metadata_path = embedding_dir / f"{stem}.metadata.tsv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing embedding metadata: {metadata_path}"
        )
    return embedding_path, metadata_path


def load_aligned_fm(
    task: pd.DataFrame,
    feature: str,
    embedding_dir: Path,
) -> FeatureData:
    embedding_path, metadata_path = find_fm_paths(
        feature,
        embedding_dir,
    )
    all_embeddings = np.load(embedding_path, mmap_mode="r")
    metadata = pd.read_csv(metadata_path, sep="\t", low_memory=False)

    if all_embeddings.ndim != 2:
        raise ValueError(
            f"{feature} embedding matrix is not 2D: {all_embeddings.shape}"
        )
    if len(metadata) != all_embeddings.shape[0]:
        raise ValueError(
            f"{feature} embedding/metadata row mismatch"
        )
    if "inchikey" not in metadata.columns:
        raise ValueError(f"Missing inchikey in {metadata_path}")
    if metadata["inchikey"].isna().any() or not metadata["inchikey"].is_unique:
        raise ValueError(
            f"Embedding metadata InChIKeys must be nonmissing and unique: "
            f"{metadata_path}"
        )

    lookup = pd.Series(
        np.arange(len(metadata), dtype=np.int64),
        index=metadata["inchikey"].astype(str),
    )
    rows = task["inchikey"].astype(str).map(lookup)
    if rows.isna().any():
        examples = (
            task.loc[rows.isna(), "inchikey"].astype(str).head(10).tolist()
        )
        raise ValueError(
            f"{feature} is missing {int(rows.isna().sum())} task compounds. "
            f"Examples: {examples}"
        )

    matrix = np.asarray(
        all_embeddings[rows.to_numpy(dtype=np.int64)],
        dtype=np.float32,
    )
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite values in {embedding_path}")
    print(f"Loaded {feature}: {matrix.shape}")

    return FeatureData(
        feature,
        "dense_embedding",
        matrix,
        str(embedding_path),
        int(matrix.shape[1]),
        False,
    )


def load_feature(
    task: pd.DataFrame,
    feature: str,
    cache_dir: Path,
    embedding_dir: Path,
    rebuild: bool,
) -> FeatureData:
    if feature == "morgan_r2_2048":
        return load_or_build_morgan(
            task,
            cache_dir,
            feature,
            radius=2,
            rebuild=rebuild,
        )
    if feature == "morgan_r3_2048":
        return load_or_build_morgan(
            task,
            cache_dir,
            feature,
            radius=3,
            rebuild=rebuild,
        )
    if feature == "rdkit_descriptors":
        return load_or_build_rdkit(task, cache_dir, rebuild)
    if feature in FM_FEATURES:
        return load_aligned_fm(task, feature, embedding_dir)
    raise ValueError(f"Unsupported feature: {feature}")


def make_estimator(
    feature_kind: str,
    C: float,
    seed: int,
    args: argparse.Namespace,
) -> Any:
    if feature_kind == "sparse_morgan":
        return LogisticRegression(
            C=C,
            solver=args.sparse_solver,
            class_weight=args.class_weight,
            max_iter=args.sparse_max_iter,
            tol=args.tol,
            random_state=seed,
        )

    logistic = LogisticRegression(
        C=C,
        solver=args.dense_solver,
        class_weight=args.class_weight,
        max_iter=args.dense_max_iter,
        tol=args.tol,
        random_state=seed,
    )
    if feature_kind == "rdkit":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", logistic),
            ]
        )
    if feature_kind == "dense_embedding":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", logistic),
            ]
        )
    raise ValueError(f"Unknown feature kind: {feature_kind}")


def fitted_classifier(estimator: Any) -> LogisticRegression:
    if isinstance(estimator, Pipeline):
        classifier = estimator.named_steps["classifier"]
    else:
        classifier = estimator
    if not isinstance(classifier, LogisticRegression):
        raise TypeError("Expected LogisticRegression")
    return classifier


def convergence_info(
    estimator: Any,
    max_iter: int,
) -> tuple[int, bool, str]:
    classifier = fitted_classifier(estimator)
    values = np.asarray(classifier.n_iter_).reshape(-1).astype(int)
    maximum = int(values.max()) if len(values) else 0
    reached = bool(np.any(values >= max_iter))
    return maximum, reached, "|".join(str(int(value)) for value in values)


def aligned_probabilities(
    estimator: Any,
    matrix: Any,
    n_classes: int,
) -> np.ndarray:
    raw = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)
    classes = np.asarray(estimator.classes_, dtype=int)
    probabilities = np.zeros(
        (matrix.shape[0], n_classes),
        dtype=np.float64,
    )
    probabilities[:, classes] = raw
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite class probabilities")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Multiclass probabilities do not sum to one")
    return probabilities


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    n_classes = len(label_names)
    classes = np.arange(n_classes)
    predictions = probabilities.argmax(axis=1)

    overall = {
        "auroc_ovr_macro": float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
        ),
        "auprc_ovr_macro": float(
            np.mean(
                [
                    average_precision_score(
                        (y_true == class_index).astype(np.int8),
                        probabilities[:, class_index],
                    )
                    for class_index in classes
                ]
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
    }

    class_rows: list[dict[str, Any]] = []
    for class_index, label_name in enumerate(label_names):
        truth = (y_true == class_index).astype(np.int8)
        class_prediction = (predictions == class_index).astype(np.int8)
        class_rows.append(
            {
                "class_index": class_index,
                "label": label_name,
                "support": int(truth.sum()),
                "auroc": float(
                    roc_auc_score(
                        truth,
                        probabilities[:, class_index],
                    )
                ),
                "auprc": float(
                    average_precision_score(
                        truth,
                        probabilities[:, class_index],
                    )
                ),
                "f1": float(
                    f1_score(
                        truth,
                        class_prediction,
                        zero_division=0,
                    )
                ),
            }
        )
    return overall, class_rows


def fit_select_C(
    feature: FeatureData,
    y: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    label_names: list[str],
    seed: int,
    args: argparse.Namespace,
) -> tuple[
    Any,
    float,
    list[dict[str, Any]],
    int,
    bool,
    str,
    float,
]:
    max_iter = (
        args.sparse_max_iter
        if feature.kind == "sparse_morgan"
        else args.dense_max_iter
    )
    best = None
    trial_rows: list[dict[str, Any]] = []

    for C in args.C_grid:
        estimator = make_estimator(feature.kind, C, seed, args)
        start = time.perf_counter()
        estimator.fit(
            feature.matrix[train_indices],
            y[train_indices],
        )
        fit_seconds = time.perf_counter() - start

        max_n_iter, reached, n_iter_values = convergence_info(
            estimator,
            max_iter,
        )
        validation_probabilities = aligned_probabilities(
            estimator,
            feature.matrix[validation_indices],
            len(label_names),
        )
        validation_metrics, _ = evaluate_probabilities(
            y[validation_indices],
            validation_probabilities,
            label_names,
        )

        trial_rows.append(
            {
                "C": float(C),
                "validation_auroc_ovr_macro": validation_metrics[
                    "auroc_ovr_macro"
                ],
                "validation_auprc_ovr_macro": validation_metrics[
                    "auprc_ovr_macro"
                ],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "fit_seconds": fit_seconds,
                "max_n_iter": max_n_iter,
                "reached_max_iter": reached,
                "n_iter_values": n_iter_values,
            }
        )

        # Prefer a converged candidate whenever at least one candidate
        # converges. Within the same convergence status, select by validation
        # macro AUROC, then macro AUPRC, then the smaller C.
        key = (
            int(not reached),
            validation_metrics["auroc_ovr_macro"],
            validation_metrics["auprc_ovr_macro"],
            -float(C),
        )
        candidate = (
            key,
            estimator,
            float(C),
            max_n_iter,
            reached,
            n_iter_values,
            fit_seconds,
        )
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        raise RuntimeError("No C candidate was fitted")
    (
        _,
        estimator,
        selected_C,
        max_n_iter,
        reached,
        n_iter_values,
        fit_seconds,
    ) = best
    return (
        estimator,
        selected_C,
        trial_rows,
        max_n_iter,
        reached,
        n_iter_values,
        fit_seconds,
    )


def save_predictions(
    output_dir: Path,
    feature: str,
    split_mode: str,
    seed: int,
    eval_set: str,
    row_indices: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    selected_C: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        f"{feature}__{split_mode}__seed{seed}__{eval_set}.npz"
    )
    np.savez_compressed(
        path,
        task_row=row_indices.astype(np.int64),
        y_true=y_true.astype(np.int16),
        probability=probabilities.astype(np.float32),
        y_pred=probabilities.argmax(axis=1).astype(np.int16),
        selected_C=np.asarray([selected_C], dtype=np.float64),
        split_seed=np.asarray([seed], dtype=np.int64),
    )


def run_condition(
    feature: FeatureData,
    split_mode: str,
    seed: int,
    indices: dict[str, np.ndarray],
    y: np.ndarray,
    label_names: list[str],
    args: argparse.Namespace,
    prediction_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
]:
    (
        estimator,
        selected_C,
        trial_results,
        max_n_iter,
        reached,
        n_iter_values,
        selected_fit_seconds,
    ) = fit_select_C(
        feature,
        y,
        indices["train"],
        indices["validation"],
        label_names,
        seed,
        args,
    )

    max_iter = (
        args.sparse_max_iter
        if feature.kind == "sparse_morgan"
        else args.dense_max_iter
    )
    base = {
        "task": "np_classification",
        "feature": feature.name,
        "feature_kind": feature.kind,
        "feature_source": feature.source_path,
        "n_features": feature.n_features,
        "sparse_matrix": feature.sparse_matrix,
        "model": "multiclass_logistic_regression",
        "split": split_mode,
        "split_column": split_spec(split_mode, seed).split_column,
        "split_seed": seed,
        "n_train": int(len(indices["train"])),
        "n_classes": len(label_names),
        "selected_C": selected_C,
        "C_grid": "|".join(str(value) for value in args.C_grid),
        "max_iter": max_iter,
        "tol": args.tol,
        "max_n_iter": max_n_iter,
        "n_iter_values": n_iter_values,
        "reached_max_iter": reached,
        "selected_fit_seconds": selected_fit_seconds,
    }

    run_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    log_lines: list[str] = []

    for eval_set in ["validation", "test"]:
        eval_indices = indices[eval_set]
        score_start = time.perf_counter()
        probabilities = aligned_probabilities(
            estimator,
            feature.matrix[eval_indices],
            len(label_names),
        )
        score_seconds = time.perf_counter() - score_start
        overall, per_class = evaluate_probabilities(
            y[eval_indices],
            probabilities,
            label_names,
        )
        eval_base = {
            **base,
            "eval_set": eval_set,
            "n_eval": int(len(eval_indices)),
            "score_seconds": score_seconds,
        }
        run_rows.append({**eval_base, **overall})
        class_rows.extend(
            {**eval_base, **row} for row in per_class
        )

        if args.save_predictions:
            save_predictions(
                prediction_dir,
                feature.name,
                split_mode,
                seed,
                eval_set,
                eval_indices,
                y[eval_indices],
                probabilities,
                selected_C,
            )

        log_lines.append(
            f"{feature.name} | {split_mode} | seed={seed} | {eval_set} | "
            f"AUROC={overall['auroc_ovr_macro']:.6f} | "
            f"AUPRC={overall['auprc_ovr_macro']:.6f} | "
            f"macro-F1={overall['macro_f1']:.6f}"
        )

    trial_rows = [
        {
            "task": "np_classification",
            "feature": feature.name,
            "split": split_mode,
            "split_seed": seed,
            **row,
        }
        for row in trial_results
    ]
    return run_rows, class_rows, trial_rows, "\n".join(log_lines)


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    test = runs[runs["eval_set"].eq("test")].copy()
    group_columns = ["task", "feature", "model", "split"]
    summary = (
        test.groupby(group_columns)[OVERALL_METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary = flatten_columns(summary)
    for metric in OVERALL_METRICS:
        summary[f"{metric}_se"] = (
            summary[f"{metric}_std"]
            / np.sqrt(summary[f"{metric}_count"])
        )
    summary["n_split_replicates"] = summary[
        "auroc_ovr_macro_count"
    ].astype(int)
    return summary


def paired_gains(
    runs: pd.DataFrame,
    reference: str = "morgan_r2_2048",
) -> pd.DataFrame:
    test = runs[runs["eval_set"].eq("test")].copy()
    keys = ["split", "split_seed"]
    reference_rows = test[test["feature"].eq(reference)][
        keys + OVERALL_METRICS
    ].copy()
    reference_rows = reference_rows.rename(
        columns={
            metric: f"reference_{metric}" for metric in OVERALL_METRICS
        }
    )
    merged = test.merge(
        reference_rows,
        on=keys,
        how="left",
        validate="many_to_one",
    )
    reference_columns = [
        f"reference_{metric}" for metric in OVERALL_METRICS
    ]
    if merged[reference_columns].isna().any().any():
        raise ValueError(f"Missing paired {reference} results")

    difference_columns: list[str] = []
    for metric in OVERALL_METRICS:
        column = f"{metric}_difference_vs_{reference}"
        merged[column] = merged[metric] - merged[f"reference_{metric}"]
        difference_columns.append(column)

    group_columns = ["task", "feature", "model", "split"]
    summary = (
        merged.groupby(group_columns)[difference_columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary = flatten_columns(summary)
    for column in difference_columns:
        summary[f"{column}_se"] = (
            summary[f"{column}_std"]
            / np.sqrt(summary[f"{column}_count"])
        )
    return summary


def result_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "runs": output_dir / "npclassification_repeated_runs.tsv",
        "per_class": output_dir / "npclassification_per_class.tsv",
        "summary": output_dir / "npclassification_main_statistics.tsv",
        "paired": output_dir / "npclassification_paired_gains.tsv",
        "split_qc": output_dir / "npclassification_split_qc.tsv",
        "trials": output_dir / "npclassification_C_trials.tsv",
        "config": output_dir / "npclassification_run_config.json",
        "predictions": output_dir / "predictions",
    }


def prepare_results(
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths["runs"].parent.mkdir(parents=True, exist_ok=True)
    managed = [
        paths["runs"],
        paths["per_class"],
        paths["summary"],
        paths["paired"],
        paths["split_qc"],
        paths["trials"],
        paths["config"],
    ]

    if args.overwrite_results:
        for path in managed:
            if path.exists():
                path.unlink()
        if paths["predictions"].exists():
            for path in paths["predictions"].glob("*.npz"):
                path.unlink()

    if not args.resume and not args.overwrite_results:
        existing = [str(path) for path in managed if path.exists()]
        if existing:
            raise FileExistsError(
                "Results already exist. Use --resume or "
                "--overwrite-results:\n" + "\n".join(existing)
            )

    def read_if_resuming(key: str) -> pd.DataFrame:
        path = paths[key]
        if args.resume and path.exists():
            return pd.read_csv(path, sep="\t")
        return pd.DataFrame()

    return (
        read_if_resuming("runs"),
        read_if_resuming("per_class"),
        read_if_resuming("trials"),
    )


def completed_keys(
    runs: pd.DataFrame,
) -> set[tuple[str, str, int]]:
    if runs.empty:
        return set()
    required = {"feature", "split", "split_seed", "eval_set"}
    if not required.issubset(runs.columns):
        raise ValueError("Existing result schema is incompatible")
    complete: set[tuple[str, str, int]] = set()
    for key, group in runs.groupby(["feature", "split", "split_seed"]):
        if set(group["eval_set"]) == {"validation", "test"}:
            complete.add(
                (str(key[0]), str(key[1]), int(key[2]))
            )
    return complete


def append_rows(
    existing: pd.DataFrame,
    rows: list[dict[str, Any]],
    path: Path,
) -> pd.DataFrame:
    if rows:
        addition = pd.DataFrame(rows)
        frame = (
            pd.concat([existing, addition], ignore_index=True)
            if not existing.empty
            else addition
        )
    else:
        frame = existing.copy()
    if not frame.empty:
        frame.to_csv(path, sep="\t", index=False)
    return frame


def warn_thread_settings(args: argparse.Namespace) -> None:
    if args.parallel_runs <= 1:
        return
    variables = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ]
    problematic = {
        name: os.environ.get(name)
        for name in variables
        if os.environ.get(name) not in {None, "1"}
    }
    if problematic:
        print(
            "WARNING: concurrent model fits are enabled, but these thread "
            "settings are not 1: "
            + ", ".join(
                f"{name}={value}"
                for name, value in problematic.items()
            )
            + ". This can make the run slower through oversubscription."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    warn_thread_settings(args)

    task_path = Path(args.task)
    embedding_dir = Path(args.embedding_dir)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    paths = result_paths(output_dir)

    task, y, label_names = load_task(task_path)
    print(f"Task rows: {len(task):,}")
    print(f"Classes: {len(label_names)}")
    print("Features:", ", ".join(args.features))

    splits, split_qc = build_splits(
        task,
        y,
        args.split_modes,
        args.split_seeds,
        len(label_names),
    )
    runs, per_class, trials = prepare_results(paths, args)
    complete = completed_keys(runs)

    config = {
        "script_version": SCRIPT_VERSION,
        "task": str(task_path),
        "embedding_dir": str(embedding_dir),
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "features": args.features,
        "excluded_features": ["nafm_contrast"],
        "split_seeds": args.split_seeds,
        "split_modes": args.split_modes,
        "C_grid": args.C_grid,
        "C_selection": (
            "validation_macro_ovr_auroc_then_auprc_then_smaller_C"
            if len(args.C_grid) > 1
            else "fixed"
        ),
        "classifier": {
            "type": "single-label multiclass LogisticRegression",
            "class_weight": args.class_weight,
            "dense_solver": args.dense_solver,
            "sparse_solver": args.sparse_solver,
            "dense_max_iter": args.dense_max_iter,
            "sparse_max_iter": args.sparse_max_iter,
            "tol": args.tol,
            "parallel_split_seed_runs": args.parallel_runs,
            "morgan_representation": "scipy CSR; no scaling",
            "dense_scaler": "training-fitted StandardScaler",
            "rdkit_imputer": "training-fitted median",
        },
        "metrics": {
            "primary": "auroc_ovr_macro",
            "secondary": "auprc_ovr_macro",
            "supplementary": [
                "macro_f1",
                "weighted_f1",
                "accuracy",
                "balanced_accuracy",
            ],
            "score_source": "multiclass predict_proba",
        },
        "n_compounds": len(task),
        "n_classes": len(label_names),
        "label_names": label_names,
        "save_predictions": args.save_predictions,
    }
    paths["config"].write_text(json.dumps(config, indent=2) + "\n")
    split_qc.to_csv(paths["split_qc"], sep="\t", index=False)

    for feature_name in args.features:
        try:
            feature = load_feature(
                task,
                feature_name,
                cache_dir,
                embedding_dir,
                args.rebuild_feature_cache,
            )
        except FileNotFoundError:
            if args.skip_missing_features:
                print(f"Skipping missing feature: {feature_name}")
                continue
            raise

        print(
            f"\nFeature: {feature.name}; kind={feature.kind}; "
            f"shape={feature.matrix.shape}; source={feature.source_path}"
        )

        conditions = [
            (mode, seed)
            for mode in args.split_modes
            for seed in args.split_seeds
            if (feature.name, mode, seed) not in complete
        ]
        if not conditions:
            print(f"All requested runs already complete for {feature.name}")
            continue

        new_runs: list[dict[str, Any]] = []
        new_classes: list[dict[str, Any]] = []
        new_trials: list[dict[str, Any]] = []

        with ThreadPoolExecutor(
            max_workers=min(args.parallel_runs, len(conditions))
        ) as executor:
            futures = {
                executor.submit(
                    run_condition,
                    feature,
                    mode,
                    seed,
                    splits[(mode, seed)],
                    y,
                    label_names,
                    args,
                    paths["predictions"],
                ): (mode, seed)
                for mode, seed in conditions
            }

            for future in as_completed(futures):
                mode, seed = futures[future]
                try:
                    run_rows, class_rows, trial_rows, log_text = future.result()
                except Exception as error:
                    raise RuntimeError(
                        f"Failed {feature.name} | {mode} | seed={seed}"
                    ) from error
                print(log_text)
                new_runs.extend(run_rows)
                new_classes.extend(class_rows)
                new_trials.extend(trial_rows)

        runs = append_rows(runs, new_runs, paths["runs"])
        per_class = append_rows(
            per_class,
            new_classes,
            paths["per_class"],
        )
        trials = append_rows(
            trials,
            new_trials,
            paths["trials"],
        )

        del feature

    if runs.empty:
        raise RuntimeError("No benchmark results are available")

    runs = runs.sort_values(
        ["feature", "split", "split_seed", "eval_set"]
    ).reset_index(drop=True)
    runs.to_csv(paths["runs"], sep="\t", index=False)

    per_class = per_class.sort_values(
        [
            "feature",
            "split",
            "split_seed",
            "eval_set",
            "class_index",
        ]
    ).reset_index(drop=True)
    per_class.to_csv(paths["per_class"], sep="\t", index=False)

    trials = trials.sort_values(
        ["feature", "split", "split_seed", "C"]
    ).reset_index(drop=True)
    trials.to_csv(paths["trials"], sep="\t", index=False)

    summary = summarize_runs(runs)
    summary.to_csv(paths["summary"], sep="\t", index=False)

    if "morgan_r2_2048" in set(runs["feature"]):
        paired = paired_gains(runs)
        paired.to_csv(paths["paired"], sep="\t", index=False)

    expected_seeds = len(args.split_seeds)
    counts = (
        runs[runs["eval_set"].eq("test")]
        .groupby(["feature", "split"])["split_seed"]
        .nunique()
    )
    incomplete = counts[counts != expected_seeds]
    if not incomplete.empty:
        print(
            "WARNING: incomplete feature/split replicate counts:\n"
            + incomplete.to_string()
        )

    print("\nWrote:", paths["runs"])
    print("Wrote:", paths["per_class"])
    print("Wrote:", paths["summary"])
    if paths["paired"].exists():
        print("Wrote:", paths["paired"])
    print("Wrote:", paths["split_qc"])
    print("Wrote:", paths["trials"])
    print("Wrote:", paths["config"])


if __name__ == "__main__":
    main()

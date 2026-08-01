#!/usr/bin/env python3
"""Repeated-split frozen-representation benchmark for PlantCyc reaction neighbors.

Primary evaluation
------------------
- Binary, undirected pair classification.
- AUROC is the primary metric.
- AUPRC is the second threshold-independent metric.
- Logistic-regression C is selected using validation AUROC by default.
- Test data are never used for model or threshold selection.

Supplementary threshold-dependent evaluation
--------------------------------------------
F1, MCC, balanced accuracy, precision, recall, specificity, and accuracy are
reported both at threshold 0.5 and at a global threshold selected using
validation F1. These metrics are supplementary and do not define the primary
model ranking.

Statistical replicates
----------------------
A replicate is a distinct seed-specific pair dataset/split file, not a repeated
fit of logistic regression on the same data. The default seeds are 2026-2030.
Every representation is evaluated on exactly the same task replicate, allowing
paired comparisons by:

    split_scheme + negative_variant + split_seed

Pair representations
--------------------
Dense compound vectors x_a and x_b:

    [x_a + x_b, abs(x_a - x_b)]

Binary Morgan fingerprints:

    [AND(x_a, x_b), XOR(x_a, x_b)]

For binary inputs, AND/XOR spans the same linear information as sum/absolute
change while remaining sparse.
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_VERSION = "plantcyc_reaction_neighbor_full_repeated_v1"
TASK_NAME = "plantcyc_reaction_neighbor"

DEFAULT_FEATURES = [
    "morgan_r2_2048",
    "morgan_r3_2048",
    #"rdkit_descriptors",
    "chemberta_77m_mlm",
    "chemberta_100m_mlm",
    "molformer_c3_1_1b",
    "nafm_graph_mean",
    "molclr_gcn",
    "molclr_gin",
    "graphmvp_gin",
    "molebert_gin",
]

DEFAULT_SPLIT_SEEDS = [2026, 2027, 2028, 2029, 2030]
PRIMARY_METRICS = ["auroc", "auprc"]

SPLIT_NAMES = {
    "random": ("train", "val", "test"),
    "scaffold_pair": (
        "scaffold_train",
        "scaffold_val",
        "scaffold_test",
    ),
}


def default_threshold_grid() -> list[float]:
    return [0.2, 0.4, 0.6, 0.8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeated-split frozen-feature benchmark for PlantCyc "
            "reaction-neighbor prediction."
        )
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("data/features/plantcyc_reaction_neighbor"),
    )
    parser.add_argument(
        "--pair-index-dir",
        type=Path,
        default=Path(
            "data/features/plantcyc_reaction_neighbor/pair_indices"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/plantcyc_reaction_neighbor/full"),
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
    )
    parser.add_argument(
        "--skip-missing-features",
        action="store_true",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["easy", "hard", "mixed"],
        default=["easy", "hard", "mixed"],
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=["random", "scaffold_pair"],
        default=["random", "scaffold_pair"],
    )
    parser.add_argument(
        "--split-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SPLIT_SEEDS,
    )
    parser.add_argument(
        "--C-grid",
        nargs="+",
        type=float,
        default=[0.01, 0.1, 1.0, 10.0],
    )
    parser.add_argument(
        "--selection-metric",
        choices=["auroc", "auprc"],
        default="auroc",
        help=(
            "Validation metric used to select C. AUROC is the default and "
            "primary benchmark metric."
        ),
    )
    parser.add_argument(
        "--threshold-grid",
        nargs="+",
        type=float,
        default=default_threshold_grid(),
    )
    parser.add_argument(
        "--solver",
        choices=["liblinear", "lbfgs", "sag", "saga"],
        default="lbfgs",
    )
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help=(
            "The generated task files are balanced 1:1 within each split, "
            "so no class weighting is used by default."
        ),
    )
    parser.add_argument("--model-seed", type=int, default=2026)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from existing replicate-level result files.",
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Do not save pair-level validation and test probabilities.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.features = list(dict.fromkeys(args.features))
    args.variants = list(dict.fromkeys(args.variants))
    args.schemes = list(dict.fromkeys(args.schemes))
    args.split_seeds = list(dict.fromkeys(args.split_seeds))

    if not args.features:
        raise ValueError("At least one feature is required")
    if not args.variants:
        raise ValueError("At least one negative variant is required")
    if not args.schemes:
        raise ValueError("At least one split scheme is required")
    if len(args.split_seeds) < 2:
        raise ValueError(
            "At least two distinct split seeds are required for a repeated "
            "benchmark; five are recommended."
        )
    if any(value <= 0 for value in args.C_grid):
        raise ValueError("All C values must be positive")
    if not args.threshold_grid:
        raise ValueError("Threshold grid cannot be empty")
    if not all(0.0 < value < 1.0 for value in args.threshold_grid):
        raise ValueError("Thresholds must be strictly between 0 and 1")
    if args.resume and args.overwrite_results:
        raise ValueError(
            "--resume and --overwrite-results cannot be used together"
        )


def class_weight_value(value: str):
    return None if value == "none" else value


def load_metadata(feature_dir: Path) -> pd.DataFrame:
    path = feature_dir / "metadata.tsv"
    if not path.exists():
        raise FileNotFoundError(path)

    metadata = pd.read_csv(path, sep="\t", low_memory=False)
    required = {
        "compound_index",
        "task_compound_id",
        "canonical_smiles",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")

    metadata = metadata.copy()
    metadata["compound_index"] = metadata["compound_index"].astype(int)
    metadata["task_compound_id"] = (
        metadata["task_compound_id"].astype(str)
    )
    metadata["canonical_smiles"] = (
        metadata["canonical_smiles"].astype(str)
    )

    if not np.array_equal(
        metadata["compound_index"].to_numpy(),
        np.arange(len(metadata), dtype=int),
    ):
        raise ValueError(
            "metadata compound_index must be contiguous, zero-based, and "
            "aligned to feature-row order"
        )
    if metadata["task_compound_id"].duplicated().any():
        raise ValueError("Duplicated task_compound_id in metadata")
    if metadata["canonical_smiles"].duplicated().any():
        raise ValueError("Duplicated canonical_smiles in metadata")

    print("Compounds:", f"{len(metadata):,}")
    return metadata


def endpoint_index_columns(
    frame: pd.DataFrame,
) -> tuple[str, str]:
    candidates = [
        ("compound_a_index", "compound_b_index"),
        ("substrate_compound_index", "product_compound_index"),
    ]
    for left, right in candidates:
        if left in frame.columns and right in frame.columns:
            return left, right
    raise ValueError(
        "Could not identify endpoint-index columns. Expected "
        "compound_a_index/compound_b_index or "
        "substrate_compound_index/product_compound_index."
    )


def pair_file_name(variant: str, scheme: str, seed: int) -> str:
    return (
        f"plantcyc_reaction_neighbor_{variant}_{scheme}_seed{seed}.tsv"
    )


def resolve_pair_file(
    pair_index_dir: Path,
    variant: str,
    scheme: str,
    seed: int,
) -> Path:
    candidates = [
        pair_index_dir / pair_file_name(variant, scheme, seed),
        pair_index_dir
        / (
            f"plantcyc_reaction_neighbor_{variant}_{scheme}_"
            f"split_seed{seed}.tsv"
        ),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Missing seed-specific pair-index file. Checked:\n"
        + "\n".join(str(path) for path in candidates)
    )


def normalized_pair_keys(
    frame: pd.DataFrame,
    left_col: str,
    right_col: str,
) -> pd.Series:
    left = frame[left_col].to_numpy(dtype=np.int64)
    right = frame[right_col].to_numpy(dtype=np.int64)
    lower = np.minimum(left, right).astype(str)
    upper = np.maximum(left, right).astype(str)
    return pd.Series(
        np.char.add(np.char.add(lower, "|"), upper),
        index=frame.index,
    )


def find_scaffold_pair_column(frame: pd.DataFrame) -> str | None:
    preferred = [
        "undirected_scaffold_pair_key",
        "scaffold_pair_key",
        "canonical_scaffold_pair_key",
    ]
    for column in preferred:
        if column in frame.columns:
            return column
    candidates = [
        column
        for column in frame.columns
        if "scaffold" in column.lower()
        and "pair" in column.lower()
        and "key" in column.lower()
    ]
    return candidates[0] if len(candidates) == 1 else None


def endpoint_set(
    frame: pd.DataFrame,
    left_col: str,
    right_col: str,
) -> set[int]:
    return set(frame[left_col].astype(int)) | set(
        frame[right_col].astype(int)
    )


def load_pair_table(
    path: Path,
    variant: str,
    scheme: str,
    seed: int,
    n_compounds: int,
) -> tuple[pd.DataFrame, str, str, list[dict]]:
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    required = {"pair_id", "split", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")

    frame = frame.copy()
    frame["pair_id"] = frame["pair_id"].astype(str)
    if frame["pair_id"].duplicated().any():
        raise ValueError(f"{path}: duplicated pair_id")

    if "split_seed" in frame.columns:
        observed_seeds = set(frame["split_seed"].astype(int).unique())
        if observed_seeds != {int(seed)}:
            raise ValueError(
                f"{path}: split_seed values {observed_seeds} do not match "
                f"filename/requested seed {seed}"
            )

    left_col, right_col = endpoint_index_columns(frame)
    for column in [left_col, right_col]:
        if frame[column].isna().any():
            raise ValueError(f"{path}: missing endpoint indices in {column}")
        frame[column] = frame[column].astype(int)
        if (
            frame[column].min() < 0
            or frame[column].max() >= n_compounds
        ):
            raise ValueError(
                f"{path}: {column} outside [0, {n_compounds - 1}]"
            )

    if np.any(frame[left_col].to_numpy() == frame[right_col].to_numpy()):
        raise ValueError(f"{path}: self-pairs remain")

    frame["label"] = frame["label"].astype(int)
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError(f"{path}: labels must be exactly {{0, 1}}")

    expected_splits = set(SPLIT_NAMES[scheme])
    observed_splits = set(frame["split"].astype(str).unique())
    if observed_splits != expected_splits:
        raise ValueError(
            f"{path}: expected splits {sorted(expected_splits)}, found "
            f"{sorted(observed_splits)}"
        )

    frame["_normalized_pair_key"] = normalized_pair_keys(
        frame,
        left_col,
        right_col,
    )
    if frame["_normalized_pair_key"].duplicated().any():
        raise ValueError(
            f"{path}: repeated undirected compound pairs remain"
        )

    split_frames = {
        split_name: frame[frame["split"].eq(split_name)].copy()
        for split_name in SPLIT_NAMES[scheme]
    }
    for split_name, subset in split_frames.items():
        if subset.empty:
            raise ValueError(f"{path}: empty split {split_name}")
        if set(subset["label"].unique()) != {0, 1}:
            raise ValueError(
                f"{path}: split {split_name} does not contain both classes"
            )

    pair_sets = {
        split_name: set(subset["_normalized_pair_key"])
        for split_name, subset in split_frames.items()
    }
    split_names = list(SPLIT_NAMES[scheme])
    for i, left_name in enumerate(split_names):
        for right_name in split_names[i + 1 :]:
            overlap = pair_sets[left_name] & pair_sets[right_name]
            if overlap:
                raise ValueError(
                    f"{path}: {len(overlap)} undirected pairs cross "
                    f"{left_name}/{right_name}"
                )

    scaffold_pair_column = find_scaffold_pair_column(frame)
    if scheme == "scaffold_pair" and scaffold_pair_column is not None:
        scaffold_sets = {
            split_name: set(
                subset[scaffold_pair_column].fillna("").astype(str)
            )
            for split_name, subset in split_frames.items()
        }
        for i, left_name in enumerate(split_names):
            for right_name in split_names[i + 1 :]:
                overlap = (
                    scaffold_sets[left_name]
                    & scaffold_sets[right_name]
                ) - {""}
                if overlap:
                    raise ValueError(
                        f"{path}: {len(overlap)} scaffold-pair groups cross "
                        f"{left_name}/{right_name}"
                    )

    qc_rows: list[dict] = []
    for split_name, subset in split_frames.items():
        qc_rows.append(
            {
                "task": TASK_NAME,
                "negative_variant": variant,
                "split_scheme": scheme,
                "split_seed": int(seed),
                "split": split_name,
                "n_pairs": int(len(subset)),
                "n_negative": int((subset["label"] == 0).sum()),
                "n_positive": int((subset["label"] == 1).sum()),
                "positive_fraction": float(subset["label"].mean()),
                "n_endpoint_compounds": int(
                    len(endpoint_set(subset, left_col, right_col))
                ),
                "source_file": str(path),
            }
        )

    # These are descriptive overlaps, not leakage criteria for pair-level and
    # scaffold-pair-group splits.
    for i, left_name in enumerate(split_names):
        for right_name in split_names[i + 1 :]:
            left_endpoints = endpoint_set(
                split_frames[left_name], left_col, right_col
            )
            right_endpoints = endpoint_set(
                split_frames[right_name], left_col, right_col
            )
            overlap_count = len(left_endpoints & right_endpoints)
            union_count = len(left_endpoints | right_endpoints)
            qc_rows.append(
                {
                    "task": TASK_NAME,
                    "negative_variant": variant,
                    "split_scheme": scheme,
                    "split_seed": int(seed),
                    "split": f"{left_name}__vs__{right_name}",
                    "n_pairs": np.nan,
                    "n_negative": np.nan,
                    "n_positive": np.nan,
                    "positive_fraction": np.nan,
                    "n_endpoint_compounds": np.nan,
                    "endpoint_compound_overlap": overlap_count,
                    "endpoint_compound_jaccard": (
                        overlap_count / union_count if union_count else 0.0
                    ),
                    "source_file": str(path),
                }
            )

    return frame, left_col, right_col, qc_rows


def feature_available(
    feature_name: str,
    feature_dir: Path,
) -> tuple[bool, str]:
    if feature_name in {"morgan_r2_2048", "morgan_r3_2048"}:
        path = feature_dir / f"{feature_name}.npz"
        return path.exists(), str(path)
    if feature_name == "rdkit_descriptors":
        path = feature_dir / "rdkit_descriptors.tsv"
        return path.exists(), str(path)

    path = feature_dir / "full" / f"{feature_name}.embeddings.npy"
    return path.exists(), str(path)


def load_rdkit_descriptors(
    feature_dir: Path,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    path = feature_dir / "rdkit_descriptors.tsv"
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    if len(frame) != len(metadata):
        raise ValueError(
            "RDKit descriptor row count does not match metadata"
        )

    if "compound_index" in frame.columns:
        if not np.array_equal(
            frame["compound_index"].astype(int).to_numpy(),
            metadata["compound_index"].to_numpy(),
        ):
            raise ValueError(
                "RDKit descriptor compound_index is not aligned to metadata"
            )

    excluded = {
        "compound_index",
        "feature_row_index",
        "task_compound_id",
        "canonical_smiles",
    }
    descriptor_columns = [
        column
        for column in frame.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not descriptor_columns:
        raise ValueError("No numeric RDKit descriptors found")

    matrix = frame[descriptor_columns].to_numpy(dtype=np.float64)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix, descriptor_columns


def load_feature(
    feature_name: str,
    feature_dir: Path,
    metadata: pd.DataFrame,
):
    n_compounds = len(metadata)

    if feature_name in {"morgan_r2_2048", "morgan_r3_2048"}:
        path = feature_dir / f"{feature_name}.npz"
        matrix = sparse.load_npz(path).tocsr()
        if matrix.shape[0] != n_compounds:
            raise ValueError(
                f"{feature_name}: row count does not match metadata"
            )
        return matrix, "sparse_binary", {
            "source_path": str(path),
            "compound_dim": int(matrix.shape[1]),
        }

    if feature_name == "rdkit_descriptors":
        matrix, columns = load_rdkit_descriptors(
            feature_dir,
            metadata,
        )
        return matrix, "dense_descriptor", {
            "source_path": str(
                feature_dir / "rdkit_descriptors.tsv"
            ),
            "compound_dim": int(matrix.shape[1]),
            "descriptor_columns": columns,
        }

    path = feature_dir / "full" / f"{feature_name}.embeddings.npy"
    matrix = np.load(path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"{feature_name}: embedding array is not 2D")
    if matrix.shape[0] != n_compounds:
        raise ValueError(
            f"{feature_name}: embedding rows do not match metadata"
        )

    embedding_metadata = (
        feature_dir / "full" / f"{feature_name}.metadata.tsv"
    )
    if embedding_metadata.exists():
        emb_meta = pd.read_csv(
            embedding_metadata,
            sep="\t",
            low_memory=False,
        )
        if len(emb_meta) != len(metadata):
            raise ValueError(
                f"{feature_name}: embedding metadata row mismatch"
            )
        if "task_compound_id" in emb_meta.columns:
            if (
                emb_meta["task_compound_id"].astype(str).tolist()
                != metadata["task_compound_id"].astype(str).tolist()
            ):
                raise ValueError(
                    f"{feature_name}: embedding ID order mismatch"
                )

    return matrix, "dense_embedding", {
        "source_path": str(path),
        "compound_dim": int(matrix.shape[1]),
    }


def fit_descriptor_preprocessor(
    compound_matrix: np.ndarray,
    train_pairs: pd.DataFrame,
    left_col: str,
    right_col: str,
) -> tuple[np.ndarray, SimpleImputer, StandardScaler]:
    train_compound_indices = np.unique(
        np.concatenate(
            [
                train_pairs[left_col].to_numpy(dtype=int),
                train_pairs[right_col].to_numpy(dtype=int),
            ]
        )
    )

    raw = np.asarray(compound_matrix, dtype=np.float64)
    raw[~np.isfinite(raw)] = np.nan

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )
    imputer.fit(raw[train_compound_indices])
    imputed = imputer.transform(raw).astype(np.float64, copy=False)
    if not np.isfinite(imputed).all():
        raise ValueError(
            "RDKit descriptors remain non-finite after imputation"
        )

    scaler = StandardScaler()
    scaler.fit(imputed[train_compound_indices])
    transformed = scaler.transform(imputed).astype(
        np.float64,
        copy=False,
    )
    if not np.isfinite(transformed).all():
        raise ValueError(
            "RDKit descriptors remain non-finite after standardization"
        )

    return transformed, imputer, scaler


def build_dense_pair_features(
    compound_matrix,
    pairs: pd.DataFrame,
    left_col: str,
    right_col: str,
) -> np.ndarray:
    left = np.asarray(
        compound_matrix[pairs[left_col].to_numpy(dtype=int)],
        dtype=np.float64,
    )
    right = np.asarray(
        compound_matrix[pairs[right_col].to_numpy(dtype=int)],
        dtype=np.float64,
    )
    matrix = np.concatenate(
        [left + right, np.abs(left - right)],
        axis=1,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Dense pair features contain non-finite values")
    return matrix


def build_sparse_binary_pair_features(
    compound_matrix: sparse.csr_matrix,
    pairs: pd.DataFrame,
    left_col: str,
    right_col: str,
) -> sparse.csr_matrix:
    left = compound_matrix[
        pairs[left_col].to_numpy(dtype=int)
    ].astype(np.int8)
    right = compound_matrix[
        pairs[right_col].to_numpy(dtype=int)
    ].astype(np.int8)

    intersection = left.multiply(right).astype(np.float32).tocsr()
    union = left.maximum(right)
    xor = (union - intersection).astype(np.float32).tocsr()
    matrix = sparse.hstack(
        [intersection, xor],
        format="csr",
        dtype=np.float32,
    )
    matrix.eliminate_zeros()
    return matrix


def make_model(
    feature_kind: str,
    C: float,
    args: argparse.Namespace,
):
    classifier = LogisticRegression(
        C=float(C),
        penalty="l2",
        solver=args.solver,
        class_weight=class_weight_value(args.class_weight),
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=args.model_seed,
    )
    if feature_kind == "sparse_binary":
        return classifier
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def fitted_classifier(model) -> LogisticRegression:
    if isinstance(model, Pipeline):
        return model.named_steps["classifier"]
    return model


def threshold_free_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    if len(np.unique(y_true)) != 2:
        return {"auroc": np.nan, "auprc": np.nan}
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
    }


def threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted,
        labels=[0, 1],
    ).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predicted)
        ),
        "precision": float(
            precision_score(y_true, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predicted, zero_division=0)
        ),
        "specificity": float(specificity),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "predicted_positive_fraction": float(np.mean(predicted)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def tune_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold_grid: Iterable[float],
) -> tuple[float, dict[str, float | int]]:
    best_threshold = None
    best_metrics = None
    best_key = None

    for threshold in threshold_grid:
        metrics = threshold_metrics(
            y_true,
            scores,
            float(threshold),
        )
        key = (
            metrics["f1"],
            metrics["mcc"],
            -abs(float(threshold) - 0.5),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_threshold is None or best_metrics is None:
        raise RuntimeError("Threshold tuning failed")
    return best_threshold, best_metrics


def selection_key(
    metrics: dict[str, float],
    C: float,
    selection_metric: str,
) -> tuple[float, float, float]:
    secondary = "auprc" if selection_metric == "auroc" else "auroc"
    return (
        metrics[selection_metric],
        metrics[secondary],
        -float(C),
    )


def tune_C(
    X_train,
    y_train: np.ndarray,
    X_validation,
    y_validation: np.ndarray,
    feature_kind: str,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    best = None
    search_rows: list[dict] = []

    for C in args.C_grid:
        model = make_model(feature_kind, C, args)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y_train)

        classifier = fitted_classifier(model)
        n_iter = int(np.max(classifier.n_iter_))
        convergence_messages = [
            str(item.message)
            for item in caught
            if issubclass(item.category, ConvergenceWarning)
        ]
        converged = not convergence_messages
        validation_scores = np.asarray(
            model.predict_proba(X_validation)[:, 1],
            dtype=np.float64,
        )
        metrics = threshold_free_metrics(
            y_validation,
            validation_scores,
        )

        search_rows.append(
            {
                "C": float(C),
                "validation_auroc": metrics["auroc"],
                "validation_auprc": metrics["auprc"],
                "solver_n_iter": n_iter,
                "solver_converged": converged,
                "convergence_messages": " | ".join(
                    convergence_messages
                ),
                "selected": False,
            }
        )
        row_index = len(search_rows) - 1

        if converged:
            key = selection_key(metrics, C, args.selection_metric)
            if best is None or key > best["key"]:
                if best is not None:
                    del best["model"]
                best = {
                    "key": key,
                    "model": model,
                    "selected_C": float(C),
                    "validation_scores": validation_scores,
                    "validation_auroc": metrics["auroc"],
                    "validation_auprc": metrics["auprc"],
                    "solver_n_iter": n_iter,
                    "solver_converged": converged,
                    "convergence_messages": convergence_messages,
                    "search_row_index": row_index,
                }
            else:
                del model
        else:
            del model

    if best is None:
        raise RuntimeError("No C candidate converged")

    search_rows[best["search_row_index"]]["selected"] = True
    return best, search_rows


def complete_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    tuned_threshold: float,
) -> dict[str, float | int]:
    result: dict[str, float | int] = threshold_free_metrics(
        y_true,
        scores,
    )
    fixed = threshold_metrics(y_true, scores, 0.5)
    tuned = threshold_metrics(y_true, scores, tuned_threshold)

    for name, value in fixed.items():
        result[f"{name}_at_0_5"] = value
    for name, value in tuned.items():
        result[f"{name}_tuned"] = value

    result["positive_prevalence"] = float(np.mean(y_true))
    result["mean_predicted_probability"] = float(np.mean(scores))
    return result


def subset_by_split(
    frame: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    subset = frame[frame["split"].eq(split_name)].copy()
    if subset.empty:
        raise ValueError(f"No rows found for split: {split_name}")
    return subset.reset_index(drop=True)


def save_predictions(
    path: Path,
    pair_ids: Iterable[str],
    y_true: np.ndarray,
    scores: np.ndarray,
    tuned_threshold: float,
    split_seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pair_id=np.asarray(list(pair_ids), dtype=str),
        y_true=y_true.astype(np.int8),
        y_score=scores.astype(np.float32),
        y_pred_05=(scores >= 0.5).astype(np.int8),
        y_pred_tuned=(scores >= tuned_threshold).astype(np.int8),
        tuned_threshold=np.asarray(
            [tuned_threshold],
            dtype=np.float64,
        ),
        split_seed=np.asarray([split_seed], dtype=np.int64),
    )


def append_rows(path: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(
        path,
        sep="\t",
        index=False,
        mode="a",
        header=not path.exists(),
    )


def completed_keys(runs: pd.DataFrame) -> set[tuple]:
    if runs.empty:
        return set()
    keys = [
        "feature",
        "negative_variant",
        "split_scheme",
        "split_seed",
    ]
    completed = set()
    for values, subset in runs.groupby(keys, dropna=False):
        if set(subset["eval_set"]) >= {"validation", "test"}:
            completed.add(tuple(values))
    return completed


def build_config(args: argparse.Namespace) -> dict:
    return {
        "script_version": SCRIPT_VERSION,
        "task": TASK_NAME,
        "feature_dir": str(args.feature_dir),
        "pair_index_dir": str(args.pair_index_dir),
        "features": args.features,
        "negative_variants": args.variants,
        "split_schemes": args.schemes,
        "split_seeds": args.split_seeds,
        "C_grid": args.C_grid,
        "C_selection_metric": args.selection_metric,
        "threshold_grid": args.threshold_grid,
        "primary_metrics": PRIMARY_METRICS,
        "primary_model_ranking_metric": "auroc",
        "threshold_dependent_metrics": "supplementary",
        "dense_pair_representation": "sum_absolute_difference",
        "morgan_pair_representation": "and_xor",
        "solver": args.solver,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "class_weight": args.class_weight,
        "model_seed": args.model_seed,
    }


def verify_resume_config(path: Path, current: dict) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"--resume requested but config is missing: {path}"
        )
    previous = json.loads(path.read_text())
    fields = [
        "feature_dir",
        "pair_index_dir",
        "features",
        "negative_variants",
        "split_schemes",
        "split_seeds",
        "C_grid",
        "C_selection_metric",
        "threshold_grid",
        "dense_pair_representation",
        "morgan_pair_representation",
        "solver",
        "max_iter",
        "tol",
        "class_weight",
    ]
    mismatches = {
        field: {
            "previous": previous.get(field),
            "current": current.get(field),
        }
        for field in fields
        if previous.get(field) != current.get(field)
    }
    if mismatches:
        raise ValueError(
            "Resume configuration mismatch:\n"
            + json.dumps(mismatches, indent=2)
        )


def add_paired_morgan_gains(
    runs: pd.DataFrame,
    reference_feature: str = "morgan_r2_2048",
) -> pd.DataFrame:
    result = runs.copy()
    keys = [
        "negative_variant",
        "split_scheme",
        "split_seed",
        "eval_set",
    ]
    metrics = ["auroc", "auprc"]

    reference = result[result["feature"].eq(reference_feature)][
        keys + metrics
    ].copy()
    if reference.empty:
        for metric in metrics:
            result[f"gain_vs_{reference_feature}_{metric}"] = np.nan
        return result

    reference = reference.rename(
        columns={metric: f"reference_{metric}" for metric in metrics}
    )
    result = result.merge(
        reference,
        on=keys,
        how="left",
        validate="many_to_one",
    )
    for metric in metrics:
        result[f"gain_vs_{reference_feature}_{metric}"] = (
            result[metric] - result[f"reference_{metric}"]
        )
        result = result.drop(columns=[f"reference_{metric}"])
    return result


def aggregate_statistics(runs_with_gains: pd.DataFrame) -> pd.DataFrame:
    test = runs_with_gains[runs_with_gains["eval_set"].eq("test")]
    group_columns = [
        "task",
        "feature",
        "feature_kind",
        "pair_representation",
        "negative_variant",
        "split_scheme",
    ]
    metric_columns = [
        "auroc",
        "auprc",
        "f1_at_0_5",
        "f1_tuned",
        "mcc_at_0_5",
        "mcc_tuned",
        "balanced_accuracy_at_0_5",
        "balanced_accuracy_tuned",
    ]
    metric_columns.extend(
        column
        for column in test.columns
        if column.startswith("gain_vs_morgan_r2_2048_")
    )

    rows: list[dict] = []
    for keys, subset in test.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row["n_split_replicates"] = int(
            subset["split_seed"].nunique()
        )
        row["n_convergence_failures"] = int(
            (~subset["solver_converged"].astype(bool)).sum()
        )

        for metric in metric_columns:
            values = subset[metric].astype(float).to_numpy()
            valid = values[np.isfinite(values)]
            row[f"{metric}_mean"] = (
                float(np.mean(valid)) if len(valid) else np.nan
            )
            row[f"{metric}_sd"] = (
                float(np.std(valid, ddof=1))
                if len(valid) > 1
                else np.nan
            )
            row[f"{metric}_se"] = (
                row[f"{metric}_sd"] / np.sqrt(len(valid))
                if len(valid) > 1
                else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_path = output_dir / "reaction_neighbor_repeated_runs.tsv"
    hyperparameter_path = output_dir / "hyperparameter_search.tsv"
    statistics_path = output_dir / "reaction_neighbor_main_statistics.tsv"
    gains_path = output_dir / "reaction_neighbor_paired_gains.tsv"
    split_qc_path = output_dir / "reaction_neighbor_split_qc.tsv"
    config_path = output_dir / "experiment_config.json"
    prediction_dir = output_dir / "predictions"

    managed = [
        runs_path,
        hyperparameter_path,
        statistics_path,
        gains_path,
        split_qc_path,
        config_path,
    ]
    if args.overwrite_results:
        for path in managed:
            if path.exists():
                path.unlink()
        if prediction_dir.exists():
            import shutil

            shutil.rmtree(prediction_dir)

    config = build_config(args)
    if args.resume:
        verify_resume_config(config_path, config)
    else:
        existing = [path for path in managed if path.exists()]
        if existing:
            raise FileExistsError(
                "Result files already exist. Use --resume or "
                "--overwrite-results:\n"
                + "\n".join(str(path) for path in existing)
            )
        config_path.write_text(json.dumps(config, indent=2))

    metadata = load_metadata(args.feature_dir)

    experiments: list[dict] = []
    split_qc_rows: list[dict] = []
    seen_source_files: set[str] = set()

    for scheme in args.schemes:
        for variant in args.variants:
            for split_seed in args.split_seeds:
                path = resolve_pair_file(
                    args.pair_index_dir,
                    variant,
                    scheme,
                    split_seed,
                )
                if str(path.resolve()) in seen_source_files:
                    raise ValueError(
                        f"The same pair file would be counted more than once: "
                        f"{path}"
                    )
                seen_source_files.add(str(path.resolve()))

                frame, left_col, right_col, qc = load_pair_table(
                    path=path,
                    variant=variant,
                    scheme=scheme,
                    seed=split_seed,
                    n_compounds=len(metadata),
                )
                split_qc_rows.extend(qc)
                experiments.append(
                    {
                        "negative_variant": variant,
                        "split_scheme": scheme,
                        "split_seed": split_seed,
                        "pair_file": path,
                        "pairs": frame,
                        "left_col": left_col,
                        "right_col": right_col,
                    }
                )

    pd.DataFrame(split_qc_rows).to_csv(
        split_qc_path,
        sep="\t",
        index=False,
    )

    existing_runs = (
        pd.read_csv(runs_path, sep="\t")
        if runs_path.exists()
        else pd.DataFrame()
    )
    done = completed_keys(existing_runs)

    selected_features: list[str] = []
    for feature_name in args.features:
        available, expected_path = feature_available(
            feature_name,
            args.feature_dir,
        )
        if not available:
            if args.skip_missing_features:
                print(
                    f"Skipping missing feature {feature_name}: "
                    f"{expected_path}"
                )
                continue
            raise FileNotFoundError(expected_path)
        selected_features.append(feature_name)

    if not selected_features:
        raise ValueError("No available features selected")

    for feature_name in selected_features:
        print(f"\n=== Loading feature: {feature_name} ===")
        compound_matrix, feature_kind, feature_info = load_feature(
            feature_name,
            args.feature_dir,
            metadata,
        )
        pair_representation = (
            "and_xor"
            if feature_kind == "sparse_binary"
            else "sum_absolute_difference"
        )

        for experiment in experiments:
            key = (
                feature_name,
                experiment["negative_variant"],
                experiment["split_scheme"],
                experiment["split_seed"],
            )
            if key in done:
                print("Skipping completed:", key)
                continue

            variant = experiment["negative_variant"]
            scheme = experiment["split_scheme"]
            split_seed = experiment["split_seed"]
            pairs = experiment["pairs"]
            left_col = experiment["left_col"]
            right_col = experiment["right_col"]
            train_name, validation_name, test_name = SPLIT_NAMES[scheme]

            train_pairs = subset_by_split(pairs, train_name)
            validation_pairs = subset_by_split(pairs, validation_name)
            test_pairs = subset_by_split(pairs, test_name)
            y_train = train_pairs["label"].to_numpy(dtype=int)
            y_validation = validation_pairs["label"].to_numpy(dtype=int)
            y_test = test_pairs["label"].to_numpy(dtype=int)

            print(
                f"\n=== feature={feature_name} variant={variant} "
                f"scheme={scheme} seed={split_seed} ==="
            )
            print(
                f"train={len(train_pairs):,} "
                f"validation={len(validation_pairs):,} "
                f"test={len(test_pairs):,}"
            )

            imputer = None
            descriptor_scaler = None
            working_compound_matrix = compound_matrix
            if feature_kind == "dense_descriptor":
                (
                    working_compound_matrix,
                    imputer,
                    descriptor_scaler,
                ) = fit_descriptor_preprocessor(
                    compound_matrix,
                    train_pairs,
                    left_col,
                    right_col,
                )

            if feature_kind == "sparse_binary":
                X_train = build_sparse_binary_pair_features(
                    working_compound_matrix,
                    train_pairs,
                    left_col,
                    right_col,
                )
                X_validation = build_sparse_binary_pair_features(
                    working_compound_matrix,
                    validation_pairs,
                    left_col,
                    right_col,
                )
                X_test = build_sparse_binary_pair_features(
                    working_compound_matrix,
                    test_pairs,
                    left_col,
                    right_col,
                )
            else:
                X_train = build_dense_pair_features(
                    working_compound_matrix,
                    train_pairs,
                    left_col,
                    right_col,
                )
                X_validation = build_dense_pair_features(
                    working_compound_matrix,
                    validation_pairs,
                    left_col,
                    right_col,
                )
                X_test = build_dense_pair_features(
                    working_compound_matrix,
                    test_pairs,
                    left_col,
                    right_col,
                )

            best, search_rows = tune_C(
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
                feature_kind=feature_kind,
                args=args,
            )

            for row in search_rows:
                row.update(
                    {
                        "task": TASK_NAME,
                        "feature": feature_name,
                        "feature_kind": feature_kind,
                        "pair_representation": pair_representation,
                        "negative_variant": variant,
                        "split_scheme": scheme,
                        "split_seed": split_seed,
                        "selection_metric": args.selection_metric,
                        "source_pair_file": str(
                            experiment["pair_file"]
                        ),
                    }
                )
            append_rows(hyperparameter_path, search_rows)

            tuned_threshold, validation_threshold_metrics = tune_threshold(
                y_validation,
                best["validation_scores"],
                args.threshold_grid,
            )
            test_scores = np.asarray(
                best["model"].predict_proba(X_test)[:, 1],
                dtype=np.float64,
            )

            common = {
                "task": TASK_NAME,
                "feature": feature_name,
                "feature_kind": feature_kind,
                "pair_representation": pair_representation,
                "negative_variant": variant,
                "split_scheme": scheme,
                "split_seed": split_seed,
                "model": "binary_logistic_regression",
                "n_train": int(len(train_pairs)),
                "n_validation": int(len(validation_pairs)),
                "n_test": int(len(test_pairs)),
                "pair_feature_dim": int(X_train.shape[1]),
                "selected_C": best["selected_C"],
                "C_grid": "|".join(str(value) for value in args.C_grid),
                "C_selection_metric": args.selection_metric,
                "selected_validation_auroc": best[
                    "validation_auroc"
                ],
                "selected_validation_auprc": best[
                    "validation_auprc"
                ],
                "threshold_grid": "|".join(
                    str(value) for value in args.threshold_grid
                ),
                "tuned_threshold": tuned_threshold,
                "validation_best_f1_tuned": (
                    validation_threshold_metrics["f1"]
                ),
                "solver": args.solver,
                "solver_tolerance": args.tol,
                "solver_n_iter": best["solver_n_iter"],
                "solver_converged": best["solver_converged"],
                "class_weight": args.class_weight,
                "source_pair_file": str(experiment["pair_file"]),
                "source_feature_file": feature_info["source_path"],
            }

            result_rows: list[dict] = []
            for eval_set, eval_pairs, y_true, scores in [
                (
                    "validation",
                    validation_pairs,
                    y_validation,
                    best["validation_scores"],
                ),
                ("test", test_pairs, y_test, test_scores),
            ]:
                row = {
                    **common,
                    "eval_set": eval_set,
                    "n_eval": int(len(y_true)),
                }
                row.update(
                    complete_metrics(
                        y_true,
                        scores,
                        tuned_threshold,
                    )
                )
                result_rows.append(row)

                if not args.skip_predictions:
                    prediction_path = (
                        prediction_dir
                        / scheme
                        / variant
                        / f"seed{split_seed}"
                        / f"{feature_name}.{eval_set}.npz"
                    )
                    save_predictions(
                        prediction_path,
                        eval_pairs["pair_id"],
                        y_true,
                        scores,
                        tuned_threshold,
                        split_seed,
                    )

            append_rows(runs_path, result_rows)
            done.add(key)

            test_row = result_rows[1]
            print(
                f"selected C={best['selected_C']:g}; "
                f"threshold={tuned_threshold:.2f}; "
                f"test AUROC={test_row['auroc']:.4f}; "
                f"AUPRC={test_row['auprc']:.4f}; "
                f"F1@0.5={test_row['f1_at_0_5']:.4f}; "
                f"F1 tuned={test_row['f1_tuned']:.4f}"
            )

            if imputer is not None and not args.skip_predictions:
                parameter_dir = (
                    prediction_dir
                    / scheme
                    / variant
                    / f"seed{split_seed}"
                    / f"{feature_name}.preprocessing"
                )
                parameter_dir.mkdir(parents=True, exist_ok=True)
                np.save(
                    parameter_dir / "descriptor_imputer_statistics.npy",
                    imputer.statistics_,
                )
                np.save(
                    parameter_dir / "descriptor_scaler_mean.npy",
                    descriptor_scaler.mean_,
                )
                np.save(
                    parameter_dir / "descriptor_scaler_scale.npy",
                    descriptor_scaler.scale_,
                )

            del X_train, X_validation, X_test
            del best
            if feature_kind == "dense_descriptor":
                del working_compound_matrix
            gc.collect()

        del compound_matrix
        gc.collect()

    runs = pd.read_csv(runs_path, sep="\t")
    expected_keys = {
        (feature, variant, scheme, seed)
        for feature in selected_features
        for variant in args.variants
        for scheme in args.schemes
        for seed in args.split_seeds
    }
    missing_keys = expected_keys - completed_keys(runs)
    if missing_keys:
        raise RuntimeError(
            f"Run ended with {len(missing_keys)} incomplete replicate keys"
        )

    gains = add_paired_morgan_gains(runs)
    gains.to_csv(gains_path, sep="\t", index=False)
    statistics = aggregate_statistics(gains)
    statistics.to_csv(statistics_path, sep="\t", index=False)

    print("\nWrote:")
    print(runs_path)
    print(hyperparameter_path)
    print(statistics_path)
    print(gains_path)
    print(split_qc_path)
    print(config_path)
    if not args.skip_predictions:
        print(prediction_dir)

    display_columns = [
        "split_scheme",
        "negative_variant",
        "feature",
        "n_split_replicates",
        "auroc_mean",
        "auroc_sd",
        "auprc_mean",
        "auprc_sd",
        "gain_vs_morgan_r2_2048_auroc_mean",
        "gain_vs_morgan_r2_2048_auprc_mean",
    ]
    display_columns = [
        column
        for column in display_columns
        if column in statistics.columns
    ]
    print("\n=== Main test statistics ===")
    print(
        statistics.sort_values(
            [
                "split_scheme",
                "negative_variant",
                "auroc_mean",
            ],
            ascending=[True, True, False],
        )[display_columns].to_string(index=False)
    )


if __name__ == "__main__":
    main()

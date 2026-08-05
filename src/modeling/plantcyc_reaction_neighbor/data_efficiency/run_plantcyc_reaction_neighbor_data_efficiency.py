#!/usr/bin/env python3
"""Standalone sample-efficiency benchmark for PlantCyc reaction-neighbor prediction.

Design
------
- Reuses the seed-specific pair tables from the full-data benchmark.
- Keeps validation and test pairs fixed for every training-set size.
- Creates class-balanced, nested training-pair subsets independently for each
  sampling seed. Every smaller subset is contained in every larger subset.
- Uses exactly the same sampled pair IDs for every molecular representation.
- Tunes logistic-regression C on validation AUROC, with validation AUPRC as the
  tie-breaker, unless --selection-metric is changed explicitly.
- Tunes one global decision threshold on validation F1 for supplementary
  threshold-dependent metrics. Test data are never used for sampling or model
  selection.
- Averages sampling replicates within each split seed before calculating the
  across-split summary and 95% confidence intervals.

This version is self-contained. It does not import the PlantCyc full-data
runner or any other project-local Python module.

Expected location
-----------------
    src/modeling/plantcyc_reaction_neighbor/data_efficiency/
        run_plantcyc_reaction_neighbor_data_efficiency_standalone_v2.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t as student_t

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


DEFAULT_FEATURES = [
    "morgan_r2_2048",
    "morgan_r3_2048",
    "rdkit_descriptors",
    "chemberta_77m_mlm",
    "chemberta_100m_mlm",
    "molformer_c3_1_1b",
    "nafm_graph_mean",
    "molclr_gcn",
    "molclr_gin",
    "graphmvp_gin",
    "molebert_gin",
]

SPLIT_NAMES = {
    "random": ("train", "val", "test"),
    "scaffold_pair": (
        "scaffold_train",
        "scaffold_val",
        "scaffold_test",
    ),
}


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
    metadata["task_compound_id"] = metadata["task_compound_id"].astype(str)
    metadata["canonical_smiles"] = metadata["canonical_smiles"].astype(str)

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


def endpoint_index_columns(frame: pd.DataFrame) -> tuple[str, str]:
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
    return f"plantcyc_reaction_neighbor_{variant}_{scheme}_seed{seed}.tsv"


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
    return set(frame[left_col].astype(int)) | set(frame[right_col].astype(int))


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
    frame["split"] = frame["split"].astype(str)
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
        if frame[column].min() < 0 or frame[column].max() >= n_compounds:
            raise ValueError(
                f"{path}: {column} outside [0, {n_compounds - 1}]"
            )

    if np.any(frame[left_col].to_numpy() == frame[right_col].to_numpy()):
        raise ValueError(f"{path}: self-pairs remain")

    frame["label"] = frame["label"].astype(int)
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError(f"{path}: labels must be exactly {{0, 1}}")

    expected_splits = set(SPLIT_NAMES[scheme])
    observed_splits = set(frame["split"].unique())
    if observed_splits != expected_splits:
        raise ValueError(
            f"{path}: expected splits {sorted(expected_splits)}, found "
            f"{sorted(observed_splits)}"
        )

    frame["_normalized_pair_key"] = normalized_pair_keys(
        frame, left_col, right_col
    )
    if frame["_normalized_pair_key"].duplicated().any():
        raise ValueError(f"{path}: repeated undirected compound pairs remain")

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
                    scaffold_sets[left_name] & scaffold_sets[right_name]
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
                "task": "plantcyc_reaction_neighbor",
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
                    "task": "plantcyc_reaction_neighbor",
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


def feature_available(feature_name: str, feature_dir: Path) -> tuple[bool, str]:
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
        raise ValueError("RDKit descriptor row count does not match metadata")

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
            raise ValueError(f"{feature_name}: row count does not match metadata")
        return matrix, "sparse_binary", {
            "source_path": str(path),
            "compound_dim": int(matrix.shape[1]),
        }

    if feature_name == "rdkit_descriptors":
        matrix, columns = load_rdkit_descriptors(feature_dir, metadata)
        return matrix, "dense_descriptor", {
            "source_path": str(feature_dir / "rdkit_descriptors.tsv"),
            "compound_dim": int(matrix.shape[1]),
            "descriptor_columns": columns,
        }

    path = feature_dir / "full" / f"{feature_name}.embeddings.npy"
    matrix = np.load(path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"{feature_name}: embedding array is not 2D")
    if matrix.shape[0] != n_compounds:
        raise ValueError(f"{feature_name}: embedding rows do not match metadata")

    embedding_metadata = feature_dir / "full" / f"{feature_name}.metadata.tsv"
    if embedding_metadata.exists():
        emb_meta = pd.read_csv(embedding_metadata, sep="\t", low_memory=False)
        if len(emb_meta) != len(metadata):
            raise ValueError(f"{feature_name}: embedding metadata row mismatch")
        if "task_compound_id" in emb_meta.columns:
            if (
                emb_meta["task_compound_id"].astype(str).tolist()
                != metadata["task_compound_id"].astype(str).tolist()
            ):
                raise ValueError(f"{feature_name}: embedding ID order mismatch")

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

    raw = np.asarray(compound_matrix, dtype=np.float64).copy()
    raw[~np.isfinite(raw)] = np.nan

    try:
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        imputer = SimpleImputer(strategy="median")
    imputer.fit(raw[train_compound_indices])
    imputed = imputer.transform(raw).astype(np.float64, copy=False)
    if not np.isfinite(imputed).all():
        raise ValueError("RDKit descriptors remain non-finite after imputation")

    scaler = StandardScaler()
    scaler.fit(imputed[train_compound_indices])
    transformed = scaler.transform(imputed).astype(np.float64, copy=False)
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
    matrix = np.concatenate([left + right, np.abs(left - right)], axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError("Dense pair features contain non-finite values")
    return matrix


def build_sparse_binary_pair_features(
    compound_matrix: sparse.csr_matrix,
    pairs: pd.DataFrame,
    left_col: str,
    right_col: str,
) -> sparse.csr_matrix:
    left = compound_matrix[pairs[left_col].to_numpy(dtype=int)].astype(np.int8)
    right = compound_matrix[pairs[right_col].to_numpy(dtype=int)].astype(np.int8)

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


def make_model(feature_kind: str, C: float, args: argparse.Namespace):
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
        y_true, predicted, labels=[0, 1]
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
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
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
        metrics = threshold_metrics(y_true, scores, float(threshold))
        prevalence_gap = abs(
            metrics["predicted_positive_fraction"] - float(np.mean(y_true))
        )
        key = (metrics["f1"], metrics["mcc"], -prevalence_gap)
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
        metrics = threshold_free_metrics(y_validation, validation_scores)

        search_rows.append(
            {
                "C": float(C),
                "validation_auroc": metrics["auroc"],
                "validation_auprc": metrics["auprc"],
                "solver_n_iter": n_iter,
                "solver_converged": converged,
                "convergence_messages": " | ".join(convergence_messages),
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
    result: dict[str, float | int] = threshold_free_metrics(y_true, scores)
    tuned = threshold_metrics(y_true, scores, tuned_threshold)
    for name, value in tuned.items():
        result[f"{name}_tuned"] = value
    result["positive_prevalence"] = float(np.mean(y_true))
    result["mean_predicted_probability"] = float(np.mean(scores))
    return result


def subset_by_split(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    subset = frame[frame["split"].eq(split_name)].copy()
    if subset.empty:
        raise ValueError(f"No rows found for split: {split_name}")
    return subset.reset_index(drop=True)


SCRIPT_VERSION = "plantcyc_reaction_neighbor_data_efficiency_standalone_v2"
TASK_NAME = "plantcyc_reaction_neighbor"

DEFAULT_SPLIT_SEEDS = [2026, 2027, 2028, 2029, 2030]
DEFAULT_SAMPLING_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_TRAIN_SIZES = [10, 30, 100, 300, 1000, 3000]

# Intentionally smaller than the full-data feature set. These cover the two
# fingerprint baselines, the descriptor baseline, and the FM families most
# relevant to the full-data PlantCyc results. Pass --features followed by
# DEFAULT_FEATURES from the full runner to evaluate every retained model.
DEFAULT_EFFICIENCY_FEATURES = [
    "morgan_r2_2048",
    "morgan_r3_2048",
    "rdkit_descriptors",
    "chemberta_77m_mlm",
    "molformer_c3_1_1b",
    "nafm_graph_mean",
    "molclr_gin",
    "graphmvp_gin",
]

RUN_KEY_COLUMNS = [
    "feature",
    "negative_variant",
    "split_scheme",
    "split_seed",
    "sampling_seed",
    "requested_n_train",
]

SUMMARY_METRICS = [
    "auroc",
    "auprc",
    "f1_tuned",
    "mcc_tuned",
    "balanced_accuracy_tuned",
    "precision_tuned",
    "recall_tuned",
    "specificity_tuned",
    "accuracy_tuned",
]


def default_threshold_grid() -> list[float]:
    return [0.2, 0.4, 0.6, 0.8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=Path(
            "results/plantcyc_reaction_neighbor/data_efficiency"
        ),
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_EFFICIENCY_FEATURES,
        help=(
            "Representations to evaluate. The default is a reduced "
            "sample-efficiency set. The full runner's retained feature names "
            "are: " + ", ".join(DEFAULT_FEATURES)
        ),
    )
    parser.add_argument("--skip-missing-features", action="store_true")
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
        "--sampling-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SAMPLING_SEEDS,
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_TRAIN_SIZES,
        help=(
            "Total labeled training pairs. Samples are approximately 1:1 "
            "positive:negative and nested within each sampling seed."
        ),
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
            "Training subsets are balanced by construction, so no class "
            "weighting is used by default."
        ),
    )
    parser.add_argument(
        "--model-seed-base",
        type=int,
        default=2026,
        help="Base value used to derive a deterministic seed for every fit.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help=(
            "Save one compressed validation/test prediction file per completed "
            "run. Disabled by default because the complete experiment contains "
            "many runs."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs already represented by a test row in the runs TSV.",
    )
    mode.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Delete the output directory before starting.",
    )
    return parser.parse_args()


def deduplicate_preserving_order(values: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def validate_args(args: argparse.Namespace) -> None:
    args.features = deduplicate_preserving_order(args.features)
    args.variants = deduplicate_preserving_order(args.variants)
    args.schemes = deduplicate_preserving_order(args.schemes)
    args.split_seeds = deduplicate_preserving_order(args.split_seeds)
    args.sampling_seeds = deduplicate_preserving_order(args.sampling_seeds)
    args.train_sizes = sorted(set(int(value) for value in args.train_sizes))
    args.C_grid = deduplicate_preserving_order(
        float(value) for value in args.C_grid
    )
    args.threshold_grid = deduplicate_preserving_order(
        float(value) for value in args.threshold_grid
    )

    if not args.features:
        raise ValueError("--features cannot be empty")
    if not args.variants:
        raise ValueError("--variants cannot be empty")
    if not args.schemes:
        raise ValueError("--schemes cannot be empty")
    if not args.split_seeds:
        raise ValueError("--split-seeds cannot be empty")
    if not args.sampling_seeds:
        raise ValueError("--sampling-seeds cannot be empty")
    if not args.train_sizes:
        raise ValueError("--train-sizes cannot be empty")
    if min(args.train_sizes) < 2:
        raise ValueError("Every training subset must contain at least two pairs")
    if any(value <= 0 or not np.isfinite(value) for value in args.C_grid):
        raise ValueError("All C values must be positive and finite")
    if not args.threshold_grid or not all(
        0.0 < value < 1.0 for value in args.threshold_grid
    ):
        raise ValueError("Thresholds must lie strictly between 0 and 1")
    if args.max_iter <= 0:
        raise ValueError("--max-iter must be positive")
    if args.tol <= 0 or not np.isfinite(args.tol):
        raise ValueError("--tol must be positive and finite")


def stable_code(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def deterministic_sampling_seed(
    negative_variant: str,
    split_scheme: str,
    split_seed: int,
    sampling_seed: int,
) -> int:
    sequence = np.random.SeedSequence(
        [
            stable_code(negative_variant),
            stable_code(split_scheme),
            int(split_seed),
            int(sampling_seed),
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def deterministic_model_seed(
    base: int,
    feature: str,
    negative_variant: str,
    split_scheme: str,
    split_seed: int,
    sampling_seed: int,
    requested_n_train: int,
) -> int:
    sequence = np.random.SeedSequence(
        [
            int(base),
            stable_code(feature),
            stable_code(negative_variant),
            stable_code(split_scheme),
            int(split_seed),
            int(sampling_seed),
            int(requested_n_train),
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def sample_hash(pair_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in pair_ids)).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def class_quotas(requested_n_train: int) -> tuple[int, int]:
    """Return positive and negative quotas for an approximately balanced set."""
    n_positive = requested_n_train // 2
    n_negative = requested_n_train - n_positive
    return n_positive, n_negative


def build_nested_pair_samples(
    train_pairs: pd.DataFrame,
    negative_variant: str,
    split_scheme: str,
    split_seed: int,
    sampling_seeds: Iterable[int],
    train_sizes: Iterable[int],
) -> tuple[
    dict[tuple[int, int], np.ndarray],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    labels = train_pairs["label"].to_numpy(dtype=int)
    observed = set(np.unique(labels).tolist())
    if observed != {0, 1}:
        raise ValueError(
            f"Training pool must contain labels {{0, 1}}; observed {observed}"
        )

    positive_positions = np.flatnonzero(labels == 1)
    negative_positions = np.flatnonzero(labels == 0)
    maximum_size = max(int(value) for value in train_sizes)
    maximum_positive, maximum_negative = class_quotas(maximum_size)
    if maximum_positive > len(positive_positions):
        raise ValueError(
            f"Requested N={maximum_size} requires {maximum_positive} positive "
            f"pairs, but the training pool contains {len(positive_positions)}"
        )
    if maximum_negative > len(negative_positions):
        raise ValueError(
            f"Requested N={maximum_size} requires {maximum_negative} negative "
            f"pairs, but the training pool contains {len(negative_positions)}"
        )

    plans: dict[tuple[int, int], np.ndarray] = {}
    manifest_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []

    for sampling_seed in sampling_seeds:
        rng = np.random.default_rng(
            deterministic_sampling_seed(
                negative_variant,
                split_scheme,
                split_seed,
                int(sampling_seed),
            )
        )
        positive_order = rng.permutation(positive_positions)
        negative_order = rng.permutation(negative_positions)
        previous: set[int] = set()

        for requested_n_train in sorted(int(value) for value in train_sizes):
            n_positive, n_negative = class_quotas(requested_n_train)
            selected = np.concatenate(
                [
                    positive_order[:n_positive],
                    negative_order[:n_negative],
                ]
            ).astype(np.int64)
            selected = np.sort(selected)
            selected_set = set(selected.tolist())
            if previous and not previous.issubset(selected_set):
                raise RuntimeError(
                    "Nested-sampling invariant failed for "
                    f"variant={negative_variant}, scheme={split_scheme}, "
                    f"split_seed={split_seed}, sampling_seed={sampling_seed}, "
                    f"N={requested_n_train}"
                )
            previous = selected_set

            selected_pairs = train_pairs.iloc[selected]
            selected_labels = selected_pairs["label"].to_numpy(dtype=int)
            observed_positive = int(np.sum(selected_labels == 1))
            observed_negative = int(np.sum(selected_labels == 0))
            if observed_positive != n_positive or observed_negative != n_negative:
                raise RuntimeError("Sample class counts do not match quotas")
            if len(selected) != requested_n_train:
                raise RuntimeError("Sample size does not match requested size")

            digest = sample_hash(selected_pairs["pair_id"].astype(str))
            plans[(int(sampling_seed), requested_n_train)] = selected
            qc_rows.append(
                {
                    "task": TASK_NAME,
                    "negative_variant": negative_variant,
                    "split_scheme": split_scheme,
                    "split_seed": int(split_seed),
                    "sampling_seed": int(sampling_seed),
                    "requested_n_train": int(requested_n_train),
                    "actual_n_train": int(len(selected)),
                    "n_positive": observed_positive,
                    "n_negative": observed_negative,
                    "positive_fraction": float(observed_positive / len(selected)),
                    "sample_hash": digest,
                }
            )

            for sample_order, (_, row) in enumerate(
                selected_pairs.iterrows(), start=1
            ):
                manifest_rows.append(
                    {
                        "task": TASK_NAME,
                        "negative_variant": negative_variant,
                        "split_scheme": split_scheme,
                        "split_seed": int(split_seed),
                        "sampling_seed": int(sampling_seed),
                        "requested_n_train": int(requested_n_train),
                        "sample_order": int(sample_order),
                        "pair_id": str(row["pair_id"]),
                        "label": int(row["label"]),
                        "train_pool_position": int(selected[sample_order - 1]),
                        "sample_hash": digest,
                    }
                )

    return plans, manifest_rows, qc_rows


def selected_features(args: argparse.Namespace) -> list[str]:
    selected: list[str] = []
    for feature_name in args.features:
        available, path = feature_available(feature_name, args.feature_dir)
        if available:
            selected.append(feature_name)
        elif args.skip_missing_features:
            print(f"Skipping missing feature {feature_name}: {path}")
        else:
            raise FileNotFoundError(
                f"Feature {feature_name!r} is unavailable: {path}"
            )
    if not selected:
        raise ValueError("No available features were selected")
    return selected


def normalized_core_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "feature_dir": str(args.feature_dir),
        "pair_index_dir": str(args.pair_index_dir),
        "features": list(args.features),
        "variants": list(args.variants),
        "schemes": list(args.schemes),
        "split_seeds": list(args.split_seeds),
        "sampling_seeds": list(args.sampling_seeds),
        "train_sizes": list(args.train_sizes),
        "C_grid": [float(value) for value in args.C_grid],
        "selection_metric": str(args.selection_metric),
        "threshold_grid": [float(value) for value in args.threshold_grid],
        "solver": str(args.solver),
        "max_iter": int(args.max_iter),
        "tol": float(args.tol),
        "class_weight": str(args.class_weight),
        "model_seed_base": int(args.model_seed_base),
        "save_predictions": bool(args.save_predictions),
    }


def prepare_output_directory(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    if args.overwrite_results and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "runs": output_dir
        / "plantcyc_reaction_neighbor_data_efficiency_repeated_runs.tsv",
        "c_search": output_dir
        / "plantcyc_reaction_neighbor_data_efficiency_C_search.tsv",
        "sample_qc": output_dir
        / "plantcyc_reaction_neighbor_data_efficiency_sample_qc.tsv",
        "sample_manifest": output_dir
        / "plantcyc_reaction_neighbor_data_efficiency_samples.tsv.gz",
        "pair_split_qc": output_dir
        / "plantcyc_reaction_neighbor_data_efficiency_pair_split_qc.tsv",
        "per_split_means": output_dir
        / "plantcyc_reaction_neighbor_data_efficiency_per_split_means.tsv",
        "summary": output_dir
        / "plantcyc_reaction_neighbor_data_efficiency_summary.tsv",
        "config": output_dir
        / "plantcyc_reaction_neighbor_data_efficiency_config.json",
        "predictions": output_dir / "predictions",
    }

    core_config = normalized_core_config(args)
    if args.resume:
        if not paths["config"].exists():
            raise FileNotFoundError(
                f"Cannot resume without configuration file: {paths['config']}"
            )
        previous = json.loads(paths["config"].read_text())
        if previous.get("core_configuration") != core_config:
            raise ValueError(
                "Resume configuration differs from the existing run. "
                "Use the original arguments or --overwrite-results."
            )
    else:
        existing = [
            path
            for name, path in paths.items()
            if name not in {"predictions", "config"} and path.exists()
        ]
        if existing:
            raise FileExistsError(
                "Output files already exist. Use --resume or "
                "--overwrite-results:\n" + "\n".join(map(str, existing))
            )
        paths["config"].write_text(
            json.dumps(
                {
                    "script_version": SCRIPT_VERSION,
                    "implementation": "standalone_v2",
                    "core_configuration": core_config,
                    "save_predictions": bool(args.save_predictions),
                },
                indent=2,
                sort_keys=True,
            )
        )

    if args.save_predictions:
        paths["predictions"].mkdir(parents=True, exist_ok=True)
    return paths


def completed_keys(runs_path: Path) -> set[tuple[Any, ...]]:
    if not runs_path.exists():
        return set()
    frame = pd.read_csv(runs_path, sep="\t", low_memory=False)
    required = set(RUN_KEY_COLUMNS + ["eval_set"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{runs_path} lacks resume columns: {sorted(missing)}"
        )
    tests = frame[frame["eval_set"].astype(str).eq("test")]
    return set(
        zip(
            tests["feature"].astype(str),
            tests["negative_variant"].astype(str),
            tests["split_scheme"].astype(str),
            tests["split_seed"].astype(int),
            tests["sampling_seed"].astype(int),
            tests["requested_n_train"].astype(int),
        )
    )


def append_rows(
    path: Path,
    rows: list[dict[str, Any]],
    key_columns: list[str],
    sort_columns: list[str],
) -> None:
    if not rows:
        return
    incoming = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path, sep="\t", low_memory=False)
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming
    if combined.duplicated(key_columns).any():
        duplicates = combined.loc[
            combined.duplicated(key_columns, keep=False), key_columns
        ]
        raise ValueError(
            f"Duplicate keys detected in {path}:\n"
            + duplicates.to_string(index=False)
        )
    combined = combined.sort_values(sort_columns)
    combined.to_csv(path, sep="\t", index=False)


def model_args_for_run(
    args: argparse.Namespace,
    model_seed: int,
) -> SimpleNamespace:
    # Keep the per-fit modeling arguments explicit and deterministic.
    return SimpleNamespace(
        C_grid=[float(value) for value in args.C_grid],
        selection_metric=str(args.selection_metric),
        solver=str(args.solver),
        class_weight=str(args.class_weight),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        model_seed=int(model_seed),
    )


def prediction_path(
    root: Path,
    feature: str,
    negative_variant: str,
    split_scheme: str,
    split_seed: int,
    sampling_seed: int,
    requested_n_train: int,
) -> Path:
    return (
        root
        / feature
        / negative_variant
        / split_scheme
        / f"split_seed{split_seed}"
        / f"sampling_seed{sampling_seed}"
        / f"n{requested_n_train}.npz"
    )


def save_run_predictions(
    path: Path,
    validation_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    y_validation: np.ndarray,
    y_test: np.ndarray,
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    selected_C: float,
    tuned_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        validation_pair_ids=validation_pairs["pair_id"].astype(str).to_numpy(),
        validation_y_true=np.asarray(y_validation, dtype=np.int8),
        validation_probability=np.asarray(validation_scores, dtype=np.float32),
        test_pair_ids=test_pairs["pair_id"].astype(str).to_numpy(),
        test_y_true=np.asarray(y_test, dtype=np.int8),
        test_probability=np.asarray(test_scores, dtype=np.float32),
        selected_C=np.asarray(float(selected_C), dtype=np.float64),
        tuned_threshold=np.asarray(float(tuned_threshold), dtype=np.float64),
    )


def build_experiments(
    args: argparse.Namespace,
    n_compounds: int,
) -> tuple[
    list[dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    experiments: list[dict[str, Any]] = []
    pair_qc_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    sample_qc_rows: list[dict[str, Any]] = []

    for negative_variant in args.variants:
        for split_scheme in args.schemes:
            for split_seed in args.split_seeds:
                pair_path = resolve_pair_file(
                    args.pair_index_dir,
                    negative_variant,
                    split_scheme,
                    int(split_seed),
                )
                pairs, left_col, right_col, qc_rows = load_pair_table(
                    pair_path,
                    negative_variant,
                    split_scheme,
                    int(split_seed),
                    n_compounds,
                )
                pair_qc_rows.extend(qc_rows)
                train_name, validation_name, test_name = SPLIT_NAMES[
                    split_scheme
                ]
                train_pairs = subset_by_split(pairs, train_name)
                validation_pairs = subset_by_split(pairs, validation_name)
                test_pairs = subset_by_split(pairs, test_name)

                plans, one_manifest, one_sample_qc = build_nested_pair_samples(
                    train_pairs=train_pairs,
                    negative_variant=negative_variant,
                    split_scheme=split_scheme,
                    split_seed=int(split_seed),
                    sampling_seeds=args.sampling_seeds,
                    train_sizes=args.train_sizes,
                )
                manifest_rows.extend(one_manifest)
                sample_qc_rows.extend(one_sample_qc)

                experiments.append(
                    {
                        "negative_variant": negative_variant,
                        "split_scheme": split_scheme,
                        "split_seed": int(split_seed),
                        "pair_path": pair_path,
                        "left_col": left_col,
                        "right_col": right_col,
                        "train_pairs": train_pairs,
                        "validation_pairs": validation_pairs,
                        "test_pairs": test_pairs,
                        "plans": plans,
                    }
                )
                print(
                    f"Prepared variant={negative_variant} "
                    f"scheme={split_scheme} split_seed={split_seed}: "
                    f"train={len(train_pairs):,}, "
                    f"validation={len(validation_pairs):,}, "
                    f"test={len(test_pairs):,}"
                )

    pair_qc = pd.DataFrame(pair_qc_rows)
    manifest = pd.DataFrame(manifest_rows)
    sample_qc = pd.DataFrame(sample_qc_rows)
    return experiments, pair_qc, manifest, sample_qc


def run_one_feature_experiment(
    args: argparse.Namespace,
    paths: dict[str, Path],
    feature_name: str,
    feature_kind: str,
    feature_info: dict[str, Any],
    compound_matrix,
    experiment: dict[str, Any],
    done: set[tuple[Any, ...]],
) -> None:
    negative_variant = experiment["negative_variant"]
    split_scheme = experiment["split_scheme"]
    split_seed = int(experiment["split_seed"])
    left_col = experiment["left_col"]
    right_col = experiment["right_col"]
    train_pairs = experiment["train_pairs"]
    validation_pairs = experiment["validation_pairs"]
    test_pairs = experiment["test_pairs"]
    plans = experiment["plans"]

    experiment_keys = {
        (
            feature_name,
            negative_variant,
            split_scheme,
            split_seed,
            int(sampling_seed),
            int(requested_n_train),
        )
        for sampling_seed in args.sampling_seeds
        for requested_n_train in args.train_sizes
    }
    if experiment_keys.issubset(done):
        print(
            "Skipping completed experiment: "
            f"feature={feature_name}, variant={negative_variant}, "
            f"scheme={split_scheme}, split_seed={split_seed}"
        )
        return

    y_validation = validation_pairs["label"].to_numpy(dtype=int)
    y_test = test_pairs["label"].to_numpy(dtype=int)
    pair_representation = (
        "and_xor"
        if feature_kind == "sparse_binary"
        else "sum_absolute_difference"
    )

    # Pair features for fingerprints and FM embeddings do not depend on the
    # sampled training subset. Build them once for this feature/split file and
    # index the training pool for every nested sample. RDKit descriptor
    # preprocessing is fitted separately on each selected training subset.
    X_train_pool = None
    X_validation_fixed = None
    X_test_fixed = None
    if feature_kind != "dense_descriptor":
        if feature_kind == "sparse_binary":
            builder = build_sparse_binary_pair_features
        else:
            builder = build_dense_pair_features
        X_train_pool = builder(
            compound_matrix, train_pairs, left_col, right_col
        )
        X_validation_fixed = builder(
            compound_matrix, validation_pairs, left_col, right_col
        )
        X_test_fixed = builder(compound_matrix, test_pairs, left_col, right_col)

    experiment_metric_rows: list[dict[str, Any]] = []
    experiment_c_rows: list[dict[str, Any]] = []

    for sampling_seed in args.sampling_seeds:
        for requested_n_train in args.train_sizes:
            key = (
                feature_name,
                negative_variant,
                split_scheme,
                split_seed,
                int(sampling_seed),
                int(requested_n_train),
            )
            if key in done:
                print("Skipping completed:", key)
                continue

            selected_positions = plans[
                (int(sampling_seed), int(requested_n_train))
            ]
            sampled_train_pairs = train_pairs.iloc[
                selected_positions
            ].reset_index(drop=True)
            y_train = sampled_train_pairs["label"].to_numpy(dtype=int)
            n_positive = int(np.sum(y_train == 1))
            n_negative = int(np.sum(y_train == 0))
            digest = sample_hash(sampled_train_pairs["pair_id"].astype(str))

            print(
                f"\nfeature={feature_name} variant={negative_variant} "
                f"scheme={split_scheme} split_seed={split_seed} "
                f"sampling_seed={sampling_seed} N={requested_n_train}"
            )

            imputer_statistics = None
            descriptor_scale_mean = None
            descriptor_scale_var = None
            if feature_kind == "dense_descriptor":
                (
                    working_compound_matrix,
                    imputer,
                    descriptor_scaler,
                ) = fit_descriptor_preprocessor(
                    compound_matrix,
                    sampled_train_pairs,
                    left_col,
                    right_col,
                )
                X_train = build_dense_pair_features(
                    working_compound_matrix,
                    sampled_train_pairs,
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
                imputer_statistics = np.asarray(
                    imputer.statistics_, dtype=np.float64
                )
                descriptor_scale_mean = np.asarray(
                    descriptor_scaler.mean_, dtype=np.float64
                )
                descriptor_scale_var = np.asarray(
                    descriptor_scaler.var_, dtype=np.float64
                )
            else:
                X_train = X_train_pool[selected_positions]
                X_validation = X_validation_fixed
                X_test = X_test_fixed

            model_seed = deterministic_model_seed(
                args.model_seed_base,
                feature_name,
                negative_variant,
                split_scheme,
                split_seed,
                int(sampling_seed),
                int(requested_n_train),
            )
            tuning_args = model_args_for_run(args, model_seed)
            best, search_rows = tune_C(
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
                feature_kind=feature_kind,
                args=tuning_args,
            )
            tuned_threshold, tuned_validation_metrics = tune_threshold(
                y_validation,
                best["validation_scores"],
                args.threshold_grid,
            )
            model = best["model"]
            validation_scores = np.asarray(
                best["validation_scores"], dtype=np.float64
            )
            test_scores = np.asarray(
                model.predict_proba(X_test)[:, 1], dtype=np.float64
            )

            common = {
                "task": TASK_NAME,
                "script_version": SCRIPT_VERSION,
                "feature": feature_name,
                "feature_kind": feature_kind,
                "pair_representation": pair_representation,
                "negative_variant": negative_variant,
                "split_scheme": split_scheme,
                "split_seed": split_seed,
                "sampling_seed": int(sampling_seed),
                "requested_n_train": int(requested_n_train),
                "actual_n_train": int(len(sampled_train_pairs)),
                "n_train_positive": n_positive,
                "n_train_negative": n_negative,
                "train_positive_fraction": float(np.mean(y_train)),
                "n_validation": int(len(validation_pairs)),
                "n_test": int(len(test_pairs)),
                "sample_hash": digest,
                "source_pair_file": str(experiment["pair_path"]),
                "feature_source_path": str(feature_info.get("source_path", "")),
                "compound_dim": int(feature_info.get("compound_dim", 0)),
                "classifier": "logistic_regression",
                "C_grid": "|".join(str(float(value)) for value in args.C_grid),
                "selected_C": float(best["selected_C"]),
                "C_selection_metric": f"validation_{args.selection_metric}",
                "validation_auroc_for_selected_C": float(
                    best["validation_auroc"]
                ),
                "validation_auprc_for_selected_C": float(
                    best["validation_auprc"]
                ),
                "solver": str(args.solver),
                "max_iter": int(args.max_iter),
                "tol": float(args.tol),
                "class_weight": str(args.class_weight),
                "model_seed": int(model_seed),
                "solver_n_iter": int(best["solver_n_iter"]),
                "solver_converged": bool(best["solver_converged"]),
                "convergence_messages": " | ".join(
                    best["convergence_messages"]
                ),
                "threshold_strategy": (
                    "global_threshold_tuned_on_validation_f1"
                ),
                "threshold_grid": "|".join(
                    str(float(value)) for value in args.threshold_grid
                ),
                "tuned_threshold": float(tuned_threshold),
                "validation_f1_for_tuned_threshold": float(
                    tuned_validation_metrics["f1"]
                ),
                "validation_mcc_for_tuned_threshold": float(
                    tuned_validation_metrics["mcc"]
                ),
            }

            metric_rows: list[dict[str, Any]] = []
            for eval_set, y_true, scores in [
                ("validation", y_validation, validation_scores),
                ("test", y_test, test_scores),
            ]:
                metrics = complete_metrics(y_true, scores, tuned_threshold)
                metric_rows.append(
                    {
                        **common,
                        "eval_set": eval_set,
                        "n_eval_positive": int(np.sum(y_true == 1)),
                        "n_eval_negative": int(np.sum(y_true == 0)),
                        **metrics,
                    }
                )

            c_rows: list[dict[str, Any]] = []
            for row in search_rows:
                c_rows.append(
                    {
                        **{
                            column: common[column]
                            for column in [
                                "task",
                                "script_version",
                                "feature",
                                "feature_kind",
                                "negative_variant",
                                "split_scheme",
                                "split_seed",
                                "sampling_seed",
                                "requested_n_train",
                                "actual_n_train",
                                "sample_hash",
                                "model_seed",
                            ]
                        },
                        **row,
                    }
                )

            experiment_metric_rows.extend(metric_rows)
            experiment_c_rows.extend(c_rows)

            if args.save_predictions:
                path = prediction_path(
                    paths["predictions"],
                    feature_name,
                    negative_variant,
                    split_scheme,
                    split_seed,
                    int(sampling_seed),
                    int(requested_n_train),
                )
                save_run_predictions(
                    path,
                    validation_pairs,
                    test_pairs,
                    y_validation,
                    y_test,
                    validation_scores,
                    test_scores,
                    best["selected_C"],
                    tuned_threshold,
                )
                if feature_kind == "dense_descriptor":
                    preprocessing_path = path.with_name(
                        path.stem + "_descriptor_preprocessing.npz"
                    )
                    np.savez_compressed(
                        preprocessing_path,
                        imputer_statistics=imputer_statistics,
                        descriptor_scale_mean=descriptor_scale_mean,
                        descriptor_scale_var=descriptor_scale_var,
                    )

            print(
                f"selected_C={best['selected_C']}; "
                f"validation_AUROC={best['validation_auroc']:.6f}; "
                f"test_AUROC={metric_rows[1]['auroc']:.6f}"
            )
            done.add(key)

            del model
            del best
            del X_train
            if feature_kind == "dense_descriptor":
                del X_validation
                del X_test
                del working_compound_matrix
            gc.collect()

    append_rows(
        paths["runs"],
        experiment_metric_rows,
        key_columns=RUN_KEY_COLUMNS + ["eval_set"],
        sort_columns=RUN_KEY_COLUMNS + ["eval_set"],
    )
    append_rows(
        paths["c_search"],
        experiment_c_rows,
        key_columns=RUN_KEY_COLUMNS + ["C"],
        sort_columns=RUN_KEY_COLUMNS + ["C"],
    )

    del X_train_pool
    del X_validation_fixed
    del X_test_fixed
    gc.collect()


def summarize_results(
    runs_path: Path,
    per_split_path: Path,
    summary_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(runs_path, sep="\t", low_memory=False)
    group_per_split = [
        "task",
        "feature",
        "feature_kind",
        "pair_representation",
        "negative_variant",
        "split_scheme",
        "split_seed",
        "requested_n_train",
        "eval_set",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "n_sampling_replicates": ("sampling_seed", "nunique"),
        "actual_n_train_mean": ("actual_n_train", "mean"),
    }
    for metric in SUMMARY_METRICS:
        if metric in runs.columns:
            aggregations[f"{metric}_mean"] = (metric, "mean")
            aggregations[f"{metric}_sd_across_sampling"] = (
                metric,
                "std",
            )
    per_split = (
        runs.groupby(group_per_split, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(group_per_split)
    )
    per_split.to_csv(per_split_path, sep="\t", index=False)

    group_summary = [
        "task",
        "feature",
        "feature_kind",
        "pair_representation",
        "negative_variant",
        "split_scheme",
        "requested_n_train",
        "eval_set",
    ]
    rows: list[dict[str, Any]] = []
    mean_columns = [
        column
        for column in per_split.columns
        if column.endswith("_mean")
        and column not in {"actual_n_train_mean"}
    ]
    for keys, subset in per_split.groupby(
        group_summary, dropna=False, sort=True
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_summary, keys))
        row["n_split_seeds"] = int(subset["split_seed"].nunique())
        row["n_sampling_replicates_per_split_min"] = int(
            subset["n_sampling_replicates"].min()
        )
        row["n_sampling_replicates_per_split_max"] = int(
            subset["n_sampling_replicates"].max()
        )
        row["actual_n_train_mean"] = float(
            subset["actual_n_train_mean"].mean()
        )

        for mean_column in mean_columns:
            metric_name = mean_column[: -len("_mean")]
            values = pd.to_numeric(
                subset[mean_column], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            n = len(values)
            if n == 0:
                row[f"{metric_name}_mean"] = np.nan
                row[f"{metric_name}_sd"] = np.nan
                row[f"{metric_name}_se"] = np.nan
                row[f"{metric_name}_ci95_low"] = np.nan
                row[f"{metric_name}_ci95_high"] = np.nan
                continue
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
            se = float(sd / np.sqrt(n)) if n > 1 else np.nan
            if n > 1:
                critical = float(student_t.ppf(0.975, df=n - 1))
                half_width = critical * se
                ci_low = mean - half_width
                ci_high = mean + half_width
            else:
                ci_low = np.nan
                ci_high = np.nan
            row[f"{metric_name}_mean"] = mean
            row[f"{metric_name}_sd"] = sd
            row[f"{metric_name}_se"] = se
            row[f"{metric_name}_ci95_low"] = ci_low
            row[f"{metric_name}_ci95_high"] = ci_high
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(group_summary)
    summary.to_csv(summary_path, sep="\t", index=False)
    return per_split, summary


def expected_run_keys(
    features: Iterable[str],
    args: argparse.Namespace,
) -> set[tuple[Any, ...]]:
    return {
        (
            feature,
            variant,
            scheme,
            int(split_seed),
            int(sampling_seed),
            int(train_size),
        )
        for feature in features
        for variant in args.variants
        for scheme in args.schemes
        for split_seed in args.split_seeds
        for sampling_seed in args.sampling_seeds
        for train_size in args.train_sizes
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    paths = prepare_output_directory(args)

    metadata = load_metadata(args.feature_dir)
    features = selected_features(args)
    print("Selected features:", ", ".join(features))

    experiments, pair_qc, sample_manifest, sample_qc = build_experiments(
        args, n_compounds=len(metadata)
    )
    pair_qc.to_csv(paths["pair_split_qc"], sep="\t", index=False)
    sample_qc.to_csv(paths["sample_qc"], sep="\t", index=False)
    sample_manifest.to_csv(
        paths["sample_manifest"],
        sep="\t",
        index=False,
        compression="gzip",
    )

    done = completed_keys(paths["runs"]) if args.resume else set()
    for feature_name in features:
        print(f"\n=== Loading feature: {feature_name} ===")
        compound_matrix, feature_kind, feature_info = load_feature(
            feature_name,
            args.feature_dir,
            metadata,
        )
        for experiment in experiments:
            run_one_feature_experiment(
                args=args,
                paths=paths,
                feature_name=feature_name,
                feature_kind=feature_kind,
                feature_info=feature_info,
                compound_matrix=compound_matrix,
                experiment=experiment,
                done=done,
            )
        del compound_matrix
        gc.collect()

    expected = expected_run_keys(features, args)
    observed = completed_keys(paths["runs"])
    missing = expected - observed
    extra = observed - expected
    if missing:
        raise RuntimeError(
            f"Experiment ended with {len(missing)} missing runs. "
            f"First missing keys: {sorted(missing)[:10]}"
        )
    if extra:
        raise RuntimeError(
            f"Runs TSV contains {len(extra)} unexpected keys. "
            f"First extra keys: {sorted(extra)[:10]}"
        )

    _, summary = summarize_results(
        paths["runs"],
        paths["per_split_means"],
        paths["summary"],
    )

    print("\nWrote:")
    for name, path in paths.items():
        if name == "predictions" and not args.save_predictions:
            continue
        print(f"  {name}: {path}")

    test_summary = summary[summary["eval_set"].astype(str).eq("test")]
    display_columns = [
        "negative_variant",
        "split_scheme",
        "requested_n_train",
        "feature",
        "auroc_mean",
        "auroc_ci95_low",
        "auroc_ci95_high",
        "auprc_mean",
    ]
    available = [
        column for column in display_columns if column in test_summary.columns
    ]
    print("\nTest summary:")
    print(test_summary[available].to_string(index=False))


if __name__ == "__main__":
    main()

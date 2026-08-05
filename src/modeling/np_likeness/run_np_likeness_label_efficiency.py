#!/usr/bin/env python

"""
Task 0 NP-likeness label-efficiency benchmark.

Design
------
- Binary task: COCONUT natural products versus ZINC synthetic decoys.
- One fixed predefined train/validation/test split per split mode.
- Validation and test sets never change with training-set size.
- For each sampling seed, class-balanced training subsets are nested:
  smaller subsets are contained in larger subsets.
- Every representation uses the exact same sampled training compounds.
- Logistic-regression C is selected using validation AUROC, with AUPRC as a tie-breaker.
- A single decision threshold is selected using validation F1.
- Test data are never used for model, C, or threshold selection.

The full-data benchmark is intentionally separate. This script focuses on
subsampled training sizes and does not include probability ensembles.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import hashlib
import json
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


SCRIPT_VERSION = "np_likeness_label_efficiency_v2_auroc_tuning"
TASK_NAME = "np_likeness"
ID_COL = "compound_id"
LABEL_COL = "is_np"

# The label-efficiency script uses Morgan r2 by default and all retained
# frozen FM embeddings. Pass --fm-patterns to use a smaller FM subset.
DEFAULT_FM_PATTERNS = [
    "chemberta_77m_mlm",
    "chemberta_100m_mlm",
    "molformer_c3_1_1b",
    "nafm_graph_mean",
    "molclr_gcn",
    "molclr_gin",
    "graphmvp_gin",
    "molebert_gin",
]

DEFAULT_TRAIN_SIZES = [100, 300, 1000, 3000, 10000, 30000]
DEFAULT_SAMPLE_SEEDS = [2026, 2027, 2028, 2029, 2030]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fixed-validation/test label-efficiency benchmark for "
            "NP-likeness prediction."
        )
    )
    parser.add_argument(
        "--task-file",
        default="data/tasks/np_likeness/np_likeness_task.tsv",
    )
    parser.add_argument(
        "--split-dir",
        default="data/splits/np_likeness",
    )
    parser.add_argument(
        "--feature-dir",
        default="data/features/np_likeness",
    )
    parser.add_argument(
        "--output-dir",
        default="results/np_likeness/label_efficiency",
    )
    parser.add_argument(
        "--split-mode",
        choices=["random", "scaffold", "both"],
        default="both",
    )
    parser.add_argument(
        "--base-split-seed",
        type=int,
        default=2026,
        help=(
            "Predefined split seed used to fix train/validation/test. "
            "Training subsampling is repeated with --sample-seeds."
        ),
    )
    parser.add_argument(
        "--sample-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SAMPLE_SEEDS,
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_TRAIN_SIZES,
        help="Total number of labeled training compounds per subset.",
    )
    parser.add_argument(
        "--fm-patterns",
        nargs="+",
        default=DEFAULT_FM_PATTERNS,
    )
    parser.add_argument(
        "--skip-missing-fm",
        action="store_true",
    )
    parser.add_argument(
        "--fp-radii",
        nargs="+",
        type=int,
        default=[2,3],
        help="Morgan radius 2 is the default label-efficiency baseline.",
    )
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument(
        "--fingerprint-chunk-size",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--overwrite-fingerprints",
        action="store_true",
    )
    parser.add_argument(
        "--C-grid",
        nargs="+",
        type=float,
        default=[0.01, 0.1],
    )
    parser.add_argument(
        "--threshold-grid",
        nargs="+",
        type=float,
        default=[0.2, 0.4, 0.6, 0.8],
    )
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the existing repeated-runs TSV. A configuration "
            "mismatch causes an error rather than silently mixing runs."
        ),
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Delete existing result files before starting.",
    )
    return parser.parse_args()


def requested_modes(split_mode: str) -> list[str]:
    return ["random", "scaffold"] if split_mode == "both" else [split_mode]


def load_task(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    task_path = Path(path)
    if not task_path.exists():
        raise FileNotFoundError(task_path)

    task = pd.read_csv(task_path, sep="\t", low_memory=False)
    required = {
        ID_COL,
        "canonical_smiles",
        "inchikey",
        "murcko_scaffold",
        "scaffold_split_key",
        LABEL_COL,
        "source",
    }
    missing = required - set(task.columns)
    if missing:
        raise ValueError(f"Task file lacks columns: {sorted(missing)}")

    task = task.copy()
    task[ID_COL] = task[ID_COL].astype(str)
    task["inchikey"] = task["inchikey"].astype(str)
    task[LABEL_COL] = task[LABEL_COL].astype(int)

    if task[ID_COL].duplicated().any():
        raise ValueError(f"Duplicated {ID_COL} values")
    if task["inchikey"].duplicated().any():
        raise ValueError("Duplicated InChIKeys")
    if task["canonical_smiles"].isna().any():
        raise ValueError("Missing canonical SMILES")
    if set(task[LABEL_COL].unique()) != {0, 1}:
        raise ValueError(f"{LABEL_COL} must contain exactly classes 0 and 1")

    print("SCRIPT_VERSION:", SCRIPT_VERSION)
    print("Task:", TASK_NAME)
    print("Compounds:", f"{len(task):,}")
    print(
        "Class counts:",
        task[LABEL_COL].value_counts().sort_index().to_dict(),
    )
    return task, task[LABEL_COL].to_numpy(dtype=np.int8)


def split_file_path(
    split_dir: str,
    split_mode: str,
    split_seed: int,
) -> Path:
    return (
        Path(split_dir)
        / f"np_likeness_{split_mode}_split_seed{split_seed}.tsv"
    )


def load_split(
    task: pd.DataFrame,
    split_dir: str,
    split_mode: str,
    split_seed: int,
) -> pd.DataFrame:
    path = split_file_path(split_dir, split_mode, split_seed)
    if not path.exists():
        raise FileNotFoundError(path)

    saved = pd.read_csv(path, sep="\t", low_memory=False)
    required = {ID_COL, "split"}
    missing = required - set(saved.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")

    saved = saved.copy()
    saved[ID_COL] = saved[ID_COL].astype(str)
    if saved[ID_COL].duplicated().any():
        raise ValueError(f"{path}: duplicated {ID_COL}")

    keep = [ID_COL, "split"]
    for column in ["inchikey", LABEL_COL]:
        if column in saved.columns:
            keep.append(column)

    renamed = {
        column: f"saved_{column}"
        for column in keep
        if column not in {ID_COL, "split"}
    }
    saved = saved[keep].rename(columns=renamed)

    result = task.merge(
        saved,
        on=ID_COL,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if result["split"].isna().any():
        raise ValueError(
            f"{path}: missing split assignment for "
            f"{int(result['split'].isna().sum())} compounds"
        )
    if not np.array_equal(
        result[ID_COL].astype(str).to_numpy(),
        task[ID_COL].astype(str).to_numpy(),
    ):
        raise ValueError(
            f"{path}: merging the split table changed task row order"
        )

    expected = {"train", "validation", "test"}
    observed = set(result["split"].astype(str).unique())
    if observed != expected:
        raise ValueError(
            f"{path}: expected {sorted(expected)}, found {sorted(observed)}"
        )

    if "saved_inchikey" in result.columns:
        mismatch = (
            result["inchikey"].astype(str)
            != result["saved_inchikey"].astype(str)
        )
        if mismatch.any():
            raise ValueError(
                f"{path}: {int(mismatch.sum())} InChIKey mismatches"
            )

    if f"saved_{LABEL_COL}" in result.columns:
        mismatch = (
            result[LABEL_COL].astype(int)
            != result[f"saved_{LABEL_COL}"].astype(int)
        )
        if mismatch.any():
            raise ValueError(
                f"{path}: {int(mismatch.sum())} label mismatches"
            )

    if split_mode == "scaffold":
        groups = {
            name: set(
                result.loc[
                    result["split"].eq(name),
                    "scaffold_split_key",
                ].astype(str)
            )
            for name in ["train", "validation", "test"]
        }
        for left, right in [
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise ValueError(
                    f"{path}: scaffold leakage between {left} and {right}: "
                    f"{len(overlap)} shared scaffold keys"
                )

    print(f"\nFixed {split_mode} split: {path}")
    print(
        result.groupby("split")[LABEL_COL]
        .agg(["size", "sum", "mean"])
        .rename(
            columns={
                "sum": "n_positive",
                "mean": "positive_fraction",
            }
        )
        .to_string()
    )
    return result


def deterministic_seed(
    split_mode: str,
    base_split_seed: int,
    sample_seed: int,
) -> int:
    mode_code = 0 if split_mode == "random" else 1
    sequence = np.random.SeedSequence(
        [mode_code, base_split_seed, sample_seed]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def sample_hash(
    compound_ids: np.ndarray,
) -> str:
    ordered = np.sort(compound_ids.astype(str))
    payload = "\n".join(ordered).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_nested_balanced_samples(
    split_task: pd.DataFrame,
    y: np.ndarray,
    split_mode: str,
    base_split_seed: int,
    sample_seeds: list[int],
    train_sizes: list[int],
) -> tuple[list[dict], list[dict], dict[str, np.ndarray]]:
    train_mask = split_task["split"].eq("train").to_numpy()
    train_indices = np.flatnonzero(train_mask)
    negative_indices = train_indices[y[train_indices] == 0]
    positive_indices = train_indices[y[train_indices] == 1]

    maximum_balanced_size = 2 * min(
        len(negative_indices),
        len(positive_indices),
    )
    if max(train_sizes) > maximum_balanced_size:
        raise ValueError(
            f"{split_mode}: requested training size {max(train_sizes):,} "
            f"exceeds maximum balanced subset size "
            f"{maximum_balanced_size:,}"
        )

    plans = []
    qc_rows = []
    arrays = {}

    for sample_seed in sample_seeds:
        rng = np.random.default_rng(
            deterministic_seed(
                split_mode,
                base_split_seed,
                sample_seed,
            )
        )
        negative_order = negative_indices.copy()
        positive_order = positive_indices.copy()
        rng.shuffle(negative_order)
        rng.shuffle(positive_order)

        previous_indices: set[int] = set()
        for requested_n_train in train_sizes:
            n_positive = requested_n_train // 2
            n_negative = requested_n_train - n_positive

            selected = np.concatenate(
                [
                    negative_order[:n_negative],
                    positive_order[:n_positive],
                ]
            )
            selected = np.sort(selected.astype(np.int64))

            selected_set = set(selected.tolist())
            if previous_indices and not previous_indices.issubset(selected_set):
                raise RuntimeError(
                    "Nested-sampling invariant failed: a smaller sample is "
                    "not contained in the next larger sample"
                )
            previous_indices = selected_set

            key = (
                f"{split_mode}_basesplit{base_split_seed}_"
                f"sampleseed{sample_seed}_n{requested_n_train}"
            )
            arrays[key] = selected

            ids = split_task.iloc[selected][ID_COL].astype(str).to_numpy()
            labels = y[selected]
            plan = {
                "split_mode": split_mode,
                "base_split_seed": base_split_seed,
                "sample_seed": sample_seed,
                "requested_n_train": requested_n_train,
                "actual_n_train": int(len(selected)),
                "train_indices": selected,
                "sample_key": key,
            }
            plans.append(plan)
            qc_rows.append(
                {
                    "task": TASK_NAME,
                    "split_mode": split_mode,
                    "base_split_seed": base_split_seed,
                    "sample_seed": sample_seed,
                    "requested_n_train": requested_n_train,
                    "actual_n_train": int(len(selected)),
                    "n_negative": int((labels == 0).sum()),
                    "n_positive": int((labels == 1).sum()),
                    "positive_fraction": float(labels.mean()),
                    "sample_sha256": sample_hash(ids),
                    "nested_within_sampling_seed": True,
                }
            )

    return plans, qc_rows, arrays


def find_embedding_files(
    pattern: str,
    feature_dir: str,
) -> tuple[str, Path, Path]:
    directory = Path(feature_dir)
    exact = directory / f"{pattern}.embeddings.npy"

    if exact.exists():
        embedding_path = exact
    else:
        matches = sorted(directory.glob(f"{pattern}*.embeddings.npy"))
        if not matches:
            raise FileNotFoundError(
                f"No embedding file found for pattern: {pattern}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple embedding files found for {pattern}: {matches}"
            )
        embedding_path = matches[0]

    stem = embedding_path.name.replace(".embeddings.npy", "")
    metadata_path = directory / f"{stem}.metadata.tsv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    return stem, embedding_path, metadata_path


def fm_pattern_available(
    pattern: str,
    feature_dir: str,
) -> tuple[bool, str]:
    try:
        find_embedding_files(pattern, feature_dir)
    except (FileNotFoundError, ValueError) as exc:
        return False, str(exc)
    return True, "ok"


def load_aligned_embedding(
    task: pd.DataFrame,
    pattern: str,
    feature_dir: str,
):
    stem, embedding_path, metadata_path = find_embedding_files(
        pattern,
        feature_dir,
    )
    X_all = np.load(embedding_path, mmap_mode="r")
    metadata = pd.read_csv(
        metadata_path,
        sep="\t",
        usecols=["inchikey"],
    )
    metadata["inchikey"] = metadata["inchikey"].astype(str)

    if X_all.ndim != 2:
        raise ValueError(f"{stem}: expected 2D array, found {X_all.shape}")
    if len(metadata) != X_all.shape[0]:
        raise ValueError(
            f"{stem}: metadata rows {len(metadata)} != "
            f"embedding rows {X_all.shape[0]}"
        )
    if metadata["inchikey"].duplicated().any():
        raise ValueError(f"{stem}: duplicated metadata InChIKeys")

    task_keys = task["inchikey"].astype(str).to_numpy()
    metadata_keys = metadata["inchikey"].to_numpy()

    if np.array_equal(task_keys, metadata_keys):
        X = X_all
        alignment = "direct_row_alignment"
    else:
        lookup = pd.Series(
            np.arange(len(metadata), dtype=np.int64),
            index=metadata["inchikey"],
        )
        rows = lookup.reindex(task_keys)
        if rows.isna().any():
            raise ValueError(
                f"{stem}: missing embeddings for "
                f"{int(rows.isna().sum())} task compounds"
            )
        X = np.asarray(
            X_all[rows.to_numpy(dtype=np.int64)],
            dtype=np.float32,
        )
        alignment = "inchikey_reordered_copy"

    print(
        f"{stem}: shape={X.shape}; alignment={alignment}; "
        f"dtype={X.dtype}"
    )
    return stem, X


def fingerprint_paths(
    feature_dir: str,
    radius: int,
    n_bits: int,
) -> tuple[Path, Path]:
    directory = Path(feature_dir)
    stem = f"morgan_r{radius}_{n_bits}"
    return (
        directory / f"{stem}.npz",
        directory / f"{stem}.metadata.tsv",
    )


def generate_morgan_cache(
    task: pd.DataFrame,
    radius: int,
    n_bits: int,
    feature_dir: str,
    chunk_size: int,
):
    feature_path, metadata_path = fingerprint_paths(
        feature_dir,
        radius,
        n_bits,
    )
    feature_path.parent.mkdir(parents=True, exist_ok=True)

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )
    chunks = []
    total = len(task)

    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        row_indices: list[int] = []
        column_indices: list[int] = []

        for local_row, smiles in enumerate(
            task.iloc[start:stop]["canonical_smiles"].astype(str)
        ):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(
                    "RDKit failed while regenerating a validated task "
                    f"molecule at row {start + local_row}: {smiles}"
                )
            fingerprint = generator.GetFingerprint(mol)
            on_bits = list(fingerprint.GetOnBits())
            row_indices.extend([local_row] * len(on_bits))
            column_indices.extend(on_bits)

        chunks.append(
            sparse.csr_matrix(
                (
                    np.ones(len(row_indices), dtype=np.uint8),
                    (
                        np.asarray(row_indices, dtype=np.int32),
                        np.asarray(column_indices, dtype=np.int32),
                    ),
                ),
                shape=(stop - start, n_bits),
                dtype=np.uint8,
            )
        )
        print(
            f"Generated Morgan r{radius}: {stop:,}/{total:,} compounds"
        )

    X = sparse.vstack(chunks, format="csr")
    sparse.save_npz(feature_path, X, compressed=True)
    task[[ID_COL, "inchikey"]].to_csv(
        metadata_path,
        sep="\t",
        index=False,
    )
    print("Wrote:", feature_path)
    print("Wrote:", metadata_path)
    return X


def load_or_create_morgan(
    task: pd.DataFrame,
    radius: int,
    n_bits: int,
    feature_dir: str,
    chunk_size: int,
    overwrite: bool,
):
    feature_path, metadata_path = fingerprint_paths(
        feature_dir,
        radius,
        n_bits,
    )
    feature_name = f"morgan_r{radius}_{n_bits}"

    if overwrite or not feature_path.exists() or not metadata_path.exists():
        X = generate_morgan_cache(
            task=task,
            radius=radius,
            n_bits=n_bits,
            feature_dir=feature_dir,
            chunk_size=chunk_size,
        )
    else:
        X = sparse.load_npz(feature_path).tocsr()
        metadata = pd.read_csv(metadata_path, sep="\t")

        if X.shape != (len(task), n_bits):
            raise ValueError(
                f"{feature_name}: expected {(len(task), n_bits)}, "
                f"found {X.shape}"
            )
        if len(metadata) != len(task):
            raise ValueError(
                f"{feature_name}: metadata rows {len(metadata)} != "
                f"{len(task)}"
            )
        if not np.array_equal(
            metadata["inchikey"].astype(str).to_numpy(),
            task["inchikey"].astype(str).to_numpy(),
        ):
            raise ValueError(
                f"{feature_name}: cached fingerprint order does not match "
                "the current task table"
            )
        print(f"{feature_name}: loaded cached shape {X.shape}")

    return feature_name, X


def class_weight_value(value: str):
    return None if value == "none" else value


def build_classifier(
    C: float,
    seed: int,
    max_iter: int,
    tol: float,
    class_weight: str,
) -> LogisticRegression:
    return LogisticRegression(
        C=float(C),
        solver="lbfgs",
        class_weight=class_weight_value(class_weight),
        max_iter=max_iter,
        tol=tol,
        random_state=seed,
    )


def prepare_features(
    X,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    feature_type: str,
):
    if feature_type == "dense_embedding":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_indices])
        X_validation = scaler.transform(X[validation_indices])
        return scaler, X_train, X_validation

    if feature_type == "fingerprint":
        return None, X[train_indices], X[validation_indices]

    raise ValueError(f"Unknown feature type: {feature_type}")


def predict_probability(
    classifier: LogisticRegression,
    scaler: StandardScaler | None,
    X,
    indices: np.ndarray,
) -> np.ndarray:
    X_eval = X[indices]
    if scaler is not None:
        X_eval = scaler.transform(X_eval)
    return np.asarray(
        classifier.predict_proba(X_eval)[:, 1],
        dtype=np.float64,
    )


def probability_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {"auprc": np.nan, "auroc": np.nan}
    return {
        "auprc": float(average_precision_score(y_true, probability)),
        "auroc": float(roc_auc_score(y_true, probability)),
    }


def threshold_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    prediction = (probability >= threshold).astype(np.int8)
    return {
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "precision": float(
            precision_score(y_true, prediction, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, prediction, zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, prediction)
        ),
        "predicted_positive_fraction": float(np.mean(prediction)),
    }


def compute_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    tuned_threshold: float,
) -> dict[str, float]:
    metrics = probability_metrics(y_true, probability)
    metrics.update(
        threshold_metrics(
            y_true,
            probability,
            tuned_threshold,
        )
    )
    fixed = threshold_metrics(y_true, probability, 0.5)
    metrics.update(
        {
            "f1_at_0_5": fixed["f1"],
            "balanced_accuracy_at_0_5": fixed[
                "balanced_accuracy"
            ],
        }
    )
    return metrics


def tune_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold_grid: Iterable[float],
) -> tuple[float, float]:
    best_threshold = None
    best_f1 = -np.inf
    best_prevalence_gap = np.inf
    true_prevalence = float(np.mean(y_true))

    for threshold in threshold_grid:
        metrics = threshold_metrics(
            y_true,
            probability,
            float(threshold),
        )
        prevalence_gap = abs(
            metrics["predicted_positive_fraction"] - true_prevalence
        )
        if (
            metrics["f1"] > best_f1
            or (
                np.isclose(metrics["f1"], best_f1)
                and prevalence_gap < best_prevalence_gap
            )
        ):
            best_threshold = float(threshold)
            best_f1 = metrics["f1"]
            best_prevalence_gap = prevalence_gap

    if best_threshold is None:
        raise RuntimeError("Threshold selection failed")
    return best_threshold, float(best_f1)


def fit_select_c_and_threshold(
    X,
    y: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    feature_type: str,
    seed: int,
    C_grid: Iterable[float],
    threshold_grid: Iterable[float],
    max_iter: int,
    tol: float,
    class_weight: str,
) -> dict:
    scaler, X_train, X_validation = prepare_features(
        X=X,
        train_indices=train_indices,
        validation_indices=validation_indices,
        feature_type=feature_type,
    )

    y_train = y[train_indices]
    y_validation = y[validation_indices]
    best = None
    best_auroc = -np.inf
    best_auprc = -np.inf

    for C in C_grid:
        classifier = build_classifier(
            C=float(C),
            seed=seed,
            max_iter=max_iter,
            tol=tol,
            class_weight=class_weight,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(X_train, y_train)

        probability = np.asarray(
            classifier.predict_proba(X_validation)[:, 1],
            dtype=np.float64,
        )
        metrics = probability_metrics(y_validation, probability)
        reached_max_iter = bool(
            classifier.n_iter_[0] >= max_iter
            or any(
                issubclass(item.category, ConvergenceWarning)
                for item in caught
            )
        )

        if (
            metrics["auroc"] > best_auroc
            or (
                np.isclose(metrics["auroc"], best_auroc)
                and metrics["auprc"] > best_auprc
            )
        ):
            best = {
                "classifier": classifier,
                "scaler": scaler,
                "selected_C": float(C),
                "validation_probability": probability,
                "validation_auprc_for_selected_C": metrics["auprc"],
                "validation_auroc_for_selected_C": metrics["auroc"],
                "n_iter": int(classifier.n_iter_[0]),
                "reached_max_iter": reached_max_iter,
            }
            best_auroc = metrics["auroc"]
            best_auprc = metrics["auprc"]

    if best is None:
        raise RuntimeError("No C value was selected")

    threshold, validation_best_f1 = tune_threshold(
        y_validation,
        best["validation_probability"],
        threshold_grid,
    )
    best["tuned_threshold"] = threshold
    best["validation_best_f1_at_tuned_threshold"] = (
        validation_best_f1
    )

    del X_train
    del X_validation
    gc.collect()
    return best


def evaluate_row(
    feature: str,
    feature_type: str,
    split_mode: str,
    base_split_seed: int,
    sample_seed: int,
    requested_n_train: int,
    actual_n_train: int,
    eval_set: str,
    y_true: np.ndarray,
    probability: np.ndarray,
    selected: dict,
    args: argparse.Namespace,
    sample_sha256: str,
) -> dict:
    row = {
        "task": TASK_NAME,
        "feature": feature,
        "feature_type": feature_type,
        "model": "binary_logistic_regression",
        "split_mode": split_mode,
        "base_split_seed": base_split_seed,
        "sample_seed": sample_seed,
        "requested_n_train": requested_n_train,
        "actual_n_train": actual_n_train,
        "eval_set": eval_set,
        "n_eval": int(len(y_true)),
        "sample_sha256": sample_sha256,
        "sampling_strategy": "nested_class_balanced",
        "C_grid": "|".join(str(float(C)) for C in args.C_grid),
        "selected_C": selected["selected_C"],
        "C_selection_metric": "validation_auroc",
        "validation_auprc_for_selected_C": selected[
            "validation_auprc_for_selected_C"
        ],
        "validation_auroc_for_selected_C": selected[
            "validation_auroc_for_selected_C"
        ],
        "threshold_grid": "|".join(
            str(float(value))
            for value in args.threshold_grid
        ),
        "threshold_strategy": (
            "global_threshold_tuned_on_validation_f1"
        ),
        "tuned_threshold": selected["tuned_threshold"],
        "validation_best_f1_at_tuned_threshold": selected[
            "validation_best_f1_at_tuned_threshold"
        ],
        "class_weight": args.class_weight,
        "solver": "lbfgs",
        "tol": args.tol,
        "n_iter": selected["n_iter"],
        "reached_max_iter": selected["reached_max_iter"],
    }
    row.update(
        compute_metrics(
            y_true,
            probability,
            selected["tuned_threshold"],
        )
    )
    return row


def feature_specs(args: argparse.Namespace) -> list[tuple]:
    specs = [
        (
            f"morgan_r{radius}_{args.n_bits}",
            "fingerprint",
            ("morgan", int(radius)),
        )
        for radius in args.fp_radii
    ]

    for pattern in args.fm_patterns:
        if args.skip_missing_fm:
            available, reason = fm_pattern_available(
                pattern,
                args.feature_dir,
            )
            if not available:
                print(f"Skipping {pattern}: {reason}")
                continue
        specs.append(
            (
                pattern,
                "dense_embedding",
                ("embedding", pattern),
            )
        )

    if not specs:
        raise ValueError("No features selected")
    return specs


def completed_keys(existing: pd.DataFrame) -> set[tuple]:
    if existing.empty:
        return set()

    key_columns = [
        "feature",
        "split_mode",
        "base_split_seed",
        "sample_seed",
        "requested_n_train",
    ]
    completed = set()
    for keys, subset in existing.groupby(key_columns, dropna=False):
        if set(subset["eval_set"]) >= {"validation", "test"}:
            completed.add(tuple(keys))
    return completed


def append_rows(path: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    write_header = not path.exists()
    frame.to_csv(
        path,
        sep="\t",
        index=False,
        mode="a",
        header=write_header,
    )


def add_paired_morgan_gains(
    runs: pd.DataFrame,
    reference_feature: str = "morgan_r2_2048",
) -> pd.DataFrame:
    result = runs.copy()
    for metric in [
        "auprc",
        "auroc",
        "f1",
        "balanced_accuracy",
    ]:
        result[f"gain_vs_{reference_feature}_{metric}"] = np.nan

    keys = [
        "split_mode",
        "base_split_seed",
        "sample_seed",
        "requested_n_train",
        "eval_set",
    ]
    reference = result[
        result["feature"].eq(reference_feature)
    ][keys + ["auprc", "auroc", "f1", "balanced_accuracy"]].copy()

    if reference.empty:
        return result

    reference = reference.rename(
        columns={
            metric: f"reference_{metric}"
            for metric in [
                "auprc",
                "auroc",
                "f1",
                "balanced_accuracy",
            ]
        }
    )
    result = result.merge(
        reference,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    for metric in [
        "auprc",
        "auroc",
        "f1",
        "balanced_accuracy",
    ]:
        result[f"gain_vs_{reference_feature}_{metric}"] = (
            result[metric] - result[f"reference_{metric}"]
        )
        result = result.drop(columns=[f"reference_{metric}"])

    return result


def aggregate_statistics(runs: pd.DataFrame) -> pd.DataFrame:
    test = runs[runs["eval_set"].eq("test")].copy()
    group_columns = [
        "task",
        "feature",
        "feature_type",
        "split_mode",
        "base_split_seed",
        "requested_n_train",
        "actual_n_train",
        "sampling_strategy",
    ]
    metric_columns = [
        "auprc",
        "auroc",
        "f1",
        "balanced_accuracy",
        "f1_at_0_5",
        "balanced_accuracy_at_0_5",
    ]
    metric_columns.extend(
        column
        for column in test.columns
        if column.startswith("gain_vs_morgan_r2_2048_")
    )

    rows = []
    for keys, subset in test.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_sampling_replicates"] = int(
            subset["sample_seed"].nunique()
        )
        row["n_convergence_failures"] = int(
            subset["reached_max_iter"].astype(bool).sum()
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


def build_config(args: argparse.Namespace) -> dict:
    return {
        "script_version": SCRIPT_VERSION,
        "task_file": args.task_file,
        "split_dir": args.split_dir,
        "feature_dir": args.feature_dir,
        "split_modes": requested_modes(args.split_mode),
        "base_split_seed": args.base_split_seed,
        "sample_seeds": sorted(set(args.sample_seeds)),
        "train_sizes": sorted(set(args.train_sizes)),
        "fm_patterns": args.fm_patterns,
        "fp_radii": args.fp_radii,
        "n_bits": args.n_bits,
        "C_grid": args.C_grid,
        "threshold_grid": args.threshold_grid,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "class_weight": args.class_weight,
        "sampling_strategy": "nested_class_balanced",
        "validation_and_test_fixed": True,
        "probability_ensembles_included": False,
    }


def verify_resume_config(
    config_path: Path,
    current_config: dict,
) -> None:
    if not config_path.exists():
        raise FileNotFoundError(
            f"--resume requested but config does not exist: {config_path}"
        )

    previous = json.loads(config_path.read_text())
    compared_fields = [
        "task_file",
        "split_dir",
        "feature_dir",
        "split_modes",
        "base_split_seed",
        "sample_seeds",
        "train_sizes",
        "fm_patterns",
        "fp_radii",
        "n_bits",
        "C_grid",
        "threshold_grid",
        "max_iter",
        "tol",
        "class_weight",
        "sampling_strategy",
    ]
    mismatches = {
        field: {
            "previous": previous.get(field),
            "current": current_config.get(field),
        }
        for field in compared_fields
        if previous.get(field) != current_config.get(field)
    }
    if mismatches:
        raise ValueError(
            "Resume configuration does not match the existing run:\n"
            + json.dumps(mismatches, indent=2)
        )


def main() -> None:
    args = parse_args()

    args.train_sizes = sorted(set(args.train_sizes))
    args.sample_seeds = sorted(set(args.sample_seeds))

    if not args.train_sizes or any(size < 2 for size in args.train_sizes):
        raise ValueError("All training sizes must be at least 2")
    if not args.sample_seeds:
        raise ValueError("At least one sample seed is required")
    if any(C <= 0 for C in args.C_grid):
        raise ValueError("All C values must be positive")
    if not all(0.0 < value < 1.0 for value in args.threshold_grid):
        raise ValueError("Thresholds must be strictly between 0 and 1")
    if any(radius <= 0 for radius in args.fp_radii):
        raise ValueError("Morgan radii must be positive")
    if args.n_bits <= 0:
        raise ValueError("--n-bits must be positive")
    if args.resume and args.overwrite_results:
        raise ValueError(
            "--resume and --overwrite-results cannot be used together"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_path = (
        output_dir
        / "np_likeness_label_efficiency_repeated_runs.tsv"
    )
    statistics_path = (
        output_dir
        / "np_likeness_label_efficiency_main_statistics.tsv"
    )
    sample_qc_path = (
        output_dir
        / "np_likeness_label_efficiency_sample_qc.tsv"
    )
    sample_indices_path = (
        output_dir
        / "np_likeness_label_efficiency_sample_indices.npz"
    )
    config_path = (
        output_dir
        / "np_likeness_label_efficiency_config.json"
    )

    managed_paths = [
        runs_path,
        statistics_path,
        sample_qc_path,
        sample_indices_path,
        config_path,
    ]
    if args.overwrite_results:
        for path in managed_paths:
            if path.exists():
                path.unlink()

    config = build_config(args)
    if args.resume:
        verify_resume_config(config_path, config)
    else:
        existing = [path for path in managed_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Output files already exist. Use --resume to continue or "
                "--overwrite-results to start over:\n"
                + "\n".join(str(path) for path in existing)
            )
        config_path.write_text(json.dumps(config, indent=2))

    task, y = load_task(args.task_file)

    split_tables: dict[str, pd.DataFrame] = {}
    all_plans: list[dict] = []
    all_qc_rows: list[dict] = []
    all_sample_arrays: dict[str, np.ndarray] = {}

    for split_mode in requested_modes(args.split_mode):
        split_task = load_split(
            task=task,
            split_dir=args.split_dir,
            split_mode=split_mode,
            split_seed=args.base_split_seed,
        )
        split_tables[split_mode] = split_task

        plans, qc_rows, sample_arrays = build_nested_balanced_samples(
            split_task=split_task,
            y=y,
            split_mode=split_mode,
            base_split_seed=args.base_split_seed,
            sample_seeds=args.sample_seeds,
            train_sizes=args.train_sizes,
        )
        all_plans.extend(plans)
        all_qc_rows.extend(qc_rows)
        all_sample_arrays.update(sample_arrays)

    current_sample_qc = pd.DataFrame(all_qc_rows)
    if args.resume:
        previous_qc = pd.read_csv(sample_qc_path, sep="\t")
        comparison_columns = [
            "split_mode",
            "base_split_seed",
            "sample_seed",
            "requested_n_train",
            "actual_n_train",
            "n_negative",
            "n_positive",
            "sample_sha256",
        ]
        left = previous_qc[comparison_columns].sort_values(
            comparison_columns[:4]
        ).reset_index(drop=True)
        right = current_sample_qc[comparison_columns].sort_values(
            comparison_columns[:4]
        ).reset_index(drop=True)
        if not left.equals(right):
            raise ValueError(
                "Regenerated training samples do not match the existing "
                "sample QC file"
            )
    else:
        current_sample_qc.to_csv(
            sample_qc_path,
            sep="\t",
            index=False,
        )
        np.savez_compressed(
            sample_indices_path,
            **all_sample_arrays,
        )

    existing_runs = (
        pd.read_csv(runs_path, sep="\t")
        if runs_path.exists()
        else pd.DataFrame()
    )
    done = completed_keys(existing_runs)

    specs = feature_specs(args)

    for requested_name, feature_type, loader in specs:
        print(f"\n=== Loading feature: {requested_name} ===")

        if loader[0] == "morgan":
            feature_name, X = load_or_create_morgan(
                task=task,
                radius=int(loader[1]),
                n_bits=args.n_bits,
                feature_dir=args.feature_dir,
                chunk_size=args.fingerprint_chunk_size,
                overwrite=args.overwrite_fingerprints,
            )
        else:
            feature_name, X = load_aligned_embedding(
                task=task,
                pattern=str(loader[1]),
                feature_dir=args.feature_dir,
            )

        for plan in all_plans:
            key = (
                feature_name,
                plan["split_mode"],
                plan["base_split_seed"],
                plan["sample_seed"],
                plan["requested_n_train"],
            )
            if key in done:
                print("Skipping completed:", key)
                continue

            split_task = split_tables[plan["split_mode"]]
            validation_indices = np.flatnonzero(
                split_task["split"].eq("validation").to_numpy()
            )
            test_indices = np.flatnonzero(
                split_task["split"].eq("test").to_numpy()
            )
            train_indices = plan["train_indices"]

            sample_qc = current_sample_qc[
                current_sample_qc["split_mode"].eq(
                    plan["split_mode"]
                )
                & current_sample_qc["sample_seed"].eq(
                    plan["sample_seed"]
                )
                & current_sample_qc["requested_n_train"].eq(
                    plan["requested_n_train"]
                )
            ]
            if len(sample_qc) != 1:
                raise RuntimeError("Could not uniquely resolve sample QC")
            sample_sha256 = str(sample_qc["sample_sha256"].iloc[0])

            print(
                f"\n=== {feature_name}: "
                f"{plan['split_mode']} "
                f"base_split={plan['base_split_seed']} "
                f"sample_seed={plan['sample_seed']} "
                f"n_train={plan['requested_n_train']:,} ==="
            )

            selected = fit_select_c_and_threshold(
                X=X,
                y=y,
                train_indices=train_indices,
                validation_indices=validation_indices,
                feature_type=feature_type,
                seed=plan["sample_seed"],
                C_grid=args.C_grid,
                threshold_grid=args.threshold_grid,
                max_iter=args.max_iter,
                tol=args.tol,
                class_weight=args.class_weight,
            )
            validation_probability = selected[
                "validation_probability"
            ]
            test_probability = predict_probability(
                classifier=selected["classifier"],
                scaler=selected["scaler"],
                X=X,
                indices=test_indices,
            )

            validation_row = evaluate_row(
                feature=feature_name,
                feature_type=feature_type,
                split_mode=plan["split_mode"],
                base_split_seed=plan["base_split_seed"],
                sample_seed=plan["sample_seed"],
                requested_n_train=plan["requested_n_train"],
                actual_n_train=plan["actual_n_train"],
                eval_set="validation",
                y_true=y[validation_indices],
                probability=validation_probability,
                selected=selected,
                args=args,
                sample_sha256=sample_sha256,
            )
            test_row = evaluate_row(
                feature=feature_name,
                feature_type=feature_type,
                split_mode=plan["split_mode"],
                base_split_seed=plan["base_split_seed"],
                sample_seed=plan["sample_seed"],
                requested_n_train=plan["requested_n_train"],
                actual_n_train=plan["actual_n_train"],
                eval_set="test",
                y_true=y[test_indices],
                probability=test_probability,
                selected=selected,
                args=args,
                sample_sha256=sample_sha256,
            )
            append_rows(runs_path, [validation_row, test_row])
            done.add(key)

            print(
                f"{feature_name:24s} "
                f"validation AUPRC={validation_row['auprc']:.4f} "
                f"AUROC={validation_row['auroc']:.4f} "
                f"F1={validation_row['f1']:.4f}"
            )
            print(
                f"{feature_name:24s} "
                f"test       AUPRC={test_row['auprc']:.4f} "
                f"AUROC={test_row['auroc']:.4f} "
                f"F1={test_row['f1']:.4f} "
                f"C={selected['selected_C']} "
                f"thr={selected['tuned_threshold']:.2f} "
                f"n_iter={selected['n_iter']} "
                f"max_iter={selected['reached_max_iter']}"
            )

            del selected
            gc.collect()

        del X
        gc.collect()

    runs = pd.read_csv(runs_path, sep="\t")
    expected_keys = {
        (
            feature,
            plan["split_mode"],
            plan["base_split_seed"],
            plan["sample_seed"],
            plan["requested_n_train"],
        )
        for feature, _, _ in [
            (
                (
                    f"morgan_r{loader[1]}_{args.n_bits}"
                    if loader[0] == "morgan"
                    else requested_name
                ),
                feature_type,
                loader,
            )
            for requested_name, feature_type, loader in specs
        ]
        for plan in all_plans
    }
    final_completed = completed_keys(runs)
    missing_keys = expected_keys - final_completed
    if missing_keys:
        raise RuntimeError(
            f"Run ended with {len(missing_keys)} incomplete experiment keys"
        )

    runs_with_gains = add_paired_morgan_gains(runs)
    runs_with_gains.to_csv(runs_path, sep="\t", index=False)

    statistics = aggregate_statistics(runs_with_gains)
    statistics.to_csv(statistics_path, sep="\t", index=False)

    print("\nWrote:")
    print(runs_path)
    print(statistics_path)
    print(sample_qc_path)
    print(sample_indices_path)
    print(config_path)

    display_columns = [
        "split_mode",
        "requested_n_train",
        "feature",
        "n_sampling_replicates",
        "auroc_mean",
        "auroc_sd",
        "auprc_mean",
        "auprc_sd",
        "gain_vs_morgan_r2_2048_auroc_mean",
    ]
    display_columns = [
        column
        for column in display_columns
        if column in statistics.columns
    ]
    if not statistics.empty:
        print("\n=== Label-efficiency test statistics ===")
        print(
            statistics.sort_values(
                ["split_mode", "requested_n_train", "auroc_mean"],
                ascending=[True, True, False],
            )[display_columns].to_string(index=False)
        )


if __name__ == "__main__":
    main()

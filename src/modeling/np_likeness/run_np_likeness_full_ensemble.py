#!/usr/bin/env python

"""
Repeated-split frozen-representation benchmark for NP-likeness prediction.

Task
----
Balanced binary classification:
- 1: COCONUT natural product
- 0: ZINC synthetic decoy

Evaluation
----------
- Five predefined stratified-random splits.
- Five predefined scaffold-disjoint splits.
- Each split uses train/validation/test partitions.
- Logistic-regression C is selected using validation AUPRC.
- A global decision threshold is selected using validation F1.
- Test data are not used for model or threshold selection.

Representations
---------------
- Morgan radius-2 and radius-3 fingerprints, generated once and cached.
- Frozen molecular-foundation-model embeddings aligned by InChIKey.
- Equal-weight probability ensembles of Morgan-r2 plus each FM and all features.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import json
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

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


SCRIPT_VERSION = "np_likeness_full_ensemble_repeated_splits_v1"
TASK_NAME = "np_likeness"
ID_COL = "compound_id"
LABEL_COL = "is_np"

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

PRIMARY_METRICS = [
    "auprc",
    "auroc",
    "f1",
    "balanced_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeated-split Morgan/frozen-FM benchmark for balanced "
            "COCONUT-versus-ZINC NP-likeness prediction."
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
        default="results/np_likeness/full",
    )
    parser.add_argument(
        "--split-mode",
        choices=["random", "scaffold", "both"],
        default="both",
    )
    parser.add_argument(
        "--split-seeds",
        nargs="+",
        type=int,
        default=[2026, 2027, 2028, 2029, 2030],
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
        default=[2, 3],
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
        "--skip-predictions",
        action="store_true",
        help="Do not write the compressed validation/test probability archive.",
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
    if not set(task[LABEL_COL].unique()).issubset({0, 1}):
        raise ValueError(f"{LABEL_COL} contains values outside {{0, 1}}")

    counts = task[LABEL_COL].value_counts().sort_index()
    if set(counts.index) != {0, 1}:
        raise ValueError("Both binary classes must be present")

    print("SCRIPT_VERSION:", SCRIPT_VERSION)
    print("Task:", TASK_NAME)
    print("Compounds:", f"{len(task):,}")
    print("Class counts:", counts.to_dict())
    print("Positive fraction:", f"{task[LABEL_COL].mean():.6f}")

    return task, task[LABEL_COL].to_numpy(dtype=np.int8)


def split_file_path(split_dir: str, split_mode: str, split_seed: int) -> Path:
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

    keep_columns = [ID_COL, "split"]
    for column in ["inchikey", LABEL_COL, "source"]:
        if column in saved.columns:
            keep_columns.append(column)

    renamed = {
        column: f"saved_{column}"
        for column in keep_columns
        if column != ID_COL and column != "split"
    }
    saved = saved[keep_columns].rename(columns=renamed)

    result = task.merge(
        saved,
        on=ID_COL,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if result["split"].isna().any():
        raise ValueError(
            f"{path}: missing assignments for "
            f"{int(result['split'].isna().sum())} task compounds"
        )

    observed = set(result["split"].astype(str).unique())
    expected = {"train", "validation", "test"}
    if observed != expected:
        raise ValueError(
            f"{path}: expected split values {sorted(expected)}, "
            f"found {sorted(observed)}"
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
        group_sets = {
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
            overlap = group_sets[left] & group_sets[right]
            if overlap:
                raise ValueError(
                    f"{path}: scaffold leakage for {left}/{right}: "
                    f"{len(overlap)} shared keys"
                )

    print(f"\nLoaded {split_mode} split seed {split_seed}: {path}")
    print(
        result.groupby("split")[LABEL_COL]
        .agg(["size", "sum", "mean"])
        .rename(columns={"sum": "n_positive", "mean": "positive_fraction"})
        .to_string()
    )
    return result


def split_qc_records(
    split_task: pd.DataFrame,
    split_mode: str,
    split_seed: int,
) -> list[dict]:
    rows = []
    for split_name in ["train", "validation", "test"]:
        subset = split_task[split_task["split"].eq(split_name)]
        rows.append(
            {
                "task": TASK_NAME,
                "split_mode": split_mode,
                "split_seed": split_seed,
                "split": split_name,
                "n_compounds": int(len(subset)),
                "n_negative": int((subset[LABEL_COL] == 0).sum()),
                "n_positive": int((subset[LABEL_COL] == 1).sum()),
                "positive_fraction": float(subset[LABEL_COL].mean()),
                "n_scaffold_keys": int(
                    subset["scaffold_split_key"].astype(str).nunique()
                ),
            }
        )
    return rows


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


def fm_pattern_available(pattern: str, feature_dir: str) -> tuple[bool, str]:
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
        raise ValueError(f"{stem}: expected 2D embeddings, found {X_all.shape}")
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
        row_lookup = pd.Series(
            np.arange(len(metadata), dtype=np.int64),
            index=metadata["inchikey"],
        )
        rows = row_lookup.reindex(task_keys)
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
                    f"RDKit failed while regenerating a validated task molecule "
                    f"at task row {start + local_row}: {smiles}"
                )
            fingerprint = generator.GetFingerprint(mol)
            on_bits = list(fingerprint.GetOnBits())
            row_indices.extend([local_row] * len(on_bits))
            column_indices.extend(on_bits)

        data = np.ones(len(row_indices), dtype=np.uint8)
        chunk = sparse.csr_matrix(
            (
                data,
                (
                    np.asarray(row_indices, dtype=np.int32),
                    np.asarray(column_indices, dtype=np.int32),
                ),
            ),
            shape=(stop - start, n_bits),
            dtype=np.uint8,
        )
        chunks.append(chunk)
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
                f"{feature_name}: expected shape {(len(task), n_bits)}, "
                f"found {X.shape}"
            )
        if len(metadata) != len(task):
            raise ValueError(
                f"{feature_name}: metadata rows {len(metadata)} != {len(task)}"
            )
        if not np.array_equal(
            metadata["inchikey"].astype(str).to_numpy(),
            task["inchikey"].astype(str).to_numpy(),
        ):
            raise ValueError(
                f"{feature_name}: cached fingerprint metadata is not aligned "
                "to the current task table"
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


def prepare_split_features(
    X,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    feature_type: str,
):
    if feature_type == "dense_embedding":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_validation = scaler.transform(X[validation_idx])
        return scaler, X_train, X_validation

    if feature_type == "fingerprint":
        return None, X[train_idx], X[validation_idx]

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
    probability = classifier.predict_proba(X_eval)[:, 1]
    return np.asarray(probability, dtype=np.float64)


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
        "true_positive_fraction": float(np.mean(y_true)),
        "predicted_positive_fraction": float(np.mean(prediction)),
    }


def compute_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    metrics = probability_metrics(y_true, probability)
    metrics.update(threshold_metrics(y_true, probability, threshold))
    return metrics


def tune_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold_grid: Iterable[float],
) -> tuple[float, dict[str, float]]:
    best_threshold = None
    best_metrics = None
    best_score = -np.inf
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
            metrics["f1"] > best_score
            or (
                np.isclose(metrics["f1"], best_score)
                and prevalence_gap < best_prevalence_gap
            )
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = metrics["f1"]
            best_prevalence_gap = prevalence_gap

    if best_threshold is None or best_metrics is None:
        raise RuntimeError("Threshold selection failed")
    return best_threshold, best_metrics


def fit_select_c_and_threshold(
    X,
    y: np.ndarray,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    feature_type: str,
    seed: int,
    C_grid: Iterable[float],
    threshold_grid: Iterable[float],
    max_iter: int,
    tol: float,
    class_weight: str,
):
    scaler, X_train, X_validation = prepare_split_features(
        X=X,
        train_idx=train_idx,
        validation_idx=validation_idx,
        feature_type=feature_type,
    )

    best = None
    best_auprc = -np.inf
    best_auroc = -np.inf
    y_train = y[train_idx]
    y_validation = y[validation_idx]

    for C in C_grid:
        classifier = build_classifier(
            C=float(C),
            seed=seed,
            max_iter=max_iter,
            tol=tol,
            class_weight=class_weight,
        )
        classifier.fit(X_train, y_train)
        probability = classifier.predict_proba(X_validation)[:, 1]
        metrics = probability_metrics(y_validation, probability)

        if (
            metrics["auprc"] > best_auprc
            or (
                np.isclose(metrics["auprc"], best_auprc)
                and metrics["auroc"] > best_auroc
            )
        ):
            best = {
                "classifier": classifier,
                "scaler": scaler,
                "selected_C": float(C),
                "validation_probability": np.asarray(
                    probability,
                    dtype=np.float64,
                ),
                "validation_auprc_for_selected_C": metrics["auprc"],
                "validation_auroc_for_selected_C": metrics["auroc"],
            }
            best_auprc = metrics["auprc"]
            best_auroc = metrics["auroc"]

    if best is None:
        raise RuntimeError("No valid C candidate was selected")

    threshold, threshold_metrics_value = tune_threshold(
        y_validation,
        best["validation_probability"],
        threshold_grid,
    )
    best["tuned_threshold"] = threshold
    best["validation_best_f1_at_tuned_threshold"] = (
        threshold_metrics_value["f1"]
    )

    del X_train
    del X_validation
    gc.collect()
    return best


def evaluate_row(
    feature: str,
    feature_type: str,
    model_name: str,
    split_mode: str,
    split_seed: int,
    eval_set: str,
    n_train: int,
    y_true: np.ndarray,
    probability: np.ndarray,
    tuned_threshold: float,
    metadata: dict,
) -> dict:
    row = {
        "task": TASK_NAME,
        "feature": feature,
        "feature_type": feature_type,
        "model": model_name,
        "split_mode": split_mode,
        "split_seed": split_seed,
        "eval_set": eval_set,
        "n_train": int(n_train),
        "n_eval": int(len(y_true)),
        "threshold_strategy": (
            "global_threshold_tuned_on_validation_f1"
        ),
        "tuned_threshold": float(tuned_threshold),
    }
    row.update(metadata)
    row.update(compute_metrics(y_true, probability, tuned_threshold))
    return row


def add_gain_columns(
    runs: pd.DataFrame,
    individual_features: list[str],
) -> pd.DataFrame:
    result = runs.copy()
    for metric in ["auprc", "auroc", "f1"]:
        result[f"gain_vs_morgan_r2_{metric}"] = np.nan
        result[f"gain_vs_best_individual_{metric}"] = np.nan

    for (mode, seed, eval_set), subset in result.groupby(
        ["split_mode", "split_seed", "eval_set"],
        dropna=False,
    ):
        morgan = subset[subset["feature"] == "morgan_r2_2048"]
        if morgan.empty:
            continue

        mask = (
            result["split_mode"].eq(mode)
            & result["split_seed"].eq(seed)
            & result["eval_set"].eq(eval_set)
        )

        for metric in ["auprc", "auroc", "f1"]:
            morgan_value = float(morgan[metric].iloc[0])
            best_individual = float(
                subset.loc[
                    subset["feature"].isin(individual_features),
                    metric,
                ].max()
            )
            result.loc[mask, f"gain_vs_morgan_r2_{metric}"] = (
                result.loc[mask, metric] - morgan_value
            )
            result.loc[mask, f"gain_vs_best_individual_{metric}"] = (
                result.loc[mask, metric] - best_individual
            )

    return result


def aggregate_main_statistics(runs: pd.DataFrame) -> pd.DataFrame:
    test = runs[runs["eval_set"] == "test"].copy()
    group_columns = [
        "task",
        "feature",
        "feature_type",
        "model",
        "split_mode",
        "eval_set",
        "threshold_strategy",
    ]
    metric_columns = [
        column for column in PRIMARY_METRICS if column in test.columns
    ]
    metric_columns.extend(
        column
        for column in test.columns
        if column.startswith("gain_vs_")
    )

    rows = []
    for keys, subset in test.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_split_replicates"] = int(
            subset["split_seed"].nunique()
        )

        for metric in metric_columns:
            values = subset[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_sd"] = (
                float(np.nanstd(values, ddof=1))
                if len(values) > 1
                else np.nan
            )
            row[f"{metric}_se"] = (
                row[f"{metric}_sd"] / np.sqrt(len(values))
                if len(values) > 1
                else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows)


def build_feature_specs(args: argparse.Namespace):
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
                print(f"Skipping FM pattern {pattern}: {reason}")
                continue
        specs.append(
            (
                pattern,
                "dense_embedding",
                ("embedding", pattern),
            )
        )

    if not specs:
        raise ValueError("No representations selected")
    return specs


def main() -> None:
    args = parse_args()

    if any(C <= 0 for C in args.C_grid):
        raise ValueError("All C values must be positive")
    if not all(0.0 < value < 1.0 for value in args.threshold_grid):
        raise ValueError("All thresholds must be between 0 and 1")
    if any(radius <= 0 for radius in args.fp_radii):
        raise ValueError("Morgan radii must be positive")
    if args.n_bits <= 0:
        raise ValueError("--n-bits must be positive")
    if args.fingerprint_chunk_size <= 0:
        raise ValueError("--fingerprint-chunk-size must be positive")

    task, y = load_task(args.task_file)

    experiments = []
    split_qc_rows = []
    for split_mode in requested_modes(args.split_mode):
        for split_seed in args.split_seeds:
            split_task = load_split(
                task=task,
                split_dir=args.split_dir,
                split_mode=split_mode,
                split_seed=split_seed,
            )
            split_qc_rows.extend(
                split_qc_records(
                    split_task,
                    split_mode,
                    split_seed,
                )
            )
            experiments.append(
                {
                    "split_mode": split_mode,
                    "split_seed": split_seed,
                    "task": split_task,
                    "probabilities": {
                        "validation": {},
                        "test": {},
                    },
                }
            )

    feature_specs = build_feature_specs(args)
    individual_features: list[str] = []
    feature_types: dict[str, str] = {}
    rows: list[dict] = []

    for requested_name, feature_type, loader in feature_specs:
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

        if feature_name in feature_types:
            raise ValueError(f"Duplicated feature name: {feature_name}")

        individual_features.append(feature_name)
        feature_types[feature_name] = feature_type

        for experiment in experiments:
            split_task = experiment["task"]
            split_mode = experiment["split_mode"]
            split_seed = experiment["split_seed"]

            train_idx = np.flatnonzero(
                split_task["split"].eq("train").to_numpy()
            )
            validation_idx = np.flatnonzero(
                split_task["split"].eq("validation").to_numpy()
            )
            test_idx = np.flatnonzero(
                split_task["split"].eq("test").to_numpy()
            )

            print(
                f"\n=== {feature_name}: {split_mode} "
                f"seed={split_seed} ==="
            )

            selected = fit_select_c_and_threshold(
                X=X,
                y=y,
                train_idx=train_idx,
                validation_idx=validation_idx,
                feature_type=feature_type,
                seed=split_seed,
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
                indices=test_idx,
            )

            experiment["probabilities"]["validation"][
                feature_name
            ] = validation_probability.astype(np.float32)
            experiment["probabilities"]["test"][
                feature_name
            ] = test_probability.astype(np.float32)

            metadata = {
                "C_grid": "|".join(str(float(C)) for C in args.C_grid),
                "selected_C": selected["selected_C"],
                "C_selection_metric": "validation_auprc",
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
                "validation_best_f1_at_tuned_threshold": selected[
                    "validation_best_f1_at_tuned_threshold"
                ],
                "class_weight": args.class_weight,
                "solver": "lbfgs",
                "tol": args.tol,
            }

            validation_row = evaluate_row(
                feature=feature_name,
                feature_type=feature_type,
                model_name="binary_logistic_regression",
                split_mode=split_mode,
                split_seed=split_seed,
                eval_set="validation",
                n_train=len(train_idx),
                y_true=y[validation_idx],
                probability=validation_probability,
                tuned_threshold=selected["tuned_threshold"],
                metadata=metadata,
            )
            test_row = evaluate_row(
                feature=feature_name,
                feature_type=feature_type,
                model_name="binary_logistic_regression",
                split_mode=split_mode,
                split_seed=split_seed,
                eval_set="test",
                n_train=len(train_idx),
                y_true=y[test_idx],
                probability=test_probability,
                tuned_threshold=selected["tuned_threshold"],
                metadata=metadata,
            )
            rows.extend([validation_row, test_row])

            for row in [validation_row, test_row]:
                print(
                    f"{feature_name:24s} {row['eval_set']:10s} "
                    f"C={selected['selected_C']} "
                    f"thr={selected['tuned_threshold']:.2f} "
                    f"AUPRC={row['auprc']:.4f} "
                    f"AUROC={row['auroc']:.4f} "
                    f"F1={row['f1']:.4f}"
                )

            del selected
            gc.collect()

        del X
        gc.collect()

    fm_features = [
        feature
        for feature in individual_features
        if not feature.startswith("morgan_r")
    ]

    for experiment in experiments:
        split_task = experiment["task"]
        split_mode = experiment["split_mode"]
        split_seed = experiment["split_seed"]
        validation_mask = split_task["split"].eq("validation").to_numpy()
        test_mask = split_task["split"].eq("test").to_numpy()
        n_train = int(split_task["split"].eq("train").sum())

        ensemble_specs = [
            (
                f"ensemble_morgan_r2_plus_{fm}",
                ["morgan_r2_2048", fm],
            )
            for fm in fm_features
            if "morgan_r2_2048" in individual_features
        ]
        ensemble_specs.append(
            (
                "ensemble_all_equal_probability",
                individual_features,
            )
        )

        for ensemble_name, members in ensemble_specs:
            validation_probability = np.mean(
                np.stack(
                    [
                        experiment["probabilities"]["validation"][member]
                        for member in members
                    ],
                    axis=0,
                ),
                axis=0,
            )
            test_probability = np.mean(
                np.stack(
                    [
                        experiment["probabilities"]["test"][member]
                        for member in members
                    ],
                    axis=0,
                ),
                axis=0,
            )

            threshold, validation_threshold_metrics = tune_threshold(
                y[validation_mask],
                validation_probability,
                args.threshold_grid,
            )
            metadata = {
                "C_grid": "base_models_independently_tuned",
                "selected_C": np.nan,
                "C_selection_metric": (
                    "base_models_validation_auprc"
                ),
                "validation_auprc_for_selected_C": np.nan,
                "validation_auroc_for_selected_C": np.nan,
                "threshold_grid": "|".join(
                    str(float(value))
                    for value in args.threshold_grid
                ),
                "validation_best_f1_at_tuned_threshold": (
                    validation_threshold_metrics["f1"]
                ),
                "class_weight": args.class_weight,
                "solver": "base_models_independently_fit",
                "tol": args.tol,
                "ensemble_members": "|".join(members),
            }

            for eval_set, eval_mask, probability in [
                (
                    "validation",
                    validation_mask,
                    validation_probability,
                ),
                ("test", test_mask, test_probability),
            ]:
                row = evaluate_row(
                    feature=ensemble_name,
                    feature_type="probability_ensemble",
                    model_name="equal_weight_probability_average",
                    split_mode=split_mode,
                    split_seed=split_seed,
                    eval_set=eval_set,
                    n_train=n_train,
                    y_true=y[eval_mask],
                    probability=probability,
                    tuned_threshold=threshold,
                    metadata=metadata,
                )
                rows.append(row)
                experiment["probabilities"][eval_set][
                    ensemble_name
                ] = probability.astype(np.float32)
                print(
                    f"{ensemble_name:48s} {eval_set:10s} "
                    f"thr={threshold:.2f} "
                    f"AUPRC={row['auprc']:.4f} "
                    f"AUROC={row['auroc']:.4f} "
                    f"F1={row['f1']:.4f}"
                )

    runs = pd.DataFrame(rows)
    runs = add_gain_columns(runs, individual_features)
    main_statistics = aggregate_main_statistics(runs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_path = (
        output_dir
        / "np_likeness_full_ensemble_repeated_runs.tsv"
    )
    statistics_path = (
        output_dir
        / "np_likeness_full_ensemble_main_statistics.tsv"
    )
    split_qc_path = (
        output_dir
        / "np_likeness_full_ensemble_split_qc.tsv"
    )
    config_path = (
        output_dir
        / "np_likeness_full_ensemble_config.json"
    )

    runs.to_csv(runs_path, sep="\t", index=False)
    main_statistics.to_csv(
        statistics_path,
        sep="\t",
        index=False,
    )
    pd.DataFrame(split_qc_rows).to_csv(
        split_qc_path,
        sep="\t",
        index=False,
    )

    config = {
        "script_version": SCRIPT_VERSION,
        "task_file": args.task_file,
        "split_dir": args.split_dir,
        "feature_dir": args.feature_dir,
        "split_modes": requested_modes(args.split_mode),
        "split_seeds": args.split_seeds,
        "individual_features": individual_features,
        "C_grid": args.C_grid,
        "threshold_grid": args.threshold_grid,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "class_weight": args.class_weight,
        "primary_metrics": PRIMARY_METRICS,
    }
    config_path.write_text(json.dumps(config, indent=2))

    prediction_path = (
        output_dir
        / "np_likeness_full_ensemble_predictions.npz"
    )
    if not args.skip_predictions:
        payload = {
            "script_version": np.asarray(
                [SCRIPT_VERSION],
                dtype=object,
            ),
            "task": np.asarray([TASK_NAME], dtype=object),
            "feature_order": np.asarray(
                individual_features,
                dtype=object,
            ),
        }

        for experiment in experiments:
            split_task = experiment["task"]
            split_key = (
                f"{experiment['split_mode']}_"
                f"seed{experiment['split_seed']}"
            )
            for eval_set in ["validation", "test"]:
                mask = split_task["split"].eq(eval_set).to_numpy()
                payload[f"{split_key}_{eval_set}_ids"] = (
                    split_task.loc[mask, ID_COL]
                    .astype(str)
                    .to_numpy()
                )
                payload[f"{split_key}_{eval_set}_y_true"] = y[mask]
                for feature, probability in experiment[
                    "probabilities"
                ][eval_set].items():
                    payload[
                        f"{split_key}_{eval_set}_"
                        f"{feature}_probability"
                    ] = probability.astype(np.float32)

        np.savez_compressed(prediction_path, **payload)

    print("\nWrote:")
    print(runs_path)
    print(statistics_path)
    print(split_qc_path)
    print(config_path)
    if not args.skip_predictions:
        print(prediction_path)

    if not main_statistics.empty:
        display_columns = [
            "split_mode",
            "feature",
            "n_split_replicates",
            "auprc_mean",
            "auprc_sd",
            "auprc_se",
            "auroc_mean",
            "auroc_sd",
            "f1_mean",
            "balanced_accuracy_mean",
        ]
        display_columns = [
            column
            for column in display_columns
            if column in main_statistics.columns
        ]
        print("\n=== Main test statistics ===")
        print(
            main_statistics.sort_values(
                ["split_mode", "auprc_mean"],
                ascending=[True, False],
            )[display_columns].to_string(index=False)
        )


if __name__ == "__main__":
    main()

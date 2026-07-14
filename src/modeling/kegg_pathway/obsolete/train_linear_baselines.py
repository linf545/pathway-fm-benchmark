#!/usr/bin/env python

"""
Train linear baseline classifiers for Bao plant pathway prediction.

Task:
    Predict Bao plant natural-product pathway labels from molecular features.

Inputs:
    data/features/bao_plant_pathway/
        metadata.tsv
        labels.npy
        label_columns.txt
        morgan_r2_2048.npz
        morgan_r3_2048.npz
        rdkit_descriptors.tsv
        full/*.embeddings.npy
        full/*.metadata.tsv

    data/tasks/bao_plant_pathway/bao_plant_pathway_splits.tsv
    data/splits/bao_plant_pathway/scaffold_split.tsv

Outputs:
    results/bao_pathway/linear_baseline_metrics.tsv
    results/bao_pathway/linear_baseline_per_label_metrics.tsv
    results/bao_pathway/predictions/*.npz
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/private/vina/FM_bench")

FEATURE_DIR = ROOT / "data/features/bao_plant_pathway"
TASK_DIR = ROOT / "data/tasks/bao_plant_pathway"
SPLIT_DIR = ROOT / "data/splits/bao_plant_pathway"

OUT_DIR = ROOT / "results/bao_pathway"
PRED_DIR = OUT_DIR / "predictions"


FEATURE_PATHS = {
    "morgan_r2_2048": FEATURE_DIR / "morgan_r2_2048.npz",
    "morgan_r3_2048": FEATURE_DIR / "morgan_r3_2048.npz",
    "rdkit_descriptors": FEATURE_DIR / "rdkit_descriptors.tsv",

    "chemberta_77m_mlm": FEATURE_DIR / "full/chemberta_77m_mlm.embeddings.npy",
    "chemberta_100m_mlm": FEATURE_DIR / "full/chemberta_100m_mlm.embeddings.npy",

    "molclr_gcn": FEATURE_DIR / "full/molclr_gcn.embeddings.npy",
    "molclr_gin": FEATURE_DIR / "full/molclr_gin.embeddings.npy",

    "molformer_c3_1_1b": FEATURE_DIR / "full/molformer_c3_1_1b.embeddings.npy",

    "nafm_contrast": FEATURE_DIR / "full/nafm_contrast.embeddings.npy",
    "nafm_graph_mean": FEATURE_DIR / "full/nafm_graph_mean.embeddings.npy",
}


ID_COL = "task_compound_id"


def load_labels_and_metadata():
    metadata = pd.read_csv(FEATURE_DIR / "metadata.tsv", sep="\t")
    labels = np.load(FEATURE_DIR / "labels.npy").astype(int)

    with open(FEATURE_DIR / "label_columns.txt") as f:
        label_names = [line.strip() for line in f if line.strip()]

    if ID_COL not in metadata.columns:
        raise ValueError(f"{FEATURE_DIR / 'metadata.tsv'} does not contain {ID_COL}")

    if labels.shape[0] != metadata.shape[0]:
        raise ValueError(
            f"labels.npy has {labels.shape[0]} rows, "
            f"but metadata.tsv has {metadata.shape[0]} rows"
        )

    if labels.shape[1] != len(label_names):
        raise ValueError(
            f"labels.npy has {labels.shape[1]} columns, "
            f"but label_columns.txt has {len(label_names)} names"
        )

    return metadata, labels, label_names


def load_split_table(metadata: pd.DataFrame) -> pd.DataFrame:
    split_table = metadata[[ID_COL]].copy()

    task_split_file = TASK_DIR / "bao_plant_pathway_splits.tsv"
    if task_split_file.exists():
        s = pd.read_csv(task_split_file, sep="\t")
        if ID_COL not in s.columns:
            raise ValueError(f"{task_split_file} does not contain {ID_COL}")
        s = s.drop_duplicates(ID_COL)
        split_table = split_table.merge(s, on=ID_COL, how="left")

    scaffold_split_file = SPLIT_DIR / "scaffold_split.tsv"
    if scaffold_split_file.exists():
        s = pd.read_csv(scaffold_split_file, sep="\t")
        if ID_COL not in s.columns:
            raise ValueError(f"{scaffold_split_file} does not contain {ID_COL}")
        keep = [c for c in [ID_COL, "scaffold_split"] if c in s.columns]
        s = s[keep].drop_duplicates(ID_COL)
        split_table = split_table.merge(s, on=ID_COL, how="left", suffixes=("", "_scaffold_file"))

    if "dataset_name" in metadata.columns and "dataset_name" not in split_table.columns:
        split_table["dataset_name"] = metadata["dataset_name"].values

    return split_table


def choose_split_column(split_table: pd.DataFrame, split_name: str) -> str:
    columns = list(split_table.columns)
    lower = {c: c.lower() for c in columns}

    if split_name == "random":
        preferred = [
            "random_split",
            "split_random",
            "bao_random_split",
            "data_split",
            "split",
        ]
        for c in preferred:
            if c in split_table.columns:
                return c

        candidates = [
            c for c in columns
            if "random" in lower[c] and "split" in lower[c]
        ]

    elif split_name == "scaffold":
        preferred = [
            "scaffold_split",
            "split_scaffold",
            "bao_scaffold_split",
        ]
        for c in preferred:
            if c in split_table.columns:
                return c

        candidates = [
            c for c in columns
            if "scaffold" in lower[c] and "split" in lower[c]
        ]

    else:
        raise ValueError(f"Unknown split name: {split_name}")

    if not candidates:
        raise ValueError(
            f"Could not find a column for {split_name} split. "
            f"Available columns: {columns}"
        )

    return candidates[0]


def normalize_split_values(split_table: pd.DataFrame, split_col: str) -> pd.Series:
    raw = split_table[split_col].astype(str).str.lower()

    if "dataset_name" in split_table.columns:
        dataset = split_table["dataset_name"].astype(str).str.lower()
    else:
        dataset = pd.Series([""] * len(split_table), index=split_table.index)

    split = pd.Series(index=split_table.index, dtype=object)

    external_mask = (
        dataset.str.contains("external", na=False)
        | dataset.str.contains("bao_b", na=False)
        | dataset.str.contains("b_external", na=False)
        | raw.str.contains("external", na=False)
        | raw.str.contains("test", na=False)
    )

    split[external_mask] = "test"

    split[
        raw.str.contains("train", na=False)
        & ~raw.str.contains("val", na=False)
        & ~external_mask
    ] = "train"

    split[
        raw.str.contains("val", na=False)
        & ~external_mask
    ] = "val"

    # Handle compact labels if they exist.
    split[raw.isin(["tr", "training"])] = "train"
    split[raw.isin(["va", "valid", "validation"])] = "val"
    split[raw.isin(["te", "testing"])] = "test"

    return split


def load_sparse_feature(path: Path):
    X = sparse.load_npz(path)
    return X, True, np.zeros(X.shape[0], dtype=bool)


def load_rdkit_descriptors(path: Path, metadata: pd.DataFrame):
    df = pd.read_csv(path, sep="\t")

    if ID_COL in df.columns:
        df = metadata[[ID_COL]].merge(df, on=ID_COL, how="left")

    non_feature_cols = {
        ID_COL,
        "dataset_name",
        "canonical_smiles",
        "smiles",
        "inchikey",
        "source",
        "source_dataset",
    }

    candidate_cols = [
        c for c in df.columns
        if c not in non_feature_cols
    ]

    numeric_cols = [
        c for c in candidate_cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(numeric_cols) == 0:
        raise ValueError(f"No numeric descriptor columns found in {path}")

    X = df[numeric_cols].to_numpy(dtype=float)
    missing_rows = np.isnan(X).all(axis=1)

    return X, False, missing_rows


def _clean_key_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )


def _choose_alignment_key(global_meta: pd.DataFrame, feature_meta: pd.DataFrame):
    """
    Choose a safe identifier for aligning embedding rows to the Bao task table.

    Preference:
        1. exact compound identifiers
        2. InChIKey
        3. canonical SMILES / SMILES

    The key must be unique in both tables. This avoids silent many-to-one alignment.
    """

    candidate_pairs = [
        ("task_compound_id", "task_compound_id"),
        ("compound_id", "compound_id"),
        ("master_compound_id", "master_compound_id"),
        ("bao_compound_id", "bao_compound_id"),
        ("source_compound_id", "source_compound_id"),
        ("inchikey", "inchikey"),
        ("canonical_smiles", "canonical_smiles"),
        ("canonical_smiles", "smiles"),
        ("smiles", "canonical_smiles"),
        ("smiles", "smiles"),
    ]

    for global_col, feature_col in candidate_pairs:
        if global_col not in global_meta.columns:
            continue
        if feature_col not in feature_meta.columns:
            continue

        g = _clean_key_series(global_meta[global_col])
        f = _clean_key_series(feature_meta[feature_col])

        if g.notna().sum() == 0 or f.notna().sum() == 0:
            continue

        if g.duplicated().any() or f.duplicated().any():
            continue

        overlap = len(set(g.dropna()) & set(f.dropna()))

        if overlap > 0:
            return global_col, feature_col, overlap

    return None, None, 0


def load_embedding_feature(path: Path, metadata: pd.DataFrame):
    X_raw = np.load(path)
    meta_file = path.with_name(path.name.replace(".embeddings.npy", ".metadata.tsv"))

    if not meta_file.exists():
        if X_raw.shape[0] != metadata.shape[0]:
            raise ValueError(
                f"{path.name} has {X_raw.shape[0]} rows, "
                f"but global metadata has {metadata.shape[0]} rows. "
                f"No metadata file was available for alignment."
            )
        return X_raw.astype(float), False, np.zeros(X_raw.shape[0], dtype=bool)

    feature_meta = pd.read_csv(meta_file, sep="\t")

    if X_raw.shape[0] != feature_meta.shape[0]:
        raise ValueError(
            f"{path.name} has {X_raw.shape[0]} rows, "
            f"but {meta_file.name} has {feature_meta.shape[0]} rows."
        )

    # If row counts match and task_compound_id is available, use direct order when possible.
    if ID_COL in feature_meta.columns and X_raw.shape[0] == metadata.shape[0]:
        global_ids = metadata[ID_COL].astype(str).to_numpy()
        feature_ids = feature_meta[ID_COL].astype(str).to_numpy()

        if np.array_equal(global_ids, feature_ids):
            return X_raw.astype(float), False, np.zeros(X_raw.shape[0], dtype=bool)

    global_col, feature_col, overlap = _choose_alignment_key(metadata, feature_meta)

    if global_col is None:
        if X_raw.shape[0] == metadata.shape[0]:
            print(
                f"  warning: {path.name}: no shared unique identifier found; "
                f"using row order because row counts match."
            )
            return X_raw.astype(float), False, np.zeros(X_raw.shape[0], dtype=bool)

        raise ValueError(
            f"Could not align {path.name} to global metadata.\n"
            f"Global metadata columns: {list(metadata.columns)}\n"
            f"Feature metadata columns: {list(feature_meta.columns)}\n"
            f"Embedding rows: {X_raw.shape[0]}; global rows: {metadata.shape[0]}"
        )

    print(
        f"  aligning {path.name} using "
        f"metadata.{global_col} -> feature_metadata.{feature_col}; "
        f"overlap={overlap}/{metadata.shape[0]}"
    )

    global_keys = _clean_key_series(metadata[global_col]).to_numpy()
    feature_keys = _clean_key_series(feature_meta[feature_col]).to_numpy()

    key_to_row = {
        key: i
        for i, key in enumerate(feature_keys)
        if pd.notna(key)
    }

    X = np.full((metadata.shape[0], X_raw.shape[1]), np.nan, dtype=float)
    missing = np.zeros(metadata.shape[0], dtype=bool)

    for i, key in enumerate(global_keys):
        j = key_to_row.get(key)
        if j is None:
            missing[i] = True
        else:
            X[i, :] = X_raw[j, :]

    missing_n = int(missing.sum())
    if missing_n > 0:
        print(f"  warning: {path.name}: {missing_n} Bao task compounds lack embeddings")

    return X, False, missing

def load_feature(feature_name: str, metadata: pd.DataFrame):
    path = FEATURE_PATHS[feature_name]

    if not path.exists():
        raise FileNotFoundError(path)

    if feature_name.startswith("morgan"):
        return load_sparse_feature(path)

    if feature_name == "rdkit_descriptors":
        return load_rdkit_descriptors(path, metadata)

    return load_embedding_feature(path, metadata)


def make_linear_classifier(is_sparse: bool):
    if is_sparse:
        return make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000,
            ),
        )

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
        ),
    )


def train_independent_binary_classifiers(
    X_train,
    y_train: np.ndarray,
    X_eval,
    is_sparse: bool,
):
    base_model = make_linear_classifier(is_sparse)

    n_labels = y_train.shape[1]
    y_pred = np.zeros((X_eval.shape[0], n_labels), dtype=int)
    y_score = np.full((X_eval.shape[0], n_labels), np.nan, dtype=float)

    for j in range(n_labels):
        y_j = y_train[:, j]
        unique_values = np.unique(y_j)

        # If a pathway label has only one class in training data, use a constant prediction.
        if len(unique_values) == 1:
            y_pred[:, j] = int(unique_values[0])
            y_score[:, j] = float(unique_values[0])
            continue

        model = clone(base_model)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            model.fit(X_train, y_j)

        y_pred[:, j] = model.predict(X_eval).astype(int)

        if hasattr(model[-1], "predict_proba"):
            y_score[:, j] = model.predict_proba(X_eval)[:, 1]

    return y_pred, y_score


def summarize_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n_eval": int(y_true.shape[0]),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "sample_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "exact_match_accuracy": accuracy_score(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
    }


def summarize_per_label(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> list[dict]:
    rows = []

    for j, label in enumerate(label_names):
        rows.append({
            "label": label,
            "support": int(y_true[:, j].sum()),
            "f1": f1_score(y_true[:, j], y_pred[:, j], zero_division=0),
            "precision": precision_score(y_true[:, j], y_pred[:, j], zero_division=0),
            "recall": recall_score(y_true[:, j], y_pred[:, j], zero_division=0),
        })

    return rows


def save_predictions(
    feature_name: str,
    split_name: str,
    eval_name: str,
    compound_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
):
    out_file = PRED_DIR / f"{feature_name}.{split_name}.{eval_name}.npz"

    np.savez_compressed(
        out_file,
        compound_ids=compound_ids.astype(str),
        y_true=y_true.astype(int),
        y_pred=y_pred.astype(int),
        y_score=y_score.astype(float),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(FEATURE_PATHS.keys()),
        help="Feature sets to evaluate.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["random", "scaffold"],
        choices=["random", "scaffold"],
        help="Data splits to evaluate.",
    )
    parser.add_argument(
        "--output-prefix",
        default="linear_baseline",
        help="Prefix for output result files.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    metadata, labels, label_names = load_labels_and_metadata()
    split_table = load_split_table(metadata)

    metric_rows = []
    per_label_rows = []

    for split_name in args.splits:
        split_col = choose_split_column(split_table, split_name)
        split_values = normalize_split_values(split_table, split_col)

        train_idx = np.where(split_values.to_numpy() == "train")[0]
        val_idx = np.where(split_values.to_numpy() == "val")[0]
        test_idx = np.where(split_values.to_numpy() == "test")[0]

        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(
                f"{split_name} split has train={len(train_idx)}, val={len(val_idx)}. "
                f"Check split column: {split_col}"
            )

        print()
        print(f"=== Split: {split_name} ===")
        print(f"split column: {split_col}")
        print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

        for feature_name in args.features:
            print()
            print(f"--- Feature: {feature_name} ---")

            X, is_sparse, missing_feature_rows = load_feature(feature_name, metadata)

            if X.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"{feature_name}: feature rows={X.shape[0]}, "
                    f"label rows={labels.shape[0]}"
                )

            missing_n = int(missing_feature_rows.sum())

            # Validation evaluation:
            # Train on training subset, evaluate on validation subset.
            y_val_pred, y_val_score = train_independent_binary_classifiers(
                X[train_idx],
                labels[train_idx],
                X[val_idx],
                is_sparse,
            )

            val_metrics = summarize_metrics(labels[val_idx], y_val_pred)
            val_metrics.update({
                "task": "bao_pathway",
                "split": split_name,
                "split_column": split_col,
                "eval_set": "validation",
                "feature": feature_name,
                "model": "linear_binary_classifiers",
                "n_train": int(len(train_idx)),
                "missing_feature_rows_total": missing_n,
            })
            metric_rows.append(val_metrics)

            for row in summarize_per_label(labels[val_idx], y_val_pred, label_names):
                row.update({
                    "task": "bao_pathway",
                    "split": split_name,
                    "eval_set": "validation",
                    "feature": feature_name,
                    "model": "linear_binary_classifiers",
                })
                per_label_rows.append(row)

            save_predictions(
                feature_name=feature_name,
                split_name=split_name,
                eval_name="validation",
                compound_ids=metadata.iloc[val_idx][ID_COL].astype(str).to_numpy(),
                y_true=labels[val_idx],
                y_pred=y_val_pred,
                y_score=y_val_score,
            )

            print(
                f"validation: macro_f1={val_metrics['macro_f1']:.4f} "
                f"micro_f1={val_metrics['micro_f1']:.4f} "
                f"sample_f1={val_metrics['sample_f1']:.4f}"
            )

            # External Bao B evaluation:
            # Train on training + validation subsets, evaluate on external test subset.
            if len(test_idx) > 0:
                trainval_idx = np.concatenate([train_idx, val_idx])

                y_test_pred, y_test_score = train_independent_binary_classifiers(
                    X[trainval_idx],
                    labels[trainval_idx],
                    X[test_idx],
                    is_sparse,
                )

                test_metrics = summarize_metrics(labels[test_idx], y_test_pred)
                test_metrics.update({
                    "task": "bao_pathway",
                    "split": split_name,
                    "split_column": split_col,
                    "eval_set": "bao_B_external_test",
                    "feature": feature_name,
                    "model": "linear_binary_classifiers",
                    "n_train": int(len(trainval_idx)),
                    "missing_feature_rows_total": missing_n,
                })
                metric_rows.append(test_metrics)

                for row in summarize_per_label(labels[test_idx], y_test_pred, label_names):
                    row.update({
                        "task": "bao_pathway",
                        "split": split_name,
                        "eval_set": "bao_B_external_test",
                        "feature": feature_name,
                        "model": "linear_binary_classifiers",
                    })
                    per_label_rows.append(row)

                save_predictions(
                    feature_name=feature_name,
                    split_name=split_name,
                    eval_name="bao_B_external_test",
                    compound_ids=metadata.iloc[test_idx][ID_COL].astype(str).to_numpy(),
                    y_true=labels[test_idx],
                    y_pred=y_test_pred,
                    y_score=y_test_score,
                )

                print(
                    f"Bao B external test: macro_f1={test_metrics['macro_f1']:.4f} "
                    f"micro_f1={test_metrics['micro_f1']:.4f} "
                    f"sample_f1={test_metrics['sample_f1']:.4f}"
                )

    metrics = pd.DataFrame(metric_rows)
    per_label = pd.DataFrame(per_label_rows)

    metric_cols = [
        "task",
        "split",
        "eval_set",
        "feature",
        "model",
        "n_train",
        "n_eval",
        "macro_f1",
        "micro_f1",
        "sample_f1",
        "macro_precision",
        "macro_recall",
        "exact_match_accuracy",
        "hamming_loss",
        "missing_feature_rows_total",
        "split_column",
    ]

    metrics = metrics[metric_cols]

    metrics.to_csv(OUT_DIR / f"{args.output_prefix}_metrics.tsv", sep="\t", index=False)
    per_label.to_csv(OUT_DIR / f"{args.output_prefix}_per_label_metrics.tsv", sep="\t", index=False)

    with open(OUT_DIR / f"{args.output_prefix}_run_config.json", "w") as f:
        json.dump(
            {
                "task": "bao_pathway",
                "features": args.features,
                "splits": args.splits,
                "model": "independent binary logistic regression classifiers",
                "feature_directory": str(FEATURE_DIR),
                "output_directory": str(OUT_DIR),
            },
            f,
            indent=2,
        )

    print()
    print("Saved:")
    print(OUT_DIR / f"{args.output_prefix}_metrics.tsv")
    print(OUT_DIR / f"{args.output_prefix}_per_label_metrics.tsv")
    print(OUT_DIR / f"{args.output_prefix}_run_config.json")
    print(PRED_DIR)


if __name__ == "__main__":
    main()

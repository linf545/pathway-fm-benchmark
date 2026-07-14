#!/usr/bin/env python

"""
Train tree-based classical molecular-feature baselines for Bao pathway prediction.

Task:
    Bao plant natural-product pathway prediction.

Features:
    - Morgan r2 fingerprints
    - Morgan r3 fingerprints
    - RDKit descriptors

Models:
    - RandomForestClassifier
    - ExtraTreesClassifier

Foundation-model embeddings are intentionally excluded.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline


ROOT = Path("/private/vina/FM_bench")
SCRIPT_DIR = ROOT / "src/modeling/bao_pathway"

sys.path.insert(0, str(SCRIPT_DIR))

from train_linear_baselines import (  # noqa: E402
    ID_COL,
    load_labels_and_metadata,
    load_split_table,
    choose_split_column,
    normalize_split_values,
    load_feature,
)


OUT_DIR = ROOT / "results/bao_pathway"
PRED_DIR = OUT_DIR / "predictions"

FEATURES = [
    "morgan_r2_2048",
    "morgan_r3_2048",
    "rdkit_descriptors",
]

MODEL_CONFIGS = {
    "random_forest": RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    ),
    "extra_trees": ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    ),
}


def prepare_feature_matrix(X, is_sparse: bool):
    """
    Tree models can use sparse matrices in some scikit-learn versions,
    but dense conversion is safer here because the dataset is small.
    """
    if is_sparse:
        X = X.toarray()

    X = np.asarray(X, dtype=float)

    return X


def make_model(model_name: str, has_missing_values: bool):
    base = clone(MODEL_CONFIGS[model_name])

    if has_missing_values:
        return make_pipeline(
            SimpleImputer(strategy="median"),
            base,
        )

    return base


def train_independent_binary_tree_models(
    X_train,
    y_train: np.ndarray,
    X_eval,
    model_name: str,
):
    """
    Train one binary tree model per pathway label.

    This keeps the multilabel task explicit:
        each pathway label is treated as an independent yes/no prediction.
    """
    n_labels = y_train.shape[1]
    y_pred = np.zeros((X_eval.shape[0], n_labels), dtype=int)
    y_score = np.full((X_eval.shape[0], n_labels), np.nan, dtype=float)

    has_missing_values = bool(np.isnan(X_train).any() or np.isnan(X_eval).any())

    for j in range(n_labels):
        y_j = y_train[:, j]
        unique_values = np.unique(y_j)

        if len(unique_values) == 1:
            y_pred[:, j] = int(unique_values[0])
            y_score[:, j] = float(unique_values[0])
            continue

        model = make_model(model_name, has_missing_values)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_j)

        y_pred[:, j] = model.predict(X_eval).astype(int)

        # Get probability for the positive class when available.
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_eval)
        else:
            proba = model[-1].predict_proba(model[:-1].transform(X_eval))

        if isinstance(proba, list):
            # Defensive fallback; should not happen because we train one label at a time.
            y_score[:, j] = proba[0][:, 1]
        else:
            if proba.shape[1] == 2:
                y_score[:, j] = proba[:, 1]
            else:
                y_score[:, j] = y_pred[:, j]

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


def summarize_per_label(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]):
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
    model_name: str,
    split_name: str,
    eval_name: str,
    compound_ids,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
):
    out_file = PRED_DIR / f"{feature_name}.{model_name}.{split_name}.{eval_name}.npz"

    np.savez_compressed(
        out_file,
        compound_ids=np.asarray(compound_ids).astype(str),
        y_true=y_true.astype(int),
        y_pred=y_pred.astype(int),
        y_score=y_score.astype(float),
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    metadata, labels, label_names = load_labels_and_metadata()
    split_table = load_split_table(metadata)

    metric_rows = []
    per_label_rows = []

    for split_name in ["random", "scaffold"]:
        split_col = choose_split_column(split_table, split_name)
        split_values = normalize_split_values(split_table, split_col)

        train_idx = np.where(split_values.to_numpy() == "train")[0]
        val_idx = np.where(split_values.to_numpy() == "val")[0]
        test_idx = np.where(split_values.to_numpy() == "test")[0]

        print()
        print(f"=== Split: {split_name} ===")
        print(f"split column: {split_col}")
        print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

        for feature_name in FEATURES:
            X_raw, is_sparse, missing_feature_rows = load_feature(feature_name, metadata)
            X = prepare_feature_matrix(X_raw, is_sparse)

            if X.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"{feature_name}: feature rows={X.shape[0]}, label rows={labels.shape[0]}"
                )

            for model_name in MODEL_CONFIGS:
                print()
                print(f"--- Feature: {feature_name} | Model: {model_name} ---")

                # Internal validation: train on train subset, evaluate on validation subset.
                y_val_pred, y_val_score = train_independent_binary_tree_models(
                    X[train_idx],
                    labels[train_idx],
                    X[val_idx],
                    model_name,
                )

                val_metrics = summarize_metrics(labels[val_idx], y_val_pred)
                val_metrics.update({
                    "task": "bao_pathway",
                    "split": split_name,
                    "split_column": split_col,
                    "eval_set": "validation",
                    "feature": feature_name,
                    "model": model_name,
                    "n_train": int(len(train_idx)),
                    "missing_feature_rows_total": int(missing_feature_rows.sum()),
                })
                metric_rows.append(val_metrics)

                for row in summarize_per_label(labels[val_idx], y_val_pred, label_names):
                    row.update({
                        "task": "bao_pathway",
                        "split": split_name,
                        "eval_set": "validation",
                        "feature": feature_name,
                        "model": model_name,
                    })
                    per_label_rows.append(row)

                save_predictions(
                    feature_name=feature_name,
                    model_name=model_name,
                    split_name=split_name,
                    eval_name="validation",
                    compound_ids=metadata.iloc[val_idx][ID_COL].to_numpy(),
                    y_true=labels[val_idx],
                    y_pred=y_val_pred,
                    y_score=y_val_score,
                )

                print(
                    f"validation: macro_f1={val_metrics['macro_f1']:.4f} "
                    f"micro_f1={val_metrics['micro_f1']:.4f} "
                    f"sample_f1={val_metrics['sample_f1']:.4f}"
                )

                # External Bao B: train on all Bao A trainval, evaluate on Bao B.
                if len(test_idx) > 0:
                    trainval_idx = np.concatenate([train_idx, val_idx])

                    y_test_pred, y_test_score = train_independent_binary_tree_models(
                        X[trainval_idx],
                        labels[trainval_idx],
                        X[test_idx],
                        model_name,
                    )

                    test_metrics = summarize_metrics(labels[test_idx], y_test_pred)
                    test_metrics.update({
                        "task": "bao_pathway",
                        "split": split_name,
                        "split_column": split_col,
                        "eval_set": "bao_B_external_test",
                        "feature": feature_name,
                        "model": model_name,
                        "n_train": int(len(trainval_idx)),
                        "missing_feature_rows_total": int(missing_feature_rows.sum()),
                    })
                    metric_rows.append(test_metrics)

                    for row in summarize_per_label(labels[test_idx], y_test_pred, label_names):
                        row.update({
                            "task": "bao_pathway",
                            "split": split_name,
                            "eval_set": "bao_B_external_test",
                            "feature": feature_name,
                            "model": model_name,
                        })
                        per_label_rows.append(row)

                    save_predictions(
                        feature_name=feature_name,
                        model_name=model_name,
                        split_name=split_name,
                        eval_name="bao_B_external_test",
                        compound_ids=metadata.iloc[test_idx][ID_COL].to_numpy(),
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

    metrics.to_csv(
        OUT_DIR / "fp_descriptor_tree_baseline_metrics.tsv",
        sep="\t",
        index=False,
    )
    per_label.to_csv(
        OUT_DIR / "fp_descriptor_tree_baseline_per_label_metrics.tsv",
        sep="\t",
        index=False,
    )

    print()
    print("Saved:")
    print(OUT_DIR / "fp_descriptor_tree_baseline_metrics.tsv")
    print(OUT_DIR / "fp_descriptor_tree_baseline_per_label_metrics.tsv")


if __name__ == "__main__":
    main()

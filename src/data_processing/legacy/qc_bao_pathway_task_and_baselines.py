#!/usr/bin/env python

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    hamming_loss,
)

PROJECT = Path("/private/vina/FM_bench")

META_PATH = PROJECT / "data/features/bao_plant_pathway/metadata.tsv"
LABELS_PATH = PROJECT / "data/features/bao_plant_pathway/labels.npy"
LABEL_NAMES_PATH = PROJECT / "data/features/bao_plant_pathway/label_columns.txt"

OUT_DIR = PROJECT / "qc/data_processing"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_label_names(path: Path):
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def multilabel_metrics(y_true, y_pred):
    return {
        "n_samples": int(y_true.shape[0]),
        "n_labels": int(y_true.shape[1]),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "sample_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "exact_match_accuracy": accuracy_score(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "hamming_accuracy": 1.0 - hamming_loss(y_true, y_pred),
    }


def get_split_schemes(meta: pd.DataFrame):
    """
    Return available split schemes as (scheme_name, column_name).

    Current task metadata may contain only one split column named 'split'.
    If scaffold_split/random_split columns are later added, they will also be used.
    """
    candidates = [
        ("metadata_split", "split"),
        ("random", "random_split"),
        ("random", "split_random"),
        ("scaffold", "scaffold_split"),
        ("scaffold", "split_scaffold"),
    ]

    schemes = []
    seen_cols = set()

    for scheme_name, col in candidates:
        if col in meta.columns and col not in seen_cols:
            values = set(meta[col].dropna().astype(str))
            if {"train", "val", "test"}.issubset(values) or values.intersection({"train", "val", "test"}):
                schemes.append((scheme_name, col))
                seen_cols.add(col)

    if not schemes:
        raise ValueError(
            "No usable split column found. Expected a column containing train/val/test. "
            f"Available columns: {list(meta.columns)}"
        )

    return schemes


def summarize_duplicates(meta: pd.DataFrame, col: str, out_prefix: str):
    if col not in meta.columns:
        return {"column": col, "available": False}

    s = meta[col].fillna("").astype(str)
    nonempty = s[s != ""]
    dup_values = nonempty[nonempty.duplicated(keep=False)]

    if len(dup_values) > 0:
        dup_df = meta.loc[s.isin(set(dup_values))].sort_values(col)
        dup_df.to_csv(OUT_DIR / f"{out_prefix}_duplicate_{col}.tsv", sep="\t", index=False)

    return {
        "column": col,
        "available": True,
        "nonempty": int(nonempty.shape[0]),
        "unique_nonempty": int(nonempty.nunique()),
        "duplicate_rows": int(dup_values.shape[0]),
        "duplicate_values": int(dup_values.nunique()),
    }


def baseline_predictions(y_train, y_eval, mode, rng=None):
    n_eval, n_labels = y_eval.shape
    prevalences = y_train.mean(axis=0)

    if mode == "all_negative":
        return np.zeros_like(y_eval, dtype=int)

    if mode == "train_prevalence_random":
        if rng is None:
            rng = np.random.default_rng(1)
        return (rng.random((n_eval, n_labels)) < prevalences[None, :]).astype(int)

    if mode == "top_k_common":
        mean_cardinality = y_train.sum(axis=1).mean()
        k = max(1, int(round(mean_cardinality)))
        top = np.argsort(-prevalences)[:k]
        y_pred = np.zeros_like(y_eval, dtype=int)
        y_pred[:, top] = 1
        return y_pred

    if mode == "top_1_common":
        top = int(np.argmax(prevalences))
        y_pred = np.zeros_like(y_eval, dtype=int)
        y_pred[:, top] = 1
        return y_pred

    raise ValueError(mode)


def run_baselines(meta, labels, label_names):
    rows = []

    for split_name, split_col in get_split_schemes(meta):

        train_mask = meta[split_col].astype(str).eq("train").to_numpy()
        val_mask = meta[split_col].astype(str).eq("val").to_numpy()
        test_mask = meta[split_col].astype(str).isin(["test", "external_test", "B_external_test"]).to_numpy()

        y_train = labels[train_mask]
        eval_sets = {
            "validation": labels[val_mask],
            "Bao_B_external_test": labels[test_mask],
        }

        for eval_name, y_eval in eval_sets.items():
            if y_train.shape[0] == 0 or y_eval.shape[0] == 0:
                continue

            for mode in ["all_negative", "top_1_common", "top_k_common"]:
                y_pred = baseline_predictions(y_train, y_eval, mode)
                m = multilabel_metrics(y_eval, y_pred)
                m.update({
                    "split": split_name,
                    "split_column": split_col,
                    "eval_set": eval_name,
                    "baseline": mode,
                    "random_replicate": "",
                })
                rows.append(m)

            # stochastic prevalence baseline: mean across replicates
            replicate_metrics = []
            for r in range(100):
                rng = np.random.default_rng(1000 + r)
                y_pred = baseline_predictions(y_train, y_eval, "train_prevalence_random", rng=rng)
                m = multilabel_metrics(y_eval, y_pred)
                replicate_metrics.append(m)

            rep_df = pd.DataFrame(replicate_metrics)
            mean_row = rep_df.mean(numeric_only=True).to_dict()
            std_row = rep_df.std(numeric_only=True).add_suffix("_std").to_dict()
            out = {**mean_row, **std_row}
            out.update({
                "split": split_name,
                "split_column": split_col,
                "eval_set": eval_name,
                "baseline": "train_prevalence_random_mean_100x",
                "random_replicate": "mean_100",
            })
            rows.append(out)

    return pd.DataFrame(rows)


def main():
    meta = pd.read_csv(META_PATH, sep="\t")
    labels = np.load(LABELS_PATH)
    label_names = read_label_names(LABEL_NAMES_PATH)

    if labels.ndim != 2:
        raise ValueError(f"labels should be 2D, got shape {labels.shape}")

    if labels.shape[0] != meta.shape[0]:
        raise ValueError(f"metadata rows {meta.shape[0]} != label rows {labels.shape[0]}")

    if labels.shape[1] != len(label_names):
        raise ValueError(f"label columns {labels.shape[1]} != label name count {len(label_names)}")

    labels = labels.astype(int)

    # Basic data summary
    label_counts = labels.sum(axis=0)
    label_freq = label_counts / labels.shape[0]
    cardinality = labels.sum(axis=1)

    label_df = pd.DataFrame({
        "label_index": range(len(label_names)),
        "label_name": label_names,
        "positive_count": label_counts.astype(int),
        "positive_frequency": label_freq,
    }).sort_values("positive_count", ascending=False)

    label_df.to_csv(OUT_DIR / "label_counts.tsv", sep="\t", index=False)

    sample_label_df = pd.DataFrame({
        "n_positive_labels": cardinality.astype(int)
    })
    sample_label_df.value_counts().reset_index(name="n_compounds").sort_values(
        "n_positive_labels"
    ).to_csv(OUT_DIR / "label_cardinality_distribution.tsv", sep="\t", index=False)

    # Split summaries
    split_rows = []
    for split_name, split_col in get_split_schemes(meta):

        for split_value, idx in meta.groupby(split_col).groups.items():
            idx = np.array(list(idx))
            y = labels[idx]
            split_rows.append({
                "split_scheme": split_name,
                "split_column": split_col,
                "split_value": split_value,
                "n_compounds": int(len(idx)),
                "mean_label_cardinality": float(y.sum(axis=1).mean()),
                "label_density": float(y.mean()),
                "zero_label_compounds": int((y.sum(axis=1) == 0).sum()),
            })

    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(OUT_DIR / "split_summary.tsv", sep="\t", index=False)

    # Duplicate checks
    duplicate_summary = []
    for col in ["task_compound_id", "inchikey", "canonical_smiles", "smiles"]:
        duplicate_summary.append(summarize_duplicates(meta, col, "all_metadata"))
    pd.DataFrame(duplicate_summary).to_csv(
        OUT_DIR / "duplicate_summary.tsv", sep="\t", index=False
    )

    # A/B overlap checks by split column
    overlap_rows = []
    for split_name, split_col in get_split_schemes(meta):

        trainval_mask = meta[split_col].astype(str).isin(["train", "val"])
        test_mask = meta[split_col].astype(str).isin(["test", "external_test", "B_external_test"])

        for col in ["inchikey", "canonical_smiles"]:
            if col not in meta.columns:
                continue

            trainval = set(meta.loc[trainval_mask, col].dropna().astype(str))
            test = set(meta.loc[test_mask, col].dropna().astype(str))
            trainval.discard("")
            test.discard("")
            overlap = sorted(trainval & test)

            overlap_rows.append({
                "split_scheme": split_name,
                "split_column": split_col,
                "id_column": col,
                "trainval_unique": len(trainval),
                "test_unique": len(test),
                "overlap_count": len(overlap),
            })

            if overlap:
                pd.DataFrame({col: overlap}).to_csv(
                    OUT_DIR / f"{split_name}_trainval_test_overlap_{col}.tsv",
                    sep="\t",
                    index=False,
                )

    pd.DataFrame(overlap_rows).to_csv(
        OUT_DIR / "trainval_test_overlap_summary.tsv", sep="\t", index=False
    )

    # Raw high-level summary
    summary = {
        "metadata_path": str(META_PATH),
        "labels_path": str(LABELS_PATH),
        "label_names_path": str(LABEL_NAMES_PATH),
        "n_compounds": int(labels.shape[0]),
        "n_labels": int(labels.shape[1]),
        "total_positive_labels": int(labels.sum()),
        "label_density": float(labels.mean()),
        "mean_label_cardinality": float(cardinality.mean()),
        "median_label_cardinality": float(np.median(cardinality)),
        "min_label_cardinality": int(cardinality.min()),
        "max_label_cardinality": int(cardinality.max()),
        "zero_label_compounds": int((cardinality == 0).sum()),
        "metadata_columns": list(meta.columns),
        "label_names": label_names,
    }

    (OUT_DIR / "bao_processed_dataset_qc_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # Baselines
    baseline_df = run_baselines(meta, labels, label_names)
    baseline_df.to_csv(OUT_DIR / "naive_multilabel_baselines.tsv", sep="\t", index=False)

    print("\n=== BASIC SUMMARY ===")
    print(json.dumps(summary, indent=2))

    print("\n=== SPLIT SUMMARY ===")
    print(split_df.to_string(index=False))

    print("\n=== TOP LABEL COUNTS ===")
    print(label_df.head(20).to_string(index=False))

    print("\n=== DUPLICATE SUMMARY ===")
    print(pd.DataFrame(duplicate_summary).to_string(index=False))

    print("\n=== TRAIN/VAL VS TEST OVERLAP ===")
    print(pd.DataFrame(overlap_rows).to_string(index=False))

    print("\n=== NAIVE BASELINES ===")
    show_cols = [
        "split", "eval_set", "baseline",
        "macro_f1", "micro_f1", "sample_f1",
        "macro_precision", "macro_recall",
        "exact_match_accuracy", "hamming_loss", "hamming_accuracy",
    ]
    show_cols = [c for c in show_cols if c in baseline_df.columns]
    print(baseline_df[show_cols].to_string(index=False))

    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()

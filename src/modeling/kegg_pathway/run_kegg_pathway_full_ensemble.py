#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import argparse
import copy
import gc
import sys
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

# Import shared Task 2 splitting, tuning, feature-loading, and metric functions.
THIS_DIR = Path(__file__).resolve().parent
DATA_EFFICIENCY_DIR = THIS_DIR / "data_efficiency"
if str(DATA_EFFICIENCY_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_EFFICIENCY_DIR))

from run_kegg_pathway_data_efficiency import (  # noqa: E402
    ALL_METRICS,
    PRIMARY_METRICS,
    compute_metrics,
    create_or_load_split,
    default_threshold_grid,
    fit_select_c_and_threshold,
    load_aligned_embedding,
    load_task,
    make_morgan,
    requested_modes,
    split_qc_record,
    tune_threshold,
)

SCRIPT_VERSION = "kegg_pathway_full_ensemble_repeated_splits_v5"

# Full-training ensemble defaults are intentionally broader than the
# data-efficiency defaults. Exclude nafm_contrast by default because it is the
# projection-head output rather than the encoder-level NaFM representation.
DEFAULT_FULL_FM_PATTERNS = [
    "chemberta_77m_mlm",
    "chemberta_100m_mlm",
    "molformer_c3_1_1b",
    "nafm_graph_mean",
    "molclr_gcn",
    "molclr_gin",
    "graphmvp_gin",
    "molebert_gin",
]

DEFAULT_FP_RADII = [2, 3]


def fm_pattern_available(pattern: str, embedding_dir: str) -> tuple[bool, str]:
    embedding_dir = Path(embedding_dir)

    exact_embedding = embedding_dir / f"{pattern}.embeddings.npy"
    if exact_embedding.exists():
        stem = pattern
    else:
        matches = sorted(embedding_dir.glob(f"{pattern}*.embeddings.npy"))
        if len(matches) == 0:
            return False, f"missing embedding file for pattern {pattern}"
        if len(matches) > 1:
            return False, f"multiple embedding files for pattern {pattern}: {matches}"
        stem = matches[0].name.replace(".embeddings.npy", "")

    metadata = embedding_dir / f"{stem}.metadata.tsv"
    if not metadata.exists():
        return False, f"missing metadata file for stem {stem}"

    return True, "ok"


def build_full_feature_specs(args) -> list[tuple[str, str, tuple[str, object]]]:
    specs: list[tuple[str, str, tuple[str, object]]] = []

    for radius in args.fp_radii:
        specs.append(
            (
                f"morgan_r{radius}_{args.n_bits}",
                "fingerprint",
                ("morgan", int(radius)),
            )
        )

    for pattern in args.fm_patterns:
        if args.skip_missing_fm:
            available, reason = fm_pattern_available(pattern, args.embedding_dir)
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
        raise ValueError("No features selected. Check --fp-radii and --fm-patterns.")

    return specs


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Repeated-split full-training analysis for the Bao C/KEGG pathway dataset. "
            "Evaluates individual fixed-representation models, Morgan-r2/FM pair "
            "ensembles, and an equal-weight all-selected-feature ensemble. C and thresholds "
            "are tuned using validation only."
        )
    )
    parser.add_argument(
        "--compound-file",
        default="data/processed_bao/C_kegg/compounds_master.tsv",
    )
    parser.add_argument(
        "--label-file",
        default=(
            "data/processed_bao/C_kegg/"
            "bao_compound_labels_multilabel.tsv"
        ),
    )
    parser.add_argument(
        "--embedding-dir",
        default="data/features/bao_plant_pathway/full",
    )
    parser.add_argument(
        "--split-dir",
        default="data/splits/kegg_pathway",
    )
    parser.add_argument(
        "--output-dir",
        default="results/kegg_pathway/full",
    )
    parser.add_argument(
        "--split-mode",
        choices=["scaffold", "random", "both"],
        default="scaffold",
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
        default=DEFAULT_FULL_FM_PATTERNS,
        help=(
            "FM embedding stems to evaluate in the full ensemble script. "
            "This default is broader than the data-efficiency script."
        ),
    )
    parser.add_argument(
        "--fp-radii",
        nargs="+",
        type=int,
        default=DEFAULT_FP_RADII,
        help=(
            "Morgan fingerprint radii to include. With --n-bits 2048, "
            "the default gives morgan_r2_2048 and morgan_r3_2048."
        ),
    )
    parser.add_argument(
        "--skip-missing-fm",
        action="store_true",
        help=(
            "Skip FM patterns whose embedding/metadata files are absent. "
            "By default, missing FM files raise an error."
        ),
    )
    parser.add_argument(
        "--C-grid",
        nargs="+",
        type=float,
        default=[0.01, 0.1, 1.0],
        help="Logistic-regression C candidates. Selected using validation macro-AUPRC.",
    )
    parser.add_argument(
        "--threshold-grid",
        nargs="+",
        type=float,
        default=default_threshold_grid(),
        help="Candidate thresholds; selected using validation macro-F1.",
    )
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--scaffold-candidates", type=int, default=200)
    parser.add_argument("--scaffold-size-weight", type=float, default=1.0)
    parser.add_argument("--scaffold-label-weight", type=float, default=0.25)
    parser.add_argument("--overwrite-splits", action="store_true")
    return parser.parse_args()


def load_feature_matrix(task: pd.DataFrame, args, requested_name: str, loader: Tuple[str, object]):
    if loader[0] == "morgan":
        feature_name = requested_name
        X = make_morgan(task, loader[1], args.n_bits)
    else:
        feature_name, X = load_aligned_embedding(task, loader[1], args.embedding_dir)
    return feature_name, X


def evaluate_probability_row(
    feature: str,
    model_name: str,
    split_mode: str,
    split_seed: int,
    eval_set: str,
    n_train: int,
    n_eval: int,
    n_labels: int,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    tuned_threshold: float,
    metadata: dict,
):
    row = {
        "task": "kegg_pathway_11label",
        "feature": feature,
        "model": model_name,
        "split_mode": split_mode,
        "split_seed": split_seed,
        "eval_set": eval_set,
        "n_train": int(n_train),
        "n_eval": int(n_eval),
        "n_labels": int(n_labels),
        "threshold_strategy": metadata.get(
            "threshold_strategy",
            "global_threshold_tuned_on_validation_macro_f1",
        ),
        "tuned_threshold": float(tuned_threshold),
    }
    row.update(metadata)
    row.update(compute_metrics(y_true, probabilities, tuned_threshold))
    return row


def aggregate_main_statistics(runs: pd.DataFrame) -> pd.DataFrame:
    test = runs[runs["eval_set"] == "test"].copy()
    group_cols = [
        "task",
        "feature",
        "model",
        "split_mode",
        "eval_set",
        "n_labels",
        "threshold_strategy",
    ]
    metric_cols = [c for c in PRIMARY_METRICS if c in test.columns]
    gain_cols = [c for c in test.columns if c.startswith("gain_vs_")]
    metric_cols = metric_cols + gain_cols

    rows = []
    for keys, sub in test.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_split_replicates"] = int(sub["split_seed"].nunique())
        for metric in metric_cols:
            values = sub[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_sd"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_se"] = row[f"{metric}_sd"] / np.sqrt(len(values)) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_gain_columns(runs: pd.DataFrame, individual_features: List[str]) -> pd.DataFrame:
    result = runs.copy()
    for metric in ["macro_auprc", "macro_auroc", "macro_f1"]:
        result[f"gain_vs_morgan_r2_{metric}"] = np.nan
        result[f"gain_vs_best_individual_{metric}"] = np.nan

    for (split_mode, split_seed, eval_set), sub in result.groupby(
        ["split_mode", "split_seed", "eval_set"], dropna=False
    ):
        for metric in ["macro_auprc", "macro_auroc", "macro_f1"]:
            morgan_rows = sub[sub["feature"] == "morgan_r2_2048"]
            if morgan_rows.empty:
                continue
            morgan_value = float(morgan_rows[metric].iloc[0])
            best_individual = float(sub.loc[sub["feature"].isin(individual_features), metric].max())
            mask = (
                result["split_mode"].eq(split_mode)
                & result["split_seed"].eq(split_seed)
                & result["eval_set"].eq(eval_set)
            )
            result.loc[mask, f"gain_vs_morgan_r2_{metric}"] = result.loc[mask, metric] - morgan_value
            result.loc[mask, f"gain_vs_best_individual_{metric}"] = result.loc[mask, metric] - best_individual
    return result


def main():
    args = parse_args()
    if any(C <= 0 for C in args.C_grid):
        raise ValueError("All --C-grid values must be positive")
    if not all(0.0 < t < 1.0 for t in args.threshold_grid):
        raise ValueError("All --threshold-grid values must be between 0 and 1")

    print("SCRIPT_VERSION:", SCRIPT_VERSION)
    task, label_cols, Y = load_task(args.compound_file, args.label_file)

    # Repeated split construction. required_train_positives is 1 here because this is full-training,
    # not N-per-label few-shot sampling.
    split_qc_rows = []
    experiments = []
    pseudo_args = copy.copy(args)
    # Use the same minimum training-positive requirement as the few-shot data-efficiency run.
    # This prevents the full-training script from creating a weaker split file first.
    pseudo_args.n_per_label = [64]
    for mode in requested_modes(args.split_mode):
        for split_seed in args.split_seeds:
            split_task = create_or_load_split(task, Y, label_cols, mode, split_seed, pseudo_args)
            split_qc_rows.extend(split_qc_record(split_task, Y, label_cols, mode, split_seed))
            experiments.append({"split_mode": mode, "split_seed": split_seed, "task": split_task})

    feature_specs = build_full_feature_specs(args)
    feature_matrices = {}
    feature_types = {}
    for requested_name, feature_type, loader in feature_specs:
        print(f"\n=== Loading feature: {requested_name} ===")
        feature_name, X = load_feature_matrix(task, args, requested_name, loader)
        feature_matrices[feature_name] = X
        feature_types[feature_name] = feature_type

    individual_features = list(feature_matrices.keys())
    fm_features = [f for f in individual_features if f not in {"morgan_r2_2048", "morgan_r3_2048"}]

    rows = []
    prediction_payload = {
        "script_version": np.asarray([SCRIPT_VERSION], dtype=object),
        "label_columns": np.asarray(label_cols, dtype=object),
        "feature_order": np.asarray(individual_features, dtype=object),
    }

    for exp in experiments:
        split_mode = exp["split_mode"]
        split_seed = exp["split_seed"]
        split_task = exp["task"]
        train_mask = split_task["split"].eq("train").to_numpy()
        validation_mask = split_task["split"].eq("validation").to_numpy()
        eval_masks = {
            "validation": validation_mask,
            "test": split_task["split"].eq("test").to_numpy(),
        }
        train_idx = np.flatnonzero(train_mask)

        print(f"\n=== Evaluating split_mode={split_mode} split_seed={split_seed} ===")
        probability_store = {eval_set: {} for eval_set in eval_masks}

        # Individual models: tune C by validation macro-AUPRC, then tune threshold by validation macro-F1.
        for feature_name in individual_features:
            X = feature_matrices[feature_name]
            feature_type = feature_types[feature_name]
            selected = fit_select_c_and_threshold(
                X=X,
                Y=Y,
                train_idx=train_idx,
                validation_mask=validation_mask,
                feature_type=feature_type,
                seed=split_seed,
                C_grid=args.C_grid,
                max_iter=args.max_iter,
                threshold_grid=args.threshold_grid,
            )
            model = selected["model"]
            tuned_threshold = selected["tuned_threshold"]
            metadata = {
                "C_grid": "|".join(str(float(c)) for c in args.C_grid),
                "selected_C": selected["selected_C"],
                "C_selection_metric": "validation_macro_auprc",
                "validation_macro_auprc_for_selected_C": selected["validation_macro_auprc_for_C"],
                "validation_macro_auroc_for_selected_C": selected["validation_macro_auroc_for_C"],
                "threshold_strategy": "global_threshold_tuned_on_validation_macro_f1",
                "threshold_grid": "|".join(str(float(t)) for t in args.threshold_grid),
                "validation_best_macro_f1_at_tuned_threshold": selected[
                    "validation_best_macro_f1_at_threshold"
                ],
            }

            for eval_set, eval_mask in eval_masks.items():
                if eval_set == "validation":
                    probabilities = selected["validation_probability"]
                else:
                    probabilities = np.asarray(model.predict_proba(X[eval_mask]), dtype=float)
                probability_store[eval_set][feature_name] = probabilities
                row = evaluate_probability_row(
                    feature=feature_name,
                    model_name="one_vs_rest_logistic_regression",
                    split_mode=split_mode,
                    split_seed=split_seed,
                    eval_set=eval_set,
                    n_train=int(train_mask.sum()),
                    n_eval=int(eval_mask.sum()),
                    n_labels=Y.shape[1],
                    y_true=Y[eval_mask],
                    probabilities=probabilities,
                    tuned_threshold=tuned_threshold,
                    metadata=metadata,
                )
                rows.append(row)
                print(
                    f"{feature_name:28s} {eval_set:10s} C={selected['selected_C']} "
                    f"thr={tuned_threshold:.2f} AUPRC={row['macro_auprc']:.4f} "
                    f"AUROC={row['macro_auroc']:.4f} macro-F1={row['macro_f1']:.4f}"
                )

        # Pair ensembles: Morgan r2 + each selected FM. Use selected base-model probabilities.
        ensemble_specs = []
        for fm in fm_features:
            ensemble_specs.append((f"ensemble_morgan_r2_plus_{fm}", ["morgan_r2_2048", fm]))
        ensemble_specs.append(("ensemble_all_equal_probability", individual_features))

        for ensemble_name, members in ensemble_specs:
            val_probability = np.mean(
                np.stack([probability_store["validation"][member] for member in members], axis=0),
                axis=0,
            )
            tuned_threshold, threshold_val_metrics = tune_threshold(
                Y[validation_mask], val_probability, args.threshold_grid
            )
            metadata = {
                "C_grid": "base_models_independently_tuned",
                "selected_C": np.nan,
                "C_selection_metric": "base_models_validation_macro_auprc",
                "validation_macro_auprc_for_selected_C": np.nan,
                "validation_macro_auroc_for_selected_C": np.nan,
                "threshold_strategy": "global_threshold_tuned_on_validation_macro_f1",
                "threshold_grid": "|".join(str(float(t)) for t in args.threshold_grid),
                "validation_best_macro_f1_at_tuned_threshold": threshold_val_metrics["macro_f1"],
                "ensemble_members": "|".join(members),
            }
            for eval_set, eval_mask in eval_masks.items():
                ensemble_probability = np.mean(
                    np.stack([probability_store[eval_set][member] for member in members], axis=0),
                    axis=0,
                )
                row = evaluate_probability_row(
                    feature=ensemble_name,
                    model_name="equal_weight_probability_average",
                    split_mode=split_mode,
                    split_seed=split_seed,
                    eval_set=eval_set,
                    n_train=int(train_mask.sum()),
                    n_eval=int(eval_mask.sum()),
                    n_labels=Y.shape[1],
                    y_true=Y[eval_mask],
                    probabilities=ensemble_probability,
                    tuned_threshold=tuned_threshold,
                    metadata=metadata,
                )
                rows.append(row)
                print(
                    f"{ensemble_name:44s} {eval_set:10s} thr={tuned_threshold:.2f} "
                    f"AUPRC={row['macro_auprc']:.4f} AUROC={row['macro_auroc']:.4f} "
                    f"macro-F1={row['macro_f1']:.4f}"
                )

        # Save compact prediction arrays for audit, keyed by split.
        split_key = f"{split_mode}_seed{split_seed}"
        for eval_set, eval_mask in eval_masks.items():
            prediction_payload[f"{split_key}_{eval_set}_ids"] = (
                split_task.loc[eval_mask, "internal_compound_id"].astype(str).to_numpy()
            )
            prediction_payload[f"{split_key}_{eval_set}_y_true"] = Y[eval_mask]
            for feature_name in individual_features:
                prediction_payload[f"{split_key}_{eval_set}_{feature_name}_probability"] = probability_store[
                    eval_set
                ][feature_name]

        gc.collect()

    runs = pd.DataFrame(rows)
    runs = add_gain_columns(runs, individual_features)
    main_stats = aggregate_main_statistics(runs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_out = output_dir / "kegg_pathway_full_ensemble_repeated_runs.tsv"
    main_out = output_dir / "kegg_pathway_full_ensemble_main_statistics.tsv"
    pred_out = output_dir / "kegg_pathway_full_ensemble_repeated_predictions.npz"
    split_qc_out = output_dir / "kegg_pathway_full_ensemble_split_qc.tsv"

    runs.to_csv(runs_out, sep="\t", index=False)
    main_stats.to_csv(main_out, sep="\t", index=False)
    np.savez_compressed(pred_out, **prediction_payload)
    pd.DataFrame(split_qc_rows).to_csv(split_qc_out, sep="\t", index=False)

    print("\nWrote:")
    print(runs_out)
    print(main_out)
    print(pred_out)
    print(split_qc_out)

    if not main_stats.empty:
        print("\n=== Main test statistics ===")
        display_cols = [
            "split_mode",
            "feature",
            "n_split_replicates",
            "macro_auprc_mean",
            "macro_auprc_sd",
            "macro_auprc_se",
            "macro_auroc_mean",
            "macro_f1_mean",
            "macro_f1_se",
        ]
        display_cols = [c for c in display_cols if c in main_stats.columns]
        print(
            main_stats.sort_values(
                ["split_mode", "macro_auprc_mean"], ascending=[True, False]
            )[display_cols].to_string(index=False)
        )

    for X in feature_matrices.values():
        del X
    gc.collect()


if __name__ == "__main__":
    main()

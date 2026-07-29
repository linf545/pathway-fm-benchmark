#!/usr/bin/env python

"""
Repeated-split data-efficiency benchmark for the PMN secondary-pathway task.

Task definition
---------------
- PMN/Bao Dataset A: repeatedly split into training and validation partitions.
- PMN/Bao Dataset B: fixed external test set for every split replicate.
- 18-label multilabel plant secondary-metabolic pathway prediction.

Data-efficiency design
----------------------
- Within each fixed Dataset-A training split, create nested training subsets.
- Each subset contains at least N positive compounds for every pathway label.
- The same sampled compounds are used for every molecular representation.
- Validation and external-test compounds never change across sampling replicates.
- Bao Dataset B is never used for sampling, model selection, or threshold tuning.

Representations
---------------
- Precomputed Morgan radius-2 and radius-3 fingerprints.
- Precomputed frozen molecular-foundation-model embeddings aligned by InChIKey.

Model selection
---------------
- Logistic-regression C is selected using validation macro-AUPRC.
- A single global decision threshold is selected using validation macro-F1.
- The fixed external test set is used only for final evaluation.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:
    from run_pmn_secondary_pathway_full_ensemble import (
        DATASET_B_NAME,
        ID_COL,
        PRIMARY_METRICS,
        compute_metrics,
        create_or_load_split,
        fit_select_c_and_threshold,
        fm_pattern_available,
        label_number,
        load_feature_matrix,
        load_task,
        requested_modes,
        split_qc_record,
    )
except ImportError as exc:
    raise ImportError(
        "Cannot import the PMN full-data script. Expected:\n"
        "  src/modeling/pmn_secondary_pathway/"
        "run_pmn_secondary_pathway_full_ensemble.py\n"
        "Place this data-efficiency script under:\n"
        "  src/modeling/pmn_secondary_pathway/data_efficiency/"
    ) from exc


SCRIPT_VERSION = "pmn_secondary_pathway_data_efficiency_repeated_splits_v1"
TASK_NAME = "pmn_secondary_pathway"
EXTERNAL_EVAL_NAME = "bao_B_external_test"

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

ALL_METRICS = PRIMARY_METRICS + [
    "macro_precision",
    "macro_recall",
    "exact_match_accuracy",
    "hamming_loss",
    "mean_true_labels",
    "mean_predicted_labels",
]


def default_threshold_grid() -> List[float]:
    return [0.2, 0.4, 0.6, 0.8]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Repeated-split data-efficiency analysis for the PMN secondary-pathway "
            "task. Dataset A provides fixed train/validation splits; Dataset B remains "
            "the fixed external test set."
        )
    )
    parser.add_argument(
        "--metadata-file",
        default="data/features/bao_plant_pathway/metadata.tsv",
    )
    parser.add_argument(
        "--labels-file",
        default="data/features/bao_plant_pathway/labels.npy",
    )
    parser.add_argument(
        "--label-columns-file",
        default="data/features/bao_plant_pathway/label_columns.txt",
    )
    parser.add_argument(
        "--morgan-r2-file",
        default="data/features/bao_plant_pathway/morgan_r2_2048.npz",
    )
    parser.add_argument(
        "--morgan-r3-file",
        default="data/features/bao_plant_pathway/morgan_r3_2048.npz",
    )
    parser.add_argument(
        "--embedding-dir",
        default="data/features/bao_plant_pathway/full",
    )
    parser.add_argument(
        "--split-dir",
        default="data/splits/pmn_secondary_pathway",
        help=(
            "Use the same repeated split files as the PMN full-data benchmark. "
            "Existing files are loaded unless --overwrite-splits is supplied."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/pmn_secondary_pathway/data_efficiency",
    )
    parser.add_argument(
        "--split-mode",
        choices=["scaffold", "random", "both"],
        default="both",
    )
    parser.add_argument(
        "--split-seeds",
        nargs="+",
        type=int,
        default=[2026, 2027, 2028, 2029, 2030],
    )
    parser.add_argument(
        "--sampling-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--n-per-label",
        nargs="+",
        type=int,
        default=[1, 2, 4],
        help=(
            "Minimum positive training compounds requested for every label. "
            "The PMN task contains rare labels, so values must not exceed the "
            "minimum label count in any selected Dataset-A training split."
        ),
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Used only if a requested repeated split file does not already exist.",
    )
    parser.add_argument(
        "--fm-patterns",
        nargs="+",
        default=DEFAULT_FM_PATTERNS,
        help=(
            "Frozen FM embeddings used in the data-efficiency experiment. "
            "The default is a reduced data-efficiency subset and includes GraphMVP-GIN and Mole-BERT."
        ),
    )
    parser.add_argument(
        "--skip-missing-fm",
        action="store_true",
    )
    parser.add_argument(
        "--C-grid",
        nargs="+",
        type=float,
        default=[0.01, 0.1,1],
    )
    parser.add_argument(
        "--threshold-grid",
        nargs="+",
        type=float,
        default=default_threshold_grid(),
    )
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Parallel one-vs-rest label fits.",
    )
    parser.add_argument(
        "--scaffold-candidates",
        type=int,
        default=200,
        help="Used only if a scaffold split file must be generated.",
    )
    parser.add_argument("--scaffold-size-weight", type=float, default=1.0)
    parser.add_argument("--scaffold-label-weight", type=float, default=0.25)
    parser.add_argument("--overwrite-splits", action="store_true")
    return parser.parse_args()


def build_feature_specs(args):
    specs = [
        ("morgan_r2_2048", "fingerprint", ("fingerprint", args.morgan_r2_file)),
        ("morgan_r3_2048", "fingerprint", ("fingerprint", args.morgan_r3_file)),
    ]

    for pattern in args.fm_patterns:
        if args.skip_missing_fm:
            available, reason = fm_pattern_available(pattern, args.embedding_dir)
            if not available:
                print(f"Skipping FM pattern {pattern}: {reason}")
                continue
        specs.append((pattern, "dense_embedding", ("embedding", pattern)))

    if not specs:
        raise ValueError("No representations were selected")
    return specs


def validate_requested_sample_sizes(
    split_task: pd.DataFrame,
    Y: np.ndarray,
    label_cols: List[str],
    n_values: Iterable[int],
    split_mode: str,
    split_seed: int,
):
    train_mask = split_task["split"].eq("train").to_numpy()
    train_counts = Y[train_mask].sum(axis=0).astype(int)
    minimum_count = int(train_counts.min())
    maximum_requested = int(max(n_values))

    if maximum_requested > minimum_count:
        limiting = [
            f"{label_cols[i]}={int(train_counts[i])}"
            for i in range(len(label_cols))
            if train_counts[i] < maximum_requested
        ]
        raise ValueError(
            f"{split_mode} split seed {split_seed} cannot support "
            f"N={maximum_requested} positives per label. "
            f"The minimum training-label count is {minimum_count}. "
            "Limiting labels: " + ", ".join(limiting) + ". "
            f"Use --n-per-label values no larger than {minimum_count}, or create "
            "different repeated splits intentionally."
        )

    return train_counts


def make_nested_samples(
    task: pd.DataFrame,
    Y: np.ndarray,
    label_cols: List[str],
    n_values: Iterable[int],
    sampling_seeds: Iterable[int],
    split_mode: str,
    split_seed: int,
):
    train_pool = np.flatnonzero(task["split"].eq("train").to_numpy())
    samples: Dict[Tuple[int, int], np.ndarray] = {}
    sample_rows = []

    for sampling_seed in sampling_seeds:
        seed_sequence = np.random.SeedSequence([int(split_seed), int(sampling_seed)])
        rng = np.random.default_rng(seed_sequence)

        positive_orders = {}
        for label_idx in range(Y.shape[1]):
            candidates = train_pool[Y[train_pool, label_idx] == 1]
            positive_orders[label_idx] = rng.permutation(candidates)

        previous_selected = set()
        for n in sorted(set(int(value) for value in n_values)):
            selected = set()
            for label_idx in range(Y.shape[1]):
                selected.update(positive_orders[label_idx][:n].tolist())

            if not previous_selected.issubset(selected):
                raise RuntimeError(
                    f"Nested-sampling invariant failed for split={split_mode}, "
                    f"split_seed={split_seed}, sampling_seed={sampling_seed}, N={n}"
                )
            previous_selected = selected.copy()

            selected_array = np.asarray(sorted(selected), dtype=int)
            selected_Y = Y[selected_array]
            positive_counts = selected_Y.sum(axis=0).astype(int)
            negative_counts = len(selected_array) - positive_counts

            if np.any(positive_counts < n):
                failed = [
                    label_cols[i]
                    for i in range(len(label_cols))
                    if positive_counts[i] < n
                ]
                raise RuntimeError(
                    f"Sampling failed to reach N={n} for: " + ", ".join(failed)
                )
            if np.any(negative_counts == 0):
                failed = [
                    label_cols[i]
                    for i in range(len(label_cols))
                    if negative_counts[i] == 0
                ]
                raise RuntimeError(
                    f"Sample N={n} has no negative examples for: "
                    + ", ".join(failed)
                )

            samples[(sampling_seed, n)] = selected_array

            for row_idx in selected_array:
                active_labels = [
                    str(label_number(label_cols[label_idx]))
                    for label_idx in range(len(label_cols))
                    if Y[row_idx, label_idx] == 1
                ]
                sample_rows.append(
                    {
                        "task": TASK_NAME,
                        "split_mode": split_mode,
                        "split_seed": split_seed,
                        "sampling_seed": sampling_seed,
                        "requested_n_per_label": n,
                        "task_row": int(row_idx),
                        ID_COL: str(task.iloc[row_idx][ID_COL]),
                        "inchikey": str(task.iloc[row_idx]["inchikey"]),
                        "murcko_scaffold": task.iloc[row_idx]["murcko_scaffold"],
                        "label_classes": "|".join(active_labels),
                    }
                )

            print(
                f"split={split_mode} split_seed={split_seed} "
                f"sampling_seed={sampling_seed} N={n}: "
                f"n_train={len(selected_array)}, "
                f"positive range={int(positive_counts.min())}-"
                f"{int(positive_counts.max())}"
            )

    return samples, pd.DataFrame(sample_rows)


def evaluate_feature(
    task: pd.DataFrame,
    Y: np.ndarray,
    X,
    feature_name: str,
    feature_type: str,
    samples: Dict[Tuple[int, int], np.ndarray],
    n_values: Iterable[int],
    sampling_seeds: Iterable[int],
    split_mode: str,
    split_seed: int,
    C_grid: Iterable[float],
    max_iter: int,
    threshold_grid: Iterable[float],
    n_jobs: int,
):
    validation_mask = task["split"].eq("validation").to_numpy()
    external_mask = task["split"].eq("external_test").to_numpy()
    eval_masks = {
        "validation": validation_mask,
        EXTERNAL_EVAL_NAME: external_mask,
    }

    rows = []
    for sampling_seed in sampling_seeds:
        for n in sorted(set(int(value) for value in n_values)):
            train_idx = samples[(sampling_seed, n)]
            model_seed = int(split_seed) * 1000 + int(sampling_seed)

            selected = fit_select_c_and_threshold(
                X=X,
                Y=Y,
                train_idx=train_idx,
                validation_mask=validation_mask,
                feature_type=feature_type,
                seed=model_seed,
                C_grid=C_grid,
                max_iter=max_iter,
                threshold_grid=threshold_grid,
                n_jobs=n_jobs,
            )
            model = selected["model"]
            tuned_threshold = selected["tuned_threshold"]
            train_positive_counts = Y[train_idx].sum(axis=0).astype(int)

            metadata = {
                "task": TASK_NAME,
                "feature": feature_name,
                "feature_type": feature_type,
                "split_mode": split_mode,
                "split_seed": split_seed,
                "sampling_seed": sampling_seed,
                "model_seed": model_seed,
                "requested_n_per_label": n,
                "n_train": int(len(train_idx)),
                "n_labels": int(Y.shape[1]),
                "min_train_positive_count": int(train_positive_counts.min()),
                "max_train_positive_count": int(train_positive_counts.max()),
                "classifier": "one_vs_rest_logistic_regression",
                "C_grid": "|".join(str(float(C)) for C in C_grid),
                "selected_C": selected["selected_C"],
                "C_selection_metric": "validation_macro_auprc",
                "validation_macro_auprc_for_selected_C": selected[
                    "validation_macro_auprc_for_C"
                ],
                "validation_macro_auroc_for_selected_C": selected[
                    "validation_macro_auroc_for_C"
                ],
                "threshold_strategy": (
                    "global_threshold_tuned_on_validation_macro_f1"
                ),
                "threshold_grid": "|".join(
                    str(float(threshold)) for threshold in threshold_grid
                ),
                "tuned_threshold": tuned_threshold,
                "validation_best_macro_f1_at_tuned_threshold": selected[
                    "validation_best_macro_f1_at_threshold"
                ],
                "external_test_source": DATASET_B_NAME,
            }

            for eval_set, eval_mask in eval_masks.items():
                if eval_set == "validation":
                    probabilities = selected["validation_probability"]
                else:
                    probabilities = np.asarray(
                        model.predict_proba(X[eval_mask]),
                        dtype=float,
                    )

                row = metadata.copy()
                row["eval_set"] = eval_set
                row["n_eval"] = int(eval_mask.sum())
                row.update(
                    compute_metrics(
                        Y[eval_mask],
                        probabilities,
                        tuned_threshold,
                    )
                )
                rows.append(row)

                print(
                    f"{split_mode:8s} split_seed={split_seed} "
                    f"{feature_name:24s} N={n:2d} "
                    f"sampling_seed={sampling_seed} {eval_set:20s} "
                    f"C={selected['selected_C']} "
                    f"thr={tuned_threshold:.2f} "
                    f"AUPRC={row['macro_auprc']:.4f} "
                    f"AUROC={row['macro_auroc']:.4f} "
                    f"macro-F1={row['macro_f1']:.4f}"
                )

    return rows


def aggregate_per_split_means(runs: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "task",
        "feature",
        "feature_type",
        "split_mode",
        "split_seed",
        "eval_set",
        "requested_n_per_label",
        "n_labels",
        "classifier",
        "C_selection_metric",
        "threshold_strategy",
    ]
    value_cols = ALL_METRICS + [
        "n_train",
        "n_eval",
        "selected_C",
        "tuned_threshold",
        "validation_macro_auprc_for_selected_C",
        "validation_macro_auroc_for_selected_C",
        "validation_best_macro_f1_at_tuned_threshold",
        "n_valid_auprc_labels",
        "n_valid_auroc_labels",
    ]

    aggregation = {
        column: "mean"
        for column in value_cols
        if column in runs.columns
    }
    aggregation["sampling_seed"] = "nunique"

    per_split = (
        runs.groupby(group_cols, dropna=False)
        .agg(aggregation)
        .reset_index()
        .rename(columns={"sampling_seed": "n_sampling_replicates"})
    )
    return per_split


def summarize_across_splits(
    per_split: pd.DataFrame,
    external_only: bool = True,
) -> pd.DataFrame:
    data = per_split.copy()
    if external_only:
        data = data[data["eval_set"] == EXTERNAL_EVAL_NAME].copy()

    group_cols = [
        "task",
        "feature",
        "feature_type",
        "split_mode",
        "eval_set",
        "requested_n_per_label",
        "n_labels",
        "classifier",
        "C_selection_metric",
        "threshold_strategy",
    ]
    metric_cols = [
        column for column in PRIMARY_METRICS if column in data.columns
    ]

    rows = []
    for keys, subset in data.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_split_replicates"] = int(
            subset["split_seed"].nunique()
        )
        row["mean_n_train"] = float(subset["n_train"].mean())

        for metric in metric_cols:
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


def main():
    args = parse_args()

    if any(n <= 0 for n in args.n_per_label):
        raise ValueError("All --n-per-label values must be positive")
    if len(set(args.n_per_label)) != len(args.n_per_label):
        raise ValueError("--n-per-label contains duplicated values")
    if any(C <= 0 for C in args.C_grid):
        raise ValueError("All --C-grid values must be positive")
    if not all(0.0 < threshold < 1.0 for threshold in args.threshold_grid):
        raise ValueError("All --threshold-grid values must be between 0 and 1")
    if args.n_jobs == 0:
        raise ValueError("--n-jobs cannot be zero")

    print("SCRIPT_VERSION:", SCRIPT_VERSION)
    print("Task:", TASK_NAME)
    print("Requested N per label:", sorted(args.n_per_label))
    print("Sampling seeds:", args.sampling_seeds)
    print("FM defaults:", args.fm_patterns)

    task, label_cols, Y = load_task(
        args.metadata_file,
        args.labels_file,
        args.label_columns_file,
    )

    experiments = []
    split_qc_rows = []
    sample_tables = []

    for split_mode in requested_modes(args.split_mode):
        for split_seed in args.split_seeds:
            split_task = create_or_load_split(
                task=task,
                Y=Y,
                label_cols=label_cols,
                split_mode=split_mode,
                split_seed=split_seed,
                args=args,
            )
            split_qc_rows.extend(
                split_qc_record(
                    split_task,
                    Y,
                    label_cols,
                    split_mode,
                    split_seed,
                )
            )
            validate_requested_sample_sizes(
                split_task=split_task,
                Y=Y,
                label_cols=label_cols,
                n_values=args.n_per_label,
                split_mode=split_mode,
                split_seed=split_seed,
            )
            samples, sample_table = make_nested_samples(
                task=split_task,
                Y=Y,
                label_cols=label_cols,
                n_values=args.n_per_label,
                sampling_seeds=args.sampling_seeds,
                split_mode=split_mode,
                split_seed=split_seed,
            )
            sample_tables.append(sample_table)
            experiments.append(
                {
                    "split_mode": split_mode,
                    "split_seed": split_seed,
                    "task": split_task,
                    "samples": samples,
                    "rows": [],
                }
            )

    feature_specs = build_feature_specs(args)

    for requested_name, feature_type, loader in feature_specs:
        print(f"\n=== Loading feature: {requested_name} ===")
        feature_name, X = load_feature_matrix(
            task=task,
            args=args,
            requested_name=requested_name,
            loader=loader,
        )

        for experiment in experiments:
            experiment["rows"].extend(
                evaluate_feature(
                    task=experiment["task"],
                    Y=Y,
                    X=X,
                    feature_name=feature_name,
                    feature_type=feature_type,
                    samples=experiment["samples"],
                    n_values=args.n_per_label,
                    sampling_seeds=args.sampling_seeds,
                    split_mode=experiment["split_mode"],
                    split_seed=experiment["split_seed"],
                    C_grid=args.C_grid,
                    max_iter=args.max_iter,
                    threshold_grid=args.threshold_grid,
                    n_jobs=args.n_jobs,
                )
            )

        del X
        gc.collect()

    rows = []
    for experiment in experiments:
        rows.extend(experiment["rows"])

    runs = pd.DataFrame(rows)
    per_split = aggregate_per_split_means(runs)
    main_statistics = summarize_across_splits(
        per_split,
        external_only=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_out = (
        output_dir
        / "pmn_secondary_pathway_data_efficiency_repeated_runs.tsv"
    )
    per_split_out = (
        output_dir
        / "pmn_secondary_pathway_data_efficiency_per_split_means.tsv"
    )
    main_out = (
        output_dir
        / "pmn_secondary_pathway_data_efficiency_main_statistics.tsv"
    )
    samples_out = (
        output_dir
        / "pmn_secondary_pathway_data_efficiency_repeated_samples.tsv"
    )
    split_qc_out = (
        output_dir
        / "pmn_secondary_pathway_data_efficiency_split_qc.tsv"
    )

    runs.to_csv(runs_out, sep="\t", index=False)
    per_split.to_csv(per_split_out, sep="\t", index=False)
    main_statistics.to_csv(main_out, sep="\t", index=False)
    pd.concat(sample_tables, ignore_index=True).to_csv(
        samples_out,
        sep="\t",
        index=False,
    )
    pd.DataFrame(split_qc_rows).to_csv(
        split_qc_out,
        sep="\t",
        index=False,
    )

    print("\nWrote:")
    print(runs_out)
    print(per_split_out)
    print(main_out)
    print(samples_out)
    print(split_qc_out)

    if not main_statistics.empty:
        display_columns = [
            "split_mode",
            "feature",
            "requested_n_per_label",
            "mean_n_train",
            "n_split_replicates",
            "macro_auprc_mean",
            "macro_auprc_sd",
            "macro_auprc_se",
            "macro_auroc_mean",
            "macro_f1_mean",
            "macro_f1_se",
        ]
        display_columns = [
            column
            for column in display_columns
            if column in main_statistics.columns
        ]
        print("\n=== Main external-test statistics ===")
        print(
            main_statistics.sort_values(
                [
                    "split_mode",
                    "requested_n_per_label",
                    "macro_auprc_mean",
                ],
                ascending=[True, True, False],
            )[display_columns].to_string(index=False)
        )


if __name__ == "__main__":
    main()

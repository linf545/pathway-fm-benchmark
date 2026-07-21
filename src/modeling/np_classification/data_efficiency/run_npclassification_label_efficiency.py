#!/usr/bin/env python
"""
Task 1 NPClassifier label-efficiency benchmark.

Design
------
- Single-label, seven-class NPClassifier task.
- One fixed predefined random split and one fixed predefined scaffold split.
- Validation and test sets remain fixed for every training-set size.
- For each sampling seed, class-stratified training subsets are nested:
  every smaller subset is contained in every larger subset.
- The same sampled compounds are used for every molecular representation.
- Morgan r2/r3 and all retained frozen FM embeddings are evaluated together.
- Logistic-regression C is selected on validation macro one-vs-rest AUROC,
  with macro AUPRC as the secondary selection criterion.
- Converged C candidates are preferred over non-converged candidates.
- Test data are never used for sampling or model selection.

The full-data benchmark remains separate. Its results provide the full-training
endpoint; this script focuses on subsampled labeled training sets.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:
    from run_npclassification_full import (
        DEFAULT_FEATURES,
        OVERALL_METRICS,
        FeatureData,
        aligned_probabilities,
        build_splits,
        evaluate_probabilities,
        fit_select_C,
        load_feature,
        load_task,
        split_spec,
    )
except ImportError as exc:
    raise ImportError(
        "Cannot import the Task 1 full-data script. Expected:\n"
        "  src/modeling/np_classification/run_npclassification_full.py\n"
        "Place this script under:\n"
        "  src/modeling/np_classification/data_efficiency/"
    ) from exc


SCRIPT_VERSION = "npclassification_label_efficiency_v1"
TASK_NAME = "np_classification"

DEFAULT_SAMPLE_SEEDS = [2026, 2027, 2028, 2029, 2030]
DEFAULT_TRAIN_SIZES = [100, 300, 1000, 3000, 10000, 30000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(DEFAULT_TRAIN_SIZES
        description=(
            "Fixed-validation/test label-efficiency benchmark for "
            "single-label NPClassifier prediction."
        )
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
        default="results/np_classification/label_efficiency",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
    )
    parser.add_argument(
        "--split-modes",
        nargs="+",
        choices=["random", "scaffold"],
        default=["random", "scaffold"],
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
        help=(
            "Total labeled training compounds. Samples are nested and "
            "approximately preserve the class distribution while guaranteeing "
            "that every class is represented."
        ),
    )
    parser.add_argument(
        "--C-grid",
        nargs="+",
        type=float,
        default=[0.01, 0.1, 1.0],
    )
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--dense-solver", default="lbfgs")
    parser.add_argument("--sparse-solver", default="lbfgs")
    parser.add_argument("--dense-max-iter", type=int, default=500)
    parser.add_argument("--sparse-max-iter", type=int, default=1000)
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Convergence tolerance; 1e-3 is the Task 1 default.",
    )
    parser.add_argument(
        "--parallel-runs",
        type=int,
        default=2,
        help=(
            "Number of sample-size/seed fits run concurrently for one feature. "
            "Set BLAS/OpenMP thread counts to 1 when using values above 1."
        ),
    )
    parser.add_argument("--skip-missing-features", action="store_true")
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite-results", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    allowed = set(DEFAULT_FEATURES)
    unknown = sorted(set(args.features) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown features: {unknown}. Allowed features: "
            + ", ".join(DEFAULT_FEATURES)
        )
    if len(set(args.features)) != len(args.features):
        raise ValueError("--features contains duplicates")
    if len(set(args.split_modes)) != len(args.split_modes):
        raise ValueError("--split-modes contains duplicates")
    if len(set(args.sample_seeds)) != len(args.sample_seeds):
        raise ValueError("--sample-seeds contains duplicates")
    if not args.sample_seeds:
        raise ValueError("--sample-seeds cannot be empty")
    if not args.train_sizes:
        raise ValueError("--train-sizes cannot be empty")
    if any(value <= 0 for value in args.train_sizes):
        raise ValueError("--train-sizes must contain positive integers")
    if len(set(args.train_sizes)) != len(args.train_sizes):
        raise ValueError("--train-sizes contains duplicates")
    if not args.C_grid:
        raise ValueError("--C-grid cannot be empty")
    if any((not np.isfinite(value)) or value <= 0 for value in args.C_grid):
        raise ValueError("All C values must be positive and finite")
    if args.parallel_runs < 1:
        raise ValueError("--parallel-runs must be >= 1")
    if args.tol <= 0 or not np.isfinite(args.tol):
        raise ValueError("--tol must be positive and finite")


def deterministic_seed(
    split_mode: str,
    base_split_seed: int,
    sample_seed: int,
) -> int:
    mode_code = 0 if split_mode == "random" else 1
    sequence = np.random.SeedSequence(
        [mode_code, int(base_split_seed), int(sample_seed)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def model_seed(
    split_mode: str,
    base_split_seed: int,
    sample_seed: int,
    requested_n_train: int,
) -> int:
    mode_code = 0 if split_mode == "random" else 1
    sequence = np.random.SeedSequence(
        [
            9173,
            mode_code,
            int(base_split_seed),
            int(sample_seed),
            int(requested_n_train),
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def sample_hash(values: np.ndarray) -> str:
    ordered = np.sort(values.astype(str))
    payload = "\n".join(ordered).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def allocate_nested_quotas(
    class_counts: np.ndarray,
    requested_sizes: list[int],
    tie_order: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    Allocate monotone per-class quotas.

    The first sample contains at least one compound from every class. Additional
    compounds are assigned to the class with the largest deficit relative to
    the natural training-set class proportions. Quotas only increase, so taking
    fixed per-class shuffled prefixes creates exactly nested samples.
    """
    n_classes = len(class_counts)
    if np.any(class_counts <= 0):
        raise ValueError("Every class must be present in the training pool")

    maximum = int(class_counts.sum())
    if max(requested_sizes) > maximum:
        raise ValueError(
            f"Requested training size {max(requested_sizes):,} exceeds "
            f"the training pool size {maximum:,}"
        )
    if min(requested_sizes) < n_classes:
        raise ValueError(
            f"Every sample must contain all {n_classes} classes; "
            f"minimum requested size is {min(requested_sizes)}"
        )

    proportions = class_counts.astype(float) / float(maximum)
    quotas = np.ones(n_classes, dtype=np.int64)
    outputs: dict[int, np.ndarray] = {}

    tie_rank = np.empty(n_classes, dtype=np.int64)
    tie_rank[tie_order] = np.arange(n_classes, dtype=np.int64)

    for requested in sorted(requested_sizes):
        while int(quotas.sum()) < requested:
            available = quotas < class_counts
            if not available.any():
                raise RuntimeError(
                    "All classes exhausted before reaching requested size "
                    f"{requested}"
                )

            targets = proportions * float(requested)
            deficits = targets - quotas.astype(float)
            deficits[~available] = -np.inf

            best_deficit = float(np.max(deficits))
            candidates = np.flatnonzero(
                np.isclose(deficits, best_deficit, atol=1e-12, rtol=0.0)
            )
            selected_class = int(
                candidates[np.argmin(tie_rank[candidates])]
            )
            quotas[selected_class] += 1

        outputs[requested] = quotas.copy()

    return outputs


def build_nested_samples(
    task: pd.DataFrame,
    y: np.ndarray,
    split_mode: str,
    base_split_seed: int,
    train_indices: np.ndarray,
    sample_seeds: list[int],
    train_sizes: list[int],
    n_classes: int,
) -> tuple[
    dict[tuple[str, int, int], np.ndarray],
    pd.DataFrame,
    dict[str, np.ndarray],
]:
    class_counts = np.bincount(
        y[train_indices],
        minlength=n_classes,
    ).astype(np.int64)

    print(
        f"\n{split_mode} fixed training pool: {len(train_indices):,}; "
        f"class counts={class_counts.tolist()}"
    )

    plans: dict[tuple[str, int, int], np.ndarray] = {}
    qc_rows: list[dict[str, Any]] = []
    saved_arrays: dict[str, np.ndarray] = {}

    for sample_seed_value in sample_seeds:
        rng = np.random.default_rng(
            deterministic_seed(
                split_mode,
                base_split_seed,
                sample_seed_value,
            )
        )

        class_orders: dict[int, np.ndarray] = {}
        for class_index in range(n_classes):
            candidates = train_indices[y[train_indices] == class_index]
            class_orders[class_index] = rng.permutation(candidates)

        tie_order = rng.permutation(np.arange(n_classes))
        quotas_by_size = allocate_nested_quotas(
            class_counts=class_counts,
            requested_sizes=sorted(train_sizes),
            tie_order=tie_order,
        )

        previous: set[int] = set()
        for requested_n_train in sorted(train_sizes):
            quotas = quotas_by_size[requested_n_train]
            selected = np.concatenate(
                [
                    class_orders[class_index][: quotas[class_index]]
                    for class_index in range(n_classes)
                ]
            ).astype(np.int64)
            selected = np.sort(selected)

            selected_set = set(selected.tolist())
            if previous and not previous.issubset(selected_set):
                raise RuntimeError(
                    "Nested-sampling invariant failed for "
                    f"{split_mode}, sample seed {sample_seed_value}, "
                    f"N={requested_n_train}"
                )
            previous = selected_set

            labels = y[selected]
            observed_counts = np.bincount(
                labels,
                minlength=n_classes,
            ).astype(np.int64)
            if not np.array_equal(observed_counts, quotas):
                raise RuntimeError("Sample class counts do not match quotas")
            if len(selected) != requested_n_train:
                raise RuntimeError(
                    f"Requested {requested_n_train}, sampled {len(selected)}"
                )

            key = (
                split_mode,
                int(sample_seed_value),
                int(requested_n_train),
            )
            plans[key] = selected

            array_key = (
                f"{split_mode}_basesplit{base_split_seed}_"
                f"sampleseed{sample_seed_value}_n{requested_n_train}"
            )
            saved_arrays[array_key] = selected

            ids = task.iloc[selected]["inchikey"].astype(str).to_numpy()
            qc_rows.append(
                {
                    "task": TASK_NAME,
                    "split": split_mode,
                    "base_split_seed": int(base_split_seed),
                    "sample_seed": int(sample_seed_value),
                    "requested_n_train": int(requested_n_train),
                    "actual_n_train": int(len(selected)),
                    "min_class_count": int(observed_counts.min()),
                    "max_class_count": int(observed_counts.max()),
                    "class_counts": "|".join(
                        str(int(value)) for value in observed_counts
                    ),
                    "sample_sha256": sample_hash(ids),
                    "nested_within_sampling_seed": True,
                }
            )

            print(
                f"{split_mode} | sample_seed={sample_seed_value} | "
                f"N={requested_n_train:,} | "
                f"class range={observed_counts.min()}-"
                f"{observed_counts.max()}"
            )

    return plans, pd.DataFrame(qc_rows), saved_arrays


def run_condition(
    feature: FeatureData,
    split_mode: str,
    base_split_seed: int,
    sample_seed_value: int,
    requested_n_train: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    args: argparse.Namespace,
    sample_sha256: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
]:
    seed = model_seed(
        split_mode,
        base_split_seed,
        sample_seed_value,
        requested_n_train,
    )

    (
        estimator,
        selected_C,
        trial_results,
        max_n_iter,
        reached,
        n_iter_values,
        selected_fit_seconds,
    ) = fit_select_C(
        feature=feature,
        y=y,
        train_indices=train_indices,
        validation_indices=validation_indices,
        label_names=label_names,
        seed=seed,
        args=args,
    )

    max_iter = (
        args.sparse_max_iter
        if feature.kind == "sparse_morgan"
        else args.dense_max_iter
    )
    train_class_counts = np.bincount(
        y[train_indices],
        minlength=len(label_names),
    ).astype(np.int64)

    base = {
        "task": TASK_NAME,
        "feature": feature.name,
        "feature_kind": feature.kind,
        "feature_source": feature.source_path,
        "n_features": int(feature.n_features),
        "sparse_matrix": bool(feature.sparse_matrix),
        "model": "single_label_multiclass_logistic_regression",
        "split": split_mode,
        "split_column": split_spec(
            split_mode,
            base_split_seed,
        ).split_column,
        "base_split_seed": int(base_split_seed),
        "sample_seed": int(sample_seed_value),
        "model_seed": int(seed),
        "requested_n_train": int(requested_n_train),
        "n_train": int(len(train_indices)),
        "n_classes": int(len(label_names)),
        "min_train_class_count": int(train_class_counts.min()),
        "max_train_class_count": int(train_class_counts.max()),
        "train_class_counts": "|".join(
            str(int(value)) for value in train_class_counts
        ),
        "sample_sha256": sample_sha256,
        "selected_C": float(selected_C),
        "C_grid": "|".join(str(float(value)) for value in args.C_grid),
        "C_selection": (
            "prefer_converged_then_validation_macro_ovr_auroc_"
            "then_auprc_then_smaller_C"
        ),
        "max_iter": int(max_iter),
        "tol": float(args.tol),
        "max_n_iter": int(max_n_iter),
        "n_iter_values": n_iter_values,
        "reached_max_iter": bool(reached),
        "selected_fit_seconds": float(selected_fit_seconds),
    }

    run_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    log_lines: list[str] = []

    for eval_set, eval_indices in [
        ("validation", validation_indices),
        ("test", test_indices),
    ]:
        start = time.perf_counter()
        probabilities = aligned_probabilities(
            estimator,
            feature.matrix[eval_indices],
            len(label_names),
        )
        score_seconds = time.perf_counter() - start

        overall, per_class = evaluate_probabilities(
            y[eval_indices],
            probabilities,
            label_names,
        )
        eval_base = {
            **base,
            "eval_set": eval_set,
            "n_eval": int(len(eval_indices)),
            "score_seconds": float(score_seconds),
        }
        run_rows.append({**eval_base, **overall})
        class_rows.extend(
            {**eval_base, **row} for row in per_class
        )

        log_lines.append(
            f"{feature.name} | {split_mode} | "
            f"sample_seed={sample_seed_value} | "
            f"N={requested_n_train:,} | {eval_set} | "
            f"C={selected_C:g} | "
            f"AUROC={overall['auroc_ovr_macro']:.6f} | "
            f"AUPRC={overall['auprc_ovr_macro']:.6f} | "
            f"macro-F1={overall['macro_f1']:.6f}"
        )

    trial_rows = [
        {
            "task": TASK_NAME,
            "feature": feature.name,
            "split": split_mode,
            "base_split_seed": int(base_split_seed),
            "sample_seed": int(sample_seed_value),
            "requested_n_train": int(requested_n_train),
            "n_train": int(len(train_indices)),
            "sample_sha256": sample_sha256,
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
    group_columns = [
        "task",
        "feature",
        "model",
        "split",
        "requested_n_train",
    ]
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
    summary["n_sampling_replicates"] = summary[
        "auroc_ovr_macro_count"
    ].astype(int)
    return summary


def paired_gains(
    runs: pd.DataFrame,
    reference: str = "morgan_r2_2048",
) -> pd.DataFrame:
    test = runs[runs["eval_set"].eq("test")].copy()
    keys = [
        "split",
        "base_split_seed",
        "sample_seed",
        "requested_n_train",
    ]
    reference_rows = test[test["feature"].eq(reference)][
        keys + OVERALL_METRICS
    ].copy()
    reference_rows = reference_rows.rename(
        columns={
            metric: f"reference_{metric}"
            for metric in OVERALL_METRICS
        }
    )

    merged = test.merge(
        reference_rows,
        on=keys,
        how="left",
        validate="many_to_one",
    )
    required = [
        f"reference_{metric}" for metric in OVERALL_METRICS
    ]
    if merged[required].isna().any().any():
        raise ValueError(f"Missing paired {reference} label-efficiency runs")

    difference_columns: list[str] = []
    for metric in OVERALL_METRICS:
        column = f"{metric}_difference_vs_{reference}"
        merged[column] = (
            merged[metric] - merged[f"reference_{metric}"]
        )
        difference_columns.append(column)

    group_columns = [
        "task",
        "feature",
        "model",
        "split",
        "requested_n_train",
    ]
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
        "runs": output_dir / "npclassification_label_efficiency_runs.tsv",
        "per_class": (
            output_dir / "npclassification_label_efficiency_per_class.tsv"
        ),
        "summary": (
            output_dir / "npclassification_label_efficiency_summary.tsv"
        ),
        "paired": (
            output_dir / "npclassification_label_efficiency_paired_gains.tsv"
        ),
        "sample_qc": (
            output_dir / "npclassification_label_efficiency_sample_qc.tsv"
        ),
        "sample_indices": (
            output_dir / "npclassification_label_efficiency_samples.npz"
        ),
        "split_qc": (
            output_dir / "npclassification_label_efficiency_split_qc.tsv"
        ),
        "trials": (
            output_dir / "npclassification_label_efficiency_C_trials.tsv"
        ),
        "config": (
            output_dir / "npclassification_label_efficiency_config.json"
        ),
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
        paths["sample_qc"],
        paths["sample_indices"],
        paths["split_qc"],
        paths["trials"],
        paths["config"],
    ]

    if args.overwrite_results:
        for path in managed:
            if path.exists():
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
) -> set[tuple[str, str, int, int]]:
    if runs.empty:
        return set()

    required = {
        "feature",
        "split",
        "sample_seed",
        "requested_n_train",
        "eval_set",
    }
    if not required.issubset(runs.columns):
        raise ValueError("Existing result schema is incompatible")

    complete: set[tuple[str, str, int, int]] = set()
    for key, group in runs.groupby(
        ["feature", "split", "sample_seed", "requested_n_train"]
    ):
        if set(group["eval_set"]) == {"validation", "test"}:
            complete.add(
                (
                    str(key[0]),
                    str(key[1]),
                    int(key[2]),
                    int(key[3]),
                )
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


def critical_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "task": str(Path(args.task)),
        "embedding_dir": str(Path(args.embedding_dir)),
        "cache_dir": str(Path(args.cache_dir)),
        "features": list(args.features),
        "split_modes": list(args.split_modes),
        "base_split_seed": int(args.base_split_seed),
        "sample_seeds": [int(value) for value in args.sample_seeds],
        "train_sizes": [int(value) for value in args.train_sizes],
        "C_grid": [float(value) for value in args.C_grid],
        "class_weight": args.class_weight,
        "dense_solver": args.dense_solver,
        "sparse_solver": args.sparse_solver,
        "dense_max_iter": int(args.dense_max_iter),
        "sparse_max_iter": int(args.sparse_max_iter),
        "tol": float(args.tol),
    }


def validate_resume_config(
    path: Path,
    expected: dict[str, Any],
    resume: bool,
) -> None:
    if not resume or not path.exists():
        return
    existing = json.loads(path.read_text())
    existing_critical = existing.get("critical_config")
    if existing_critical != expected:
        raise ValueError(
            "Existing label-efficiency configuration does not match the "
            "requested run. Use --overwrite-results instead of --resume."
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

    expected_config = critical_config(args)
    validate_resume_config(
        paths["config"],
        expected_config,
        args.resume,
    )

    task, y, label_names = load_task(task_path)
    n_classes = len(label_names)

    print("SCRIPT_VERSION:", SCRIPT_VERSION)
    print(f"Task rows: {len(task):,}")
    print(f"Classes: {n_classes}")
    print("Features:", ", ".join(args.features))
    print("Train sizes:", ", ".join(f"{n:,}" for n in args.train_sizes))
    print("C grid:", args.C_grid)
    print("Tolerance:", args.tol)

    splits, split_qc = build_splits(
        task=task,
        y=y,
        modes=args.split_modes,
        seeds=[args.base_split_seed],
        n_classes=n_classes,
    )

    plans: dict[tuple[str, int, int], np.ndarray] = {}
    sample_qc_frames: list[pd.DataFrame] = []
    sample_arrays: dict[str, np.ndarray] = {}

    for split_mode in args.split_modes:
        mode_plans, mode_qc, mode_arrays = build_nested_samples(
            task=task,
            y=y,
            split_mode=split_mode,
            base_split_seed=args.base_split_seed,
            train_indices=splits[
                (split_mode, args.base_split_seed)
            ]["train"],
            sample_seeds=args.sample_seeds,
            train_sizes=sorted(args.train_sizes),
            n_classes=n_classes,
        )
        plans.update(mode_plans)
        sample_qc_frames.append(mode_qc)
        sample_arrays.update(mode_arrays)

    sample_qc = pd.concat(sample_qc_frames, ignore_index=True)

    runs, per_class, trials = prepare_results(paths, args)
    complete = completed_keys(runs)

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_qc.to_csv(paths["sample_qc"], sep="\t", index=False)
    np.savez_compressed(paths["sample_indices"], **sample_arrays)
    split_qc.to_csv(paths["split_qc"], sep="\t", index=False)

    config = {
        "critical_config": expected_config,
        "excluded_features": ["nafm_contrast", "rdkit_descriptors"],
        "sampling": {
            "strategy": (
                "nested_class_stratified_total_size; approximately preserves "
                "the fixed training-pool class distribution; at least one "
                "compound per class"
            ),
            "fixed_validation_and_test": True,
            "sample_identity": "InChIKey SHA256",
            "same_samples_across_representations": True,
            "full_data_endpoint": (
                "Use results/np_classification/full/"
                "npclassification_repeated_runs.tsv separately."
            ),
        },
        "model_selection": {
            "primary": "validation_macro_ovr_auroc",
            "secondary": "validation_macro_ovr_auprc",
            "prefer_converged_candidates": True,
            "final_tie_break": "smaller_C",
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
        },
        "n_compounds": int(len(task)),
        "n_classes": int(n_classes),
        "label_names": label_names,
    }
    paths["config"].write_text(json.dumps(config, indent=2) + "\n")

    sample_hash_lookup = {
        (
            str(row.split),
            int(row.sample_seed),
            int(row.requested_n_train),
        ): str(row.sample_sha256)
        for row in sample_qc.itertuples(index=False)
    }

    for feature_name in args.features:
        try:
            feature = load_feature(
                task=task,
                feature=feature_name,
                cache_dir=cache_dir,
                embedding_dir=embedding_dir,
                rebuild=args.rebuild_feature_cache,
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
            (
                split_mode,
                int(sample_seed_value),
                int(requested_n_train),
            )
            for split_mode in args.split_modes
            for sample_seed_value in args.sample_seeds
            for requested_n_train in sorted(args.train_sizes)
            if (
                feature.name,
                split_mode,
                int(sample_seed_value),
                int(requested_n_train),
            )
            not in complete
        ]

        if not conditions:
            print(f"All requested runs already complete for {feature.name}")
            del feature
            gc.collect()
            continue

        new_runs: list[dict[str, Any]] = []
        new_classes: list[dict[str, Any]] = []
        new_trials: list[dict[str, Any]] = []

        with ThreadPoolExecutor(
            max_workers=min(args.parallel_runs, len(conditions))
        ) as executor:
            futures = {}
            for (
                split_mode,
                sample_seed_value,
                requested_n_train,
            ) in conditions:
                fixed_indices = splits[
                    (split_mode, args.base_split_seed)
                ]
                train_indices = plans[
                    (
                        split_mode,
                        sample_seed_value,
                        requested_n_train,
                    )
                ]
                sha256 = sample_hash_lookup[
                    (
                        split_mode,
                        sample_seed_value,
                        requested_n_train,
                    )
                ]

                future = executor.submit(
                    run_condition,
                    feature,
                    split_mode,
                    args.base_split_seed,
                    sample_seed_value,
                    requested_n_train,
                    train_indices,
                    fixed_indices["validation"],
                    fixed_indices["test"],
                    y,
                    label_names,
                    args,
                    sha256,
                )
                futures[future] = (
                    split_mode,
                    sample_seed_value,
                    requested_n_train,
                )

            for future in as_completed(futures):
                condition = futures[future]
                try:
                    (
                        run_rows,
                        class_rows,
                        trial_rows,
                        log_text,
                    ) = future.result()
                except Exception as error:
                    raise RuntimeError(
                        f"Failed {feature.name} | split={condition[0]} | "
                        f"sample_seed={condition[1]} | N={condition[2]}"
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
        gc.collect()

    if runs.empty:
        raise RuntimeError("No label-efficiency results are available")

    runs = runs.sort_values(
        [
            "feature",
            "split",
            "requested_n_train",
            "sample_seed",
            "eval_set",
        ]
    ).reset_index(drop=True)
    runs.to_csv(paths["runs"], sep="\t", index=False)

    per_class = per_class.sort_values(
        [
            "feature",
            "split",
            "requested_n_train",
            "sample_seed",
            "eval_set",
            "class_index",
        ]
    ).reset_index(drop=True)
    per_class.to_csv(paths["per_class"], sep="\t", index=False)

    trials = trials.sort_values(
        [
            "feature",
            "split",
            "requested_n_train",
            "sample_seed",
            "C",
        ]
    ).reset_index(drop=True)
    trials.to_csv(paths["trials"], sep="\t", index=False)

    summary = summarize_runs(runs)
    summary.to_csv(paths["summary"], sep="\t", index=False)

    if "morgan_r2_2048" in set(runs["feature"]):
        paired = paired_gains(runs)
        paired.to_csv(paths["paired"], sep="\t", index=False)

    expected_replicates = len(args.sample_seeds)
    counts = (
        runs[runs["eval_set"].eq("test")]
        .groupby(
            ["feature", "split", "requested_n_train"]
        )["sample_seed"]
        .nunique()
    )
    incomplete = counts[counts != expected_replicates]
    if not incomplete.empty:
        print(
            "WARNING: incomplete sampling-replicate counts:\n"
            + incomplete.to_string()
        )

    nonconverged = (
        runs[runs["eval_set"].eq("test")]
        .groupby(
            ["feature", "split", "requested_n_train"]
        )["reached_max_iter"]
        .sum()
    )
    problematic = nonconverged[nonconverged > 0]
    if not problematic.empty:
        print(
            "WARNING: selected non-converged models remain:\n"
            + problematic.to_string()
        )

    print("\nWrote:", paths["runs"])
    print("Wrote:", paths["per_class"])
    print("Wrote:", paths["summary"])
    if paths["paired"].exists():
        print("Wrote:", paths["paired"])
    print("Wrote:", paths["sample_qc"])
    print("Wrote:", paths["sample_indices"])
    print("Wrote:", paths["split_qc"])
    print("Wrote:", paths["trials"])
    print("Wrote:", paths["config"])


if __name__ == "__main__":
    main()

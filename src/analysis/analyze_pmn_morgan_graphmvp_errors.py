#!/usr/bin/env python
"""
Compare false-positive and false-negative errors from Morgan r2 and GraphMVP-GIN
on the PMN/Bao B external test set.

The script uses saved probabilities from the full-data benchmark and each model's
validation-selected threshold from the repeated-runs TSV. It does not retrain any
model or recalculate AUROC.

Outputs
-------
1. error_overlap_by_seed.tsv
   Shared and model-specific FP/FN counts for every split seed.
2. error_overlap_summary.tsv
   Mean, SD, minimum, and maximum counts across split seeds.
3. unique_error_pathways_by_seed.tsv
   Pathway counts for Morgan-only and GraphMVP-only FP/FN errors.
4. unique_error_pathways_summary.tsv
   Mean pathway-specific counts across split seeds.
5. unique_error_details.tsv
   Compound-pathway records for every model-specific error.

For false positives, the pathway is the incorrectly predicted pathway because the
true compound-pathway label is 0. For false negatives, the pathway is the true
pathway missed by the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MORGAN_FEATURE = "morgan_r2_2048"
GRAPHMVP_FEATURE = "graphmvp_gin"
DEFAULT_EVAL_SET = "bao_B_external_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze overlap and model-specific FP/FN errors for Morgan r2 and "
            "GraphMVP-GIN in the PMN secondary-pathway benchmark."
        )
    )
    parser.add_argument(
        "--results-dir",
        default="results/pmn_secondary_pathway/full",
        help="Directory containing the full-data repeated-run outputs.",
    )
    parser.add_argument(
        "--predictions-file",
        default=None,
        help=(
            "Saved prediction NPZ. When omitted, the script searches --results-dir "
            "for the unique compatible file."
        ),
    )
    parser.add_argument(
        "--runs-file",
        default=None,
        help=(
            "Repeated-runs TSV containing tuned_threshold. When omitted, the script "
            "searches --results-dir for the unique compatible file."
        ),
    )
    parser.add_argument(
        "--metadata-file",
        default="data/features/bao_plant_pathway/metadata.tsv",
        help="Optional compound metadata added to unique_error_details.tsv.",
    )
    parser.add_argument(
        "--metadata-id-column",
        default="task_compound_id",
    )
    parser.add_argument(
        "--eval-set",
        default=DEFAULT_EVAL_SET,
        help="Evaluation-set name used in both the runs TSV and NPZ keys.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "results/pmn_secondary_pathway/full/"
            "morgan_r2_graphmvp_error_overlap"
        ),
    )
    return parser.parse_args()


def compatible_prediction_file(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=True) as payload:
            keys = payload.files
            return (
                "label_columns" in keys
                and any(
                    key.endswith(f"_{MORGAN_FEATURE}_probability")
                    for key in keys
                )
                and any(
                    key.endswith(f"_{GRAPHMVP_FEATURE}_probability")
                    for key in keys
                )
                and any(key.endswith("_y_true") for key in keys)
            )
    except Exception:
        return False


def compatible_runs_file(path: Path) -> bool:
    try:
        columns = set(pd.read_csv(path, sep="\t", nrows=2).columns)
    except Exception:
        return False
    required = {
        "feature",
        "split_mode",
        "split_seed",
        "eval_set",
    }
    return required.issubset(columns) and (
        "tuned_threshold" in columns or "threshold" in columns
    )


def discover_unique_file(
    explicit: str | None,
    results_dir: Path,
    suffix: str,
    compatibility_check,
    description: str,
) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        if not compatibility_check(path):
            raise ValueError(f"Incompatible {description}: {path}")
        return path

    candidates = [
        path
        for path in sorted(results_dir.rglob(f"*{suffix}"))
        if compatibility_check(path)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No compatible {description} found under {results_dir}. "
            f"Provide it explicitly."
        )
    if len(candidates) > 1:
        formatted = "\n  ".join(str(path) for path in candidates)
        raise ValueError(
            f"Multiple compatible {description} files were found:\n  {formatted}\n"
            "Provide the intended file explicitly."
        )
    return candidates[0]


def load_optional_metadata(path: str, id_column: str) -> pd.DataFrame | None:
    metadata_path = Path(path)
    if not metadata_path.exists():
        print(f"Warning: metadata file not found; details will contain IDs only: {path}")
        return None

    metadata = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    if id_column not in metadata.columns:
        raise ValueError(
            f"Metadata file lacks ID column {id_column!r}: {metadata_path}"
        )
    metadata = metadata.copy()
    metadata[id_column] = metadata[id_column].astype(str)
    if metadata[id_column].duplicated().any():
        raise ValueError(f"Duplicated metadata IDs in {metadata_path}")

    preferred_columns = [
        id_column,
        "dataset_name",
        "canonical_smiles",
        "inchikey",
        "murcko_scaffold",
        "pathway_names",
        "label_classes",
    ]
    retained = [column for column in preferred_columns if column in metadata.columns]
    return metadata[retained].copy()


def one_threshold(
    runs: pd.DataFrame,
    split_mode: str,
    split_seed: int,
    eval_set: str,
    feature: str,
    threshold_column: str,
) -> float:
    subset = runs[
        runs["split_mode"].astype(str).eq(str(split_mode))
        & runs["split_seed"].astype(int).eq(int(split_seed))
        & runs["eval_set"].astype(str).eq(str(eval_set))
        & runs["feature"].astype(str).eq(str(feature))
    ]
    if len(subset) != 1:
        raise ValueError(
            "Expected exactly one threshold row for "
            f"split_mode={split_mode}, split_seed={split_seed}, "
            f"eval_set={eval_set}, feature={feature}; found {len(subset)}"
        )
    threshold = float(subset[threshold_column].iloc[0])
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError(
            f"Invalid threshold for {feature}, {split_mode}, {split_seed}: {threshold}"
        )
    return threshold


def summarize_numeric_counts(
    frame: pd.DataFrame,
    group_columns: list[str],
    count_columns: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, subset in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_split_seeds"] = int(subset["split_seed"].nunique())
        for column in count_columns:
            values = subset[column].astype(float).to_numpy()
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            )
            row[f"{column}_min"] = int(np.min(values))
            row[f"{column}_max"] = int(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows)


def append_detail_rows(
    rows: list[dict],
    mask: np.ndarray,
    compound_ids: np.ndarray,
    label_columns: list[str],
    split_mode: str,
    split_seed: int,
    eval_set: str,
    error_type: str,
    unique_to: str,
    morgan_probability: np.ndarray,
    graphmvp_probability: np.ndarray,
    morgan_prediction: np.ndarray,
    graphmvp_prediction: np.ndarray,
    morgan_threshold: float,
    graphmvp_threshold: float,
) -> None:
    compound_indices, label_indices = np.where(mask)
    pathway_role = (
        "incorrectly_predicted_pathway"
        if error_type == "false_positive"
        else "missed_true_pathway"
    )
    true_label = 0 if error_type == "false_positive" else 1

    for compound_index, label_index in zip(compound_indices, label_indices):
        rows.append(
            {
                "split_mode": split_mode,
                "split_seed": int(split_seed),
                "eval_set": eval_set,
                "error_type": error_type,
                "unique_to": unique_to,
                "compound_id": str(compound_ids[compound_index]),
                "pathway_index": int(label_index),
                "pathway": label_columns[label_index],
                "pathway_role": pathway_role,
                "true_label": true_label,
                "morgan_probability": float(
                    morgan_probability[compound_index, label_index]
                ),
                "morgan_threshold": float(morgan_threshold),
                "morgan_prediction": int(
                    morgan_prediction[compound_index, label_index]
                ),
                "graphmvp_probability": float(
                    graphmvp_probability[compound_index, label_index]
                ),
                "graphmvp_threshold": float(graphmvp_threshold),
                "graphmvp_prediction": int(
                    graphmvp_prediction[compound_index, label_index]
                ),
            }
        )


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(results_dir)

    predictions_file = discover_unique_file(
        args.predictions_file,
        results_dir,
        ".npz",
        compatible_prediction_file,
        "prediction NPZ",
    )
    runs_file = discover_unique_file(
        args.runs_file,
        results_dir,
        ".tsv",
        compatible_runs_file,
        "repeated-runs TSV",
    )

    print("Predictions:", predictions_file)
    print("Runs:", runs_file)

    runs = pd.read_csv(runs_file, sep="\t", low_memory=False)
    threshold_column = (
        "tuned_threshold" if "tuned_threshold" in runs.columns else "threshold"
    )

    available_eval_sets = sorted(runs["eval_set"].dropna().astype(str).unique())
    if args.eval_set not in available_eval_sets:
        raise ValueError(
            f"Eval set {args.eval_set!r} not found. Available: {available_eval_sets}"
        )

    model_rows = runs[
        runs["eval_set"].astype(str).eq(args.eval_set)
        & runs["feature"].astype(str).isin(
            [MORGAN_FEATURE, GRAPHMVP_FEATURE]
        )
    ].copy()
    if model_rows.empty:
        raise ValueError(
            "No Morgan r2 or GraphMVP-GIN rows were found for the requested eval set"
        )

    split_pairs = (
        model_rows[["split_mode", "split_seed"]]
        .drop_duplicates()
        .sort_values(["split_mode", "split_seed"])
    )

    metadata = load_optional_metadata(
        args.metadata_file,
        args.metadata_id_column,
    )

    overlap_rows: list[dict] = []
    pathway_rows: list[dict] = []
    detail_rows: list[dict] = []

    with np.load(predictions_file, allow_pickle=True) as payload:
        label_columns = [str(value) for value in payload["label_columns"]]

        for split_mode, split_seed in split_pairs.itertuples(index=False, name=None):
            split_mode = str(split_mode)
            split_seed = int(split_seed)
            prefix = f"{split_mode}_seed{split_seed}_{args.eval_set}"
            required_keys = {
                "ids": f"{prefix}_ids",
                "y_true": f"{prefix}_y_true",
                "morgan_probability": (
                    f"{prefix}_{MORGAN_FEATURE}_probability"
                ),
                "graphmvp_probability": (
                    f"{prefix}_{GRAPHMVP_FEATURE}_probability"
                ),
            }
            missing = [
                key for key in required_keys.values() if key not in payload.files
            ]
            if missing:
                raise KeyError(
                    f"Missing prediction arrays for {split_mode} seed {split_seed}: "
                    + ", ".join(missing)
                )

            compound_ids = payload[required_keys["ids"]].astype(str)
            y_true = np.asarray(payload[required_keys["y_true"]], dtype=int)
            morgan_probability = np.asarray(
                payload[required_keys["morgan_probability"]], dtype=float
            )
            graphmvp_probability = np.asarray(
                payload[required_keys["graphmvp_probability"]], dtype=float
            )

            expected_shape = (len(compound_ids), len(label_columns))
            for name, array in [
                ("y_true", y_true),
                ("morgan_probability", morgan_probability),
                ("graphmvp_probability", graphmvp_probability),
            ]:
                if array.shape != expected_shape:
                    raise ValueError(
                        f"{prefix}: {name} has shape {array.shape}, "
                        f"expected {expected_shape}"
                    )
            if np.any((y_true != 0) & (y_true != 1)):
                raise ValueError(f"{prefix}: y_true contains non-binary values")

            morgan_threshold = one_threshold(
                runs,
                split_mode,
                split_seed,
                args.eval_set,
                MORGAN_FEATURE,
                threshold_column,
            )
            graphmvp_threshold = one_threshold(
                runs,
                split_mode,
                split_seed,
                args.eval_set,
                GRAPHMVP_FEATURE,
                threshold_column,
            )

            morgan_prediction = (
                morgan_probability >= morgan_threshold
            ).astype(np.uint8)
            graphmvp_prediction = (
                graphmvp_probability >= graphmvp_threshold
            ).astype(np.uint8)

            error_masks = {
                "false_positive": {
                    "morgan": (y_true == 0) & (morgan_prediction == 1),
                    "graphmvp": (y_true == 0) & (graphmvp_prediction == 1),
                },
                "false_negative": {
                    "morgan": (y_true == 1) & (morgan_prediction == 0),
                    "graphmvp": (y_true == 1) & (graphmvp_prediction == 0),
                },
            }

            for error_type, masks in error_masks.items():
                morgan_mask = masks["morgan"]
                graphmvp_mask = masks["graphmvp"]
                shared = morgan_mask & graphmvp_mask
                morgan_only = morgan_mask & ~graphmvp_mask
                graphmvp_only = graphmvp_mask & ~morgan_mask
                union = morgan_mask | graphmvp_mask

                overlap_rows.append(
                    {
                        "split_mode": split_mode,
                        "split_seed": split_seed,
                        "eval_set": args.eval_set,
                        "error_type": error_type,
                        "morgan_threshold": morgan_threshold,
                        "graphmvp_threshold": graphmvp_threshold,
                        "morgan_total": int(morgan_mask.sum()),
                        "graphmvp_total": int(graphmvp_mask.sum()),
                        "shared": int(shared.sum()),
                        "morgan_only": int(morgan_only.sum()),
                        "graphmvp_only": int(graphmvp_only.sum()),
                        "union": int(union.sum()),
                    }
                )

                pathway_role = (
                    "incorrectly_predicted_pathway"
                    if error_type == "false_positive"
                    else "missed_true_pathway"
                )
                for unique_to, unique_mask in [
                    ("morgan_r2", morgan_only),
                    ("graphmvp_gin", graphmvp_only),
                ]:
                    counts = unique_mask.sum(axis=0).astype(int)
                    for label_index, (pathway, count) in enumerate(
                        zip(label_columns, counts)
                    ):
                        pathway_rows.append(
                            {
                                "split_mode": split_mode,
                                "split_seed": split_seed,
                                "eval_set": args.eval_set,
                                "error_type": error_type,
                                "unique_to": unique_to,
                                "pathway_index": label_index,
                                "pathway": pathway,
                                "pathway_role": pathway_role,
                                "n_errors": int(count),
                            }
                        )

                append_detail_rows(
                    detail_rows,
                    morgan_only,
                    compound_ids,
                    label_columns,
                    split_mode,
                    split_seed,
                    args.eval_set,
                    error_type,
                    "morgan_r2",
                    morgan_probability,
                    graphmvp_probability,
                    morgan_prediction,
                    graphmvp_prediction,
                    morgan_threshold,
                    graphmvp_threshold,
                )
                append_detail_rows(
                    detail_rows,
                    graphmvp_only,
                    compound_ids,
                    label_columns,
                    split_mode,
                    split_seed,
                    args.eval_set,
                    error_type,
                    "graphmvp_gin",
                    morgan_probability,
                    graphmvp_probability,
                    morgan_prediction,
                    graphmvp_prediction,
                    morgan_threshold,
                    graphmvp_threshold,
                )

    overlap_by_seed = pd.DataFrame(overlap_rows).sort_values(
        ["split_mode", "split_seed", "error_type"]
    )
    overlap_summary = summarize_numeric_counts(
        overlap_by_seed,
        group_columns=["split_mode", "eval_set", "error_type"],
        count_columns=[
            "morgan_total",
            "graphmvp_total",
            "shared",
            "morgan_only",
            "graphmvp_only",
            "union",
        ],
    ).sort_values(["split_mode", "error_type"])

    pathways_by_seed = pd.DataFrame(pathway_rows).sort_values(
        [
            "split_mode",
            "error_type",
            "unique_to",
            "pathway_index",
            "split_seed",
        ]
    )
    pathway_summary_rows: list[dict] = []
    group_columns = [
        "split_mode",
        "eval_set",
        "error_type",
        "unique_to",
        "pathway_index",
        "pathway",
        "pathway_role",
    ]
    for keys, subset in pathways_by_seed.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        row = dict(zip(group_columns, keys))
        values = subset["n_errors"].astype(float).to_numpy()
        row["n_split_seeds"] = int(subset["split_seed"].nunique())
        row["n_seeds_with_error"] = int((values > 0).sum())
        row["n_errors_mean"] = float(np.mean(values))
        row["n_errors_sd"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        )
        row["n_errors_min"] = int(np.min(values))
        row["n_errors_max"] = int(np.max(values))
        pathway_summary_rows.append(row)
    pathways_summary = pd.DataFrame(pathway_summary_rows).sort_values(
        [
            "split_mode",
            "error_type",
            "unique_to",
            "n_errors_mean",
            "pathway_index",
        ],
        ascending=[True, True, True, False, True],
    )

    details = pd.DataFrame(detail_rows)
    if not details.empty and metadata is not None:
        details = details.merge(
            metadata,
            left_on="compound_id",
            right_on=args.metadata_id_column,
            how="left",
            validate="many_to_one",
            sort=False,
        )
        if args.metadata_id_column != "compound_id":
            details = details.drop(columns=[args.metadata_id_column])
    if not details.empty:
        details = details.sort_values(
            [
                "split_mode",
                "split_seed",
                "error_type",
                "unique_to",
                "pathway_index",
                "compound_id",
            ]
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "error overlap by seed": output_dir / "error_overlap_by_seed.tsv",
        "error overlap summary": output_dir / "error_overlap_summary.tsv",
        "unique pathways by seed": (
            output_dir / "unique_error_pathways_by_seed.tsv"
        ),
        "unique pathways summary": (
            output_dir / "unique_error_pathways_summary.tsv"
        ),
        "unique error details": output_dir / "unique_error_details.tsv",
    }

    overlap_by_seed.to_csv(output_paths["error overlap by seed"], sep="\t", index=False)
    overlap_summary.to_csv(output_paths["error overlap summary"], sep="\t", index=False)
    pathways_by_seed.to_csv(output_paths["unique pathways by seed"], sep="\t", index=False)
    pathways_summary.to_csv(output_paths["unique pathways summary"], sep="\t", index=False)
    details.to_csv(output_paths["unique error details"], sep="\t", index=False)

    print("\n=== Error overlap by split seed ===")
    display_columns = [
        "split_mode",
        "split_seed",
        "error_type",
        "morgan_total",
        "graphmvp_total",
        "shared",
        "morgan_only",
        "graphmvp_only",
    ]
    print(overlap_by_seed[display_columns].to_string(index=False))

    print("\nWrote:")
    for description, path in output_paths.items():
        print(f"  {description}: {path}")


if __name__ == "__main__":
    main()

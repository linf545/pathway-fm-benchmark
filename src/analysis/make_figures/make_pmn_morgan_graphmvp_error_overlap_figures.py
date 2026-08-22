#!/usr/bin/env python3
"""Create main figures for Morgan r2 vs GraphMVP-GIN error overlap.

This script summarizes PMN secondary-pathway false-positive and false-negative
errors on the Bao Dataset B external test set. It generates:

1. A pathway-resolved stacked-bar figure in which each pathway has two bars
   (false positive and false negative), and each bar is partitioned into
   GraphMVP-only errors, shared errors, and Morgan-only errors.
2. A global stacked-bar figure with the same three components but aggregated
   across pathways.

Counts are averaged across split seeds separately for random and scaffold
splits.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MORGAN_FEATURE = "morgan_r2_2048"
GRAPHMVP_FEATURE = "graphmvp_gin"
DEFAULT_EVAL_SET = "bao_B_external_test"

COMPONENT_ORDER = ["graphmvp_only", "shared", "morgan_only"]
COMPONENT_LABELS = {
    "graphmvp_only": "Unique to GraphMVP-GIN",
    "shared": "Overlapping errors",
    "morgan_only": "Unique to Morgan r2",
}
COMPONENT_COLORS = {
    "graphmvp_only": "#4C78A8",
    "shared": "#9E9E9E",
    "morgan_only": "#F58518",
}
ERROR_ORDER = ["false_positive", "false_negative"]
ERROR_LABELS = {
    "false_positive": "FP",
    "false_negative": "FN",
}
SPLIT_ORDER = ["random", "scaffold"]
SPLIT_LABELS = {"random": "Random split", "scaffold": "Scaffold split"}


# ---------- shared utilities ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--results-dir",
        default="results/pmn_secondary_pathway/full",
        help="Directory containing the PMN full-data benchmark outputs.",
    )
    parser.add_argument(
        "--predictions-file",
        default=None,
        help="Saved prediction NPZ. If omitted, the script searches --results-dir.",
    )
    parser.add_argument(
        "--runs-file",
        default=None,
        help="Repeated-runs TSV with thresholds. If omitted, the script searches --results-dir.",
    )
    parser.add_argument("--eval-set", default=DEFAULT_EVAL_SET)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for figures. Default: results/figures/main",
    )
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def parse_formats(value: str) -> tuple[str, ...]:
    return tuple(fmt.strip() for fmt in value.split(",") if fmt.strip())


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def compatible_prediction_file(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=True) as payload:
            keys = payload.files
            return (
                "label_columns" in keys
                and any(key.endswith(f"_{MORGAN_FEATURE}_probability") for key in keys)
                and any(key.endswith(f"_{GRAPHMVP_FEATURE}_probability") for key in keys)
                and any(key.endswith("_y_true") for key in keys)
            )
    except Exception:
        return False


def compatible_runs_file(path: Path) -> bool:
    try:
        columns = set(pd.read_csv(path, sep="\t", nrows=2).columns)
    except Exception:
        return False
    required = {"feature", "split_mode", "split_seed", "eval_set"}
    return required.issubset(columns) and ("tuned_threshold" in columns or "threshold" in columns)


def discover_unique_file(explicit: str | None, results_dir: Path, suffix: str, check, description: str) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        if not check(path):
            raise ValueError(f"Incompatible {description}: {path}")
        return path
    candidates = [path for path in sorted(results_dir.rglob(f"*{suffix}")) if check(path)]
    if not candidates:
        raise FileNotFoundError(f"No compatible {description} found under {results_dir}")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple compatible {description} files found under {results_dir}:\n  "
            + "\n  ".join(str(p) for p in candidates)
        )
    return candidates[0]


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
            f"Expected one threshold row for {split_mode}, seed {split_seed}, {eval_set}, {feature}; found {len(subset)}"
        )
    value = float(subset[threshold_column].iloc[0])
    if not np.isfinite(value):
        raise ValueError(f"Invalid threshold {value} for {feature}")
    return value


def prettify_pathway(name: str) -> str:
    text = str(name)
    text = text.replace("label_", "")
    parts = text.split("_", 2)
    if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
        text = parts[2]
    text = text.replace("_", " ")
    return text


# ---------- data preparation ----------

def compute_error_counts(predictions_file: Path, runs_file: Path, eval_set: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(runs_file, sep="\t", low_memory=False)
    threshold_column = "tuned_threshold" if "tuned_threshold" in runs.columns else "threshold"
    model_rows = runs[
        runs["eval_set"].astype(str).eq(eval_set)
        & runs["feature"].astype(str).isin([MORGAN_FEATURE, GRAPHMVP_FEATURE])
    ].copy()
    split_pairs = model_rows[["split_mode", "split_seed"]].drop_duplicates().sort_values(["split_mode", "split_seed"])

    overall_rows: list[dict] = []
    pathway_rows: list[dict] = []

    with np.load(predictions_file, allow_pickle=True) as payload:
        label_columns = [str(value) for value in payload["label_columns"]]
        for split_mode, split_seed in split_pairs.itertuples(index=False, name=None):
            split_mode = str(split_mode)
            split_seed = int(split_seed)
            prefix = f"{split_mode}_seed{split_seed}_{eval_set}"
            ids_key = f"{prefix}_ids"
            y_key = f"{prefix}_y_true"
            m_key = f"{prefix}_{MORGAN_FEATURE}_probability"
            g_key = f"{prefix}_{GRAPHMVP_FEATURE}_probability"
            for key in [ids_key, y_key, m_key, g_key]:
                if key not in payload.files:
                    raise KeyError(f"Missing array in prediction file: {key}")

            y_true = np.asarray(payload[y_key], dtype=int)
            morgan_probability = np.asarray(payload[m_key], dtype=float)
            graphmvp_probability = np.asarray(payload[g_key], dtype=float)

            morgan_threshold = one_threshold(runs, split_mode, split_seed, eval_set, MORGAN_FEATURE, threshold_column)
            graphmvp_threshold = one_threshold(runs, split_mode, split_seed, eval_set, GRAPHMVP_FEATURE, threshold_column)
            morgan_prediction = (morgan_probability >= morgan_threshold).astype(np.uint8)
            graphmvp_prediction = (graphmvp_probability >= graphmvp_threshold).astype(np.uint8)

            masks = {
                "false_positive": {
                    "morgan": (y_true == 0) & (morgan_prediction == 1),
                    "graphmvp": (y_true == 0) & (graphmvp_prediction == 1),
                },
                "false_negative": {
                    "morgan": (y_true == 1) & (morgan_prediction == 0),
                    "graphmvp": (y_true == 1) & (graphmvp_prediction == 0),
                },
            }

            for error_type, entry in masks.items():
                morgan_mask = entry["morgan"]
                graph_mask = entry["graphmvp"]
                components = {
                    "graphmvp_only": graph_mask & ~morgan_mask,
                    "shared": graph_mask & morgan_mask,
                    "morgan_only": morgan_mask & ~graph_mask,
                }
                for component, mask in components.items():
                    overall_rows.append(
                        {
                            "split_mode": split_mode,
                            "split_seed": split_seed,
                            "eval_set": eval_set,
                            "error_type": error_type,
                            "component": component,
                            "n_errors": int(mask.sum()),
                        }
                    )
                    counts = mask.sum(axis=0).astype(int)
                    for pathway_index, (pathway, count) in enumerate(zip(label_columns, counts)):
                        pathway_rows.append(
                            {
                                "split_mode": split_mode,
                                "split_seed": split_seed,
                                "eval_set": eval_set,
                                "error_type": error_type,
                                "component": component,
                                "pathway_index": int(pathway_index),
                                "pathway": pathway,
                                "pathway_label": prettify_pathway(pathway),
                                "n_errors": int(count),
                            }
                        )
    return pd.DataFrame(overall_rows), pd.DataFrame(pathway_rows)


def summarize_means(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    summary = (
        frame.groupby(group_columns, dropna=False, sort=True)["n_errors"]
        .agg([("n_split_seeds", "count"), ("n_errors_mean", "mean"), ("n_errors_sd", "std")])
        .reset_index()
    )
    return summary


# ---------- plotting ----------

def plot_pathway_stacks(ax: plt.Axes, summary: pd.DataFrame, split_mode: str) -> None:
    subset = summary[summary["split_mode"] == split_mode].copy()
    pathways = (
        subset[["pathway_index", "pathway_label"]]
        .drop_duplicates()
        .sort_values("pathway_index")
    )
    x = np.arange(len(pathways))
    width = 0.38
    offsets = {"false_positive": -width / 2, "false_negative": width / 2}

    for error_type in ERROR_ORDER:
        bottom = np.zeros(len(pathways), dtype=float)
        for component in COMPONENT_ORDER:
            current = subset[
                (subset["error_type"] == error_type)
                & (subset["component"] == component)
            ][["pathway_index", "n_errors_mean"]].copy()
            merged = pathways.merge(current, on="pathway_index", how="left").fillna({"n_errors_mean": 0.0})
            values = merged["n_errors_mean"].to_numpy(dtype=float)
            ax.bar(
                x + offsets[error_type],
                values,
                width=width,
                bottom=bottom,
                color=COMPONENT_COLORS[component],
                edgecolor="white",
                linewidth=0.4,
                hatch="" if error_type == "false_positive" else "///",
                label=f"{COMPONENT_LABELS[component]} ({ERROR_LABELS[error_type]})",
            )
            bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(pathways["pathway_label"], rotation=75, ha="right")
    ax.set_ylabel("Mean error count across split seeds")
    ax.set_title(SPLIT_LABELS.get(split_mode, split_mode))
    ax.grid(axis="y", color="0.90", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_overall_stacks(ax: plt.Axes, summary: pd.DataFrame, split_mode: str) -> None:
    subset = summary[summary["split_mode"] == split_mode].copy()
    x = np.arange(len(ERROR_ORDER))
    width = 0.6
    bottom = np.zeros(len(ERROR_ORDER), dtype=float)
    for component in COMPONENT_ORDER:
        values = []
        for error_type in ERROR_ORDER:
            current = subset[
                (subset["error_type"] == error_type)
                & (subset["component"] == component)
            ]
            values.append(float(current["n_errors_mean"].iloc[0]) if not current.empty else 0.0)
        ax.bar(
            x,
            values,
            width=width,
            bottom=bottom,
            color=COMPONENT_COLORS[component],
            edgecolor="white",
            linewidth=0.5,
            label=COMPONENT_LABELS[component],
        )
        bottom += np.asarray(values)
    ax.set_xticks(x)
    ax.set_xticklabels([ERROR_LABELS[e] for e in ERROR_ORDER])
    ax.set_ylabel("Mean error count across split seeds")
    ax.set_title(SPLIT_LABELS.get(split_mode, split_mode))
    ax.grid(axis="y", color="0.90", linewidth=0.6)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: Path, formats: Iterable[str], dpi: int) -> list[Path]:
    outputs: list[Path] = []
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    project_root = args.project_root.resolve()
    results_dir = (project_root / args.results_dir).resolve()
    output_dir = (args.output_dir or (project_root / "results/figures/main")).resolve()
    source_dir = output_dir / "source_data"

    predictions_file = discover_unique_file(args.predictions_file, results_dir, ".npz", compatible_prediction_file, "prediction NPZ")
    runs_file = discover_unique_file(args.runs_file, results_dir, ".tsv", compatible_runs_file, "repeated-runs TSV")

    overall_by_seed, pathways_by_seed = compute_error_counts(predictions_file, runs_file, args.eval_set)
    overall_summary = summarize_means(overall_by_seed, ["split_mode", "eval_set", "error_type", "component"])
    pathway_summary = summarize_means(
        pathways_by_seed,
        ["split_mode", "eval_set", "error_type", "component", "pathway_index", "pathway", "pathway_label"],
    )

    # Write source data used by the plots.
    source_dir.mkdir(parents=True, exist_ok=True)
    overall_by_seed.to_csv(source_dir / "pmn_morgan_graphmvp_error_overlap_overall_by_seed.tsv", sep="\t", index=False)
    overall_summary.to_csv(source_dir / "pmn_morgan_graphmvp_error_overlap_overall_summary.tsv", sep="\t", index=False)
    pathways_by_seed.to_csv(source_dir / "pmn_morgan_graphmvp_error_overlap_pathway_by_seed.tsv", sep="\t", index=False)
    pathway_summary.to_csv(source_dir / "pmn_morgan_graphmvp_error_overlap_pathway_summary.tsv", sep="\t", index=False)

    formats = parse_formats(args.formats)
    outputs: list[Path] = []

    # Figure 1: pathway-specific stacked bars.
    fig1, axes1 = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, sharex=True)
    for ax, split_mode in zip(axes1, SPLIT_ORDER):
        plot_pathway_stacks(ax, pathway_summary, split_mode)
    # One combined legend: colors identify model-specific/shared errors, and
    # hatching identifies false-positive versus false-negative bars.
    component_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COMPONENT_COLORS[c])
        for c in COMPONENT_ORDER
    ]
    component_labels = [COMPONENT_LABELS[c] for c in COMPONENT_ORDER]
    error_handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor="white",
            edgecolor="black",
            hatch="",
        ),
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor="white",
            edgecolor="black",
            hatch="///",
        ),
    ]
    error_labels = ["FP", "FN"]
    fig1.legend(
        component_handles + error_handles,
        component_labels + error_labels,
        loc="outside upper center",
        ncol=5,
        frameon=False,
        columnspacing=1.5,
        handlelength=1.8,
    )
    outputs.extend(save_figure(fig1, output_dir / "pmn_morgan_graphmvp_error_overlap_by_pathway", formats, args.dpi))

    # Figure 2: overall stacked bars.
    fig2, axes2 = plt.subplots(1, 2, figsize=(7.15, 3.6), constrained_layout=True, sharey=True)
    for ax, split_mode in zip(axes2, SPLIT_ORDER):
        plot_overall_stacks(ax, overall_summary, split_mode)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COMPONENT_COLORS[c]) for c in COMPONENT_ORDER]
    labels = [COMPONENT_LABELS[c] for c in COMPONENT_ORDER]
    fig2.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    outputs.extend(save_figure(fig2, output_dir / "pmn_morgan_graphmvp_error_overlap_overall", formats, args.dpi))

    print("Predictions:", predictions_file)
    print("Runs:", runs_file)
    print("Wrote source data to:", source_dir)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

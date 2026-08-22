#!/usr/bin/env python3
"""Plot PlantCyc reaction-neighbor sample-efficiency results by negative variant.

This script reads the summary table produced by the PlantCyc data-efficiency
runner and generates one standalone two-panel figure per requested negative
variant. Each figure contains random and scaffold-pair split panels and is
saved under results/figures/main by default.

Default outputs
---------------
results/figures/main/plantcyc_reaction_neighbor_sample_efficiency_easy.pdf
results/figures/main/plantcyc_reaction_neighbor_sample_efficiency_mixed.pdf
results/figures/main/plantcyc_reaction_neighbor_sample_efficiency_hard.pdf
and matching PNG files.

Source-data files are written to:
results/figures/source_data/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FEATURE_ORDER = [
    "morgan_r2_2048",
    "morgan_r3_2048",
    "rdkit_descriptors",
    "chemberta_77m_mlm",
    "chemberta_100m_mlm",
    "molformer_c3_1_1b",
    "nafm_graph_mean",
    "molclr_gcn",
    "molclr_gin",
    "graphmvp_gcn",
    "graphmvp_gin",
    "molebert_gin",
]

DISPLAY_NAME = {
    "morgan_r2_2048": "Morgan r2",
    "morgan_r3_2048": "Morgan r3",
    "rdkit_descriptors": "RDKit descriptors",
    "chemberta_77m_mlm": "ChemBERTa-77M",
    "chemberta_100m_mlm": "ChemBERTa-100M",
    "molformer_c3_1_1b": "MolFormer",
    "nafm_graph_mean": "NaFM",
    "molclr_gcn": "MolCLR-GCN",
    "molclr_gin": "MolCLR-GIN",
    "graphmvp_gcn": "GraphMVP-GCN",
    "graphmvp_gin": "GraphMVP-GIN",
    "molebert_gin": "Mole-BERT-GIN",
}

COLOR = {
    "morgan_r2_2048": "#222222",
    "morgan_r3_2048": "#666666",
    "rdkit_descriptors": "#8c564b",
    "chemberta_77m_mlm": "#1f77b4",
    "chemberta_100m_mlm": "#17becf",
    "molformer_c3_1_1b": "#d62728",
    "nafm_graph_mean": "#9467bd",
    "molclr_gcn": "#bcbd22",
    "molclr_gin": "#2ca02c",
    "graphmvp_gcn": "#ff7f0e",
    "graphmvp_gin": "#e377c2",
    "molebert_gin": "#7f7f7f",
}

MARKER = {
    "morgan_r2_2048": "o",
    "morgan_r3_2048": "s",
    "rdkit_descriptors": "D",
    "chemberta_77m_mlm": "^",
    "chemberta_100m_mlm": "v",
    "molformer_c3_1_1b": "P",
    "nafm_graph_mean": "X",
    "molclr_gcn": "<",
    "molclr_gin": ">",
    "graphmvp_gcn": "h",
    "graphmvp_gin": "H",
    "molebert_gin": "*",
}

SPLIT_LABEL = {
    "random": "Random split",
    "scaffold": "Scaffold-pair split",
    "scaffold_pair": "Scaffold-pair split",
}

VARIANT_LABEL = {
    "easy": "Easy negatives",
    "mixed": "Mixed negatives",
    "hard": "Hard negatives",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-tsv",
        type=Path,
        default=Path(
            "results/plantcyc_reaction_neighbor/data_efficiency/"
            "plantcyc_reaction_neighbor_data_efficiency_summary.tsv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/main"),
    )
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        default=Path("results/figures/source_data"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["easy", "mixed", "hard"],
        choices=["easy", "mixed", "hard"],
        help="Negative variants to plot. One figure is produced per variant.",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional subset of features to plot.",
    )
    parser.add_argument(
        "--metric",
        default="auroc",
        choices=["auroc", "auprc", "f1", "balanced_accuracy", "mcc"],
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--xscale",
        default="log",
        choices=["log", "linear"],
    )
    parser.add_argument(
        "--legend-columns",
        type=int,
        default=3,
    )
    return parser.parse_args()


def choose_feature_order(
    features_in_data: Iterable[str],
    requested: list[str] | None,
) -> list[str]:
    present = set(features_in_data)
    if requested:
        missing = [f for f in requested if f not in present]
        if missing:
            raise ValueError(
                "Requested features are absent from the summary table: "
                + ", ".join(missing)
            )
        return [f for f in requested if f in present]
    ordered = [f for f in DEFAULT_FEATURE_ORDER if f in present]
    remainder = sorted(present - set(ordered))
    return ordered + remainder


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    frame["split_scheme"] = frame["split_scheme"].replace(
        {"scaffold_pair": "scaffold"}
    )
    return frame


def filter_plot_table(frame: pd.DataFrame, variant: str, metric: str) -> pd.DataFrame:
    required = {
        "feature",
        "negative_variant",
        "split_scheme",
        "requested_n_train",
        "eval_set",
        f"{metric}_mean",
        f"{metric}_ci95_low",
        f"{metric}_ci95_high",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Summary table is missing required columns: {sorted(missing)}")

    plot = frame[
        frame["negative_variant"].astype(str).eq(variant)
        & frame["eval_set"].astype(str).eq("test")
        & frame["split_scheme"].astype(str).isin(["random", "scaffold"])
    ].copy()
    if plot.empty:
        raise ValueError(f"No rows found for negative variant '{variant}'.")
    plot["requested_n_train"] = pd.to_numeric(plot["requested_n_train"], errors="raise").astype(int)
    plot = plot.sort_values(["feature", "split_scheme", "requested_n_train"])
    return plot


def export_source_data(plot: pd.DataFrame, metric: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "feature",
        "split_scheme",
        "negative_variant",
        "requested_n_train",
        f"{metric}_mean",
        f"{metric}_ci95_low",
        f"{metric}_ci95_high",
        "n_split_seeds",
        "n_sampling_replicates_per_split_min",
        "n_sampling_replicates_per_split_max",
        "actual_n_train_mean",
    ]
    out = plot[cols].copy()
    out.insert(1, "feature_display_name", out["feature"].map(lambda x: DISPLAY_NAME.get(x, x)))
    out.insert(3, "split_display_name", out["split_scheme"].map(lambda x: SPLIT_LABEL.get(x, x)))
    out.insert(5, "negative_variant_display_name", out["negative_variant"].map(lambda x: VARIANT_LABEL.get(x, x)))
    out.to_csv(path, sep="\t", index=False)


def plot_metric_panel(ax, subset: pd.DataFrame, feature_order: list[str], metric: str, xscale: str) -> None:
    for feature in feature_order:
        one = subset[subset["feature"].astype(str).eq(feature)].copy()
        if one.empty:
            continue
        x = one["requested_n_train"].to_numpy(dtype=float)
        y = one[f"{metric}_mean"].to_numpy(dtype=float)
        y_low = one[f"{metric}_ci95_low"].to_numpy(dtype=float)
        y_high = one[f"{metric}_ci95_high"].to_numpy(dtype=float)
        color = COLOR.get(feature)
        marker = MARKER.get(feature, "o")
        label = DISPLAY_NAME.get(feature, feature)
        ax.plot(x, y, marker=marker, label=label, linewidth=1.8, markersize=5.5, color=color)
        if np.all(np.isfinite(y_low)) and np.all(np.isfinite(y_high)):
            ax.fill_between(x, y_low, y_high, alpha=0.18, linewidth=0, color=color)

    ax.set_xscale(xscale)
    ax.set_xlabel("Training pairs")
    ax.set_ylabel({
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "f1": "F1",
        "balanced_accuracy": "Balanced accuracy",
        "mcc": "MCC",
    }[metric])
    ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)
    ax.set_axisbelow(True)

    finite = pd.concat([subset[f"{metric}_ci95_low"], subset[f"{metric}_ci95_high"]], ignore_index=True)
    finite = pd.to_numeric(finite, errors="coerce")
    finite = finite[np.isfinite(finite.to_numpy())]
    if len(finite) > 0:
        ymin = max(0.0, float(finite.min()) - 0.03)
        ymax = min(1.0, float(finite.max()) + 0.03)
        if ymax - ymin < 0.15:
            midpoint = 0.5 * (ymin + ymax)
            ymin = max(0.0, midpoint - 0.08)
            ymax = min(1.0, midpoint + 0.08)
        ax.set_ylim(ymin, ymax)


def make_variant_figure(plot: pd.DataFrame, variant: str, args: argparse.Namespace):
    feature_order = choose_feature_order(plot["feature"], args.features)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True, constrained_layout=False)
    split_order = ["random", "scaffold"]
    panel_labels = ["A", "B"]
    for ax, split_scheme, panel_label in zip(axes, split_order, panel_labels):
        subset = plot[plot["split_scheme"].astype(str).eq(split_scheme)].copy()
        if subset.empty:
            raise ValueError(f"No rows found for split scheme {split_scheme} in variant {variant}")
        plot_metric_panel(ax, subset, feature_order, args.metric, args.xscale)
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
        ax.text(0.5, -0.28, SPLIT_LABEL.get(split_scheme, split_scheme), transform=ax.transAxes, ha="center", va="top", fontsize=10)
        if args.xscale == "log":
            ticks = sorted(subset["requested_n_train"].astype(int).unique())
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(v) for v in ticks])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=args.legend_columns, frameon=False, fontsize=9, handlelength=2.2, columnspacing=1.2)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.27, top=0.76, wspace=0.15)
    return fig


def main() -> None:
    args = parse_args()
    summary = load_summary(args.summary_tsv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.source_data_dir.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        plot = filter_plot_table(summary, variant=variant, metric=args.metric)
        fig = make_variant_figure(plot, variant, args)
        stem = f"plantcyc_reaction_neighbor_sample_efficiency_{variant}"
        for fmt in args.formats:
            out_path = args.output_dir / f"{stem}.{fmt}"
            save_kwargs = {"dpi": args.dpi} if fmt.lower() in {"png", "jpg", "jpeg", "tif", "tiff"} else {}
            fig.savefig(out_path, bbox_inches="tight", **save_kwargs)
            print(f"Saved figure: {out_path}")
        source_path = args.source_data_dir / f"{stem}_source_data.tsv"
        export_source_data(plot, metric=args.metric, path=source_path)
        print(f"Saved source data: {source_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()

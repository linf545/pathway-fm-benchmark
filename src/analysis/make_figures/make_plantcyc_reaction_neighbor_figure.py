#!/usr/bin/env python3
"""Generate the main PlantCyc reaction-neighbor figure.

The current project has no PlantCyc label-efficiency output. This figure therefore
focuses on full-data performance across negative-set difficulty and split scheme.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    NEGATIVE_LABELS,
    SPLIT_LABELS,
    assert_finite,
    assert_unique,
    configure_matplotlib,
    feature_label,
    ordered_features,
    parse_formats,
    read_tsv,
    resolve_paths,
    save_figure,
    set_panel_label,
    stable_jitter,
    uncertainty_half_width,
    write_source_data,
)


def plot_negative_panel(
    ax,
    data: pd.DataFrame,
    feature_order: list[str],
    negative_variant: str,
) -> pd.DataFrame:
    subset = data[data["negative_variant"] == negative_variant].copy()
    split_order = ("random", "scaffold_pair")
    split_colors = {"random": "0.15", "scaffold_pair": "0.60"}
    offsets = {"random": -0.10, "scaffold_pair": 0.10}
    rows: list[dict] = []

    for y_index, feature in enumerate(feature_order):
        for split in split_order:
            current = subset[(subset["feature"] == feature) & (subset["split_scheme"] == split)]
            if current.empty:
                continue
            y_center = y_index + offsets[split]
            jitter = stable_jitter(current["split_seed"].tolist(), width=0.035)
            ax.scatter(
                current["auroc"],
                np.full(len(current), y_center) + jitter,
                s=12,
                facecolors="none",
                edgecolors=split_colors[split],
                linewidths=0.65,
                alpha=0.8,
                zorder=2,
            )
            mean = float(current["auroc"].mean())
            sd = float(current["auroc"].std(ddof=1)) if len(current) > 1 else 0.0
            n = int(len(current))
            se = sd / np.sqrt(n) if n else 0.0
            half_width = float(uncertainty_half_width(
                pd.DataFrame({"sd": [sd], "se": [se], "n": [n]}), "ci95"
            )[0])
            ax.errorbar(
                mean,
                y_center,
                xerr=half_width,
                fmt="o",
                color=split_colors[split],
                markerfacecolor=split_colors[split],
                markeredgecolor="white",
                markeredgewidth=0.4,
                capsize=2.2,
                linewidth=1.0,
                markersize=4.5,
                label=SPLIT_LABELS[split] if y_index == 0 else None,
                zorder=3,
            )
            rows.append(
                {
                    "feature": feature,
                    "feature_label": feature_label(feature),
                    "negative_variant": negative_variant,
                    "split_scheme": split,
                    "metric": "auroc",
                    "mean": mean,
                    "sd": sd,
                    "se": se,
                    "n": n,
                    "uncertainty": "ci95",
                    "uncertainty_half_width": half_width,
                }
            )

    ax.axvline(0.5, color="0.65", linestyle=":", linewidth=0.8)
    ax.set_yticks(range(len(feature_order)))
    ax.set_yticklabels([feature_label(feature) for feature in feature_order])
    ax.invert_yaxis()
    ax.set_xlabel("AUROC")
    ax.set_title(NEGATIVE_LABELS[negative_variant])
    ax.grid(axis="x", color="0.90", linewidth=0.6)
    ax.set_axisbelow(True)
    return pd.DataFrame(rows)


def make_figure(
    project_root: Path,
    output_dir: Path | None = None,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
) -> list[Path]:
    paths = resolve_paths(__file__, project_root, output_dir)
    configure_matplotlib()

    data = read_tsv(
        paths.project_root / "results/plantcyc_reaction_neighbor/full/reaction_neighbor_repeated_runs.tsv",
        [
            "feature",
            "negative_variant",
            "split_scheme",
            "split_seed",
            "eval_set",
            "auroc",
            "solver_converged",
        ],
    )
    data = data[data["eval_set"] == "test"].copy()
    
    # Exclude RDKit descriptors from the manuscript figure and source-data files.
    data = data[
        data["feature"].astype(str) != "rdkit_descriptors"
    ].copy()
    
    if not data["solver_converged"].all():
        raise ValueError("At least one selected PlantCyc model did not converge")
    assert_unique(
        data,
        ["feature", "negative_variant", "split_scheme", "split_seed"],
        "PlantCyc test results",
    )
    assert_finite(data, ["auroc"], "PlantCyc test results")

    features = ordered_features(data["feature"])
    variants = [variant for variant in ("easy", "mixed", "hard") if variant in set(data["negative_variant"])]

    fig, axes = plt.subplots(
        1,
        len(variants),
        figsize=(7.15, 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(variants) == 1:
        axes = [axes]

    sources: list[pd.DataFrame] = []
    for panel_index, (ax, variant) in enumerate(zip(axes, variants)):
        sources.append(plot_negative_panel(ax, data, features, variant))
        set_panel_label(ax, chr(ord("A") + panel_index))
        if panel_index > 0:
            ax.tick_params(labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)

    source = pd.concat(sources, ignore_index=True)
    write_source_data(source, paths.source_data_dir / "plantcyc_reaction_neighbor_main_source_data.tsv")
    write_source_data(data, paths.source_data_dir / "plantcyc_reaction_neighbor_raw.tsv")
    return save_figure(fig, paths.output_dir / "plantcyc_reaction_neighbor_main", formats, dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    project_root = args.project_root or Path(__file__).resolve().parents[3]
    outputs = make_figure(project_root, args.output_dir, parse_formats(args.formats), args.dpi)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

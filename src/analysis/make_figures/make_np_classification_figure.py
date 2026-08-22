#!/usr/bin/env python3
"""Generate the main NP-classification figure from current result tables."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    add_chance_line,
    assert_unique,
    configure_matplotlib,
    ordered_features,
    parse_formats,
    plot_full_dot_summary,
    plot_learning_curves,
    padded_limits,
    read_tsv,
    resolve_paths,
    save_figure,
    set_panel_label,
    summarize_replicates,
    write_source_data,
)


def make_figure(
    project_root: Path,
    output_dir: Path | None = None,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    curve_uncertainty: str = "ci95",
) -> list[Path]:
    paths = resolve_paths(__file__, project_root, output_dir)
    configure_matplotlib()

    full = read_tsv(
        paths.project_root / "results/np_classification/full/npclassification_repeated_runs.tsv",
        ["feature", "split", "split_seed", "eval_set", "auroc_ovr_macro", "reached_max_iter"],
    )
    efficiency = read_tsv(
        paths.project_root / "results/np_classification/label_efficiency/npclassification_label_efficiency_runs.tsv",
        [
            "feature",
            "split",
            "sample_seed",
            "requested_n_train",
            "eval_set",
            "auroc_ovr_macro",
            "reached_max_iter",
        ],
    )
    full = full[full["eval_set"] == "test"].copy()
    efficiency = efficiency[efficiency["eval_set"] == "test"].copy()
    if full["reached_max_iter"].any() or efficiency["reached_max_iter"].any():
        raise ValueError("At least one selected NP-classification model reached max_iter")

    assert_unique(full, ["feature", "split", "split_seed"], "NP-classification full test")
    assert_unique(
        efficiency,
        ["feature", "split", "sample_seed", "requested_n_train"],
        "NP-classification label-efficiency test",
    )

    features = ordered_features(set(full["feature"]) & set(efficiency["feature"]))
    summary = summarize_replicates(
        efficiency,
        ["feature", "split", "requested_n_train"],
        "auroc_ovr_macro",
        "sample_seed",
    )

    fig = plt.figure(figsize=(7.15, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0])
    ax_full = fig.add_subplot(grid[0, :])
    ax_random = fig.add_subplot(grid[1, 0])
    ax_scaffold = fig.add_subplot(grid[1, 1])

    full_source = plot_full_dot_summary(
        ax_full,
        full.rename(columns={"split": "split_mode"}),
        metric="auroc_ovr_macro",
        feature_order=features,
        split_column="split_mode",
        split_order=("random", "scaffold"),
        replicate_column="split_seed",
        ylabel="Macro AUROC",
        title=None,
        uncertainty="ci95",
    )
    ax_full.set_ylim(*padded_limits(full["auroc_ovr_macro"], floor=0.5))
    set_panel_label(ax_full, "A")

    curve_sources: list[pd.DataFrame] = []
    for ax, split, panel in [(ax_random, "random", "B"), (ax_scaffold, "scaffold", "C")]:
        raw = efficiency[efficiency["split"] == split]
        summed = summary[summary["split"] == split]
        exported = plot_learning_curves(
            ax,
            raw,
            summed,
            metric="auroc_ovr_macro",
            x_column="requested_n_train",
            feature_order=features,
            replicate_column="sample_seed",
            xlabel="Number of training examples",
            ylabel="Macro AUROC",
            title=f"{split.capitalize()} split",
            log_x=True,
            uncertainty=curve_uncertainty,
            show_replicates=False,
        )
        exported["panel_split"] = split
        curve_sources.append(exported)
        add_chance_line(ax)
        set_panel_label(ax, panel)

    handles, labels = ax_random.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)

    source = pd.concat(
        [
            full_source.assign(panel="full_data"),
            pd.concat(curve_sources, ignore_index=True).assign(panel="label_efficiency"),
        ],
        ignore_index=True,
        sort=False,
    )
    write_source_data(source, paths.source_data_dir / "np_classification_main_source_data.tsv")
    write_source_data(full, paths.source_data_dir / "np_classification_full_raw.tsv")
    write_source_data(efficiency, paths.source_data_dir / "np_classification_efficiency_raw.tsv")
    return save_figure(fig, paths.output_dir / "np_classification_main", formats, dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--curve-uncertainty",
        choices=["sd", "se", "ci90", "ci95", "none"],
        default="ci95",
        help="Uncertainty ribbon for label-efficiency curves",
    )
    args = parser.parse_args()
    project_root = args.project_root or Path(__file__).resolve().parents[3]
    outputs = make_figure(
        project_root,
        args.output_dir,
        parse_formats(args.formats),
        args.dpi,
        args.curve_uncertainty,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

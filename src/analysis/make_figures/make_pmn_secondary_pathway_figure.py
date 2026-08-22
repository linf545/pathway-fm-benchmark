#!/usr/bin/env python3
"""Generate the main PMN secondary-pathway figure.

The main PMN figure focuses on the primary task result: full-data performance on
Dataset B external test and label-efficiency performance evaluated on the same
external set. Probability ensembles are moved to a separate pathway-ensemble
figure.
"""
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
    summarize_pmn_efficiency,
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
        paths.project_root
        / "results/pmn_secondary_pathway/full/"
        "pmn_secondary_pathway_full_ensemble_repeated_runs.tsv",
        ["feature", "feature_type", "split_mode", "split_seed", "eval_set", "macro_auroc"],
    )
    efficiency = read_tsv(
        paths.project_root
        / "results/pmn_secondary_pathway/data_efficiency/"
        "pmn_secondary_pathway_data_efficiency_repeated_runs.tsv",
        [
            "feature",
            "split_mode",
            "split_seed",
            "sampling_seed",
            "requested_n_per_label",
            "eval_set",
            "macro_auroc",
        ],
    )

    full = full[
        (full["feature_type"] != "probability_ensemble")
        & (full["eval_set"] == "bao_B_external_test")
    ].copy()
    efficiency = efficiency[efficiency["eval_set"] == "bao_B_external_test"].copy()

    assert_unique(
        full,
        ["feature", "split_mode", "split_seed"],
        "PMN Dataset B external test: individual representations",
    )
    assert_unique(
        efficiency,
        [
            "feature",
            "split_mode",
            "split_seed",
            "sampling_seed",
            "requested_n_per_label",
        ],
        "PMN label efficiency: Dataset B external test",
    )

    features = ordered_features(set(full["feature"]) & set(efficiency["feature"]))
    per_split, efficiency_summary = summarize_pmn_efficiency(efficiency, "macro_auroc")
    per_split_plot = per_split.rename(columns={"per_split_mean": "macro_auroc"})

    fig = plt.figure(figsize=(7.15, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0])
    ax_full = fig.add_subplot(grid[0, :])
    ax_random = fig.add_subplot(grid[1, 0])
    ax_scaffold = fig.add_subplot(grid[1, 1])

    full_source = plot_full_dot_summary(
        ax_full,
        full,
        metric="macro_auroc",
        feature_order=features,
        split_column="split_mode",
        split_order=("random", "scaffold"),
        replicate_column="split_seed",
        ylabel="Macro AUROC",
        title=None,
        uncertainty="ci95",
    )
    ax_full.set_ylim(*padded_limits(full["macro_auroc"], floor=0.5))
    set_panel_label(ax_full, "A")

    curve_sources: list[pd.DataFrame] = []
    for ax, split, panel in [
        (ax_random, "random", "B"),
        (ax_scaffold, "scaffold", "C"),
    ]:
        raw = per_split_plot[per_split_plot["split_mode"] == split]
        summed = efficiency_summary[efficiency_summary["split_mode"] == split]
        exported = plot_learning_curves(
            ax,
            raw,
            summed,
            metric="macro_auroc",
            x_column="requested_n_per_label",
            feature_order=features,
            replicate_column="split_seed",
            xlabel="Training examples per pathway",
            ylabel="Macro AUROC",
            title=f"{split.capitalize()} split",
            log_x=False,
            uncertainty=curve_uncertainty,
            show_replicates=False,
        )
        ax.set_xticks(sorted(efficiency["requested_n_per_label"].unique()))
        exported["panel_split"] = split
        curve_sources.append(exported)
        add_chance_line(ax)
        set_panel_label(ax, panel)

    handles, labels = ax_random.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)

    source = pd.concat(
        [
            full_source.assign(panel="full_data_external_test"),
            pd.concat(curve_sources, ignore_index=True).assign(panel="label_efficiency_external_test"),
        ],
        ignore_index=True,
        sort=False,
    )
    write_source_data(
        source,
        paths.source_data_dir / "pmn_secondary_pathway_main_source_data.tsv",
    )
    write_source_data(
        per_split,
        paths.source_data_dir / "pmn_secondary_pathway_efficiency_per_split.tsv",
    )
    write_source_data(
        full,
        paths.source_data_dir / "pmn_secondary_pathway_individual_full_raw.tsv",
    )
    write_source_data(
        efficiency,
        paths.source_data_dir / "pmn_secondary_pathway_efficiency_raw.tsv",
    )
    return save_figure(
        fig,
        paths.output_dir / "pmn_secondary_pathway_main",
        formats,
        dpi,
    )


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

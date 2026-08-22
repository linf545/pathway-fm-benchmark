#!/usr/bin/env python3
"""Generate pathway-ensemble showcase figures for PMN and/or KEGG.

Use --tasks pmn to create a PMN-only figure, or --tasks pmn,kegg for the
original two-panel figure.
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
    assert_unique,
    configure_matplotlib,
    ordered_features,
    padded_limits,
    parse_formats,
    plot_full_dot_summary,
    read_tsv,
    resolve_paths,
    save_figure,
    set_panel_label,
    write_source_data,
)

TASK_CONFIG = {
    "pmn": {
        "title": "PMN secondary-pathway task",
        "repeated_runs": "results/pmn_secondary_pathway/full/pmn_secondary_pathway_full_ensemble_repeated_runs.tsv",
        "eval_set": "bao_B_external_test",
        "metric": "macro_auroc",
        "ylabel": "Macro AUROC",
    },
    "kegg": {
        "title": "KEGG pathway task",
        "repeated_runs": "results/kegg_pathway/full/kegg_pathway_full_ensemble_repeated_runs.tsv",
        "eval_set": "test",
        "metric": "macro_auroc",
        "ylabel": "Macro AUROC",
    },
}


def _load_pathway_full(path: Path, eval_set: str) -> pd.DataFrame:
    df = read_tsv(path, ["feature", "split_mode", "split_seed", "eval_set", "macro_auroc"])
    if "feature_type" not in df.columns:
        df["feature_type"] = df["feature"].astype(str).map(
            lambda name: "probability_ensemble" if name.startswith("ensemble_") else "individual_representation"
        )
    return df[df["eval_set"] == eval_set].copy()


def _ensemble_panel_features(df: pd.DataFrame) -> list[str]:
    available = set(df["feature"])
    ensembles = ordered_features(
        [f for f in available if str(f).startswith("ensemble_")],
        include_ensembles=True,
    )
    keep = ["morgan_r2_2048"] + ensembles
    return [f for f in keep if f in available]


def make_figure(
    project_root: Path,
    output_dir: Path | None = None,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    tasks: tuple[str, ...] = ("pmn", "kegg"),
) -> list[Path]:
    paths = resolve_paths(__file__, project_root, output_dir)
    configure_matplotlib()

    selected = [task for task in tasks if task in TASK_CONFIG]
    if not selected:
        raise ValueError("No valid tasks selected. Use some combination of: pmn,kegg")

    nrows = len(selected)
    fig_height = 3.6 if nrows == 1 else 7.0
    fig, axes = plt.subplots(nrows, 1, figsize=(7.15, fig_height), constrained_layout=True)
    if nrows == 1:
        axes = [axes]

    panel_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    source_frames = []

    for i, (ax, task) in enumerate(zip(axes, selected)):
        cfg = TASK_CONFIG[task]
        df = _load_pathway_full(paths.project_root / cfg["repeated_runs"], cfg["eval_set"])
        df = df[(df["feature"] == "morgan_r2_2048") | (df["feature_type"] == "probability_ensemble")].copy()
        assert_unique(df, ["feature", "split_mode", "split_seed"], f"{task} ensemble showcase")
        features = _ensemble_panel_features(df)
        plotted = plot_full_dot_summary(
            ax,
            df,
            metric=cfg["metric"],
            feature_order=features,
            split_column="split_mode",
            split_order=("random", "scaffold"),
            replicate_column="split_seed",
            ylabel=cfg["ylabel"],
            title=cfg["title"],
            uncertainty="ci95",
        )
        ax.set_ylim(*padded_limits(df[cfg["metric"]], floor=0.5))
        set_panel_label(ax, panel_labels[i])
        source_frames.append(plotted.assign(panel=task))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)

    source = pd.concat(source_frames, ignore_index=True, sort=False)
    stem_name = "pathway_ensemble_showcase" if len(selected) > 1 else f"{selected[0]}_ensemble_showcase"
    write_source_data(source, paths.source_data_dir / f"{stem_name}_source_data.tsv")
    return save_figure(fig, paths.output_dir / stem_name, formats, dpi)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--tasks",
        default="pmn,kegg",
        help="Comma-separated task list: pmn, kegg, or pmn,kegg",
    )
    args = parser.parse_args()
    project_root = args.project_root or Path(__file__).resolve().parents[3]
    tasks = tuple(task.strip().lower() for task in args.tasks.split(",") if task.strip())
    outputs = make_figure(project_root, args.output_dir, parse_formats(args.formats), args.dpi, tasks)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

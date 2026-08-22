#!/usr/bin/env python3
"""Create the LoRA adaptation comparison figure for the manuscript appendix.

The figure compares Morgan r2, frozen ChemBERTa-77M and MolFormer
representations, and their LoRA-adapted counterparts for:

1. natural-product-likeness prediction, using AUROC;
2. natural-product chemical-class prediction, using macro-AUROC;

under random and scaffold splits.

The script also writes seed-level source data, summary statistics, and paired
LoRA-minus-frozen differences.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


FEATURE_LABELS = {
    "morgan_r2_2048": "Morgan r2",
    "chemberta_77m_mlm": "ChemBERTa-77M",
    "molformer_c3_1_1b": "MolFormer",
}

CATEGORY_ORDER = [
    "Morgan r2",
    "ChemBERTa-77M\nFrozen",
    "ChemBERTa-77M\nLoRA",
    "MolFormer\nFrozen",
    "MolFormer\nLoRA",
]

TASK_ORDER = ["np_likeness", "np_classification"]
SPLIT_ORDER = ["random", "scaffold"]

TASK_LABELS = {
    "np_likeness": "NP-likeness",
    "np_classification": "NP chemical class",
}

METRIC_LABELS = {
    "np_likeness": "AUROC",
    "np_classification": "Macro-AUROC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <project-root>/results/figures/main",
    )
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        default=None,
        help="Default: <project-root>/results/figures/source_data",
    )
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated output formats. Default: pdf,png",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--allow-stale-np-likeness-selection",
        action="store_true",
        help=(
            "Allow NP-likeness LoRA rows not marked as "
            "validation_auroc_then_auprc. By default, stale AUPRC-first "
            "outputs cause an error."
        ),
    )
    return parser.parse_args()


def read_tsv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def normalize_seed(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="raise").astype(int)


def load_np_likeness(project_root: Path, allow_stale: bool) -> pd.DataFrame:
    frozen_path = (
        project_root
        / "results/np_likeness/full/"
        "np_likeness_full_ensemble_repeated_runs.tsv"
    )
    frozen = read_tsv(
        frozen_path,
        ["feature", "split_mode", "split_seed", "eval_set", "auroc"],
    )
    frozen = frozen.loc[
        frozen["eval_set"].astype(str).eq("test")
        & frozen["feature"].astype(str).isin(
            ["morgan_r2_2048", "chemberta_77m_mlm", "molformer_c3_1_1b"]
        ),
        ["feature", "split_mode", "split_seed", "auroc"],
    ].copy()
    frozen["task"] = "np_likeness"
    frozen["adaptation"] = "Frozen"
    frozen.loc[frozen["feature"].eq("morgan_r2_2048"), "adaptation"] = "Baseline"
    frozen["metric_value"] = pd.to_numeric(frozen["auroc"], errors="raise")
    frozen["split_seed"] = normalize_seed(frozen["split_seed"])

    lora_frames: list[pd.DataFrame] = []
    for feature in ["chemberta_77m_mlm", "molformer_c3_1_1b"]:
        path = (
            project_root
            / f"results/np_likeness/lora/{feature}/"
            "np_likeness_adaptation_repeated_runs.tsv"
        )
        frame = read_tsv(
            path,
            [
                "feature",
                "adaptation_mode",
                "split_mode",
                "split_seed",
                "eval_set",
                "selected_by",
                "auroc",
            ],
        )
        frame = frame.loc[
            frame["eval_set"].astype(str).eq("test")
            & frame["adaptation_mode"].astype(str).eq("lora"),
            [
                "feature",
                "split_mode",
                "split_seed",
                "selected_by",
                "auroc",
            ],
        ].copy()

        selection_values = set(frame["selected_by"].dropna().astype(str))
        expected = {"validation_auroc_then_auprc"}
        if not allow_stale and selection_values != expected:
            raise ValueError(
                f"{path} has checkpoint-selection labels {sorted(selection_values)}; "
                f"expected exactly {sorted(expected)}. This usually means the "
                "AUPRC-first outputs were not replaced by the AUROC-first rerun."
            )

        frame["task"] = "np_likeness"
        frame["adaptation"] = "LoRA"
        frame["metric_value"] = pd.to_numeric(frame["auroc"], errors="raise")
        frame["split_seed"] = normalize_seed(frame["split_seed"])
        lora_frames.append(frame)

    columns = [
        "task",
        "feature",
        "adaptation",
        "split_mode",
        "split_seed",
        "metric_value",
    ]
    return pd.concat(
        [frozen[columns], *(frame[columns] for frame in lora_frames)],
        ignore_index=True,
    )


def load_np_classification(project_root: Path) -> pd.DataFrame:
    frozen_path = (
        project_root
        / "results/np_classification/full/"
        "npclassification_repeated_runs.tsv"
    )
    frozen = read_tsv(
        frozen_path,
        ["feature", "split", "split_seed", "eval_set", "auroc_ovr_macro"],
    )
    frozen = frozen.loc[
        frozen["eval_set"].astype(str).eq("test")
        & frozen["feature"].astype(str).isin(
            ["morgan_r2_2048", "chemberta_77m_mlm", "molformer_c3_1_1b"]
        ),
        ["feature", "split", "split_seed", "auroc_ovr_macro"],
    ].copy()
    frozen = frozen.rename(columns={"split": "split_mode"})
    frozen["task"] = "np_classification"
    frozen["adaptation"] = "Frozen"
    frozen.loc[frozen["feature"].eq("morgan_r2_2048"), "adaptation"] = "Baseline"
    frozen["metric_value"] = pd.to_numeric(
        frozen["auroc_ovr_macro"], errors="raise"
    )
    frozen["split_seed"] = normalize_seed(frozen["split_seed"])

    lora_frames: list[pd.DataFrame] = []
    for feature in ["chemberta_77m_mlm", "molformer_c3_1_1b"]:
        path = (
            project_root
            / f"results/np_classification/lora/{feature}/"
            "npclassification_adaptation_repeated_runs.tsv"
        )
        frame = read_tsv(
            path,
            [
                "feature",
                "adaptation_mode",
                "split",
                "split_seed",
                "eval_set",
                "selected_by",
                "auroc_ovr_macro",
            ],
        )
        frame = frame.loc[
            frame["eval_set"].astype(str).eq("test")
            & frame["adaptation_mode"].astype(str).eq("lora"),
            [
                "feature",
                "split",
                "split_seed",
                "selected_by",
                "auroc_ovr_macro",
            ],
        ].copy()

        expected = {"validation_macro_auroc_then_macro_auprc"}
        selection_values = set(frame["selected_by"].dropna().astype(str))
        if selection_values != expected:
            raise ValueError(
                f"{path} has checkpoint-selection labels {sorted(selection_values)}; "
                f"expected exactly {sorted(expected)}."
            )

        frame = frame.rename(columns={"split": "split_mode"})
        frame["task"] = "np_classification"
        frame["adaptation"] = "LoRA"
        frame["metric_value"] = pd.to_numeric(
            frame["auroc_ovr_macro"], errors="raise"
        )
        frame["split_seed"] = normalize_seed(frame["split_seed"])
        lora_frames.append(frame)

    columns = [
        "task",
        "feature",
        "adaptation",
        "split_mode",
        "split_seed",
        "metric_value",
    ]
    return pd.concat(
        [frozen[columns], *(frame[columns] for frame in lora_frames)],
        ignore_index=True,
    )


def category_label(feature: str, adaptation: str) -> str:
    model = FEATURE_LABELS[feature]
    if feature == "morgan_r2_2048":
        return model
    return f"{model}\n{adaptation}"


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_columns = ["task", "split_mode", "feature", "adaptation", "category"]
    for keys, group in frame.groupby(group_columns, sort=False):
        values = group["metric_value"].to_numpy(dtype=float)
        n = len(values)
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 1 else np.nan
        t_critical = float(student_t.ppf(0.975, n - 1)) if n > 1 else np.nan
        margin = t_critical * se if n > 1 else np.nan
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_seeds": n,
                "mean": mean,
                "sd": sd,
                "se": se,
                "ci_lower": mean - margin if n > 1 else np.nan,
                "ci_upper": mean + margin if n > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def paired_differences(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for task in TASK_ORDER:
        for split_mode in SPLIT_ORDER:
            for feature in ["chemberta_77m_mlm", "molformer_c3_1_1b"]:
                subset = frame.loc[
                    frame["task"].eq(task)
                    & frame["split_mode"].eq(split_mode)
                    & frame["feature"].eq(feature)
                    & frame["adaptation"].isin(["Frozen", "LoRA"]),
                    ["split_seed", "adaptation", "metric_value"],
                ].copy()

                wide = subset.pivot_table(
                    index="split_seed",
                    columns="adaptation",
                    values="metric_value",
                    aggfunc="mean",
                )
                if not {"Frozen", "LoRA"}.issubset(wide.columns):
                    raise ValueError(
                        f"Missing frozen or LoRA rows for {task}, "
                        f"{split_mode}, {feature}"
                    )
                wide = wide[["Frozen", "LoRA"]].dropna().copy()
                if len(wide) < 2:
                    raise ValueError(
                        f"Fewer than two paired seeds for {task}, "
                        f"{split_mode}, {feature}"
                    )

                wide["difference_lora_minus_frozen"] = (
                    wide["LoRA"] - wide["Frozen"]
                )
                wide = wide.reset_index()
                wide.insert(0, "feature", feature)
                wide.insert(0, "split_mode", split_mode)
                wide.insert(0, "task", task)
                paired_rows.append(wide)

                differences = wide["difference_lora_minus_frozen"].to_numpy(float)
                n = len(differences)
                mean = float(np.mean(differences))
                sd = float(np.std(differences, ddof=1))
                se = sd / np.sqrt(n)
                t_critical = float(student_t.ppf(0.975, n - 1))
                margin = t_critical * se
                summary_rows.append(
                    {
                        "task": task,
                        "split_mode": split_mode,
                        "feature": feature,
                        "model_label": FEATURE_LABELS[feature],
                        "n_paired_seeds": n,
                        "mean_frozen": float(wide["Frozen"].mean()),
                        "mean_lora": float(wide["LoRA"].mean()),
                        "mean_difference_lora_minus_frozen": mean,
                        "sd_difference": sd,
                        "se_difference": se,
                        "ci_lower": mean - margin,
                        "ci_upper": mean + margin,
                    }
                )

    return pd.concat(paired_rows, ignore_index=True), pd.DataFrame(summary_rows)


def validate_complete_design(frame: pd.DataFrame) -> None:
    expected_categories = set(CATEGORY_ORDER)
    for task in TASK_ORDER:
        for split_mode in SPLIT_ORDER:
            subset = frame.loc[
                frame["task"].eq(task) & frame["split_mode"].eq(split_mode)
            ]
            categories = set(subset["category"].astype(str))
            if categories != expected_categories:
                raise ValueError(
                    f"{task}/{split_mode} categories are {sorted(categories)}; "
                    f"expected {sorted(expected_categories)}"
                )

            seed_counts = subset.groupby("category")["split_seed"].nunique()
            if not (seed_counts == 5).all():
                raise ValueError(
                    f"{task}/{split_mode} does not have five seeds per category:\n"
                    f"{seed_counts}"
                )


def plot(frame: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.2, 6.8),
        constrained_layout=True,
        sharex=True,
    )

    x = np.arange(len(CATEGORY_ORDER), dtype=float)
    jitter_offsets = np.linspace(-0.10, 0.10, 5)

    for row_index, task in enumerate(TASK_ORDER):
        for column_index, split_mode in enumerate(SPLIT_ORDER):
            ax = axes[row_index, column_index]
            panel = frame.loc[
                frame["task"].eq(task)
                & frame["split_mode"].eq(split_mode)
            ].copy()
            panel_summary = summary.loc[
                summary["task"].eq(task)
                & summary["split_mode"].eq(split_mode)
            ].set_index("category")

            for category_index, category in enumerate(CATEGORY_ORDER):
                values = (
                    panel.loc[panel["category"].eq(category)]
                    .sort_values("split_seed")["metric_value"]
                    .to_numpy(dtype=float)
                )
                ax.scatter(
                    np.full(len(values), x[category_index]) + jitter_offsets[: len(values)],
                    values,
                    s=22,
                    facecolors="none",
                    edgecolors="0.45",
                    linewidths=0.8,
                    zorder=2,
                )

                row = panel_summary.loc[category]
                lower = float(row["mean"] - row["ci_lower"])
                upper = float(row["ci_upper"] - row["mean"])
                ax.errorbar(
                    x[category_index],
                    row["mean"],
                    yerr=np.asarray([[lower], [upper]]),
                    fmt="o",
                    markersize=6,
                    capsize=3,
                    linewidth=1.4,
                    zorder=3,
                )

            ax.set_title(
                f"{TASK_LABELS[task]} — {split_mode.capitalize()} split",
                fontsize=10,
            )
            ax.set_ylabel(METRIC_LABELS[task])
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)

            values = panel["metric_value"].to_numpy(dtype=float)
            data_min = float(np.min(values))
            data_max = float(np.max(values))
            padding = max((data_max - data_min) * 0.18, 0.004)
            ax.set_ylim(max(0.0, data_min - padding), min(1.0, data_max + padding))

    for ax in axes[-1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(CATEGORY_ORDER, rotation=35, ha="right")

    panel_labels = ["A", "B", "C", "D"]
    for label, ax in zip(panel_labels, axes.flat):
        ax.text(
            -0.12,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            va="bottom",
        )

    return fig


def parse_formats(value: str) -> list[str]:
    formats = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not formats:
        raise ValueError("At least one output format is required")
    return formats


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root / "results/figures/main"
    )
    source_data_dir = (
        args.source_data_dir.resolve()
        if args.source_data_dir is not None
        else project_root / "results/figures/source_data"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_data_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.concat(
        [
            load_np_likeness(
                project_root,
                allow_stale=args.allow_stale_np_likeness_selection,
            ),
            load_np_classification(project_root),
        ],
        ignore_index=True,
    )
    frame["category"] = [
        category_label(feature, adaptation)
        for feature, adaptation in zip(frame["feature"], frame["adaptation"])
    ]

    validate_complete_design(frame)

    summary = summarize(frame)
    paired, paired_summary = paired_differences(frame)

    source_by_seed = source_data_dir / "lora_adaptation_by_seed.tsv"
    source_summary = source_data_dir / "lora_adaptation_summary.tsv"
    source_paired = source_data_dir / "lora_adaptation_paired_differences.tsv"
    source_paired_summary = (
        source_data_dir / "lora_adaptation_paired_difference_summary.tsv"
    )

    frame.sort_values(
        ["task", "split_mode", "category", "split_seed"]
    ).to_csv(source_by_seed, sep="\t", index=False)
    summary.sort_values(
        ["task", "split_mode", "category"]
    ).to_csv(source_summary, sep="\t", index=False)
    paired.sort_values(
        ["task", "split_mode", "feature", "split_seed"]
    ).to_csv(source_paired, sep="\t", index=False)
    paired_summary.sort_values(
        ["task", "split_mode", "feature"]
    ).to_csv(source_paired_summary, sep="\t", index=False)

    figure = plot(frame, summary)
    outputs: list[Path] = []
    for extension in parse_formats(args.formats):
        path = output_dir / f"lora_adaptation_comparison.{extension}"
        figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)

    print("\nPaired LoRA-minus-frozen differences")
    display_columns = [
        "task",
        "split_mode",
        "model_label",
        "mean_frozen",
        "mean_lora",
        "mean_difference_lora_minus_frozen",
        "ci_lower",
        "ci_upper",
    ]
    print(
        paired_summary[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nWrote source data:")
    for path in [
        source_by_seed,
        source_summary,
        source_paired,
        source_paired_summary,
    ]:
        print(path)

    print("\nWrote figures:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Shared utilities for manuscript figure generation.

The functions in this module deliberately read replicate-level result tables and
recompute plotted summaries. Pre-aggregated result tables are used only for
independent QC, not as plotting inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import hashlib
import json
import math
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


# Fixed order prevents performance-based reordering from changing between tasks.
FEATURE_ORDER: tuple[str, ...] = (
    "morgan_r2_2048",
    "morgan_r3_2048",
    "rdkit_descriptors",
    "chemberta_77m_mlm",
    "chemberta_100m_mlm",
    "molformer_c3_1_1b",
    "nafm_graph_mean",
    "molclr_gcn",
    "molclr_gin",
    "graphmvp_gin",
    "molebert_gin",
)

FEATURE_LABELS: dict[str, str] = {
    "morgan_r2_2048": "Morgan r2",
    "morgan_r3_2048": "Morgan r3",
    "rdkit_descriptors": "RDKit descriptors",
    "chemberta_77m_mlm": "ChemBERTa-77M",
    "chemberta_100m_mlm": "ChemBERTa-100M",
    "molformer_c3_1_1b": "MolFormer",
    "nafm_graph_mean": "NaFM",
    "molclr_gcn": "MolCLR-GCN",
    "molclr_gin": "MolCLR-GIN",
    "graphmvp_gin": "GraphMVP-GIN",
    "molebert_gin": "Mole-BERT",
    "ensemble_morgan_r2_plus_chemberta_77m_mlm": "Morgan r2 + ChemBERTa-77M",
    "ensemble_morgan_r2_plus_chemberta_100m_mlm": "Morgan r2 + ChemBERTa-100M",
    "ensemble_morgan_r2_plus_molformer_c3_1_1b": "Morgan r2 + MolFormer",
    "ensemble_morgan_r2_plus_nafm_graph_mean": "Morgan r2 + NaFM",
    "ensemble_morgan_r2_plus_molclr_gcn": "Morgan r2 + MolCLR-GCN",
    "ensemble_morgan_r2_plus_molclr_gin": "Morgan r2 + MolCLR-GIN",
    "ensemble_morgan_r2_plus_graphmvp_gin": "Morgan r2 + GraphMVP-GIN",
    "ensemble_morgan_r2_plus_molebert_gin": "Morgan r2 + Mole-BERT",
    "ensemble_all_equal_probability": "All-model ensemble",
}

ENSEMBLE_ORDER: tuple[str, ...] = tuple(
    key for key in FEATURE_LABELS if key.startswith("ensemble_")
)

SPLIT_LABELS: dict[str, str] = {
    "random": "Random split",
    "scaffold": "Scaffold split",
    "scaffold_pair": "Scaffold-pair split",
}

NEGATIVE_LABELS: dict[str, str] = {
    "easy": "Easy negatives",
    "mixed": "Mixed negatives",
    "hard": "Hard negatives",
}

# Baselines use neutral colors and distinct line styles; FMs use a categorical palette.
# This makes fingerprint baselines identifiable even in dense learning-curve panels.
BASELINE_STYLES: dict[str, dict[str, object]] = {
    "morgan_r2_2048": {
        "color": "#111111",
        "linestyle": "-",
        "marker": "s",
        "linewidth": 1.8,
    },
    "morgan_r3_2048": {
        "color": "#777777",
        "linestyle": "--",
        "marker": "s",
        "linewidth": 1.6,
    },
    "rdkit_descriptors": {
        "color": "#8C564B",
        "linestyle": ":",
        "marker": "D",
        "linewidth": 1.6,
    },
}

FM_FEATURE_ORDER: tuple[str, ...] = tuple(
    feature for feature in FEATURE_ORDER if feature not in BASELINE_STYLES
)


@dataclass(frozen=True)
class FigurePaths:
    project_root: Path
    output_dir: Path
    source_data_dir: Path


def infer_project_root(script_file: str | Path) -> Path:
    """Infer project root for scripts stored in src/analysis/make_figures/."""
    return Path(script_file).resolve().parents[3]


def resolve_paths(
    script_file: str | Path,
    project_root: str | Path | None,
    output_dir: str | Path | None,
) -> FigurePaths:
    root = Path(project_root).resolve() if project_root else infer_project_root(script_file)
    out = (
        Path(output_dir).resolve()
        if output_dir
        else root / "results" / "figures" / "main"
    )
    source = out.parent / "source_data"
    out.mkdir(parents=True, exist_ok=True)
    source.mkdir(parents=True, exist_ok=True)
    return FigurePaths(root, out, source)


def configure_matplotlib() -> None:
    """Set compact publication-oriented defaults without requiring custom fonts."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "figure.titlesize": 10.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def read_tsv(path: Path, required_columns: Iterable[str] = ()) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")
    df = pd.read_csv(path, sep="\t")
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_unique(df: pd.DataFrame, columns: Sequence[str], table_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Cannot check {table_name}: missing key columns {missing}")
    duplicated = df.duplicated(list(columns), keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, list(columns)].head(10).to_dict("records")
        raise ValueError(
            f"{table_name} contains duplicate replicate rows for key {columns}. "
            f"Examples: {examples}"
        )


def assert_finite(df: pd.DataFrame, columns: Sequence[str], table_name: str) -> None:
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"{table_name} is missing metric column {column}")
        values = pd.to_numeric(df[column], errors="coerce")
        bad = ~np.isfinite(values.to_numpy(dtype=float))
        if bad.any():
            raise ValueError(
                f"{table_name}.{column} contains {int(bad.sum())} non-finite values"
            )


def ordered_features(
    available: Iterable[str], include_ensembles: bool = False
) -> list[str]:
    available_set = set(available)
    order = [feature for feature in FEATURE_ORDER if feature in available_set]
    if include_ensembles:
        order.extend(feature for feature in ENSEMBLE_ORDER if feature in available_set)
    extras = sorted(available_set - set(order))
    if extras:
        warnings.warn(
            "Unrecognized feature names will be appended alphabetically: " + ", ".join(extras),
            stacklevel=2,
        )
        order.extend(extras)
    return order


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def stable_jitter(values: Sequence[object], width: float = 0.10) -> np.ndarray:
    """Return deterministic jitter derived from replicate identifiers."""
    offsets: list[float] = []
    for value in values:
        digest = hashlib.sha256(str(value).encode("utf-8")).digest()
        unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        offsets.append((unit - 0.5) * 2.0 * width)
    return np.asarray(offsets, dtype=float)


def summarize_replicates(
    df: pd.DataFrame,
    group_columns: Sequence[str],
    metric: str,
    replicate_column: str,
) -> pd.DataFrame:
    """Calculate mean, sample SD, count, and SE from independent replicates."""
    assert_finite(df, [metric], "replicate table")
    grouped = (
        df.groupby(list(group_columns), dropna=False, as_index=False)[metric]
        .agg(mean="mean", sd="std", n="count")
    )
    grouped["sd"] = grouped["sd"].fillna(0.0)
    grouped["se"] = grouped["sd"] / np.sqrt(grouped["n"].clip(lower=1))
    grouped["metric"] = metric
    grouped["replicate_column"] = replicate_column
    return grouped


def summarize_pmn_efficiency(
    df: pd.DataFrame,
    metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Avoid treating sampling replicates within one split as independent splits.

    First average across sampling seeds within each split seed, then calculate
    mean and SD across split seeds. Both levels are returned for source-data export.
    """
    per_split = (
        df.groupby(
            ["feature", "split_mode", "split_seed", "requested_n_per_label"],
            as_index=False,
        )[metric]
        .mean()
        .rename(columns={metric: "per_split_mean"})
    )
    summary = (
        per_split.groupby(
            ["feature", "split_mode", "requested_n_per_label"], as_index=False
        )["per_split_mean"]
        .agg(mean="mean", sd="std", n="count")
    )
    summary["sd"] = summary["sd"].fillna(0.0)
    summary["se"] = summary["sd"] / np.sqrt(summary["n"].clip(lower=1))
    summary["metric"] = metric
    return per_split, summary


def feature_style(feature: str) -> dict[str, object]:
    """Return a stable visual style for one molecular representation."""
    if feature in BASELINE_STYLES:
        return dict(BASELINE_STYLES[feature])

    cmap = plt.get_cmap("tab10")
    try:
        index = FM_FEATURE_ORDER.index(feature)
    except ValueError:
        # Unknown representations remain deterministic without altering known colors.
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big")
    return {
        "color": cmap(index % cmap.N),
        "linestyle": "-",
        "marker": "o",
        "linewidth": 1.25,
    }


def feature_colors(features: Sequence[str]) -> Mapping[str, object]:
    """Compatibility wrapper returning the stable color for each feature."""
    return {feature: feature_style(feature)["color"] for feature in features}


def uncertainty_half_width(summary: pd.DataFrame, uncertainty: str) -> np.ndarray:
    """Calculate the plotted half-width from summary columns sd, se, and n.

    ci90 and ci95 use a two-sided Student-t interval because the number of
    independent replicates is small (typically 4 or 5).
    """
    uncertainty = uncertainty.lower()
    sd = summary["sd"].to_numpy(dtype=float)
    se = summary["se"].to_numpy(dtype=float)
    n = summary["n"].to_numpy(dtype=float)

    if uncertainty == "sd":
        return sd
    if uncertainty == "se":
        return se
    if uncertainty in {"ci90", "ci95"}:
        confidence = 0.90 if uncertainty == "ci90" else 0.95
        df = np.maximum(n - 1.0, 1.0)
        critical = student_t.ppf((1.0 + confidence) / 2.0, df)
        half_width = critical * se
        half_width[n < 2] = 0.0
        return half_width
    if uncertainty == "none":
        return np.zeros_like(sd)
    raise ValueError(
        f"Unknown uncertainty mode {uncertainty!r}; use sd, se, ci90, ci95, or none"
    )


def set_panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def add_chance_line(ax: mpl.axes.Axes, chance: float = 0.5) -> None:
    ax.axhline(chance, color="0.65", linewidth=0.8, linestyle=":", zorder=0)


def padded_limits(values: Iterable[float], floor: float | None = None) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return (0.0, 1.0)
    low = float(array.min())
    high = float(array.max())
    span = max(high - low, 0.02)
    lower = low - 0.12 * span
    upper = high + 0.12 * span
    if floor is not None:
        lower = max(floor, lower)
    return lower, min(1.0, upper)


def plot_full_dot_summary(
    ax: mpl.axes.Axes,
    df: pd.DataFrame,
    metric: str,
    feature_order: Sequence[str],
    split_column: str,
    split_order: Sequence[str],
    replicate_column: str,
    ylabel: str,
    title: str | None,
    uncertainty: str = "ci95",
) -> pd.DataFrame:
    """Plot replicate points and a mean uncertainty interval.

    The default error bar is a two-sided 95% Student-t confidence interval for
    the mean across independent split seeds. Individual split-seed points remain
    visible, so readers can separately assess run-to-run variability.
    """
    data = df[df["feature"].isin(feature_order)].copy()
    assert_unique(data, ["feature", split_column, replicate_column], title)
    assert_finite(data, [metric], title)

    split_colors = {
        split_order[0]: "0.15",
        split_order[1]: "0.60" if len(split_order) > 1 else "0.15",
    }
    offsets = np.linspace(-0.18, 0.18, len(split_order))
    rows: list[dict] = []

    for feature_index, feature in enumerate(feature_order):
        for split_index, split in enumerate(split_order):
            subset = data[(data["feature"] == feature) & (data[split_column] == split)]
            if subset.empty:
                continue
            x_center = feature_index + offsets[split_index]
            jitter = stable_jitter(subset[replicate_column].tolist(), width=0.045)
            ax.scatter(
                np.full(len(subset), x_center) + jitter,
                subset[metric],
                s=12,
                facecolors="none",
                edgecolors=split_colors[split],
                linewidths=0.65,
                alpha=0.80,
                zorder=2,
            )
            mean = float(subset[metric].mean())
            sd = float(subset[metric].std(ddof=1)) if len(subset) > 1 else 0.0
            n = int(len(subset))
            se = sd / math.sqrt(n) if n else 0.0
            summary_row = pd.DataFrame({"sd": [sd], "se": [se], "n": [n]})
            half_width = float(uncertainty_half_width(summary_row, uncertainty)[0])
            ax.errorbar(
                x_center,
                mean,
                yerr=half_width,
                fmt="o",
                color=split_colors[split],
                markerfacecolor=split_colors[split],
                markeredgecolor="white",
                markeredgewidth=0.4,
                markersize=4.5,
                capsize=2.2,
                linewidth=1.0,
                zorder=3,
                label=SPLIT_LABELS.get(split, split) if feature_index == 0 else None,
            )
            rows.append(
                {
                    "feature": feature,
                    "feature_label": feature_label(feature),
                    "split": split,
                    "metric": metric,
                    "mean": mean,
                    "sd": sd,
                    "se": se,
                    "n": n,
                    "uncertainty": uncertainty,
                    "uncertainty_half_width": half_width,
                    "uncertainty_lower": max(0.0, mean - half_width),
                    "uncertainty_upper": min(1.0, mean + half_width),
                }
            )

    ax.set_xticks(range(len(feature_order)))
    ax.set_xticklabels([feature_label(feature) for feature in feature_order], rotation=55, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(axis="y", color="0.90", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="best")
    return pd.DataFrame(rows)

def plot_learning_curves(
    ax: mpl.axes.Axes,
    df: pd.DataFrame,
    summary: pd.DataFrame,
    metric: str,
    x_column: str,
    feature_order: Sequence[str],
    replicate_column: str,
    xlabel: str,
    ylabel: str,
    title: str | None,
    log_x: bool = True,
    uncertainty: str = "ci95",
    show_replicates: bool = False,
) -> pd.DataFrame:
    """Plot mean learning curves with a configurable uncertainty ribbon.

    Individual replicate trajectories are disabled by default. For ci90/ci95,
    the ribbon is a two-sided Student-t confidence interval for the mean across
    the independent replicate unit represented by ``summary``.
    """
    plotted_rows: list[pd.DataFrame] = []

    for feature in feature_order:
        raw_feature = df[df["feature"] == feature].sort_values(x_column)
        sum_feature = summary[summary["feature"] == feature].sort_values(x_column).copy()
        if sum_feature.empty:
            continue

        style = feature_style(feature)
        if show_replicates:
            for _, replicate in raw_feature.groupby(replicate_column):
                replicate = replicate.sort_values(x_column)
                ax.plot(
                    replicate[x_column],
                    replicate[metric],
                    color=style["color"],
                    alpha=0.12,
                    linewidth=0.65,
                    zorder=1,
                )

        x = sum_feature[x_column].to_numpy(dtype=float)
        mean = sum_feature["mean"].to_numpy(dtype=float)
        half_width = uncertainty_half_width(sum_feature, uncertainty)
        lower = np.clip(mean - half_width, 0.0, 1.0)
        upper = np.clip(mean + half_width, 0.0, 1.0)

        if uncertainty != "none":
            ax.fill_between(
                x,
                lower,
                upper,
                color=style["color"],
                alpha=0.13,
                linewidth=0,
                zorder=1,
            )
        ax.plot(
            x,
            mean,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=float(style["linewidth"]),
            color=style["color"],
            label=feature_label(feature),
            zorder=4 if feature in BASELINE_STYLES else 3,
        )
        sum_feature["uncertainty"] = uncertainty
        sum_feature["uncertainty_half_width"] = half_width
        sum_feature["uncertainty_lower"] = lower
        sum_feature["uncertainty_upper"] = upper
        sum_feature["feature_label"] = feature_label(feature)
        plotted_rows.append(sum_feature)

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(color="0.90", linewidth=0.6)
    ax.set_axisbelow(True)
    return pd.concat(plotted_rows, ignore_index=True) if plotted_rows else pd.DataFrame()


def save_figure(
    fig: mpl.figure.Figure,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    paths: list[Path] = []
    for extension in formats:
        extension = extension.lower().lstrip(".")
        path = output_stem.with_suffix(f".{extension}")
        kwargs = {"dpi": dpi} if extension in {"png", "tif", "tiff", "jpg", "jpeg"} else {}
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def write_source_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def parse_formats(value: str) -> tuple[str, ...]:
    formats = tuple(part.strip().lower().lstrip(".") for part in value.split(",") if part.strip())
    allowed = {"pdf", "png", "svg", "tif", "tiff"}
    invalid = sorted(set(formats) - allowed)
    if invalid:
        raise ValueError(f"Unsupported output formats: {invalid}. Allowed: {sorted(allowed)}")
    if not formats:
        raise ValueError("At least one output format is required")
    return formats

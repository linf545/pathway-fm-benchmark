#!/usr/bin/env python3
"""Generate full-data per-label AUROC heatmaps for NP class, PMN, and KEGG.

For each task, rows are class/pathway labels, columns are molecular
representations, and cell color is mean one-vs-rest AUROC across split seeds.
Two panels show random and scaffold splits. By default, the heatmap includes
Morgan r2, Morgan r3, and the five frozen FMs with the highest mean per-label
AUROC across both split schemes.

The script uses the existing NP-class per-class table and reconstructs PMN/KEGG
per-label AUROC from the saved full-data prediction NPZ files. It writes both
seed-level and summarized CSV source data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import textwrap
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from common import FEATURE_LABELS, configure_matplotlib, parse_formats, resolve_paths, save_figure
except ImportError:
    FEATURE_LABELS = {
        "morgan_r2_2048": "Morgan r2",
        "morgan_r3_2048": "Morgan r3",
        "chemberta_77m_mlm": "ChemBERTa-77M",
        "chemberta_100m_mlm": "ChemBERTa-100M",
        "molformer_c3_1_1b": "MolFormer",
        "nafm_graph_mean": "NaFM",
        "molclr_gcn": "MolCLR-GCN",
        "molclr_gin": "MolCLR-GIN",
        "graphmvp_gin": "GraphMVP-GIN",
        "molebert_gin": "Mole-BERT",
    }

    def configure_matplotlib() -> None:
        plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})

    def parse_formats(value: str) -> tuple[str, ...]:
        return tuple(item.strip() for item in value.split(",") if item.strip())

    class _Paths:
        def __init__(self, root: Path, output: Path):
            self.project_root = root
            self.output_dir = output
            self.source_data_dir = output.parent / "source_data"

    def resolve_paths(script_file, project_root, output_dir):
        root = Path(project_root).resolve() if project_root else Path(script_file).resolve().parents[3]
        output = Path(output_dir).resolve() if output_dir else root / "results/figures/main"
        output.mkdir(parents=True, exist_ok=True)
        (output.parent / "source_data").mkdir(parents=True, exist_ok=True)
        return _Paths(root, output)

    def save_figure(fig, stem, formats, dpi):
        paths = []
        for fmt in formats:
            path = Path(stem).with_suffix(f".{fmt}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)
        return paths


BASELINES = ("morgan_r2_2048", "morgan_r3_2048")
TASKS = ("np_classification", "pmn_secondary_pathway", "kegg_pathway")
TASK_TITLES = {
    "np_classification": "NP chemical-class prediction",
    "pmn_secondary_pathway": "PMN secondary-pathway prediction",
    "kegg_pathway": "KEGG pathway prediction",
}
SPLIT_TITLES = {"random": "Random split", "scaffold": "Scaffold split"}

NP_CLASS_LABELS = {
    1: "Alkaloids",
    2: "Amino acids and peptides",
    3: "Carbohydrates",
    4: "Fatty acids",
    5: "Polyketides",
    6: "Shikimates and phenylpropanoids",
    7: "Terpenoids",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--tasks", default="all", help="Comma-separated task names or 'all'.")
    parser.add_argument(
        "--top-fms",
        type=int,
        default=5,
        help="Number of highest-performing frozen FMs shown in addition to Morgan r2/r3. Use 0 for all FMs.",
    )
    parser.add_argument(
        "--label-order",
        choices=["support", "morgan", "original"],
        default="support",
        help="Order heatmap rows by mean positive support, Morgan-r2 AUROC, or original label index.",
    )
    parser.add_argument("--annotate-values", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def selected_tasks(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(TASKS)
    tasks = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(tasks) - set(TASKS))
    if invalid:
        raise ValueError(f"Unknown tasks: {invalid}; valid tasks: {TASKS}")
    return tasks


def read_tsv(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t", low_memory=False)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def clean_label(label: str) -> str:
    text = str(label)
    text = re.sub(r"^label_\d+_", "", text)
    text = re.sub(r"^\d+_", "", text)
    text = text.replace("_", " ")
    text = text.replace("  ", " ").strip()
    return "\n".join(textwrap.wrap(text, width=32, break_long_words=False))


def map_np_class_label(raw_label: object, label_index: object) -> str:
    """Map numeric NPClassifier labels to their seven pathway-level names."""
    try:
        numeric_label = int(float(str(raw_label).strip()))
    except (TypeError, ValueError):
        numeric_label = None
    if numeric_label in NP_CLASS_LABELS:
        return NP_CLASS_LABELS[numeric_label]

    try:
        index = int(label_index)
    except (TypeError, ValueError):
        return str(raw_label)
    # Multiclass outputs commonly store zero-based class indices.
    if (index + 1) in NP_CLASS_LABELS:
        return NP_CLASS_LABELS[index + 1]
    if index in NP_CLASS_LABELS:
        return NP_CLASS_LABELS[index]
    return str(raw_label)


def load_np_classification(root: Path) -> pd.DataFrame:
    path = root / "results/np_classification/full/npclassification_per_class.tsv"
    df = read_tsv(path, ("feature", "split", "split_seed", "eval_set", "class_index", "label", "support", "auroc"))
    df = df[df["eval_set"].astype(str) == "test"].copy()
    df = df.rename(
        columns={
            "split": "split_mode",
            "class_index": "label_index",
            "label": "label",
            "support": "positive_support",
        }
    )
    df["task"] = "np_classification"
    df["label"] = [
        map_np_class_label(raw_label, label_index)
        for raw_label, label_index in zip(df["label"], df["label_index"])
    ]
    return df[
        ["task", "split_mode", "split_seed", "feature", "label_index", "label", "positive_support", "auroc"]
    ]


def compatible_prediction_file(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=True) as payload:
            return "label_columns" in payload.files and any(key.endswith("_y_true") for key in payload.files)
    except Exception:
        return False


def find_prediction_file(results_dir: Path, preferred_names: tuple[str, ...]) -> Path:
    for name in preferred_names:
        path = results_dir / name
        if path.exists() and compatible_prediction_file(path):
            return path
    candidates = [path for path in sorted(results_dir.glob("*.npz")) if compatible_prediction_file(path)]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No compatible prediction NPZ under {results_dir}")
    raise ValueError("Multiple compatible prediction NPZ files found: " + ", ".join(map(str, candidates)))


def per_label_from_predictions(
    root: Path,
    *,
    task: str,
    results_dir: str,
    runs_name: str,
    prediction_names: tuple[str, ...],
    eval_set: str,
) -> pd.DataFrame:
    directory = root / results_dir
    runs = read_tsv(directory / runs_name, ("feature", "split_mode", "split_seed", "eval_set"))
    runs = runs[runs["eval_set"].astype(str) == eval_set].copy()
    runs = runs[~runs["feature"].astype(str).str.startswith("ensemble_")].copy()
    if "feature_type" in runs.columns:
        runs = runs[runs["feature_type"].astype(str) != "probability_ensemble"].copy()
    prediction_path = find_prediction_file(directory, prediction_names)

    rows: list[dict] = []
    with np.load(prediction_path, allow_pickle=True) as payload:
        labels = [str(value) for value in payload["label_columns"]]
        for split_mode, split_seed in (
            runs[["split_mode", "split_seed"]].drop_duplicates().sort_values(["split_mode", "split_seed"]).itertuples(index=False, name=None)
        ):
            split_mode = str(split_mode)
            split_seed = int(split_seed)
            prefix = f"{split_mode}_seed{split_seed}_{eval_set}"
            y_key = f"{prefix}_y_true"
            if y_key not in payload.files:
                raise KeyError(f"Missing {y_key} in {prediction_path}")
            y_true = np.asarray(payload[y_key], dtype=int)
            if y_true.ndim != 2 or y_true.shape[1] != len(labels):
                raise ValueError(f"Unexpected y_true shape for {prefix}: {y_true.shape}")

            features = sorted(runs.loc[(runs["split_mode"] == split_mode) & (runs["split_seed"] == split_seed), "feature"].astype(str).unique())
            for feature in features:
                probability_key = f"{prefix}_{feature}_probability"
                if probability_key not in payload.files:
                    warnings.warn(f"Skipping missing probability array: {probability_key}")
                    continue
                probability = np.asarray(payload[probability_key], dtype=float)
                if probability.shape != y_true.shape:
                    raise ValueError(f"{probability_key} shape {probability.shape} != y_true {y_true.shape}")
                for label_index, label in enumerate(labels):
                    y = y_true[:, label_index]
                    support = int(y.sum())
                    auroc = float(roc_auc_score(y, probability[:, label_index])) if np.unique(y).size == 2 else np.nan
                    rows.append(
                        {
                            "task": task,
                            "split_mode": split_mode,
                            "split_seed": split_seed,
                            "feature": feature,
                            "label_index": label_index,
                            "label": label,
                            "positive_support": support,
                            "auroc": auroc,
                        }
                    )
    return pd.DataFrame(rows)


def load_pmn(root: Path) -> pd.DataFrame:
    return per_label_from_predictions(
        root,
        task="pmn_secondary_pathway",
        results_dir="results/pmn_secondary_pathway/full",
        runs_name="pmn_secondary_pathway_full_ensemble_repeated_runs.tsv",
        prediction_names=("pmn_secondary_pathway_full_ensemble_predictions.npz",),
        eval_set="bao_B_external_test",
    )


def load_kegg(root: Path) -> pd.DataFrame:
    return per_label_from_predictions(
        root,
        task="kegg_pathway",
        results_dir="results/kegg_pathway/full",
        runs_name="kegg_pathway_full_ensemble_repeated_runs.tsv",
        prediction_names=(
            "kegg_pathway_full_ensemble_repeated_predictions.npz",
            "kegg_pathway_full_ensemble_predictions.npz",
        ),
        eval_set="test",
    )


LOADERS = {
    "np_classification": load_np_classification,
    "pmn_secondary_pathway": load_pmn,
    "kegg_pathway": load_kegg,
}


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(["task", "split_mode", "feature", "label_index", "label"], dropna=False)
        .agg(
            n_split_seeds=("split_seed", "nunique"),
            positive_support_mean=("positive_support", "mean"),
            positive_support_min=("positive_support", "min"),
            positive_support_max=("positive_support", "max"),
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
        )
        .reset_index()
    )
    return summary


def select_features(summary: pd.DataFrame, top_fms: int) -> list[str]:
    available = set(summary["feature"].astype(str))
    baselines = [feature for feature in BASELINES if feature in available]
    fms = [feature for feature in available if feature not in BASELINES and not feature.startswith("ensemble_") and feature != "rdkit_descriptors"]
    fm_scores = summary[summary["feature"].isin(fms)].groupby("feature")["auroc_mean"].mean().sort_values(ascending=False)
    selected_fms = list(fm_scores.index if top_fms == 0 else fm_scores.index[:top_fms])
    return baselines + selected_fms


def label_order(summary: pd.DataFrame, selected: list[str], mode: str) -> list[int]:
    del selected
    labels = (
        summary.groupby(["label_index", "label"], as_index=False)
        .agg(positive_support_mean=("positive_support_mean", "mean"))
    )
    if mode == "original":
        return labels.sort_values("label_index")["label_index"].astype(int).tolist()
    if mode == "support":
        return labels.sort_values(
            ["positive_support_mean", "label_index"], ascending=[False, True]
        )["label_index"].astype(int).tolist()
    morgan = (
        summary[summary["feature"] == "morgan_r2_2048"]
        .groupby("label_index")["auroc_mean"]
        .mean()
    )
    labels = labels.assign(morgan_auroc=labels["label_index"].map(morgan))
    return labels.sort_values(
        ["morgan_auroc", "label_index"], ascending=[False, True]
    )["label_index"].astype(int).tolist()


def make_task_heatmap(
    task: str,
    summary: pd.DataFrame,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
    top_fms: int,
    order_mode: str,
    annotate: bool,
) -> list[Path]:
    task_df = summary[summary["task"] == task].copy()
    selected = select_features(task_df, top_fms)

    # Exclude labels for which AUROC is undefined in the evaluation set, such
    # as labels with zero positive or zero negative examples. These labels are
    # retained in the source CSVs with NaN AUROC but are not shown as empty rows
    # in the heatmap.
    evaluable = (
        task_df[task_df["feature"].isin(selected)]
        .groupby(["label_index", "label"], dropna=False)
        .agg(
            has_finite_auroc=("auroc_mean", lambda values: bool(np.isfinite(values).any())),
            positive_support_mean=("positive_support_mean", "mean"),
        )
        .reset_index()
    )
    omitted = evaluable.loc[~evaluable["has_finite_auroc"], "label"].astype(str).tolist()
    if omitted:
        print(
            f"Omitting {task} labels with undefined AUROC from heatmap: "
            + ", ".join(omitted)
        )
    valid_label_indices = set(
        evaluable.loc[evaluable["has_finite_auroc"], "label_index"].astype(int)
    )
    task_df = task_df[task_df["label_index"].astype(int).isin(valid_label_indices)].copy()

    order = label_order(task_df, selected, order_mode)
    label_map = task_df[["label_index", "label"]].drop_duplicates().set_index("label_index")["label"].to_dict()

    fig_height = max(3.8, 0.33 * len(order) + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, fig_height), constrained_layout=True, sharey=True)
    image = None
    for ax, split_mode in zip(axes, ("random", "scaffold")):
        split_df = task_df[task_df["split_mode"] == split_mode]
        matrix = (
            split_df.pivot_table(index="label_index", columns="feature", values="auroc_mean", aggfunc="mean")
            .reindex(index=order, columns=selected)
        )
        image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", vmin=0.5, vmax=1.0, cmap="viridis")
        ax.set_title(SPLIT_TITLES[split_mode])
        ax.set_xticks(np.arange(len(selected)))
        ax.set_xticklabels([feature_label(feature) for feature in selected], rotation=55, ha="right")
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([clean_label(label_map[index]) for index in order])
        ax.set_xlabel("Molecular representation")
        if ax is axes[0]:
            ax.set_ylabel("Class or pathway label")
        if annotate:
            values = matrix.to_numpy(dtype=float)
            for row in range(values.shape[0]):
                for col in range(values.shape[1]):
                    if np.isfinite(values[row, col]):
                        ax.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if values[row, col] < 0.75 else "black")

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, location="right", shrink=0.8, pad=0.02)
        colorbar.set_label("AUROC")
    stem = output_dir / f"{task}_per_label_auroc_heatmap"
    return save_figure(fig, stem, formats, dpi)


def main() -> None:
    args = parse_args()
    paths = resolve_paths(__file__, args.project_root, args.output_dir)
    configure_matplotlib()
    formats = parse_formats(args.formats)

    raw_frames: list[pd.DataFrame] = []
    loaded: list[str] = []
    for task in selected_tasks(args.tasks):
        try:
            frame = LOADERS[task](paths.project_root)
        except (FileNotFoundError, KeyError) as exc:
            if args.strict:
                raise
            print(f"WARNING: skipping {task}: {exc}")
            continue
        raw_frames.append(frame)
        loaded.append(task)
        print(f"Loaded {task}: {len(frame):,} per-label seed rows")

    if not raw_frames:
        raise RuntimeError("No per-label results were loaded")
    raw = pd.concat(raw_frames, ignore_index=True)
    summary = summarize(raw)

    raw_path = paths.source_data_dir / "per_label_auroc_by_seed.csv"
    summary_path = paths.source_data_dir / "per_label_auroc_summary.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    outputs: list[Path] = []
    for task in loaded:
        outputs.extend(
            make_task_heatmap(
                task,
                summary,
                paths.output_dir,
                formats,
                args.dpi,
                args.top_fms,
                args.label_order,
                args.annotate_values,
            )
        )

    print("Wrote source data:")
    print(raw_path)
    print(summary_path)
    print("Wrote figures:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

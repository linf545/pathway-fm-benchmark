#!/usr/bin/env python3
"""Run repeated-measures benchmark statistics for the molecular FM manuscript.

The script performs Friedman omnibus tests across all individual molecular
representations for:

* full-data results under each split scheme;
* the three smallest label-efficiency training levels under each split scheme;
* PlantCyc reaction-neighbor full-data results for each negative-set difficulty.

For the four tasks with a prespecified focal FM, the script also performs paired
Wilcoxon signed-rank comparisons between that FM and Morgan r2/r3. PMN
Morgan-r2 + GraphMVP-GIN probability ensembles are compared with Morgan r2 in a
separate planned analysis. No multiple-testing adjustment is applied.

For pathway label-efficiency experiments, sampling replicates are averaged
within each split seed before statistical testing. For NP-likeness and NP-class
label-efficiency experiments, the matched sample seed is the repeated-measure
block because the current outputs use a fixed base split with repeated matched
training samples.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import warnings

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


MORGAN_FEATURES = ("morgan_r2_2048", "morgan_r3_2048")
FOCAL_FMS = {
    "np_likeness": "molformer_c3_1_1b",
    "np_classification": "chemberta_77m_mlm",
    "pmn_secondary_pathway": "graphmvp_gin",
    "kegg_pathway": "molformer_c3_1_1b",
}
FEATURE_LABELS = {
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
    "ensemble_morgan_r2_plus_graphmvp_gin": "Morgan r2 + GraphMVP-GIN",
}


@dataclass(frozen=True)
class TaskFiles:
    full: str
    efficiency: str | None


TASK_FILES = {
    "np_likeness": TaskFiles(
        "results/np_likeness/full/np_likeness_full_ensemble_repeated_runs.tsv",
        "results/np_likeness/label_efficiency/np_likeness_label_efficiency_repeated_runs.tsv",
    ),
    "np_classification": TaskFiles(
        "results/np_classification/full/npclassification_repeated_runs.tsv",
        "results/np_classification/label_efficiency/npclassification_label_efficiency_runs.tsv",
    ),
    "pmn_secondary_pathway": TaskFiles(
        "results/pmn_secondary_pathway/full/pmn_secondary_pathway_full_ensemble_repeated_runs.tsv",
        "results/pmn_secondary_pathway/data_efficiency/pmn_secondary_pathway_data_efficiency_per_split_means.tsv",
    ),
    "kegg_pathway": TaskFiles(
        "results/kegg_pathway/full/kegg_pathway_full_ensemble_repeated_runs.tsv",
        "results/kegg_pathway/data_efficiency/kegg_pathway_data_efficiency_per_split_means.tsv",
    ),
    "plantcyc_reaction_neighbor": TaskFiles(
        "results/plantcyc_reaction_neighbor/full/reaction_neighbor_repeated_runs.tsv",
        None,
    ),
}

ALL_TASKS = tuple(TASK_FILES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: results/statistics/friedman_benchmark",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated task names or 'all'.",
    )
    parser.add_argument(
        "--n-smallest",
        type=int,
        default=3,
        help="Number of smallest training levels included for label-efficiency tests.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail rather than skip a task when an input file is missing.",
    )
    return parser.parse_args()


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def read_tsv(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t", low_memory=False)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df


def exclude_ensembles(df: pd.DataFrame) -> pd.DataFrame:
    out = df[~df["feature"].astype(str).str.startswith("ensemble_")].copy()
    if "feature_type" in out.columns:
        out = out[out["feature_type"].astype(str) != "probability_ensemble"].copy()
    return out


def standardize_long(
    df: pd.DataFrame,
    *,
    task: str,
    split_column: str,
    replicate_column: str,
    metric_column: str,
    regime: str,
    training_size_column: str | None = None,
    variant_column: str | None = None,
) -> pd.DataFrame:
    columns = ["feature", split_column, replicate_column, metric_column]
    if training_size_column:
        columns.append(training_size_column)
    if variant_column:
        columns.append(variant_column)
    out = df[columns].copy()
    out = out.rename(
        columns={
            split_column: "split",
            replicate_column: "replicate",
            metric_column: "metric_value",
        }
    )
    out["task"] = task
    out["regime"] = regime
    out["training_size"] = (
        out[training_size_column] if training_size_column else pd.NA
    )
    if training_size_column and training_size_column != "training_size":
        out = out.drop(columns=[training_size_column])
    out["variant"] = out[variant_column] if variant_column else ""
    if variant_column and variant_column != "variant":
        out = out.drop(columns=[variant_column])
    out["metric_name"] = "AUROC" if task in {"np_likeness", "plantcyc_reaction_neighbor"} else "Macro AUROC"
    out["feature"] = out["feature"].astype(str)
    out["replicate"] = out["replicate"].astype(str)
    out["metric_value"] = pd.to_numeric(out["metric_value"], errors="coerce")
    return out[
        [
            "task",
            "regime",
            "split",
            "training_size",
            "variant",
            "replicate",
            "feature",
            "metric_name",
            "metric_value",
        ]
    ]


def load_np_likeness(root: Path, n_smallest: int) -> pd.DataFrame:
    files = TASK_FILES["np_likeness"]
    full = read_tsv(root / files.full, ("feature", "split_mode", "split_seed", "eval_set", "auroc"))
    full = exclude_ensembles(full[full["eval_set"].astype(str) == "test"])
    full_long = standardize_long(
        full,
        task="np_likeness",
        split_column="split_mode",
        replicate_column="split_seed",
        metric_column="auroc",
        regime="full_data",
    )

    eff = read_tsv(
        root / str(files.efficiency),
        ("feature", "split_mode", "sample_seed", "requested_n_train", "eval_set", "auroc"),
    )
    eff = eff[eff["eval_set"].astype(str) == "test"].copy()
    smallest = sorted(pd.to_numeric(eff["requested_n_train"]).dropna().unique())[:n_smallest]
    eff = eff[eff["requested_n_train"].isin(smallest)].copy()
    if "base_split_seed" in eff.columns:
        eff["replicate_id"] = eff["base_split_seed"].astype(str) + ":" + eff["sample_seed"].astype(str)
    else:
        eff["replicate_id"] = eff["sample_seed"].astype(str)
    eff_long = standardize_long(
        eff,
        task="np_likeness",
        split_column="split_mode",
        replicate_column="replicate_id",
        metric_column="auroc",
        regime="label_efficiency",
        training_size_column="requested_n_train",
    )
    return pd.concat([full_long, eff_long], ignore_index=True)


def load_np_classification(root: Path, n_smallest: int) -> pd.DataFrame:
    files = TASK_FILES["np_classification"]
    full = read_tsv(root / files.full, ("feature", "split", "split_seed", "eval_set", "auroc_ovr_macro"))
    full = full[full["eval_set"].astype(str) == "test"].copy()
    full_long = standardize_long(
        full,
        task="np_classification",
        split_column="split",
        replicate_column="split_seed",
        metric_column="auroc_ovr_macro",
        regime="full_data",
    )

    eff = read_tsv(
        root / str(files.efficiency),
        ("feature", "split", "sample_seed", "requested_n_train", "eval_set", "auroc_ovr_macro"),
    )
    eff = eff[eff["eval_set"].astype(str) == "test"].copy()
    smallest = sorted(pd.to_numeric(eff["requested_n_train"]).dropna().unique())[:n_smallest]
    eff = eff[eff["requested_n_train"].isin(smallest)].copy()
    if "base_split_seed" in eff.columns:
        eff["replicate_id"] = eff["base_split_seed"].astype(str) + ":" + eff["sample_seed"].astype(str)
    else:
        eff["replicate_id"] = eff["sample_seed"].astype(str)
    eff_long = standardize_long(
        eff,
        task="np_classification",
        split_column="split",
        replicate_column="replicate_id",
        metric_column="auroc_ovr_macro",
        regime="label_efficiency",
        training_size_column="requested_n_train",
    )
    return pd.concat([full_long, eff_long], ignore_index=True)


def load_pathway_task(root: Path, task: str, n_smallest: int) -> pd.DataFrame:
    files = TASK_FILES[task]
    eval_set = "bao_B_external_test" if task == "pmn_secondary_pathway" else "test"
    full = read_tsv(
        root / files.full,
        ("feature", "split_mode", "split_seed", "eval_set", "macro_auroc"),
    )
    full = exclude_ensembles(full[full["eval_set"].astype(str) == eval_set])
    full_long = standardize_long(
        full,
        task=task,
        split_column="split_mode",
        replicate_column="split_seed",
        metric_column="macro_auroc",
        regime="full_data",
    )

    eff_path = root / str(files.efficiency)
    if eff_path.exists():
        eff = read_tsv(
            eff_path,
            ("feature", "split_mode", "split_seed", "requested_n_per_label", "eval_set", "macro_auroc"),
        )
        eff = eff[eff["eval_set"].astype(str) == eval_set].copy()
    else:
        # Backward-compatible fallback to replicate-level output.
        repeated_path = eff_path.with_name(eff_path.name.replace("_per_split_means", "_repeated_runs"))
        eff = read_tsv(
            repeated_path,
            ("feature", "split_mode", "split_seed", "requested_n_per_label", "eval_set", "macro_auroc"),
        )
        eff = eff[eff["eval_set"].astype(str) == eval_set].copy()
        sampling_column = "sampling_seed" if "sampling_seed" in eff.columns else "sample_seed"
        group_cols = ["feature", "split_mode", "split_seed", "requested_n_per_label", "eval_set"]
        eff = eff.groupby(group_cols, as_index=False)["macro_auroc"].mean()

    smallest = sorted(pd.to_numeric(eff["requested_n_per_label"]).dropna().unique())[:n_smallest]
    eff = eff[eff["requested_n_per_label"].isin(smallest)].copy()
    eff_long = standardize_long(
        eff,
        task=task,
        split_column="split_mode",
        replicate_column="split_seed",
        metric_column="macro_auroc",
        regime="label_efficiency",
        training_size_column="requested_n_per_label",
    )
    return pd.concat([full_long, eff_long], ignore_index=True)


def load_reaction_neighbor(root: Path, n_smallest: int) -> pd.DataFrame:
    del n_smallest
    files = TASK_FILES["plantcyc_reaction_neighbor"]
    full = read_tsv(
        root / files.full,
        ("feature", "split_scheme", "split_seed", "negative_variant", "eval_set", "auroc"),
    )
    full = full[full["eval_set"].astype(str) == "test"].copy()
    return standardize_long(
        full,
        task="plantcyc_reaction_neighbor",
        split_column="split_scheme",
        replicate_column="split_seed",
        metric_column="auroc",
        regime="full_data",
        variant_column="negative_variant",
    )


LOADERS: dict[str, Callable[[Path, int], pd.DataFrame]] = {
    "np_likeness": load_np_likeness,
    "np_classification": load_np_classification,
    "pmn_secondary_pathway": lambda root, n: load_pathway_task(root, "pmn_secondary_pathway", n),
    "kegg_pathway": lambda root, n: load_pathway_task(root, "kegg_pathway", n),
    "plantcyc_reaction_neighbor": load_reaction_neighbor,
}


def condition_columns() -> list[str]:
    return ["task", "regime", "split", "training_size", "variant", "metric_name"]


def complete_pivot(group: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
    pivot = group.pivot_table(index="replicate", columns="feature", values="metric_value", aggfunc="mean")
    n_before = len(pivot)
    # Retain only models measured in every candidate block, then retain complete blocks.
    pivot = pivot.dropna(axis=1, how="any").dropna(axis=0, how="any")
    features = [str(column) for column in pivot.columns]
    return pivot, features, n_before


def mean_ranks(pivot: pd.DataFrame) -> pd.Series:
    # Highest performance receives rank 1.
    ranks = np.vstack([rankdata(-row, method="average") for row in pivot.to_numpy(dtype=float)])
    return pd.Series(ranks.mean(axis=0), index=pivot.columns)


def run_friedman(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict] = []
    mean_rows: list[dict] = []

    for keys, group in long_df.groupby(condition_columns(), dropna=False, sort=True):
        condition = dict(zip(condition_columns(), keys))
        pivot, features, n_before = complete_pivot(group)
        if len(features) < 3 or len(pivot) < 2:
            warnings.warn(f"Skipping Friedman test with insufficient complete data: {condition}")
            continue

        arrays = [pivot[feature].to_numpy(dtype=float) for feature in features]
        try:
            statistic, p_value = friedmanchisquare(*arrays)
        except ValueError as exc:
            warnings.warn(f"Friedman test failed for {condition}: {exc}")
            statistic, p_value = np.nan, np.nan

        means = pivot.mean(axis=0).sort_values(ascending=False)
        ranks = mean_ranks(pivot)
        result_rows.append(
            {
                **condition,
                "n_replicates_available": int(n_before),
                "n_replicates_complete": int(len(pivot)),
                "n_models": int(len(features)),
                "friedman_statistic": float(statistic),
                "p_value": float(p_value),
                "best_mean_feature": str(means.index[0]),
                "best_mean_feature_label": feature_label(str(means.index[0])),
                "best_mean_metric": float(means.iloc[0]),
            }
        )
        for feature in features:
            mean_rows.append(
                {
                    **condition,
                    "feature": feature,
                    "feature_label": feature_label(feature),
                    "n_replicates": int(len(pivot)),
                    "mean_metric": float(pivot[feature].mean()),
                    "sd_metric": float(pivot[feature].std(ddof=1)) if len(pivot) > 1 else np.nan,
                    "mean_rank": float(ranks[feature]),
                }
            )

    return pd.DataFrame(result_rows), pd.DataFrame(mean_rows)


def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) != len(y):
        raise ValueError("Paired arrays have unequal lengths")
    if np.allclose(x, y, equal_nan=False):
        return 0.0, 1.0
    try:
        result = wilcoxon(x, y, alternative="two-sided", method="auto")
    except TypeError:  # Older SciPy without method keyword.
        result = wilcoxon(x, y, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def run_planned_posthoc(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for task, focal_fm in FOCAL_FMS.items():
        task_df = long_df[long_df["task"] == task]
        for keys, group in task_df.groupby(condition_columns(), dropna=False, sort=True):
            condition = dict(zip(condition_columns(), keys))
            pivot = group.pivot_table(index="replicate", columns="feature", values="metric_value", aggfunc="mean")
            for morgan in MORGAN_FEATURES:
                if focal_fm not in pivot.columns or morgan not in pivot.columns:
                    continue
                pair = pivot[[focal_fm, morgan]].dropna()
                if len(pair) < 2:
                    continue
                statistic, p_value = paired_wilcoxon(
                    pair[focal_fm].to_numpy(dtype=float),
                    pair[morgan].to_numpy(dtype=float),
                )
                means = pair.mean(axis=0)
                higher = str(means.idxmax())
                rows.append(
                    {
                        **condition,
                        "comparison_family": "focal_fm_vs_morgan",
                        "model_1": focal_fm,
                        "model_1_label": feature_label(focal_fm),
                        "model_2": morgan,
                        "model_2_label": feature_label(morgan),
                        "test": "paired Wilcoxon signed-rank",
                        "n_replicates": int(len(pair)),
                        "statistic": statistic,
                        "p_value": p_value,
                        "model_1_mean": float(means[focal_fm]),
                        "model_2_mean": float(means[morgan]),
                        "higher_mean_model": higher,
                        "higher_mean_model_label": feature_label(higher),
                    }
                )
    return pd.DataFrame(rows)


def run_pmn_ensemble_tests(root: Path) -> pd.DataFrame:
    path = root / TASK_FILES["pmn_secondary_pathway"].full
    if not path.exists():
        return pd.DataFrame()
    df = read_tsv(path, ("feature", "split_mode", "split_seed", "eval_set", "macro_auroc"))
    df = df[df["eval_set"].astype(str) == "bao_B_external_test"].copy()
    model_1 = "ensemble_morgan_r2_plus_graphmvp_gin"
    model_2 = "morgan_r2_2048"
    rows: list[dict] = []
    for split, group in df[df["feature"].isin([model_1, model_2])].groupby("split_mode"):
        pivot = group.pivot_table(index="split_seed", columns="feature", values="macro_auroc", aggfunc="mean")
        if model_1 not in pivot.columns or model_2 not in pivot.columns:
            continue
        pair = pivot[[model_1, model_2]].dropna()
        if len(pair) < 2:
            continue
        statistic, p_value = paired_wilcoxon(pair[model_1].to_numpy(), pair[model_2].to_numpy())
        means = pair.mean(axis=0)
        higher = str(means.idxmax())
        rows.append(
            {
                "task": "pmn_secondary_pathway",
                "regime": "full_data",
                "split": split,
                "training_size": pd.NA,
                "variant": "",
                "metric_name": "Macro AUROC",
                "comparison_family": "pmn_ensemble_vs_morgan_r2",
                "model_1": model_1,
                "model_1_label": feature_label(model_1),
                "model_2": model_2,
                "model_2_label": feature_label(model_2),
                "test": "paired Wilcoxon signed-rank",
                "n_replicates": int(len(pair)),
                "statistic": statistic,
                "p_value": p_value,
                "model_1_mean": float(means[model_1]),
                "model_2_mean": float(means[model_2]),
                "higher_mean_model": higher,
                "higher_mean_model_label": feature_label(higher),
            }
        )
    return pd.DataFrame(rows)


def requested_tasks(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(ALL_TASKS)
    tasks = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(tasks) - set(ALL_TASKS))
    if invalid:
        raise ValueError(f"Unknown tasks: {invalid}. Valid tasks: {ALL_TASKS}")
    return tasks


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output_dir = (args.output_dir or root / "results/statistics/friedman_benchmark").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    loaded_tasks: list[str] = []
    for task in requested_tasks(args.tasks):
        try:
            frame = LOADERS[task](root, args.n_smallest)
        except FileNotFoundError as exc:
            if args.strict:
                raise
            print(f"WARNING: skipping {task}; missing input: {exc}")
            continue
        frames.append(frame)
        loaded_tasks.append(task)
        print(f"Loaded {task}: {len(frame):,} standardized rows")

    if not frames:
        raise RuntimeError("No task results were loaded")
    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df[np.isfinite(long_df["metric_value"].to_numpy(dtype=float))].copy()

    friedman, model_summary = run_friedman(long_df)
    posthoc = run_planned_posthoc(long_df)
    ensemble = run_pmn_ensemble_tests(root) if "pmn_secondary_pathway" in loaded_tasks else pd.DataFrame()
    if not ensemble.empty:
        posthoc = pd.concat([posthoc, ensemble], ignore_index=True, sort=False)

    long_path = output_dir / "friedman_input_long.csv"
    friedman_path = output_dir / "friedman_omnibus_results.csv"
    model_path = output_dir / "friedman_model_means_and_ranks.csv"
    posthoc_path = output_dir / "planned_wilcoxon_results.csv"

    long_df.to_csv(long_path, index=False)
    friedman.to_csv(friedman_path, index=False)
    model_summary.to_csv(model_path, index=False)
    posthoc.to_csv(posthoc_path, index=False)

    print("\nWrote:")
    for path in [long_path, friedman_path, model_path, posthoc_path]:
        print(path)

    if not friedman.empty:
        display = friedman[
            [
                "task",
                "regime",
                "split",
                "training_size",
                "variant",
                "n_replicates_complete",
                "n_models",
                "friedman_statistic",
                "p_value",
                "best_mean_feature_label",
                "best_mean_metric",
            ]
        ]
        print("\n=== Friedman omnibus results ===")
        print(display.to_string(index=False))


if __name__ == "__main__":
    main()

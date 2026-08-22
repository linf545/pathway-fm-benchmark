#!/usr/bin/env python3
"""Run paired t-tests for the four main results under both split types.

Comparisons
-----------
1. NP-likeness full data: MolFormer versus Morgan r2
2. NP-class sample efficiency at 10 training examples:
   ChemBERTa-77M versus Morgan r2
3. PMN sample efficiency at one positive example per pathway:
   GraphMVP-GIN versus Morgan r2
4. PMN full-data ensemble:
   Morgan r2 + GraphMVP-GIN versus Morgan r2

Each comparison is tested separately under random and scaffold splitting.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel


LABELS = {
    "molformer_c3_1_1b": "MolFormer",
    "chemberta_77m_mlm": "ChemBERTa-77M",
    "graphmvp_gin": "GraphMVP-GIN",
    "morgan_r2_2048": "Morgan r2",
    "ensemble_morgan_r2_plus_graphmvp_gin": "Morgan r2 + GraphMVP-GIN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="FM_bench project root. Default: current directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Default: <project-root>/results/statistics/"
            "main_result_paired_t_tests"
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def read_tsv(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def make_pair_table(
    df: pd.DataFrame,
    *,
    comparison_id: str,
    replicate_column: str,
    value_column: str,
    model_1: str,
    model_2: str,
) -> pd.DataFrame:
    subset = df.loc[
        df["feature"].astype(str).isin([model_1, model_2]),
        [replicate_column, "feature", value_column],
    ].copy()
    subset[value_column] = pd.to_numeric(subset[value_column], errors="coerce")
    subset = subset.dropna(subset=[replicate_column, "feature", value_column])

    pair = subset.pivot_table(
        index=replicate_column,
        columns="feature",
        values=value_column,
        aggfunc="mean",
    )

    missing_models = [
        model for model in (model_1, model_2) if model not in pair.columns
    ]
    if missing_models:
        raise ValueError(
            f"{comparison_id}: missing models after filtering: {missing_models}"
        )

    pair = pair[[model_1, model_2]].dropna().copy()
    pair.index = pair.index.astype(str)
    pair = pair.sort_index()

    if len(pair) < 2:
        raise ValueError(f"{comparison_id}: fewer than two matched pairs")
    return pair


def load_np_likeness(root: Path, split: str) -> tuple[pd.DataFrame, dict]:
    path = (
        root
        / "results/np_likeness/full/"
        "np_likeness_full_ensemble_repeated_runs.tsv"
    )
    df = read_tsv(
        path,
        ("feature", "split_mode", "split_seed", "eval_set", "auroc"),
    )
    df = df.loc[
        (df["eval_set"].astype(str) == "test")
        & (df["split_mode"].astype(str) == split)
    ].copy()

    model_1 = "molformer_c3_1_1b"
    model_2 = "morgan_r2_2048"
    comparison_id = f"np_likeness_{split}"

    pair = make_pair_table(
        df,
        comparison_id=comparison_id,
        replicate_column="split_seed",
        value_column="auroc",
        model_1=model_1,
        model_2=model_2,
    )
    return pair, {
        "comparison_id": comparison_id,
        "task": "np_likeness",
        "regime": "full_data",
        "split": split,
        "training_level": np.nan,
        "metric_name": "AUROC",
        "model_1": model_1,
        "model_2": model_2,
        "input_file": str(path),
    }


def load_np_class_efficiency(
    root: Path,
    split: str,
) -> tuple[pd.DataFrame, dict]:
    path = (
        root
        / "results/np_classification/label_efficiency/"
        "npclassification_label_efficiency_runs.tsv"
    )
    df = read_tsv(
        path,
        (
            "feature",
            "split",
            "sample_seed",
            "requested_n_train",
            "eval_set",
            "auroc_ovr_macro",
        ),
    )
    training_size = pd.to_numeric(df["requested_n_train"], errors="coerce")
    df = df.loc[
        (df["eval_set"].astype(str) == "test")
        & (df["split"].astype(str) == split)
        & np.isclose(training_size, 10)
    ].copy()

    if "base_split_seed" in df.columns:
        df["replicate_id"] = (
            df["base_split_seed"].astype(str)
            + ":"
            + df["sample_seed"].astype(str)
        )
    else:
        df["replicate_id"] = df["sample_seed"].astype(str)

    model_1 = "chemberta_77m_mlm"
    model_2 = "morgan_r2_2048"
    comparison_id = f"np_class_{split}_n10"

    pair = make_pair_table(
        df,
        comparison_id=comparison_id,
        replicate_column="replicate_id",
        value_column="auroc_ovr_macro",
        model_1=model_1,
        model_2=model_2,
    )
    return pair, {
        "comparison_id": comparison_id,
        "task": "np_classification",
        "regime": "sample_efficiency",
        "split": split,
        "training_level": 10,
        "metric_name": "Macro AUROC",
        "model_1": model_1,
        "model_2": model_2,
        "input_file": str(path),
    }


def load_pmn_efficiency(root: Path, split: str) -> tuple[pd.DataFrame, dict]:
    summary_path = (
        root
        / "results/pmn_secondary_pathway/data_efficiency/"
        "pmn_secondary_pathway_data_efficiency_per_split_means.tsv"
    )
    repeated_path = (
        root
        / "results/pmn_secondary_pathway/data_efficiency/"
        "pmn_secondary_pathway_data_efficiency_repeated_runs.tsv"
    )
    path = summary_path if summary_path.exists() else repeated_path
    df = read_tsv(
        path,
        (
            "feature",
            "split_mode",
            "split_seed",
            "requested_n_per_label",
            "eval_set",
            "macro_auroc",
        ),
    )
    training_size = pd.to_numeric(
        df["requested_n_per_label"],
        errors="coerce",
    )
    df = df.loc[
        (df["eval_set"].astype(str) == "bao_B_external_test")
        & (df["split_mode"].astype(str) == split)
        & np.isclose(training_size, 1)
    ].copy()

    # Average sampling replicates within each split seed when needed.
    df = (
        df.groupby(["split_seed", "feature"], as_index=False)["macro_auroc"]
        .mean()
    )

    model_1 = "graphmvp_gin"
    model_2 = "morgan_r2_2048"
    comparison_id = f"pmn_{split}_n1"

    pair = make_pair_table(
        df,
        comparison_id=comparison_id,
        replicate_column="split_seed",
        value_column="macro_auroc",
        model_1=model_1,
        model_2=model_2,
    )
    return pair, {
        "comparison_id": comparison_id,
        "task": "pmn_secondary_pathway",
        "regime": "sample_efficiency",
        "split": split,
        "training_level": 1,
        "metric_name": "Macro AUROC",
        "model_1": model_1,
        "model_2": model_2,
        "input_file": str(path),
    }


def load_pmn_ensemble(root: Path, split: str) -> tuple[pd.DataFrame, dict]:
    path = (
        root
        / "results/pmn_secondary_pathway/full/"
        "pmn_secondary_pathway_full_ensemble_repeated_runs.tsv"
    )
    df = read_tsv(
        path,
        (
            "feature",
            "split_mode",
            "split_seed",
            "eval_set",
            "macro_auroc",
        ),
    )
    df = df.loc[
        (df["eval_set"].astype(str) == "bao_B_external_test")
        & (df["split_mode"].astype(str) == split)
    ].copy()

    model_1 = "ensemble_morgan_r2_plus_graphmvp_gin"
    model_2 = "morgan_r2_2048"
    comparison_id = f"pmn_{split}_ensemble"

    pair = make_pair_table(
        df,
        comparison_id=comparison_id,
        replicate_column="split_seed",
        value_column="macro_auroc",
        model_1=model_1,
        model_2=model_2,
    )
    return pair, {
        "comparison_id": comparison_id,
        "task": "pmn_secondary_pathway",
        "regime": "full_data_ensemble",
        "split": split,
        "training_level": np.nan,
        "metric_name": "Macro AUROC",
        "model_1": model_1,
        "model_2": model_2,
        "input_file": str(path),
    }


def paired_t_test(
    pair: pd.DataFrame,
    metadata: dict,
    *,
    alpha: float,
) -> tuple[dict, pd.DataFrame]:
    model_1 = metadata["model_1"]
    model_2 = metadata["model_2"]
    x = pair[model_1].to_numpy(dtype=float)
    y = pair[model_2].to_numpy(dtype=float)
    differences = x - y

    n_pairs = len(differences)
    dfree = n_pairs - 1
    test = ttest_rel(x, y)
    mean_difference = float(np.mean(differences))
    sd_difference = float(np.std(differences, ddof=1))
    se_difference = sd_difference / np.sqrt(n_pairs)

    if se_difference > 0:
        ci_low, ci_high = t.interval(
            1.0 - alpha,
            df=dfree,
            loc=mean_difference,
            scale=se_difference,
        )
    else:
        ci_low = mean_difference
        ci_high = mean_difference

    cohen_dz = (
        mean_difference / sd_difference if sd_difference > 0 else np.nan
    )

    if n_pairs != 5:
        warnings.warn(
            f"{metadata['comparison_id']}: expected 5 matched pairs but "
            f"found {n_pairs}. Replicates: {', '.join(pair.index)}"
        )

    result = {
        **metadata,
        "model_1_label": LABELS.get(model_1, model_1),
        "model_2_label": LABELS.get(model_2, model_2),
        "test": "two-sided paired t-test",
        "n_pairs": n_pairs,
        "degrees_freedom": dfree,
        "model_1_mean": float(np.mean(x)),
        "model_2_mean": float(np.mean(y)),
        "mean_difference": mean_difference,
        "sd_difference": sd_difference,
        "ci_level": 1.0 - alpha,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "t_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "significant_at_alpha": bool(test.pvalue < alpha),
        "cohen_dz": float(cohen_dz),
        "all_differences_positive": bool(np.all(differences > 0)),
        "matched_replicates": ",".join(pair.index),
        "multiplicity_adjustment": "none",
    }

    matched = pd.DataFrame(
        {
            "comparison_id": metadata["comparison_id"],
            "task": metadata["task"],
            "split": metadata["split"],
            "replicate": pair.index,
            "model_1": model_1,
            "model_1_label": LABELS.get(model_1, model_1),
            "model_1_value": x,
            "model_2": model_2,
            "model_2_label": LABELS.get(model_2, model_2),
            "model_2_value": y,
            "difference_model_1_minus_model_2": differences,
        }
    )
    return result, matched


def write_text_report(results: pd.DataFrame, path: Path) -> None:
    lines = [
        "Main-result paired t-tests",
        "=" * 26,
        "",
        "Each focal comparison was tested separately under random and ",
        "scaffold splitting.",
        "All tests are two-sided paired t-tests on matched replicate-level ",
        "estimates.",
        "No multiplicity adjustment was applied.",
        "",
    ]
    for row in results.itertuples(index=False):
        lines.extend(
            [
                row.comparison_id,
                "-" * len(row.comparison_id),
                f"Split: {row.split}",
                f"{row.model_1_label}: {row.model_1_mean:.6f}",
                f"{row.model_2_label}: {row.model_2_mean:.6f}",
                f"Mean paired difference: {row.mean_difference:.6f}",
                f"95% CI: [{row.ci_low:.6f}, {row.ci_high:.6f}]",
                f"t({row.degrees_freedom}) = {row.t_statistic:.6f}",
                f"Two-sided P = {row.p_value:.8f}",
                f"Cohen dz = {row.cohen_dz:.6f}",
                f"Matched pairs = {row.n_pairs}",
                f"Matched replicates: {row.matched_replicates}",
                f"Significant at alpha=0.05: {row.significant_at_alpha}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else root / "results/statistics/main_result_paired_t_tests"
    )
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = [
        load_np_likeness,
        load_np_class_efficiency,
        load_pmn_efficiency,
        load_pmn_ensemble,
    ]

    result_rows: list[dict] = []
    matched_frames: list[pd.DataFrame] = []
    for split in ("random", "scaffold"):
        for loader in loaders:
            pair, metadata = loader(root, split)
            result, matched = paired_t_test(
                pair,
                metadata,
                alpha=args.alpha,
            )
            result_rows.append(result)
            matched_frames.append(matched)

    results = pd.DataFrame(result_rows).sort_values(
        ["task", "regime", "split"]
    )
    matched_values = pd.concat(matched_frames, ignore_index=True)

    summary_path = output_dir / "main_result_paired_t_tests.csv"
    matched_path = output_dir / "main_result_paired_values.tsv"
    report_path = output_dir / "main_result_paired_t_tests.txt"

    results.to_csv(summary_path, index=False)
    matched_values.to_csv(matched_path, sep="\t", index=False)
    write_text_report(results, report_path)

    display_columns = [
        "comparison_id",
        "split",
        "n_pairs",
        "model_1_mean",
        "model_2_mean",
        "mean_difference",
        "ci_low",
        "ci_high",
        "t_statistic",
        "degrees_freedom",
        "p_value",
        "significant_at_alpha",
    ]
    print("\n=== Main-result paired t-tests ===")
    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.8f}",
        )
    )
    print("\nWrote:")
    print(summary_path)
    print(matched_path)
    print(report_path)


if __name__ == "__main__":
    main()

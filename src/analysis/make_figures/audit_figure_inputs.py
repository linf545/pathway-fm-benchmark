#!/usr/bin/env python3
"""Audit manuscript figure inputs without generating figures."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import assert_finite, assert_unique, infer_project_root, read_tsv


INDIVIDUAL_FEATURES = {
    "morgan_r2_2048",
    "morgan_r3_2048",
    "chemberta_77m_mlm",
    "chemberta_100m_mlm",
    "molformer_c3_1_1b",
    "nafm_graph_mean",
    "molclr_gcn",
    "molclr_gin",
    "graphmvp_gin",
    "molebert_gin",
}
PMN_EFFICIENCY_FEATURES = set(INDIVIDUAL_FEATURES)
KEGG_EFFICIENCY_FEATURES = set(INDIVIDUAL_FEATURES)
PLANTCYC_FEATURES = INDIVIDUAL_FEATURES | {"rdkit_descriptors"}
PMN_ENSEMBLES = {
    "ensemble_morgan_r2_plus_chemberta_77m_mlm",
    "ensemble_morgan_r2_plus_chemberta_100m_mlm",
    "ensemble_morgan_r2_plus_molformer_c3_1_1b",
    "ensemble_morgan_r2_plus_nafm_graph_mean",
    "ensemble_morgan_r2_plus_molclr_gcn",
    "ensemble_morgan_r2_plus_molclr_gin",
    "ensemble_morgan_r2_plus_graphmvp_gin",
    "ensemble_morgan_r2_plus_molebert_gin",
    "ensemble_all_equal_probability",
}


@dataclass(frozen=True)
class TableSpec:
    relative_path: str
    key: tuple[str, ...]
    metrics: tuple[str, ...]
    expected_rows: int
    expected_values: Mapping[str, set]


TABLES = {
    "np_likeness_full": TableSpec(
        "results/np_likeness/full/np_likeness_full_ensemble_repeated_runs.tsv",
        ("feature", "split_mode", "split_seed", "eval_set"),
        ("auroc", "auprc"),
        380,
        {
            "feature": INDIVIDUAL_FEATURES | PMN_ENSEMBLES,
            "split_mode": {"random", "scaffold"},
            "split_seed": {2026, 2027, 2028, 2029, 2030},
            "eval_set": {"validation", "test"},
        },
    ),
    "np_likeness_efficiency": TableSpec(
        "results/np_likeness/label_efficiency/np_likeness_label_efficiency_repeated_runs.tsv",
        ("feature", "split_mode", "sample_seed", "requested_n_train", "eval_set"),
        ("auroc", "auprc"),
        2000,
        {
            "feature": INDIVIDUAL_FEATURES,
            "split_mode": {"random", "scaffold"},
            "sample_seed": {2026, 2027, 2028, 2029, 2030},
            "requested_n_train": {3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000},
            "eval_set": {"validation", "test"},
        },
    ),
    "np_classification_full": TableSpec(
        "results/np_classification/full/npclassification_repeated_runs.tsv",
        ("feature", "split", "split_seed", "eval_set"),
        ("auroc_ovr_macro", "auprc_ovr_macro"),
        200,
        {
            "feature": INDIVIDUAL_FEATURES,
            "split": {"random", "scaffold"},
            "split_seed": {2026, 2027, 2028, 2029, 2030},
            "eval_set": {"validation", "test"},
        },
    ),
    "np_classification_efficiency": TableSpec(
        "results/np_classification/label_efficiency/npclassification_label_efficiency_runs.tsv",
        ("feature", "split", "sample_seed", "requested_n_train", "eval_set"),
        ("auroc_ovr_macro", "auprc_ovr_macro"),
        1200,
        {
            "feature": INDIVIDUAL_FEATURES,
            "split": {"random", "scaffold"},
            "sample_seed": {2026, 2027, 2028, 2029, 2030},
            "requested_n_train": {10, 30, 100, 300, 1000, 10000},
            "eval_set": {"validation", "test"},
        },
    ),
    "pmn_full": TableSpec(
        "results/pmn_secondary_pathway/full/pmn_secondary_pathway_full_ensemble_repeated_runs.tsv",
        ("feature", "split_mode", "split_seed", "eval_set"),
        ("macro_auroc", "macro_auprc"),
        380,
        {
            "feature": INDIVIDUAL_FEATURES | PMN_ENSEMBLES,
            "split_mode": {"random", "scaffold"},
            "split_seed": {2026, 2027, 2028, 2029, 2030},
            "eval_set": {"validation", "bao_B_external_test"},
        },
    ),
    "pmn_efficiency": TableSpec(
        "results/pmn_secondary_pathway/data_efficiency/pmn_secondary_pathway_data_efficiency_repeated_runs.tsv",
        (
            "feature",
            "split_mode",
            "split_seed",
            "sampling_seed",
            "requested_n_per_label",
            "eval_set",
        ),
        ("macro_auroc", "macro_auprc"),
        2400,
        {
            "feature": PMN_EFFICIENCY_FEATURES,
            "split_mode": {"random", "scaffold"},
            "split_seed": {2026, 2027, 2028, 2030},
            "sampling_seed": {0, 1, 2, 3, 4},
            "requested_n_per_label": {1, 2, 4},
            "eval_set": {"validation", "bao_B_external_test"},
        },
    ),
    "kegg_full": TableSpec(
        "results/kegg_pathway/full/kegg_pathway_full_ensemble_repeated_runs.tsv",
        ("feature", "split_mode", "split_seed", "eval_set"),
        ("macro_auroc", "macro_auprc"),
        380,
        {
            "feature": INDIVIDUAL_FEATURES | PMN_ENSEMBLES,
            "split_mode": {"random", "scaffold"},
            "split_seed": {2026, 2027, 2028, 2029, 2030},
            "eval_set": {"validation", "test"},
        },
    ),
    "kegg_efficiency": TableSpec(
        "results/kegg_pathway/data_efficiency/kegg_pathway_data_efficiency_repeated_runs.tsv",
        (
            "feature", "split_mode", "split_seed", "sampling_seed",
            "requested_n_per_label", "eval_set",
        ),
        ("macro_auroc", "macro_auprc"),
        6000,
        {
            "feature": KEGG_EFFICIENCY_FEATURES,
            "split_mode": {"random", "scaffold"},
            "split_seed": {2026, 2027, 2028, 2029, 2030},
            "sampling_seed": {0, 1, 2, 3, 4},
            "requested_n_per_label": {4, 8, 16, 24, 32, 64},
            "eval_set": {"validation", "test"},
        },
    ),
    "plantcyc_full": TableSpec(
        "results/plantcyc_reaction_neighbor/full/reaction_neighbor_repeated_runs.tsv",
        ("feature", "negative_variant", "split_scheme", "split_seed", "eval_set"),
        ("auroc", "auprc"),
        660,
        {
            "feature": PLANTCYC_FEATURES,
            "negative_variant": {"easy", "mixed", "hard"},
            "split_scheme": {"random", "scaffold_pair"},
            "split_seed": {2026, 2027, 2028, 2029, 2030},
            "eval_set": {"validation", "test"},
        },
    ),
}


def assert_expected_values(table_name: str, df, expected_values: Mapping[str, set]) -> None:
    for column, expected in expected_values.items():
        observed = set(df[column].dropna().tolist())
        if observed != expected:
            missing = sorted(expected - observed, key=str)
            extra = sorted(observed - expected, key=str)
            raise ValueError(
                f"{table_name}.{column} differs from the current design; "
                f"missing={missing}, extra={extra}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=infer_project_root(__file__),
        help="FM_bench project root",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()

    failures: list[str] = []
    optional_tables = {"kegg_full", "kegg_efficiency"}
    for name, spec in TABLES.items():
        path = root / spec.relative_path
        if name in optional_tables and not path.exists():
            print(f"SKIP {name}: {path} does not exist")
            continue
        try:
            required = [*spec.key, *spec.metrics, *spec.expected_values.keys()]
            df = read_tsv(path, required)
            if len(df) != spec.expected_rows:
                raise ValueError(f"expected {spec.expected_rows:,} rows, found {len(df):,}")
            assert_unique(df, spec.key, name)
            assert_finite(df, spec.metrics, name)
            assert_expected_values(name, df, spec.expected_values)
            if "reached_max_iter" in df.columns and bool(df["reached_max_iter"].any()):
                raise ValueError(f"{int(df['reached_max_iter'].sum())} selected fits reached max_iter")
            if "solver_converged" in df.columns and not bool(df["solver_converged"].all()):
                raise ValueError(f"{int((~df['solver_converged']).sum())} selected fits did not converge")
            print(f"PASS {name}: {len(df):,} rows")
        except Exception as exc:  # report all failures in one run
            failures.append(f"{name}: {exc}")
            print(f"FAIL {name}: {exc}")

    if failures:
        raise SystemExit("\nInput audit failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()

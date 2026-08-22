#!/usr/bin/env python3
"""
Compute a 95% confidence interval for a paired performance difference
across matched split seeds.

For each split seed i:
    d_i = metric(model_a, seed_i) - metric(model_b, seed_i)

The reported interval is:
    mean(d) +/- t_(0.975, n-1) * sd(d) / sqrt(n)

For sample-efficiency experiments, provide --replicate-col so repeated
training-subset replicates are averaged within each seed before pairing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t


def parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_where(items: list[str]) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --where value {item!r}; expected COLUMN=VALUE")
        column, raw_value = item.split("=", 1)
        filters[column] = parse_value(raw_value)
    return filters


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute a t-based 95% CI for paired model differences across split seeds."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--model-col", default="feature")
    parser.add_argument("--seed-col", default="split_seed")
    parser.add_argument("--metric-col", default="test_auroc")
    parser.add_argument(
        "--replicate-col",
        default=None,
        help="Optional sampling-replicate column. Replicates are averaged within seed and model.",
    )
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help="Filter rows before comparison; may be supplied multiple times.",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output-prefix", type=Path, default=None)
    args = parser.parse_args()

    if not 0 < args.confidence < 1:
        raise ValueError("--confidence must be between 0 and 1")

    df = read_table(args.input)
    required = {args.model_col, args.seed_col, args.metric_col}
    if args.replicate_col:
        required.add(args.replicate_col)
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    filters = parse_where(args.where)
    for column, value in filters.items():
        if column not in df.columns:
            raise KeyError(f"Filter column not found: {column}")
        df = df.loc[df[column] == value].copy()

    df = df.loc[df[args.model_col].isin([args.model_a, args.model_b])].copy()
    if df.empty:
        raise ValueError("No rows remain after filtering.")

    per_seed = (
        df.groupby([args.seed_col, args.model_col], as_index=False)[args.metric_col]
        .mean()
    )

    paired = per_seed.pivot(
        index=args.seed_col,
        columns=args.model_col,
        values=args.metric_col,
    )

    missing_models = [m for m in [args.model_a, args.model_b] if m not in paired.columns]
    if missing_models:
        raise ValueError(f"Models absent after filtering: {missing_models}")

    paired = paired[[args.model_a, args.model_b]].dropna().sort_index()
    paired["difference"] = paired[args.model_a] - paired[args.model_b]

    n = len(paired)
    if n < 2:
        raise ValueError(f"At least two complete paired seeds are required; found {n}.")

    differences = paired["difference"].to_numpy(dtype=float)
    mean_difference = float(np.mean(differences))
    sd_difference = float(np.std(differences, ddof=1))
    standard_error = sd_difference / math.sqrt(n)

    alpha = 1.0 - args.confidence
    t_critical = float(t.ppf(1.0 - alpha / 2.0, df=n - 1))
    margin = t_critical * standard_error
    ci_lower = mean_difference - margin
    ci_upper = mean_difference + margin

    summary = pd.DataFrame(
        [{
            "model_a": args.model_a,
            "model_b": args.model_b,
            "metric": args.metric_col,
            "n_complete_seeds": n,
            "mean_model_a": paired[args.model_a].mean(),
            "mean_model_b": paired[args.model_b].mean(),
            "mean_paired_difference": mean_difference,
            "sd_paired_difference": sd_difference,
            "standard_error": standard_error,
            "confidence_level": args.confidence,
            "t_critical": t_critical,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }]
    )

    print("\nPaired seed-level values")
    print(paired.to_string(float_format=lambda x: f"{x:.8f}"))
    print("\nSummary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.8f}"))

    if args.output_prefix is not None:
        prefix = args.output_prefix
        prefix.parent.mkdir(parents=True, exist_ok=True)
        paired_path = prefix.with_name(prefix.name + "_paired.csv")
        summary_path = prefix.with_name(prefix.name + "_summary.csv")
        paired.reset_index().to_csv(paired_path, index=False)
        summary.to_csv(summary_path, index=False)
        print(f"\nWrote: {paired_path}")
        print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()

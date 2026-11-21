#!/usr/bin/env python3
"""E5: PMN rare-label sensitivity for individual representations only.

Reads saved predictions from the existing PMN benchmark; no fitting.
Compares Morgan fingerprints and standalone foundation models.
Does not evaluate probability ensembles.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EXTERNAL = "bao_B_external_test"
PRED_FILENAME = "pmn_secondary_pathway_full_ensemble_predictions.npz"
RUNS_FILENAME = "pmn_secondary_pathway_full_ensemble_repeated_runs.tsv"


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, default=Path("results/pmn_secondary_pathway/full") / PRED_FILENAME)
    p.add_argument("--runs", type=Path, default=Path("results/pmn_secondary_pathway/full") / RUNS_FILENAME)
    p.add_argument("--output-dir", type=Path, default=Path("results/e5_pmn_rare_labels"))
    p.add_argument("--min-positives", type=int, default=5,
                   help="Retain pathways with at least this many external-test positives (default: 5).")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def weighted_bootstrap_auc(y, scores, bootstrap_counts):
    """Bootstrap AUROC for one label, with ties and repeated rows handled exactly.

    bootstrap_counts[b, i] gives the number of copies of compound i in
    resample b; this avoids explicitly constructing thousands of index arrays.
    Returns NaN for resamples containing only one class, matching the original
    evaluation's rule to skip non-evaluable labels.
    """
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    y_sorted = y[order].astype(np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    counts = bootstrap_counts[:, order]
    pos = np.add.reduceat(counts * y_sorted, starts, axis=1)
    neg = np.add.reduceat(counts * (1.0 - y_sorted), starts, axis=1)
    concordant = np.sum(pos * (np.cumsum(neg, axis=1) - 0.5 * neg), axis=1)
    denom = np.sum(pos, axis=1) * np.sum(neg, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(concordant, denom, out=np.full(len(denom), np.nan), where=denom > 0)


def macro_values(label_aucs, mask):
    if not np.any(mask):
        raise ValueError("No evaluable pathways in one of the analysis groups")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(label_aucs[:, mask], axis=1)


def main():
    args = args_parser()
    if args.min_positives < 2 or args.n_bootstrap < 100:
        raise ValueError("Use --min-positives >= 2 and --n-bootstrap >= 100")
    runs = pd.read_csv(args.runs, sep="\t", low_memory=False)
    runs = runs.loc[runs["eval_set"].eq(EXTERNAL)].copy()
    required = {"split_mode", "split_seed", "feature", "macro_auroc"}
    if not required.issubset(runs.columns) or runs.empty:
        raise ValueError("PMN repeated-runs file is missing required external-test results")
    if runs.duplicated(["split_mode", "split_seed", "feature"]).any():
        raise ValueError("Duplicate PMN model/seed/external-test rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts_records, result_records = [], []
    with np.load(args.predictions, allow_pickle=True) as saved:
        # The original PMN pipeline records standalone representations here.
        # This list contains Morgan r2/r3 and individual FMs, excluding
        # fingerprint+FM and all-model probability ensembles.
        individual_features = [str(x) for x in saved["feature_order"].tolist()]
        if len(individual_features) != len(set(individual_features)):
            raise ValueError("Duplicate feature names in saved feature_order")
        if not {"morgan_r2_2048", "morgan_r3_2048"}.issubset(individual_features):
            raise ValueError("Expected Morgan fingerprint baselines in feature_order")
        if any(name.startswith("ensemble_") for name in individual_features):
            raise ValueError("Unexpected ensemble in saved standalone feature_order")
        runs = runs.loc[runs["feature"].isin(individual_features)].copy()
        if runs.empty:
            raise ValueError("No standalone model results in repeated-runs file")
        if set(runs["feature"]) != set(individual_features):
            raise ValueError("Standalone features missing from repeated-runs file")
        print(f"Standalone representations: {len(individual_features)} "
              "(Morgan fingerprints and individual FMs; no ensembles)")

        labels = [str(x) for x in saved["label_columns"].tolist()]
        if len(labels) != 18:
            raise ValueError(f"Expected 18 PMN pathways; got {len(labels)}")

        # All random/scaffold repetitions must share the identical fixed Bao B test set.
        ref_ids, ref_y = None, None
        for (mode, seed), _ in runs.groupby(["split_mode", "split_seed"], sort=True):
            prefix = f"{mode}_seed{int(seed)}_{EXTERNAL}"
            ids = np.asarray(saved[f"{prefix}_ids"]).astype(str)
            y = np.asarray(saved[f"{prefix}_y_true"], dtype=int)
            if ref_ids is None:
                ref_ids, ref_y = ids, y
            elif not np.array_equal(ids, ref_ids) or not np.array_equal(y, ref_y):
                raise ValueError("The fixed external test set differs between split repetitions")
        n_test, n_labels = ref_y.shape
        if n_labels != len(labels) or not np.isin(ref_y, [0, 1]).all():
            raise ValueError("Invalid test-label matrix")
        positive = ref_y.sum(axis=0)
        negative = n_test - positive
        full = (positive > 0) & (negative > 0)
        retained = (positive >= args.min_positives) & (negative > 0)
        if not retained.any():
            raise ValueError("No pathway meets the minimum positive count")

        for j, name in enumerate(labels):
            counts_records.append({
                "pathway": name, "n_test": n_test,
                "n_positive": int(positive[j]), "n_negative": int(negative[j]),
                "in_original_macro_auroc": bool(full[j]),
                f"in_min{args.min_positives}_macro_auroc": bool(retained[j]),
            })

        print(f"Fixed external test: {n_test} compounds, {n_labels} pathways")
        print(f"Evaluable original pathways: {int(full.sum())}; "
              f"retained with >= {args.min_positives} positives: {int(retained.sum())}")
        # All models / seeds / split modes use the SAME bootstrap resamples.
        # Resampling compounds (not labels or training seeds) captures Bao B sampling uncertainty.
        rng = np.random.default_rng(args.seed)
        weights = rng.multinomial(n_test, np.full(n_test, 1.0 / n_test), size=args.n_bootstrap)

        for (mode, feature), group in runs.groupby(["split_mode", "feature"], sort=True):
            per_seed_point = []
            per_seed_boot = []
            for row in group.itertuples(index=False):
                seed = int(row.split_seed)
                key = f"{mode}_seed{seed}_{EXTERNAL}_{feature}_probability"
                if key not in saved:
                    raise KeyError(f"Missing saved PMN prediction: {key}")
                p = np.asarray(saved[key], dtype=float)
                if p.shape != ref_y.shape or not np.isfinite(p).all():
                    raise ValueError(f"Invalid prediction shape/values for {key}")
                point_aucs = np.full(n_labels, np.nan)
                boot_aucs = np.full((args.n_bootstrap, n_labels), np.nan)
                for j in np.flatnonzero(full):
                    point_aucs[j] = roc_auc_score(ref_y[:, j], p[:, j])
                    boot_aucs[:, j] = weighted_bootstrap_auc(ref_y[:, j], p[:, j], weights)
                point_macro = float(np.mean(point_aucs[full]))
                if not np.isclose(point_macro, float(row.macro_auroc), atol=1e-3, rtol=0):
                    raise ValueError(f"Original macro-AUROC mismatch for {key}: "
                                     f"recomputed={point_macro:.6f}, saved={row.macro_auroc:.6f}")
                per_seed_point.append(point_aucs)
                per_seed_boot.append(boot_aucs)

            seed_points = np.stack(per_seed_point)    # [seed, pathway]
            seed_boot = np.stack(per_seed_boot)       # [seed, bootstrap, pathway]
            for scenario, mask in [("original", full), (f"min{args.min_positives}", retained)]:
                # Preserve original statistic: AUROC averaged over pathways, then over seeds.
                observed = float(np.mean(seed_points[:, mask]))
                boot_seed_means = np.stack([macro_values(b, mask) for b in seed_boot], axis=0)
                boot_mean = np.nanmean(boot_seed_means, axis=0)
                boot_mean = boot_mean[np.isfinite(boot_mean)]
                if len(boot_mean) < args.n_bootstrap * 0.95:
                    raise ValueError("Too many degenerate bootstrap resamples")
                lo, hi = np.percentile(boot_mean, [2.5, 97.5])
                result_records.append({
                    "split_mode": mode, "feature": feature, "scenario": scenario,
                    "n_pathways": int(mask.sum()), "n_training_seeds": len(per_seed_point),
                    "mean_macro_auroc": observed,
                    "bootstrap_ci95_low": float(lo), "bootstrap_ci95_high": float(hi),
                })

    counts_out = args.output_dir / "pmn_external_pathway_counts.tsv"
    scores_out = args.output_dir / "pmn_rare_label_sensitivity.tsv"
    pd.DataFrame(counts_records).to_csv(counts_out, sep="\t", index=False)
    result = pd.DataFrame(result_records)
    result.sort_values(["split_mode", "feature", "scenario"], inplace=True)
    result.to_csv(scores_out, sep="\t", index=False, float_format="%.6f")
    print(f"Wrote: {counts_out}")
    print(f"Wrote: {scores_out}")
    print("CI: 2.5th–97.5th percentiles from compound-level test-set bootstrap; "
          "single-class pathways in any resample are skipped, as in the original evaluation.")


if __name__ == "__main__":
    main()

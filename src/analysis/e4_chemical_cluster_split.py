#!/usr/bin/env python3
"""E4 — chemically separated KEGG evaluation (Reviewer 1, Comment 3).

Uses the *existing* KEGG data and full-benchmark fitting/metric functions.
- Build exact connected components from pairs with Morgan r2 Tanimoto >= cutoff.
- Assign entire components to 80/10/10 train/validation/test sets, preserving
  pathway-label support and approximately matching prevalences.
- Independently audit exact maximum test-to-training Tanimoto similarity.
- Retrain individual Morgan/frozen-FM logistic-regression probes with the same
  validation-based C/threshold selection used by the published KEGG benchmark.

The graph construction is O(N^2). This implementation intentionally targets
KEGG (~5,288 compounds), NOT the 411k-compound NP-likeness dataset.

Usage:
  cd /private/vina/FM_bench
  conda activate plantfm_data
  python src/analysis/e4_chemical_cluster_split.py --split-only
  python src/analysis/e4_chemical_cluster_split.py --skip-missing-fm
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import GroupShuffleSplit

DEFAULT_FEATURES = [
    "morgan_r2_2048", "morgan_r3_2048",
    "chemberta_77m_mlm", "chemberta_100m_mlm", "molformer_c3_1_1b",
    "nafm_graph_mean", "molclr_gcn", "molclr_gin",
    "graphmvp_gin", "molebert_gin",
]
SPLIT_NAMES = ("train", "validation", "test")
TARGETS = (0.80, 0.10, 0.10)


def benchmark_functions(root: Path):
    module_dir = root / "src/modeling/kegg_pathway/data_efficiency"
    if not (module_dir / "run_kegg_pathway_data_efficiency.py").exists():
        raise FileNotFoundError(f"Missing original KEGG runner: {module_dir}")
    sys.path.insert(0, str(module_dir))
    from run_kegg_pathway_data_efficiency import (  # pylint: disable=import-outside-toplevel
        load_task, make_morgan, load_aligned_embedding,
        fit_select_c_and_threshold, compute_metrics,
    )
    return load_task, make_morgan, load_aligned_embedding, fit_select_c_and_threshold, compute_metrics


def make_rdkit_fps(smiles: pd.Series) -> list:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            raise ValueError(f"Invalid SMILES at task row {i}: {s}")
        fps.append(gen.GetFingerprint(mol))
        if (i + 1) % 1000 == 0:
            print(f"  fingerprints {i + 1:,}/{len(smiles):,}", flush=True)
    return fps


class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, a):
        while a != self.parent[a]:
            self.parent[a] = self.parent[self.parent[a]]
            a = int(self.parent[a])
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def exact_similarity_components(fps: list, cutoff: float):
    """Connected components of the exact Tanimoto>=cutoff graph.

    Compared with Butina clustering, this ensures NO above-cutoff molecular pair
    lies in different components (Butina does not provide that guarantee).
    """
    n = len(fps)
    uf = UnionFind(n)
    edge_count = 0
    for i in range(1, n):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
        neighbors = np.flatnonzero(sims >= cutoff - 1e-12)
        for j in neighbors:
            uf.union(i, int(j))
        edge_count += len(neighbors)
        if (i + 1) % 1000 == 0 or i + 1 == n:
            print(f"  compared {i + 1:,}/{n:,}; similarity edges {edge_count:,}", flush=True)

    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int64)
    _, ids = np.unique(roots, return_inverse=True)
    sizes = np.bincount(ids)
    print(f"  {len(sizes):,} components; largest={sizes.max():,} "
          f"({100*sizes.max()/n:.1f}%); high-similarity edges={edge_count:,}", flush=True)
    return ids, sizes


def valid_split(y: np.ndarray, split: np.ndarray,
                min_validation_positives: int = 5,
                min_test_positives: int = 5):
    """Check support for every label without using test outcomes for model tuning."""
    required = {"train": 1, "validation": min_validation_positives,
                "test": min_test_positives}
    for name in SPLIT_NAMES:
        group = y[split == name]
        if len(group) < 2:
            return False
        positive = group.sum(axis=0)
        if np.any(positive < required[name]) or np.any(positive >= len(group)):
            return False
    return True


def candidate_score(y: np.ndarray, split: np.ndarray) -> float:
    overall = y.mean(axis=0)
    size_penalty = 0.0
    label_penalty = 0.0
    for name, fraction in zip(SPLIT_NAMES, TARGETS):
        group = y[split == name]
        size_penalty += abs(len(group) / len(y) - fraction)
        label_penalty += float(np.mean(np.abs(group.mean(axis=0) - overall)))
    return size_penalty + 0.25 * label_penalty


def select_group_split(y: np.ndarray, clusters: np.ndarray, seed: int,
                       n_candidates: int, min_validation_positives: int,
                       min_test_positives: int):
    n = len(y)
    if len(np.unique(clusters)) < 10:
        raise RuntimeError("Fewer than 10 chemical components; no reliable 80/10/10 split")
    indices = np.arange(n)
    best, best_score = None, np.inf
    first = GroupShuffleSplit(n_splits=n_candidates, test_size=0.20, random_state=seed)
    for trial, (tr, hold) in enumerate(first.split(indices, groups=clusters)):
        second = GroupShuffleSplit(n_splits=1, test_size=0.50,
                                   random_state=seed * 100000 + trial)
        va_rel, te_rel = next(second.split(hold, groups=clusters[hold]))
        split = np.full(n, "", dtype="<U10")
        split[tr], split[hold[va_rel]], split[hold[te_rel]] = SPLIT_NAMES
        if not valid_split(y, split, min_validation_positives,
                           min_test_positives):
            continue
        score = candidate_score(y, split)
        if score < best_score:
            best, best_score = split.copy(), score
    if best is None:
        raise RuntimeError(
            f"Seed {seed}: no split meeting min validation/test label counts "
            f"({min_validation_positives}/{min_test_positives}) after "
            f"{n_candidates} candidates. Try --candidates 20000, or inspect "
            "per-label component support. Do not silently relax the cutoff."
        )
    return best, best_score


def max_similarities(fps: list, reference_idx: np.ndarray, query_idx: np.ndarray):
    reference = [fps[int(i)] for i in reference_idx]
    return np.array([
        max(DataStructs.BulkTanimotoSimilarity(fps[int(j)], reference))
        for j in query_idx
    ], dtype=np.float64)


def write_tsv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    df.to_csv(temp, sep="\t", index=False)
    temp.replace(path)


def summarize_metrics(path: Path):
    if not path.exists():
        return
    runs = pd.read_csv(path, sep="\t")
    test = runs[runs.eval_set.eq("test")].copy()
    if test.empty:
        return
    if "morgan_r2_2048" in test.feature.values:
        baseline = test.loc[test.feature.eq("morgan_r2_2048")].set_index("seed")["macro_auroc"]
        test["delta_auroc_vs_morgan_r2"] = test.macro_auroc - test.seed.map(baseline)
    summary = test.groupby("feature", as_index=False).agg(
        n_seeds=("seed", "nunique"),
        macro_auroc_mean=("macro_auroc", "mean"),
        macro_auroc_sd=("macro_auroc", "std"),
        macro_auprc_mean=("macro_auprc", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
        **({"delta_auroc_vs_morgan_r2_mean": ("delta_auroc_vs_morgan_r2", "mean")}
           if "delta_auroc_vs_morgan_r2" in test else {}),
    ).sort_values("macro_auroc_mean", ascending=False)
    write_tsv(summary, path.parent / "metrics_summary.tsv")
    print("\nTest AUROC summary:\n", summary.to_string(index=False), flush=True)


def compare_with_original(root: Path, current_metrics: Path, output: Path):
    """Side-by-side individual-model ranks; different test sets are not paired."""
    original_path = root / "results/kegg_pathway/full/kegg_pathway_full_ensemble_repeated_runs.tsv"
    if not original_path.exists() or not current_metrics.exists():
        print("Original KEGG full-run metrics unavailable; comparison not generated.", flush=True)
        return
    old = pd.read_csv(original_path, sep="\t")
    current = pd.read_csv(current_metrics, sep="\t")
    required = {"feature", "split_mode", "eval_set", "macro_auroc"}
    if not required.issubset(old.columns):
        raise RuntimeError(f"Unexpected KEGG reference metrics schema: {original_path}")
    old = old.loc[old.eval_set.eq("test") & ~old.feature.str.startswith("ensemble_")]
    current = current.loc[current.eval_set.eq("test")]
    old = old[["feature", "split_mode", "macro_auroc"]]
    current = current[["feature", "split_mode", "macro_auroc"]]
    both = pd.concat([old, current], ignore_index=True)
    comparison = both.groupby(["split_mode", "feature"], as_index=False).agg(
        auroc_mean=("macro_auroc", "mean"), auroc_sd=("macro_auroc", "std"),
        n_runs=("macro_auroc", "size"))
    comparison["rank_within_split"] = comparison.groupby("split_mode")["auroc_mean"].rank(
        method="min", ascending=False).astype(int)
    base = comparison.loc[comparison.feature.eq("morgan_r2_2048")].set_index("split_mode")["auroc_mean"]
    comparison["delta_auroc_vs_morgan_r2"] = comparison.auroc_mean - comparison.split_mode.map(base)
    write_tsv(comparison.sort_values(["split_mode", "rank_within_split"]),
              output / "comparison_random_scaffold_cluster.tsv")
    print("Wrote individual-model rank comparison (different test partitions):",
          output / "comparison_random_scaffold_cluster.tsv", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Connect every pair whose exact Morgan r2 Tanimoto is >= this cutoff")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--candidates", type=int, default=5000)
    parser.add_argument("--min-validation-positives", type=int, default=5,
                        help="Minimum positives per label in validation (default 5)")
    parser.add_argument("--min-test-positives", type=int, default=5,
                        help="Minimum positives per label in test (default 5)")
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--C-grid", nargs="+", type=float, default=[0.01, 0.1, 1.0])
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=Path("results/e4_chemical_cluster"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits/kegg_pathway"))
    parser.add_argument("--split-only", action="store_true", help="Write audited partitions only")
    parser.add_argument("--skip-missing-fm", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recalculate completed metrics")
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.candidates < 1 or not args.seeds:
        parser.error("threshold must be in (0,1), candidates >=1, and seeds nonempty")
    if args.min_validation_positives < 1 or args.min_test_positives < 1:
        parser.error("Minimum validation/test positives must be >=1")
    if any(c <= 0 for c in args.C_grid):
        parser.error("All C values must be positive")
    root = args.root.resolve()
    out = args.output if args.output.is_absolute() else root / args.output
    split_dir = args.split_dir if args.split_dir.is_absolute() else root / args.split_dir
    out.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    config = dict(task="kegg", method="exact_morgan_r2_tanimoto_connected_components",
                  threshold=args.threshold, C_grid=args.C_grid,
                  max_iter=args.max_iter, fractions=list(TARGETS), n_candidates=args.candidates,
                  min_validation_positives=args.min_validation_positives,
                  min_test_positives=args.min_test_positives)
    config_file = out / "config.json"
    if config_file.exists() and not args.force:
        previous = json.loads(config_file.read_text())
        if previous != config:
            raise RuntimeError("Existing E4 output uses a different configuration; choose a new --output and --split-dir")
    else:
        config_file.write_text(json.dumps(config, indent=2) + "\n")

    load_task, make_morgan, load_embedding, fit_select, compute_metrics = benchmark_functions(root)
    task, label_cols, y = load_task(
        str(root / "data/processed_bao/C_kegg/compounds_master.tsv"),
        str(root / "data/processed_bao/C_kegg/bao_compound_labels_multilabel.tsv"),
    )
    if len(task) > 20000:
        raise RuntimeError("Exact all-pairs E4 is restricted to <=20k compounds; NP-likeness needs a scalable strategy")
    fps = make_rdkit_fps(task.canonical_smiles)
    clusters, sizes = exact_similarity_components(fps, args.threshold)
    ids = task.internal_compound_id.astype(str).to_numpy()
    cluster_df = pd.DataFrame({"internal_compound_id": ids, "cluster_id": clusters})
    write_tsv(cluster_df, out / "clusters.tsv")
    print("Chemical components with any positive label (minimum across 11 labels):",
          min(np.unique(clusters[y[:, k] == 1]).size for k in range(y.shape[1])), flush=True)

    split_map = {}
    qcs = []
    for seed in args.seeds:
        print(f"\n=== Split seed {seed} ===", flush=True)
        split, score = select_group_split(
            y, clusters, seed, args.candidates,
            args.min_validation_positives, args.min_test_positives)
        assignments = pd.DataFrame({"internal_compound_id": ids, "inchikey": task.inchikey,
                                    "cluster_id": clusters, "split": split,
                                    "split_mode": "chemical_cluster", "split_seed": seed,
                                    "tanimoto_threshold": args.threshold})
        path = split_dir / f"kegg_pathway_cluster_split_seed{seed}.tsv"
        if path.exists() and not args.force:
            old = pd.read_csv(path, sep="\t")
            if not (np.array_equal(old.internal_compound_id.astype(str), ids)
                    and np.array_equal(old.split.astype(str), split)
                    and np.array_equal(old.cluster_id.to_numpy(), clusters)
                    and np.allclose(old.tanimoto_threshold, args.threshold)):
                raise RuntimeError(f"Existing split disagrees with current configuration: {path}")
        else:
            write_tsv(assignments, path)
        parts = {name: np.flatnonzero(split == name) for name in SPLIT_NAMES}
        # Independent exact QC, in addition to the mathematical component guarantee.
        test_sims = max_similarities(fps, parts["train"], parts["test"])
        val_sims = max_similarities(fps, parts["train"], parts["validation"])
        assert max(test_sims.max(), val_sims.max()) < args.threshold - 1e-12, (
            "Chemical similarity leakage above the cluster cutoff")
        train_cluster_set = set(clusters[parts["train"]])
        assert not (train_cluster_set & set(clusters[parts["test"]]))
        assert not (train_cluster_set & set(clusters[parts["validation"]]))
        assert not (set(clusters[parts["test"]]) & set(clusters[parts["validation"]]))
        rec = dict(task="kegg", seed=seed, threshold=args.threshold,
                   n_clusters=len(sizes), largest_cluster=int(sizes.max()),
                   split_score=score, n_train=len(parts["train"]),
                   n_validation=len(parts["validation"]), n_test=len(parts["test"]),
                   test_to_train_max_sim=float(test_sims.max()),
                   test_to_train_median_max_sim=float(np.median(test_sims)),
                   test_to_train_pct_ge_0p7=float(100 * np.mean(test_sims >= 0.7)),
                   validation_to_train_max_sim=float(val_sims.max()),
                   min_train_label_positives=int(y[parts["train"]].sum(axis=0).min()),
                   min_val_label_positives=int(y[parts["validation"]].sum(axis=0).min()),
                   min_test_label_positives=int(y[parts["test"]].sum(axis=0).min()))
        for k, col in enumerate(label_cols):
            rec[f"train_pos_{col}"] = int(y[parts["train"], k].sum())
            rec[f"val_pos_{col}"] = int(y[parts["validation"], k].sum())
            rec[f"test_pos_{col}"] = int(y[parts["test"], k].sum())
        qcs.append(rec)
        split_map[seed] = parts
        print(f"  n={len(parts['train'])}/{len(parts['validation'])}/{len(parts['test'])}; "
              f"max(test->train)={test_sims.max():.4f}, "
              f"median={np.median(test_sims):.4f}, "
              f"min test label+={rec['min_test_label_positives']}", flush=True)
    write_tsv(pd.DataFrame(qcs), out / "split_qc.tsv")
    if args.split_only:
        print("\nSplit-only complete; no models trained.", flush=True)
        return

    metrics_path = out / "metrics_by_seed.tsv"
    prior = pd.read_csv(metrics_path, sep="\t") if metrics_path.exists() else pd.DataFrame()
    rows = prior.copy() if not args.force else pd.DataFrame()
    threshold_grid = [0.2, 0.4, 0.6, 0.8]  # same as original KEGG full benchmark
    for feature in dict.fromkeys(args.features):
        remaining = [seed for seed in args.seeds if rows.empty or not (
            (rows.feature == feature) & (rows.seed == seed) & (rows.eval_set == "test")).any()]
        if not remaining:
            print(f"Skipping completed feature {feature}", flush=True)
            continue
        print(f"\n=== Evaluating {feature} ({len(remaining)} seeds) ===", flush=True)
        match = re.fullmatch(r"morgan_r(\d+)_(\d+)", feature)
        if match:
            radius, bits = map(int, match.groups())
            X = make_morgan(task, radius, bits)
            feature_type = "fingerprint"
        else:
            try:
                returned, X = load_embedding(task, feature, str(root / "data/features/bao_plant_pathway/full"))
            except (FileNotFoundError, ValueError) as exc:
                if args.skip_missing_fm and isinstance(exc, FileNotFoundError):
                    print(f"  Skipping missing FM: {exc}", flush=True)
                    continue
                raise
            if returned != feature:
                raise RuntimeError(f"Expected feature stem {feature}, got {returned}")
            feature_type = "dense_embedding"
        for seed in remaining:
            tr, va, te = (split_map[seed][p] for p in SPLIT_NAMES)
            val_mask = np.zeros(len(y), dtype=bool)
            val_mask[va] = True
            selected = fit_select(
                X=X, Y=y, train_idx=tr, validation_mask=val_mask,
                feature_type=feature_type, seed=seed, C_grid=args.C_grid,
                max_iter=args.max_iter, threshold_grid=threshold_grid,
            )
            for name, idx in (("validation", va), ("test", te)):
                if name == "validation":
                    pred = selected["validation_probability"]
                else:
                    pred = np.asarray(selected["model"].predict_proba(X[idx]), dtype=np.float64)
                metric = compute_metrics(y[idx], pred, selected["tuned_threshold"])
                row = dict(task="kegg", split_mode="chemical_cluster", seed=seed,
                           threshold=args.threshold, feature=feature, eval_set=name,
                           n_train=len(tr), n_validation=len(va), n_test=len(te),
                           n_eval=len(idx), selected_C=selected["selected_C"],
                           tuned_threshold=selected["tuned_threshold"],
                           validation_macro_auroc_for_C=selected["validation_macro_auroc_for_C"])
                row.update(metric)
                rows = pd.concat([rows, pd.DataFrame([row])], ignore_index=True)
                print(f"  seed={seed} {name}: macro_AUROC={metric['macro_auroc']:.4f}", flush=True)
            del selected
            rows = rows.drop_duplicates(["task", "split_mode", "seed", "feature", "eval_set"], keep="last")
            rows = rows.sort_values(["feature", "seed", "eval_set"]).reset_index(drop=True)
            write_tsv(rows, metrics_path)
            gc.collect()
        del X
        gc.collect()
    summarize_metrics(metrics_path)
    compare_with_original(root, metrics_path, out)
    print(f"\nE4 results: {out}", flush=True)


if __name__ == "__main__":
    main()

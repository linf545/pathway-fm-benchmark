#!/usr/bin/env python3
"""E3: exact, CPU-only Tanimoto k-nearest-neighbor baselines for the FM benchmark.

Run alongside src/analysis/e2_test_train_similarity.py in plantfm_data:
    python src/analysis/e3_tanimoto_knn.py --tasks pmn kegg --seeds 2026
    python src/analysis/e3_tanimoto_knn.py --tasks all --workers 8

Reads original Morgan r2/2048 features through the validated E2 loader, without
creating another fingerprint cache. KEGG fingerprints are created in RAM only
when the benchmark has no compatible precomputed file.

For all tasks, train only on the recorded training partition (exclude
validation). PMN tests on fixed external Bao Dataset B while the random or
scaffold partitions are defined within Bao Dataset A.

Uses RDKit BulkTanimotoSimilarity and an exact, popcount-based upper bound to
prune groups that cannot enter the top K. Scores are never approximated.
K=1, 5, 10 are *prespecified* sensitivity settings, not selected on test data.
Predictions use similarity-weighted neighbor labels. For queries with all-zero
neighbor similarities, use training-set class prevalences as a fallback.

Writes small human-readable TSV metrics by seed and summary. Per-compound
predictions are optional (--save-predictions) and not created by default.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs
from sklearn.metrics import roc_auc_score

try:
    from e2_test_train_similarity import (TASKS, task_table, assignments,
                                         load_fingerprints, _popcount_upper)
except ImportError as e:
    raise ImportError("Place e3_tanimoto_knn.py beside e2_test_train_similarity.py") from e

# Filled in the parent then inherited by Linux fork workers. No data copying
# or per-query serialization of a 400k-molecule fingerprint matrix.
_FPS = None
_GROUPS = None
_COUNTS = None
_ORDERS = None
_MAX_K = None


def best_k(scores: np.ndarray, ids: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact top K, descending similarity; ascending original row ID breaks ties.

    Uses selection rather than full sorting of large training groups. Handles
    all ties at the cutoff deterministically (important for binary FP matches).
    """
    scores = np.asarray(scores, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    if len(scores) > k:
        threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
        high = np.flatnonzero(scores > threshold)
        tie = np.flatnonzero(scores == threshold)
        needed = k - len(high)
        if needed < len(tie):
            keep = np.argpartition(ids[tie], needed - 1)[:needed]
            tie = tie[keep]
        pick = np.concatenate((high, tie))
    else:
        pick = np.arange(len(scores))
    order = np.lexsort((ids[pick], -scores[pick]))
    return scores[pick[order]], ids[pick[order]]


def _worker(chunk):
    global _FPS, _GROUPS, _COUNTS, _ORDERS, _MAX_K
    start, query_indices = chunk
    ns = len(query_indices)
    out_idx = np.full((ns, _MAX_K), -1, dtype=np.int64)
    out_sim = np.full((ns, _MAX_K), -1.0, dtype=np.float32)
    for j, qidx in enumerate(query_indices):
        fp = _FPS[int(qidx)]
        qc = fp.GetNumOnBits()
        if qc not in _ORDERS:
            bounds = _popcount_upper(qc, _COUNTS)
            order = np.argsort(-bounds, kind="stable")
            _ORDERS[qc] = (order, bounds)
        order, bounds = _ORDERS[qc]
        cur_scores = np.empty(0, dtype=np.float64)
        cur_ids = np.empty(0, dtype=np.int64)
        for g in order:
            # Any molecule with these bit counts satisfies Tanimoto <= bound.
            # Strict inequality is necessary to handle cutoff ties correctly.
            if len(cur_scores) == _MAX_K and bounds[g] < cur_scores[-1] - 1e-12:
                break
            indices, candidates = _GROUPS[int(g)]
            values = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, candidates),
                                dtype=np.float64)
            if len(cur_scores) == _MAX_K:
                eligible = values >= cur_scores[-1] - 1e-12
                if not np.any(eligible):
                    continue
                values, indices = values[eligible], indices[eligible]
            # Only the best K in each group can possibly enter the global top K.
            group_scores, group_ids = best_k(values, indices, _MAX_K)
            cur_scores, cur_ids = best_k(
                np.concatenate((cur_scores, group_scores)),
                np.concatenate((cur_ids, group_ids)), _MAX_K)
        if len(cur_scores) != _MAX_K:
            raise RuntimeError("Top-K search exhausted all candidates prematurely")
        out_idx[j] = cur_ids
        out_sim[j] = cur_scores.astype(np.float32)
    return start, out_idx, out_sim


class ExactTopK:
    def __init__(self, fps: list, workers: int, chunk_size: int, progress: int):
        if os.name != "posix" or "fork" not in mp.get_all_start_methods():
            raise RuntimeError("This script's multiprocessing requires Linux/fork")
        self.fps = fps
        self.workers = workers
        self.chunk_size = chunk_size
        self.progress = progress

    def search(self, train: np.ndarray, test: np.ndarray, k: int):
        global _FPS, _GROUPS, _COUNTS, _ORDERS, _MAX_K
        if len(train) < k or not len(test):
            raise ValueError(f"Need at least {k} training samples and a nonempty test set")
        if len(np.unique(train)) != len(train) or len(np.unique(test)) != len(test):
            raise ValueError("Repeated indices in train/test partition")
        if np.intersect1d(train, test).size:
            raise ValueError("Training and test sets overlap")
        counts = np.fromiter((self.fps[int(i)].GetNumOnBits() for i in train),
                             dtype=np.int32, count=len(train))
        unique_counts = np.unique(counts)
        groups = []
        for c in unique_counts:
            idx = train[counts == c]
            groups.append((idx, [self.fps[int(i)] for i in idx]))
        _FPS = self.fps
        _GROUPS = groups
        _COUNTS = unique_counts
        _ORDERS = {}
        _MAX_K = k
        out_idx = np.empty((len(test), k), dtype=np.int64)
        out_sim = np.empty((len(test), k), dtype=np.float32)
        chunks = ((start, test[start:start + self.chunk_size])
                  for start in range(0, len(test), self.chunk_size))
        done, next_report = 0, self.progress
        started = time.monotonic()
        try:
            if self.workers == 1:
                iterator = map(_worker, chunks)
                for start, ids, scores in iterator:
                    out_idx[start:start+len(ids)] = ids
                    out_sim[start:start+len(scores)] = scores
                    done += len(ids)
                    if done >= next_report or done == len(test):
                        print(f"    {done:,}/{len(test):,} queries "
                              f"({(time.monotonic()-started)/60:.1f} min)", flush=True)
                        next_report = done + self.progress
            else:
                with mp.get_context("fork").Pool(processes=self.workers) as pool:
                    for start, ids, scores in pool.imap(_worker, chunks, chunksize=1):
                        out_idx[start:start+len(ids)] = ids
                        out_sim[start:start+len(scores)] = scores
                        done += len(ids)
                        if done >= next_report or done == len(test):
                            print(f"    {done:,}/{len(test):,} queries "
                                  f"({(time.monotonic()-started)/60:.1f} min)", flush=True)
                            next_report = done + self.progress
        finally:
            _FPS = None
            _GROUPS = None
            _COUNTS = None
            _ORDERS = None
            _MAX_K = None
        return out_idx, out_sim


def labels_for_task(root: Path, task: str, table: pd.DataFrame, key: str):
    """Return one-hot/binary labels in E2 task-row order and stable class names."""
    if task == "np_likeness":
        p = root / "data/tasks/np_likeness/np_likeness_task.tsv"
        label_table = pd.read_csv(p, sep="\t", usecols=[key, "is_np"])
        target_cols = ["is_np"]
    elif task == "np_classification":
        label_table = table
        target_cols = sorted((c for c in table if re.fullmatch(r"label_\d+", c)),
                             key=lambda c: int(c.split("_")[1]))
        if len(target_cols) != 7:
            raise ValueError(f"NP classification expects 7 one-hot labels; found {target_cols}")
    elif task == "pmn":
        p = root / "data/tasks/bao_plant_pathway/bao_plant_pathway_task.tsv"
        label_table = pd.read_csv(p, sep="\t")
        target_cols = sorted((c for c in label_table if re.match(r"^label_\d+_", c)),
                             key=lambda c: int(c.split("_")[1]))
        if len(target_cols) != 18:
            raise ValueError(f"PMN expects 18 labels; found {len(target_cols)}")
    elif task == "kegg":
        p = root / "data/processed_bao/C_kegg/bao_compound_labels_multilabel.tsv"
        if not p.exists():
            raise FileNotFoundError(f"KEGG original label file missing: {p}")
        label_table = pd.read_csv(p, sep="\t")
        target_cols = sorted((c for c in label_table if re.match(r"^label_\d+_", c)),
                             key=lambda c: int(c.split("_")[1]))
        if len(target_cols) != 11:
            raise ValueError(f"KEGG expects 11 pathway labels; found {len(target_cols)}")
    else:
        raise ValueError(task)
    if label_table[key].duplicated().any() or label_table[key].isna().any():
        raise ValueError(f"{task}: duplicate/missing label IDs")
    values = label_table.set_index(key).reindex(table[key])[target_cols]
    if values.isna().any().any():
        bad = table[key].loc[values.isna().any(axis=1).to_numpy()].iloc[:5].tolist()
        raise ValueError(f"{task}: missing target labels for compound IDs: {bad}")
    y = values.to_numpy(dtype=np.float32)
    if not np.all((y == 0) | (y == 1)):
        raise ValueError(f"{task}: labels are not binary 0/1")
    if task == "np_classification" and not np.all(y.sum(axis=1) == 1):
        raise ValueError("NP classification must be single-label; sum(row) must equal 1")
    if task in ("pmn", "kegg") and np.any(y.sum(axis=1) < 1):
        raise ValueError(f"{task}: some molecules have no pathway labels")
    return y, target_cols


def partitions(root, task, table, key, mode, seed):
    assign = assignments(root, task, table, key, mode, seed)
    if task == "pmn":
        is_a = table[key].str.startswith("BAO_A_").to_numpy()
        is_b = table[key].str.startswith("BAO_B_").to_numpy()
        train = np.flatnonzero(is_a & (assign == "train"))
        test = np.flatnonzero(is_b)
    else:
        train = np.flatnonzero(assign == "train")
        test = np.flatnonzero(assign == "test")
    if not len(train) or not len(test):
        raise ValueError(f"{task}/{mode}/{seed}: empty train/test. Check split labels.")
    return train, test


def vote_probabilities(neighbors, sims, y, train, k):
    weights = sims[:, :k].astype(np.float64)
    values = y[neighbors[:, :k]].astype(np.float64, copy=False)
    totals = weights.sum(axis=1)
    probs = np.einsum("nk,nkc->nc", weights, values, optimize=True)
    valid = totals > 0
    probs[valid] /= totals[valid, None]
    if (~valid).any():
        probs[~valid] = y[train].mean(axis=0)
    return probs, int((~valid).sum())


def macro_auroc(y: np.ndarray, prob: np.ndarray):
    aucs, skipped = [], []
    for col in range(y.shape[1]):
        if len(np.unique(y[:, col])) < 2:
            skipped.append(col)
            continue
        aucs.append(float(roc_auc_score(y[:, col], prob[:, col])))
    return (float(np.mean(aucs)) if aucs else float("nan"),
            len(aucs), skipped)


def summarize(output: Path):
    p = output / "metrics_by_seed.tsv"
    if not p.exists():
        return
    df = pd.read_csv(p, sep="\t")
    grouped = df.groupby(["task", "split_mode", "reference", "k"], as_index=False).agg(
        n_seeds=("seed", "nunique"), n_test_mean=("n_test", "mean"),
        auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
        valid_labels_mean=("n_valid_labels", "mean"),
        pct_zero_sim_mean=("pct_zero_sim", "mean"))
    grouped.to_csv(output / "metrics_summary.tsv", sep="\t", index=False)
    print(f"\nSummary: {output / 'metrics_summary.tsv'}", flush=True)
    print(grouped[["task", "split_mode", "k", "n_seeds", "auroc_mean", "auroc_sd"]].to_string(index=False), flush=True)


def atomic_tsv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    df.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--output", type=Path, default=Path("results/e3_tanimoto_knn"))
    ap.add_argument("--tasks", nargs="+", choices=TASKS + ("all",), default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(2026, 2031)))
    ap.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--chunk-size", type=int, default=24)
    ap.add_argument("--progress", type=int, default=2000)
    ap.add_argument("--check-data", action="store_true", help="Verify label/split mapping without computing similarities")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--save-predictions", action="store_true", help="Optional compressed test predictions (may be large)")
    ap.add_argument("--force", action="store_true", help="Recalculate existing seed/mode metrics")
    args = ap.parse_args()
    if any(k < 1 for k in args.k) or any(n < 1 for n in (args.workers, args.chunk_size, args.progress)):
        ap.error("k, workers, chunk-size and progress must be positive")
    ks = sorted(set(args.k))
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(output)
        return
    tasks = TASKS if "all" in args.tasks else tuple(dict.fromkeys(args.tasks))
    out_path = output / "metrics_by_seed.tsv"
    old = pd.read_csv(out_path, sep="\t") if out_path.exists() else pd.DataFrame()
    print(f"Exact CPU Tanimoto k-NN; tasks={', '.join(tasks)}; k={ks}; workers={args.workers}", flush=True)
    for task in tasks:
        print(f"\n=== {task} ===", flush=True)
        table, key = task_table(root, task)
        y, classes = labels_for_task(root, task, table, key)
        print(f"  Compounds={len(table):,}, labels={len(classes)}", flush=True)
        work = []
        for mode in ("random", "scaffold"):
            for seed in args.seeds:
                train, test = partitions(root, task, table, key, mode, seed)
                print(f"  {mode} {seed}: train={len(train):,} test={len(test):,}", flush=True)
                if len(train) < max(ks):
                    raise ValueError(f"{task}/{mode}/{seed} training set smaller than max k")
                work.append((mode, seed, train, test))
        if args.check_data:
            continue
        needed = []
        for mode, seed, train, test in work:
            if not args.force and not old.empty and all(
                    ((old.task == task) & (old.split_mode == mode) &
                     (old.seed == seed) & (old.k == k) &
                     (old.reference == "train")).any() for k in ks):
                if not args.save_predictions or (output / "predictions" / f"{task}_{mode}_seed{seed}.tsv.gz").exists():
                    print(f"  Reusing {mode} seed{seed} metrics", flush=True)
                    continue
            needed.append((mode, seed, train, test))
        if not needed:
            continue
        fps = load_fingerprints(root, task, table, key)
        engine = ExactTopK(fps, args.workers, args.chunk_size, args.progress)
        for mode, seed, train, test in needed:
            print(f"  Exact top-{max(ks)}: {mode} seed{seed}", flush=True)
            neighbors, sims = engine.search(train, test, max(ks))
            yy = y[test]
            rows = []
            prediction_df = pd.DataFrame({"test_id": table[key].iloc[test].to_numpy()}) if args.save_predictions else None
            for k in ks:
                pred, n_zero = vote_probabilities(neighbors, sims, y, train, k)
                auc, valid, skipped = macro_auroc(yy, pred)
                row = dict(task=task, split_mode=mode, reference="train", seed=seed,
                           k=k, weighting="tanimoto", fp="morgan_r2_2048",
                           n_train=len(train), n_test=len(test),
                           n_labels=y.shape[1], n_valid_labels=valid,
                           skipped_labels=",".join(classes[i] for i in skipped),
                           auroc=auc, pct_zero_sim=100 * n_zero / len(test))
                rows.append(row)
                print(f"    k={k}: AUROC={auc:.4f} ({valid}/{len(classes)} valid labels), "
                      f"zero-sim={n_zero}", flush=True)
                if prediction_df is not None:
                    for c, class_name in enumerate(classes):
                        prediction_df[f"k{k}_{class_name}"] = pred[:, c].astype(np.float32)
            if prediction_df is not None:
                pred_file = output / "predictions" / f"{task}_{mode}_seed{seed}.tsv.gz"
                pred_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = pred_file.with_name(pred_file.name + ".partial")
                prediction_df.to_csv(tmp, sep="\t", index=False, compression="gzip")
                tmp.replace(pred_file)
            new = pd.DataFrame(rows)
            if old.empty:
                old = new
            else:
                old = pd.concat([old, new], ignore_index=True)
                old = old.drop_duplicates(["task", "split_mode", "reference", "seed", "k"], keep="last")
            old = old.sort_values(["task", "split_mode", "seed", "k"]).reset_index(drop=True)
            atomic_tsv(old, out_path)
        del engine, fps
    if args.check_data:
        print("\nAll requested labels and splits verified; no fingerprints computed.", flush=True)
        return
    summarize(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E1: make property-matched COCONUT/ZINC benchmark using existing NP-likeness task.

Run from the project root. The original task and its split files remain unchanged.
The matched task keeps original compound_id and InChIKey so frozen FM embeddings
can be reused by the existing modeling script without new inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

# Reuse the EXACT split implementations used for the published benchmark.
from make_np_likeness_task import (
    make_random_split,
    make_scaffold_split,
    split_qc_lines,
)

PROPERTIES = ("molecular_weight", "logp", "ring_count")
SEEDS = (2026, 2027, 2028, 2029, 2030)
FEATURE_STEM = "mw_logp_rings"


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-file", default="data/tasks/np_likeness/np_likeness_task.tsv")
    p.add_argument("--original-feature-dir", default="data/features/np_likeness")
    p.add_argument("--task-dir", default="data/tasks/np_likeness_matched")
    p.add_argument("--split-dir", default="data/splits/np_likeness_matched")
    p.add_argument("--feature-dir", default="data/features/np_likeness_matched")
    p.add_argument("--result-dir", default="results/np_likeness/matched")
    p.add_argument("--figure-dir", default="results/figures/supplementary")
    p.add_argument("--mw-bin", type=float, default=25.0, help="Molecular-weight bin width in Da")
    p.add_argument("--logp-bin", type=float, default=0.5, help="LogP bin width")
    p.add_argument("--seed", type=int, default=2026, help="Within-bin sampling seed")
    p.add_argument("--split-seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--scaffold-candidates", type=int, default=10)
    p.add_argument("--no-link-fms", action="store_true", help="Do not symlink original FM embeddings")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def descriptors_one(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES in existing task: {smiles}")
    return (Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcNumRings(mol))


def calculate_descriptors(df):
    values = np.empty((len(df), 3), dtype=np.float32)
    for i, smi in enumerate(df["canonical_smiles"]):
        values[i] = descriptors_one(smi)
        if (i + 1) % 25000 == 0:
            print(f"Calculated RDKit descriptors: {i+1:,}/{len(df):,}", flush=True)
    for j, prop in enumerate(PROPERTIES):
        df[prop] = values[:, j] if j < 2 else values[:, j].astype(np.int32)
    return df


def matched_by_coarse_bins(df, mw_width, logp_width, seed):
    """Coarsened exact matching, with exact ring counts and no replacement."""
    work = df.copy()
    work["mw_bin"] = np.floor(work["molecular_weight"].to_numpy() / mw_width).astype(np.int32)
    work["logp_bin"] = np.floor(work["logp"].to_numpy() / logp_width).astype(np.int32)
    work["ring_bin"] = work["ring_count"].astype(np.int32)
    rng = np.random.default_rng(seed)
    keep = []
    n_matched_bins = 0
    for _, g in work.groupby(["mw_bin", "logp_bin", "ring_bin"], sort=True):
        pos = g.index[g["is_np"].eq(1)].to_numpy()
        neg = g.index[g["is_np"].eq(0)].to_numpy()
        n = min(len(pos), len(neg))
        if n == 0:
            continue
        n_matched_bins += 1
        keep.extend(rng.choice(pos, size=n, replace=False).tolist())
        keep.extend(rng.choice(neg, size=n, replace=False).tolist())
    if not keep:
        raise RuntimeError("No matches found; increase --mw-bin or --logp-bin.")
    # Preserve order and original compound IDs to simplify embedding alignment.
    matched = df.loc[sorted(keep)].copy().reset_index(drop=True)
    assert matched["is_np"].value_counts().nunique() == 1
    return matched, n_matched_bins


def smd(x, y):
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    denom = np.sqrt((vx + vy) / 2)
    delta = float(np.mean(x) - np.mean(y))
    return abs(delta) / denom if denom > 0 else (0.0 if delta == 0 else float("inf"))


def balance_summary(original, matched):
    rows = []
    for group_name, df in [("original", original), ("matched", matched)]:
        a = df.loc[df.is_np.eq(1)]
        b = df.loc[df.is_np.eq(0)]
        for prop in PROPERTIES:
            x, y = a[prop].to_numpy(dtype=float), b[prop].to_numpy(dtype=float)
            rows.append({
                "dataset": group_name, "property": prop,
                "n_np": len(x), "n_zinc": len(y),
                "mean_np": float(np.mean(x)), "mean_zinc": float(np.mean(y)),
                "sd_np": float(np.std(x, ddof=1)), "sd_zinc": float(np.std(y, ddof=1)),
                "standardized_mean_difference_abs": smd(x, y),
            })
    return pd.DataFrame(rows)


def plot_distributions(original, matched, output):
    fig, axs = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)

    for i, (name, df) in enumerate([
        ("Original balanced dataset", original),
        ("Property-matched subset", matched),
    ]):
        for j, prop in enumerate(PROPERTIES):
            ax = axs[i, j]
            col = np.asarray(original[prop], dtype=float)

            if prop == "ring_count":
                edges = np.arange(0, max(2, int(np.max(col)) + 2)) - 0.5
            else:
                lo, hi = float(np.min(col)), float(np.max(col))
                edges = np.linspace(lo, hi if hi > lo else lo + 1, 60)

            for label, class_name, color, linestyle in [
                (1, "COCONUT (NP)", "tab:blue", "-"),
                (0, "ZINC (synthetic)", "tab:orange", "--"),
            ]:
                x = df.loc[df["is_np"].eq(label), prop].to_numpy(dtype=float)

                ax.hist(
                    x,
                    bins=edges,
                    density=True,
                    histtype="step",
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    label=class_name,
                )

            ax.set_xlabel(prop.replace("_", " "))
            ax.set_ylabel("Density")

            if i == 0:
                ax.set_title(prop.replace("_", " ").title())
            if j == 0:
                ax.text(
                    0.01, 0.97, name,
                    transform=ax.transAxes,
                    va="top", fontsize=9,
                )
            if i == 0 and j == 2:
                ax.legend(loc="upper right", fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250)
    plt.close(fig)


def write_descriptor_feature(df, feature_dir, overwrite):
    feature_dir.mkdir(parents=True, exist_ok=True)
    npy = feature_dir / f"{FEATURE_STEM}.embeddings.npy"
    meta = feature_dir / f"{FEATURE_STEM}.metadata.tsv"
    if not overwrite and (npy.exists() or meta.exists()):
        raise FileExistsError(f"Descriptor files already exist: {npy}, {meta}. Use --overwrite.")
    matrix = df[list(PROPERTIES)].to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite RDKit descriptor in task")
    np.save(npy, matrix)
    df[["compound_id", "inchikey"]].to_csv(meta, sep="\t", index=False)
    print(f"Wrote descriptor feature: {npy} ({matrix.shape})")


def link_existing_fms(original_feature_dir, matched_feature_dir):
    links = 0
    for source in sorted(original_feature_dir.glob("*.embeddings.npy")):
        stem = source.name.removesuffix(".embeddings.npy")
        if stem == FEATURE_STEM:
            continue
        metadata = original_feature_dir / f"{stem}.metadata.tsv"
        if not metadata.is_file():
            print(f"Skipping {stem}: missing metadata")
            continue
        for item in (source, metadata):
            target = matched_feature_dir / item.name
            if target.is_symlink() and target.resolve() == item.resolve():
                continue
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Cannot create FM symlink, target exists: {target}")
            target.symlink_to(item.resolve())
        links += 1
    print(f"Linked {links} pretrained FM feature sets; no FM inference needed.")


def main():
    args = args_parser()
    if args.mw_bin <= 0 or args.logp_bin <= 0 or args.scaffold_candidates <= 0:
        raise ValueError("Bin widths and scaffold candidates must be positive")
    task_dir = Path(args.task_dir)
    split_dir = Path(args.split_dir)
    feature_dir = Path(args.feature_dir)
    original_feature_dir = Path(args.original_feature_dir)
    result_dir = Path(args.result_dir)
    fig_path = Path(args.figure_dir) / "np_likeness_property_matching.png"
    task_out = task_dir / "np_likeness_task.tsv"
    summary_path = result_dir / "property_balance_summary.tsv"
    qc_path = result_dir / "matching_qc.json"
    split_paths = [split_dir / f"np_likeness_{mode}_split_seed{seed}.tsv"
                   for mode in ("random", "scaffold") for seed in args.split_seeds]
    protected = [task_out, summary_path, qc_path, fig_path, *split_paths]
    if not args.overwrite:
        existing = [p for p in protected if p.exists()]
        if existing:
            raise FileExistsError("Output(s) exist; use --overwrite: " + ", ".join(map(str, existing[:5])))

    original = pd.read_csv(args.task_file, sep="\t", low_memory=False)
    required = {"compound_id", "inchikey", "canonical_smiles", "is_np",
                "murcko_scaffold", "scaffold_split_key", "source"}
    missing = required - set(original.columns)
    if missing:
        raise ValueError(f"Original task missing columns: {sorted(missing)}")
    if original["inchikey"].duplicated().any() or original["compound_id"].duplicated().any():
        raise ValueError("Original task has duplicated compound IDs or InChIKeys")
    original = calculate_descriptors(original)
    matched, n_cells = matched_by_coarse_bins(original, args.mw_bin, args.logp_bin, args.seed)
    print(f"Original: {len(original):,}; matched: {len(matched):,} "
          f"({100*len(matched)/len(original):.1f}% retained); matched bins: {n_cells:,}")

    for folder in (task_dir, split_dir, feature_dir, result_dir):
        folder.mkdir(parents=True, exist_ok=True)
    matched.to_csv(task_out, sep="\t", index=False)
    print("Wrote matched task:", task_out)

    fractions = np.array([0.8, 0.1, 0.1], dtype=float)
    split_qc = []
    for seed in args.split_seeds:
        a = make_random_split(matched, seed, validation_fraction=0.1, test_fraction=0.1)
        b = make_scaffold_split(matched, seed, fractions=fractions,
                                n_candidates=args.scaffold_candidates,
                                size_weight=1.0, label_weight=1.0)
        a.to_csv(split_dir / f"np_likeness_random_split_seed{seed}.tsv", sep="\t", index=False)
        b.to_csv(split_dir / f"np_likeness_scaffold_split_seed{seed}.tsv", sep="\t", index=False)
        split_qc += split_qc_lines(a, "random", seed) + split_qc_lines(b, "scaffold", seed)
    (result_dir / "split_qc.txt").write_text("\n".join(split_qc))

    summary = balance_summary(original, matched)
    summary.to_csv(summary_path, sep="\t", index=False)
    plot_distributions(original, matched, fig_path)
    print("Wrote QC:", summary_path, "and", fig_path)
    print(summary[["dataset", "property", "standardized_mean_difference_abs"]].to_string(index=False))

    # Descriptor LR is run by the existing benchmark runner, treating these
    # three columns as a dense feature embedding; no new training code needed.
    write_descriptor_feature(matched, feature_dir, args.overwrite)
    write_descriptor_feature(original, original_feature_dir, args.overwrite)
    if not args.no_link_fms:
        link_existing_fms(original_feature_dir, feature_dir)

    qc = {
        "original_task": str(args.task_file),
        "matching_method": "coarsened_exact_matching_without_replacement",
        "mw_bin_width_Da": args.mw_bin, "logp_bin_width": args.logp_bin,
        "ring_count_matching": "exact",
        "sampling_seed": args.seed, "split_seeds": args.split_seeds,
        "n_original": len(original), "n_matched": len(matched),
        "original_by_class": {str(k): int(v) for k, v in original.is_np.value_counts().items()},
        "matched_by_class": {str(k): int(v) for k, v in matched.is_np.value_counts().items()},
        "matched_bin_count": n_cells,
        "retained_fraction": len(matched) / len(original),
        "split_fraction_targets": [0.8, 0.1, 0.1],
        "descriptor_feature_stem": FEATURE_STEM,
    }
    qc_path.write_text(json.dumps(qc, indent=2))
    print("Wrote:", qc_path)


if __name__ == "__main__":
    main()

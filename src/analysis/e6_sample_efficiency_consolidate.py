#!/usr/bin/env python3
"""Consolidate E6 sample-efficiency results with the original benchmarks.

Usage (from the repository root, after all five E6 sample-efficiency runs finish):
  python src/analysis/e6_sample_efficiency_consolidate.py
  python src/analysis/e6_sample_efficiency_consolidate.py --no-plots

Requires src/analysis/e6_sample_efficiency.py: its read-only QC verifies that
E6 used identical sampling manifests, split QC, and modeling budgets.

IMPORTANT: Original results are the ONLY source of Morgan/legacy-model rows.
E6 result directories contribute CheMeleon and Mol-JEPA rows ONLY.

Accuracy aggregation matches each benchmark's published semantics:
  KEGG, PMN, PlantCyc: average 5 sampling seeds within each of 5 split seeds,
                     THEN compute mean/SD across the 5 split seeds.
  NP classification, NP-likeness: average across 5 sampling seeds of one fixed
                                    base split; SD is across sampling seeds.
No statistical significance is claimed from descriptive rankings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS))
import e6_sample_efficiency as runner  # noqa: E402

NEW_FMS = tuple(runner.NEW_FMS)  # exact extraction names
OLD_FMS = {
    "chemberta_77m_mlm", "chemberta_100m_mlm", "molformer_c3_1_1b",
    "nafm_graph_mean", "molclr_gcn", "molclr_gin", "graphmvp_gin",
    "molebert_gin",
}
MORGANS = {"morgan_r2_2048", "morgan_r3_2048"}
DESCRIPTORS = {"rdkit_descriptors"}
ALL_ALLOWED = OLD_FMS | MORGANS | DESCRIPTORS
OUT = ROOT / "results/e6/sample_efficiency_consolidated"

# Respect actual columns in original per-run files. Do not rename on disk.
COLS = {
    "kegg": ("split_mode", "split_seed", "sampling_seed",
             "requested_n_per_label", "test", "macro_auroc", "macro_auprc", ()),
    "pmn": ("split_mode", "split_seed", "sampling_seed",
            "requested_n_per_label", "bao_B_external_test", "macro_auroc",
            "macro_auprc", ()),
    "np_classification": ("split", "base_split_seed", "sample_seed",
                          "requested_n_train", "test", "auroc_ovr_macro",
                          "auprc_ovr_macro", ()),
    "np_likeness": ("split_mode", "base_split_seed", "sample_seed",
                    "requested_n_train", "test", "auroc", "auprc", ()),
    "plantcyc": ("split_scheme", "split_seed", "sampling_seed",
                 "requested_n_train", "test", "auroc", "auprc",
                 ("negative_variant",)),
}


def check(test: bool, message: str) -> None:
    if not test:
        raise AssertionError(message)


def read(path: Path) -> pd.DataFrame:
    check(path.is_file(), f"Missing {path}")
    return pd.read_csv(path, sep="\t", low_memory=False)


def standardize(task: str, source: str, spec: dict) -> pd.DataFrame:
    old = source == "original"
    path = ROOT / (spec["old"] if old else spec["new"]) / spec["runs"]
    frame = read(path)
    mode, seed, sampling, size, eval_name, auc, prc, extras = COLS[task]
    required = {"feature", mode, seed, sampling, size, "eval_set", auc, prc, *extras}
    check(not required - set(frame.columns),
          f"{task}/{source} missing columns: {sorted(required-set(frame.columns))}")
    frame = frame.loc[frame.eval_set.eq(eval_name)].copy()
    features = set(frame.feature.astype(str))
    allowed = ALL_ALLOWED if old else set(NEW_FMS)
    check(features <= allowed,
          f"{task}/{source} unrecognized feature labels: {sorted(features-allowed)}")
    check(not set(NEW_FMS) & features if old else features == set(NEW_FMS),
          f"{task}/{source}: unexpected model roster {sorted(features)}")
    if old:
        check(MORGANS <= features, f"{task}: original Morgan r2/r3 missing")
        check(OLD_FMS & features, f"{task}: no original FMs found")

    cols = {mode:"split", seed:"split_seed", sampling:"sampling_seed",
            size:"requested_n", auc:"auroc", prc:"auprc"}
    frame = frame.rename(columns=cols)
    frame["task"] = task
    frame["condition"] = (frame[list(extras)].astype(str).agg("/".join,axis=1)
                          if extras else "all")
    frame["source"] = source
    for col in ("split_seed","sampling_seed","requested_n"):
        frame[col] = pd.to_numeric(frame[col],errors="raise").astype(int)
    for col in ("auroc","auprc"):
        frame[col] = pd.to_numeric(frame[col],errors="raise")
        check(frame[col].notna().all() and frame[col].between(0,1).all(),
              f"{task}/{source}: invalid {col}")

    key = ["task", "split", "condition", "split_seed", "sampling_seed",
           "requested_n", "feature"]
    check(not frame.duplicated(key).any(),
          f"{task}/{source}: duplicate model/sample key")
    if "n_train" in frame.columns:
        frame["actual_n_train"] = pd.to_numeric(frame["n_train"],errors="raise")
    elif "actual_n_train" in frame.columns:
        frame["actual_n_train"] = pd.to_numeric(frame["actual_n_train"],errors="raise")
    else:
        # Some evaluators do not expose n_train in the same column.
        frame["actual_n_train"] = np.nan
    if "reached_max_iter" in frame.columns:
        frame["convergence_warning"] = frame["reached_max_iter"].astype(str).str.lower().isin(["true","1"])
    elif "solver_converged" in frame.columns:
        frame["convergence_warning"] = ~frame["solver_converged"].astype(str).str.lower().isin(["true","1"])
    else:
        frame["convergence_warning"] = False
    frame["model_group"] = np.where(
        frame.feature.isin(MORGANS), "Morgan",
        np.where(frame.feature.isin(DESCRIPTORS), "descriptors",
                 np.where(frame.feature.isin(NEW_FMS), "E6 FM", "original FM")),
    )
    return frame[[*key,"source","model_group","auroc","auprc",
                  "actual_n_train","convergence_warning"]].copy()


def published_original_summary_qc(task: str, original: pd.DataFrame, spec: dict) -> float:
    """Reproduce original summary AUROCs to catch wrong grouping/metric names."""
    ref = read(ROOT / spec["old"] / spec["summary"])
    c = COLS[task]
    expected_cols = {"feature",c[0],c[3]}
    check(expected_cols <= set(ref.columns), f"{task}: original summary schema changed")
    metric = ("auroc_ovr_macro_mean" if task == "np_classification"
              else "macro_auroc_mean" if task in ("kegg","pmn") else "auroc_mean")
    check(metric in ref.columns, f"{task}: no {metric} in published original summary")
    if "eval_set" in ref.columns:
        ref = ref.loc[ref.eval_set.eq(c[4])].copy()
    ref["condition"] = (ref[list(c[7])].astype(str).agg("/".join,axis=1)
                        if c[7] else "all")
    ref = ref.rename(columns={c[0]:"split",c[3]:"requested_n"})
    check(not ref.duplicated(["feature","split","condition","requested_n"]).any(),
          f"{task}: published result has duplicate summary rows")
    keys = ["feature","split","condition","requested_n"]
    # KEGG/PMN/PlantCyc aggregate over split seeds; both NP tasks use sample seeds.
    per_unit = reduce_to_independent_units(task, original)
    calculated = (per_unit.groupby(keys,as_index=False).auroc.mean()
                  .rename(columns={"auroc":"computed"}))
    merged = ref[keys+[metric]].merge(calculated,on=keys,how="outer",
                                     indicator=True,validate="one_to_one")
    check(merged["_merge"].eq("both").all(),
          f"{task}: original summary rows do not match reconstructed raw runs")
    diff = (merged[metric]-merged["computed"]).abs()
    max_diff = float(diff.max())
    check(max_diff <= 1e-8, f"{task}: original AUROC summary mismatch max {max_diff:.9g}; "
          "check metric and aggregation rather than changing tolerance")
    return max_diff


def reduce_to_independent_units(task: str, frame: pd.DataFrame) -> pd.DataFrame:
    units = ["split_seed"] if task in ("kegg", "pmn", "plantcyc") else ["sampling_seed"]
    keys = ["task","split","condition","requested_n","feature","model_group",*units]
    return (frame.groupby(keys,as_index=False,dropna=False)
                 .agg(auroc=("auroc","mean"),auprc=("auprc","mean"),
                      n_runs=("auroc","size"),
                      n_convergence_warnings=("convergence_warning","sum")))


def ensure_matched_grid(task: str, original: pd.DataFrame, e6: pd.DataFrame) -> tuple[int,int]:
    condition_key = ["split","condition","split_seed","sampling_seed","requested_n"]
    # Every feature must have EXACT same condition/sample keys as original Morgan r2.
    reference = original.loc[original.feature.eq("morgan_r2_2048"),condition_key]
    check(not reference.duplicated().any(),f"{task}: duplicate original Morgan sample keys")
    check(len(reference)>0, f"{task}: no original Morgan samples")
    master = reference.sort_values(condition_key).reset_index(drop=True)
    for frame in (original,e6):
        for fm, group in frame.groupby("feature",sort=True):
            keys = group[condition_key].sort_values(condition_key).reset_index(drop=True)
            check(master.equals(keys),
                  f"{task}/{fm}: sampled conditions do not exactly match original Morgan")
    return len(master),int(reference["split_seed"].nunique())


def merge_new_and_original(task: str, spec: dict) -> tuple[pd.DataFrame,dict]:
    print(f"\n[{task}] checking original/E6 provenance...",flush=True)
    # Uses original launcher validation of full manifests, split QC, training
    # sample identities, tuning grids and train/test sizes, not merely filenames.
    launcher_qc = runner.do_qc(task,spec,write=False)
    orig = standardize(task,"original",spec)
    new = standardize(task,"E6",spec)
    n_grid, n_split = ensure_matched_grid(task,orig,new)
    max_diff = published_original_summary_qc(task,orig,spec)
    data = pd.concat([orig,new],ignore_index=True)
    keys = ["task","split","condition","split_seed","sampling_seed", "requested_n","feature"]
    check(not data.duplicated(keys).any(),f"{task}: duplicate feature/model after merge")
    expected_new = n_grid * len(NEW_FMS)
    check(len(new)==expected_new,
          f"{task}: E6 rows {len(new)} expected {expected_new}")
    check(set(new.feature)==set(NEW_FMS), f"{task}: wrong E6 feature names")
    record = {"task":task,"status":"PASS","source_original":spec["old"],
              "source_e6":spec["new"],"original_models":orig.feature.nunique(),
              "e6_models":new.feature.nunique(),"samples_per_feature":n_grid,
              "n_split_seeds":n_split,"max_original_summary_auroc_diff":max_diff,
              "e6_test_rows":len(new),
              "sample_and_split_qc":launcher_qc["status"]}
    print(f"PASS {task}: {len(orig.feature.unique())} existing models + 2 E6 FMs; "
          f"{n_grid} matched sampling conditions per model; "
          f"summary AUROC delta {max_diff:.2g}",flush=True)
    return data,record


def summarize(units: pd.DataFrame) -> pd.DataFrame:
    by = ["task","split","condition","requested_n","feature","model_group"]
    out = (units.groupby(by,as_index=False,dropna=False)
            .agg(n_units=("auroc","size"),
                 auroc_mean=("auroc","mean"),auroc_sd=("auroc","std"),
                 auprc_mean=("auprc","mean"),auprc_sd=("auprc","std"),
                 n_convergence_warnings=("n_convergence_warnings","sum")))
    out["auroc_se"] = out["auroc_sd"]/np.sqrt(out["n_units"])
    out["auprc_se"] = out["auprc_sd"]/np.sqrt(out["n_units"])
    idx = ["task","split","condition","requested_n"]
    out["rank_all"] = out.groupby(idx).auroc_mean.rank(method="min",ascending=False).astype(int)
    fm = out.model_group.isin(["original FM","E6 FM"])
    out["rank_fm"] = pd.Series(pd.NA,index=out.index,dtype="Int64")
    out.loc[fm,"rank_fm"] = (out.loc[fm].groupby(idx).auroc_mean
                            .rank(method="min",ascending=False).astype(int))
    return out


def comparisons(units: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    groups = ["task","split","condition","requested_n"]
    output=[]
    for key,group in summary.groupby(groups,sort=True):
        orig_fm = group.loc[group.model_group.eq("original FM")].sort_values(
            ["auroc_mean","feature"],ascending=[False,True])
        check(not orig_fm.empty,f"No original FMs in {key}")
        best_original = str(orig_fm.iloc[0].feature)
        g = units.loc[np.logical_and.reduce([units[k].eq(v) for k,v in zip(groups,key)])]
        task = str(key[0])
        pairing = ["split_seed"] if task in ("kegg","pmn","plantcyc") else ["sampling_seed"]
        for new in NEW_FMS:
            current = g[g.feature.eq(new)].copy()
            check(not current.empty, f"{key}: missing {new}")
            row = dict(zip(groups,key))
            row["feature"]=new
            row["best_original_fm"]=best_original
            score = group.loc[group.feature.eq(new)].iloc[0]
            row["rank_all"]=int(score.rank_all)
            row["rank_fm"]=int(score.rank_fm)
            row["auroc_mean"]=float(score.auroc_mean)
            row["auprc_mean"]=float(score.auprc_mean)
            for ref in ["morgan_r2_2048",best_original]:
                right = g.loc[g.feature.eq(ref),pairing+["auroc","auprc"]]
                merged = current[pairing+["auroc","auprc"]].merge(
                    right,on=pairing,how="outer",suffixes=("_e6","_ref"),
                    indicator=True,validate="one_to_one")
                check(merged["_merge"].eq("both").all(),f"Unmatched paired seeds {key}/{new}/{ref}")
                label = "morgan_r2" if ref=="morgan_r2_2048" else "best_original_fm"
                for metric in ["auroc","auprc"]:
                    delta = merged[f"{metric}_e6"] - merged[f"{metric}_ref"]
                    row[f"delta_{metric}_vs_{label}_mean"] = float(delta.mean())
                    row[f"delta_{metric}_vs_{label}_sd"] = float(delta.std(ddof=1))
                    row[f"positive_units_vs_{label}_{metric}"] = int((delta>0).sum())
            output.append(row)
    return pd.DataFrame(output)


def plot_curves(summary: pd.DataFrame) -> int:
    """One figure per task; 1x2 random/scaffold, faceted PlantCyc negatives."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("NOTE: matplotlib unavailable; tables are complete, plots skipped")
        return 0
    count=0
    chart_dir = OUT / "plots"
    chart_dir.mkdir(exist_ok=True,parents=True)
    # To keep figures readable, use Morgan r2, E6 models, and top old FM
    # at the largest training size of each condition (identified descriptively).
    for task,tdf in summary.groupby("task"):
        conds=sorted(tdf.condition.unique())
        modes=sorted(tdf.split.unique())
        for cond in conds:
            fig,axes = plt.subplots(1,len(modes),figsize=(6*len(modes),4.5),squeeze=False)
            for ax, mode in zip(axes[0],modes):
                sub=tdf[tdf.split.eq(mode)&tdf.condition.eq(cond)]
                biggest=sub.requested_n.max()
                old=sub[(sub.requested_n==biggest)&sub.model_group.eq("original FM")]
                best=str(old.sort_values("auroc_mean",ascending=False).iloc[0].feature)
                features=["morgan_r2_2048",*NEW_FMS,best]
                for fm in dict.fromkeys(features):
                    v=sub[sub.feature.eq(fm)].sort_values("requested_n")
                    if v.empty: continue
                    ax.plot(v.requested_n,v.auroc_mean,marker="o",label=fm)
                ax.set_xscale("log")
                ax.set_ylim(0,1.01)
                ax.set_xlabel("Training set size (task-specific units)")
                ax.set_ylabel("Macro AUROC")
                ax.set_title(f"{task} — {mode} — {cond}")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=7)
            fig.tight_layout()
            path=chart_dir/f"{task}_{cond}_auroc.png"
            fig.savefig(path,dpi=170)
            plt.close(fig)
            count+=1
    return count


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-plots",action="store_true",help="Skip PNG learning curves")
    args=ap.parse_args()
    check(ROOT.joinpath("src/analysis/e6_sample_efficiency.py").is_file(),
          "Run from FM_bench root with E6 sampling launcher in src/analysis")
    data=[]
    qc=[]
    for name,spec in runner.TASKS.items():
        one,record=merge_new_and_original(name,spec)
        data.append(one)
        qc.append(record)
    all_runs=pd.concat(data,ignore_index=True)
    all_units=pd.concat([reduce_to_independent_units(task, frame)
                         for task,frame in all_runs.groupby("task")],ignore_index=True)
    summary=summarize(all_units)
    compare=comparisons(all_units,summary)
    OUT.mkdir(parents=True,exist_ok=True)
    all_runs.sort_values(["task","split","condition","requested_n","feature","split_seed","sampling_seed"]).to_csv(
        OUT/"e6_efficiency_all_models_per_run.tsv",sep="\t",index=False)
    all_units.sort_values(["task","split","condition","requested_n","feature"]).to_csv(
        OUT/"e6_efficiency_independent_units.tsv",sep="\t",index=False)
    summary.sort_values(["task","split","condition","requested_n","rank_all"]).to_csv(
        OUT/"e6_efficiency_learning_curves.tsv",sep="\t",index=False)
    compare.sort_values(["task","split","condition","requested_n","feature"]).to_csv(
        OUT/"e6_efficiency_new_models_comparisons.tsv",sep="\t",index=False)
    pd.DataFrame(qc).to_csv(OUT/"e6_efficiency_qc.tsv",sep="\t",index=False)
    nplots=0 if args.no_plots else plot_curves(summary)
    print("\n=== E6 sample efficiency vs baselines (smallest and largest N per condition) ===")
    where=compare.groupby(["task","split","condition"]).requested_n.transform(
        lambda x: x.eq(x.min())|x.eq(x.max()))
    show=compare.loc[where,["task","split","condition","requested_n","feature",
                  "rank_fm","auroc_mean","delta_auroc_vs_morgan_r2_mean",
                  "delta_auroc_vs_best_original_fm_mean"]]
    print(show.round(4).to_string(index=False))
    print(f"\nPASS: all 5 tasks consolidated into {OUT} ({nplots} learning-curve PNGs)")


if __name__=="__main__":
    main()

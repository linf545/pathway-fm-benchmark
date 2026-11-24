#!/usr/bin/env python3
"""E6 sample/label-efficiency extension: CheMeleon and Mol-JEPA only.

Run from the FM_bench repository root (plantfm_data conda environment).
Uses each task's original evaluator and published result files as the protocol.
Original results are read-only; E6 outputs use results/e6/<task>/<setting>/.

The KEGG, PMN, and NP-likeness evaluators include Morgan baselines by default;
this launcher filters their feature specifications IN MEMORY before modeling.
The source evaluators themselves are not modified.

Examples:
    python src/analysis/e6_sample_efficiency.py --task all --dry-run
    python src/analysis/e6_sample_efficiency.py --task kegg
    python src/analysis/e6_sample_efficiency.py --task all
    python src/analysis/e6_sample_efficiency.py --task all --qc-only
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd

NEW_FMS = ("chemeleon", "moljepa_cls")
OLD_REFERENCE_FM = "chemberta_77m_mlm"  # dense-FM comparator for QC, NOT used as training input
REPO = Path.cwd().resolve()

TASKS = {
    "kegg": {
        "source": "src/modeling/kegg_pathway/data_efficiency/run_kegg_pathway_data_efficiency.py",
        "old": "results/kegg_pathway/data_efficiency",
        "new": "results/e6/kegg_pathway/data_efficiency",
        "stem": "kegg_pathway_data_efficiency",
        "summary": "kegg_pathway_data_efficiency_main_statistics.tsv",
        "runs": "kegg_pathway_data_efficiency_repeated_runs.tsv",
        "size": "requested_n_per_label", "mode": "split_mode",
        "keys": ("split_mode", "split_seed", "sampling_seed", "requested_n_per_label", "eval_set"),
        "manifests": ("kegg_pathway_data_efficiency_repeated_samples.tsv",
                      "kegg_pathway_data_efficiency_split_qc.tsv"),
        "override": "build_feature_specs", "fm_dir": "data/features/bao_plant_pathway/full",
    },
    "pmn": {
        "source": "src/modeling/pmn_secondary_pathway/data_efficiency/run_pmn_secondary_pathway_data_efficiency.py",
        "old": "results/pmn_secondary_pathway/data_efficiency",
        "new": "results/e6/pmn_secondary_pathway/data_efficiency",
        "stem": "pmn_secondary_pathway_data_efficiency",
        "summary": "pmn_secondary_pathway_data_efficiency_main_statistics.tsv",
        "runs": "pmn_secondary_pathway_data_efficiency_repeated_runs.tsv",
        "size": "requested_n_per_label", "mode": "split_mode",
        "keys": ("split_mode", "split_seed", "sampling_seed", "requested_n_per_label", "eval_set"),
        "manifests": ("pmn_secondary_pathway_data_efficiency_repeated_samples.tsv",
                      "pmn_secondary_pathway_data_efficiency_split_qc.tsv"),
        "override": "build_feature_specs", "fm_dir": "data/features/bao_plant_pathway/full",
    },
    "np_classification": {
        "source": "src/modeling/np_classification/data_efficiency/run_npclassification_label_efficiency.py",
        "old": "results/np_classification/label_efficiency",
        "new": "results/e6/np_classification/label_efficiency",
        "summary": "npclassification_label_efficiency_summary.tsv",
        "runs": "npclassification_label_efficiency_runs.tsv",
        "size": "requested_n_train", "mode": "split",
        "keys": ("split", "base_split_seed", "sample_seed", "requested_n_train", "eval_set"),
        "manifests": ("npclassification_label_efficiency_sample_qc.tsv",
                      "npclassification_label_efficiency_split_qc.tsv"),
        "config": "npclassification_label_efficiency_config.json",
        "fm_dir": "data/features/bao_plant_pathway/full",
    },
    "np_likeness": {
        "source": "src/modeling/np_likeness/run_np_likeness_label_efficiency.py",
        "old": "results/np_likeness/label_efficiency",
        "new": "results/e6/np_likeness/label_efficiency",
        "summary": "np_likeness_label_efficiency_main_statistics.tsv",
        "runs": "np_likeness_label_efficiency_repeated_runs.tsv",
        "size": "requested_n_train", "mode": "split_mode",
        "keys": ("split_mode", "base_split_seed", "sample_seed", "requested_n_train", "eval_set"),
        "manifests": ("np_likeness_label_efficiency_sample_qc.tsv",),
        "config": "np_likeness_label_efficiency_config.json",
        "override": "feature_specs", "fm_dir": "data/features/np_likeness",
    },
    "plantcyc": {
        "source": "src/modeling/plantcyc_reaction_neighbor/data_efficiency/run_plantcyc_reaction_neighbor_data_efficiency.py",
        "old": "results/plantcyc_reaction_neighbor/data_efficiency",
        "new": "results/e6/plantcyc_reaction_neighbor/data_efficiency",
        "summary": "plantcyc_reaction_neighbor_data_efficiency_summary.tsv",
        "runs": "plantcyc_reaction_neighbor_data_efficiency_repeated_runs.tsv",
        "size": "requested_n_train", "mode": "split_scheme",
        "keys": ("negative_variant", "split_scheme", "split_seed", "sampling_seed", "requested_n_train", "eval_set"),
        "manifests": ("plantcyc_reaction_neighbor_data_efficiency_sample_qc.tsv",
                      "plantcyc_reaction_neighbor_data_efficiency_pair_split_qc.tsv"),
        "config": "plantcyc_reaction_neighbor_data_efficiency_config.json",
        "fm_dir": "data/features/plantcyc_reaction_neighbor/full",
    },
}


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_tsv(path: Path, **kw) -> pd.DataFrame:
    ensure(path.is_file(), f"Missing original result: {path}")
    return pd.read_csv(path, sep="\t", low_memory=False, **kw)


def ints(items) -> list[int]:
    return sorted({int(x) for x in items})


def parse_grid(value: Any) -> list[float]:
    return [float(x) for x in str(value).split("|")]


def single_grid(df: pd.DataFrame, field: str) -> list[float]:
    ensure(field in df.columns, f"Original runs lack {field}")
    grids = {tuple(parse_grid(s)) for s in df[field].dropna().unique()}
    ensure(len(grids) == 1, f"Original results have differing {field}: {grids}")
    return list(grids.pop())


def reference_protocol(name: str, spec: dict) -> dict:
    old = REPO / spec["old"]
    summary = read_tsv(old / spec["summary"])
    ensure({"feature", spec["size"], spec["mode"]}.issubset(summary.columns),
           f"Unexpected original summary schema: {old / spec['summary']}")
    sizes = ints(summary[spec["size"]])
    modes = sorted(summary[spec["mode"]].astype(str).unique().tolist())
    original_features = set(summary["feature"].unique())
    ensure(not (original_features & set(NEW_FMS)),
           f"{name}: original results already contain E6 models; check paths")
    ensure(OLD_REFERENCE_FM in original_features,
           f"{name}: cannot find original dense-FM QC reference {OLD_REFERENCE_FM}")
    ensure(modes == (["random", "scaffold_pair"] if name == "plantcyc" else ["random", "scaffold"]),
           f"{name}: unexpected split modes {modes}")

    result = {"sizes": sizes, "modes": modes, "original_models": len(original_features)}
    if name in ("kegg", "pmn"):
        original = read_tsv(old / spec["runs"])
        reference = original[original["feature"].eq(OLD_REFERENCE_FM)]
        ensure(not reference.empty, f"{name}: original per-seed FM results missing")
        result.update({
            "split_seeds": ints(reference["split_seed"]),
            "sampling_seeds": ints(reference["sampling_seed"]),
            "C_grid": single_grid(reference, "C_grid"),
            "threshold_grid": single_grid(reference, "threshold_grid"),
        })
        keys = list(spec["keys"])
        r = reference[reference["eval_set"].ne("validation")]
        ensure(not r.duplicated(keys).any(), f"{name}: original repeated runs contain duplicates")
        expected = len(modes) * len(result["split_seeds"]) * len(result["sampling_seeds"]) * len(sizes)
        ensure(len(r) == expected, f"{name}: incomplete original results ({len(r)}/{expected})")
    else:
        config = json.loads((old / spec["config"]).read_text())
        core = config.get("critical_config", config.get("core_configuration", config))
        result["settings"] = core
        key = "train_sizes"
        ensure(ints(core[key]) == sizes,
               f"{name}: original config train_sizes differ from summary")
        cmodes = (core.get("split_modes") or core.get("schemes"))
        ensure(sorted(cmodes) == modes,
               f"{name}: original config split modes differ from summary")
    return result


def flag(name: str, values) -> list[str]:
    if isinstance(values, (tuple, list)):
        return [name, *[str(v) for v in values]]
    return [name, str(values)]


def cli_options(name: str, p: dict, spec: dict) -> list[str]:
    options = ["--output-dir", spec["new"]]
    if name in ("kegg", "pmn"):
        options += flag("--fm-patterns", NEW_FMS)
        options += flag("--split-mode", "both")
        options += flag("--split-seeds", p["split_seeds"])
        options += flag("--sampling-seeds", p["sampling_seeds"])
        options += flag("--n-per-label", p["sizes"])
        options += flag("--C-grid", p["C_grid"])
        options += flag("--threshold-grid", p["threshold_grid"])
        if name == "pmn":
            options += ["--n-jobs", "4"]
    elif name == "np_classification":
        c = p["settings"]
        for k, v in [
            ("--features", NEW_FMS),
            ("--split-modes", c["split_modes"]),
            ("--base-split-seed", c["base_split_seed"]),
            ("--sample-seeds", c["sample_seeds"]),
            ("--train-sizes", p["sizes"]),
            ("--C-grid", c["C_grid"]),
            ("--class-weight", c["class_weight"]),
            ("--dense-solver", c["dense_solver"]),
            ("--sparse-solver", c["sparse_solver"]),
            ("--dense-max-iter", c["dense_max_iter"]),
            ("--sparse-max-iter", c["sparse_max_iter"]),
            ("--tol", c["tol"]),
        ]:
            options += flag(k, v)
        options += ["--parallel-runs", "2"]
    elif name == "np_likeness":
        c = p["settings"]
        for k, v in [
            ("--fm-patterns", NEW_FMS),
            ("--split-mode", "both"),
            ("--base-split-seed", c["base_split_seed"]),
            ("--sample-seeds", c["sample_seeds"]),
            ("--train-sizes", p["sizes"]),
            ("--C-grid", c["C_grid"]),
            ("--threshold-grid", c["threshold_grid"]),
            ("--max-iter", c["max_iter"]),
            ("--tol", c["tol"]),
            ("--class-weight", c["class_weight"]),
        ]:
            options += flag(k, v)
    elif name == "plantcyc":
        c = p["settings"]
        for k, v in [
            ("--features", NEW_FMS),
            ("--variants", c["variants"]),
            ("--schemes", c["schemes"]),
            ("--split-seeds", c["split_seeds"]),
            ("--sampling-seeds", c["sampling_seeds"]),
            ("--train-sizes", p["sizes"]),
            ("--C-grid", c["C_grid"]),
            ("--threshold-grid", c["threshold_grid"]),
            ("--selection-metric", c["selection_metric"]),
            ("--solver", c["solver"]),
            ("--max-iter", c["max_iter"]),
            ("--tol", c["tol"]),
            ("--class-weight", c["class_weight"]),
            ("--model-seed-base", c["model_seed_base"]),
        ]:
            options += flag(k, v)
    return options


def verify_features(name: str, spec: dict) -> None:
    d = REPO / spec["fm_dir"]
    for fm in NEW_FMS:
        for ext in ("embeddings.npy", "metadata.tsv"):
            path = d / f"{fm}.{ext}"
            ensure(path.is_file(), f"{name}: missing {path}")


def qc_table_equal(original: Path, new: Path) -> int:
    a = read_tsv(original, dtype=str, keep_default_na=False)
    b = read_tsv(new, dtype=str, keep_default_na=False)
    ensure(set(a.columns) == set(b.columns), f"Manifest schema mismatch: {new}")
    cols = sorted(a.columns)
    a = a[cols].sort_values(cols).reset_index(drop=True)
    b = b[cols].sort_values(cols).reset_index(drop=True)
    ensure(a.equals(b),
           f"Sample/split manifest differs from original: {new} (original rows={len(a)}, new={len(b)})")
    return len(a)


def qc_runs(name: str, spec: dict) -> int:
    ref = read_tsv(REPO / spec["old"] / spec["runs"])
    out = read_tsv(REPO / spec["new"] / spec["runs"])
    ensure(set(out["feature"].unique()) == set(NEW_FMS),
           f"{name}: unexpected E6 features: {sorted(out['feature'].unique())}")
    keys = list(spec["keys"])
    for fm in NEW_FMS:
        x = out[out.feature.eq(fm)].copy()
        r = ref[ref.feature.eq(OLD_REFERENCE_FM)].copy()
        ensure(not x.duplicated(keys).any(), f"{name}/{fm}: duplicated run keys")
        ensure(not r.duplicated(keys).any(), f"{name}: original repeated runs have duplicate keys")
        matched = r.merge(x, on=keys, how="outer", validate="one_to_one",
                          indicator=True, suffixes=("_old", "_e6"))
        ensure(matched["_merge"].eq("both").all(),
               f"{name}/{fm}: missing/mismatched split or sampling conditions "
               f"({matched['_merge'].value_counts().to_dict()})")
        # Compare protocol fields; never compare selected_C because models select
        # different regularization strengths on the same validation data.
        for field in ("C_grid", "threshold_grid", "class_weight", "solver",
                      "tol", "max_iter", "n_train", "n_eval", "n_validation", "n_test",
                      "selection_metric", "C_selection_metric", "threshold_strategy"):
            if field not in ref.columns or field not in out.columns:
                continue
            left, right = matched[f"{field}_old"], matched[f"{field}_e6"]
            if field in ("n_train", "n_eval", "n_validation", "n_test", "tol", "max_iter"):
                equal = pd.to_numeric(left, errors="raise").equals(pd.to_numeric(right, errors="raise"))
            elif field in ("C_grid", "threshold_grid"):
                equal = all(tuple(parse_grid(a)) == tuple(parse_grid(b)) for a, b in zip(left, right))
            else:
                equal = left.astype(str).equals(right.astype(str))
            ensure(equal, f"{name}/{fm}: {field} differs from original evaluator")
    eval_name = "bao_B_external_test" if name == "pmn" else "test"
    expected_per_model = len(ref[
        ref["feature"].eq(OLD_REFERENCE_FM) & ref["eval_set"].eq(eval_name)
    ])
    ensure(expected_per_model > 0,
           f"{name}: original reference has no {eval_name} rows")
    for fm in NEW_FMS:
        actual = len(out[
            out["feature"].eq(fm) & out["eval_set"].eq(eval_name)
        ])
        ensure(actual == expected_per_model,
               f"{name}/{fm}: expected {expected_per_model} {eval_name} rows, got {actual}")
    return expected_per_model * len(NEW_FMS)


def do_qc(name: str, spec: dict, write: bool = True) -> dict:
    print(f"\nQC {name}:", flush=True)
    manifest_counts = {}
    for fname in spec["manifests"]:
        manifest_counts[fname] = qc_table_equal(REPO/spec["old"]/fname, REPO/spec["new"]/fname)
    n_test = qc_runs(name, spec)
    result = {"task": name, "status": "PASS", "n_e6_test_rows": n_test,
              "new_fms": "|".join(NEW_FMS),
              "manifest_counts": json.dumps(manifest_counts, sort_keys=True)}
    print(f"PASS {name}: {n_test} E6 test rows; identical original samples/splits", flush=True)
    if write:
        path = REPO / "results/e6/e6_sample_efficiency_qc.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            q = pd.read_csv(path, sep="\t")
            q = q[q.task.ne(name)]
            q = pd.concat([q, pd.DataFrame([result])], ignore_index=True)
        else:
            q = pd.DataFrame([result])
        q.sort_values("task").to_csv(path, sep="\t", index=False)
    return result


def run_original_evaluator(name: str, spec: dict, options: list[str], resume: bool) -> None:
    source_path = REPO / spec["source"]
    ensure(source_path.is_file(), f"Missing evaluator: {source_path}")
    if spec.get("override"):
        # In-memory wrapper means no modified copies of source scripts are needed.
        module_name = "e6_" + name + "_evaluator"
        sys.path.insert(0, str(source_path.parent))
        info = importlib.util.spec_from_file_location(module_name, source_path)
        ensure(info is not None and info.loader is not None,
               f"Cannot import original evaluator {source_path}")
        module = importlib.util.module_from_spec(info)
        sys.modules[module_name] = module
        info.loader.exec_module(module)
        function_name = spec["override"]
        original_selector = getattr(module, function_name)

        def only_new_fms(*args, **kwargs):
            available = original_selector(*args, **kwargs)
            selected = [row for row in available if row[0] in NEW_FMS]
            ensure({row[0] for row in selected} == set(NEW_FMS),
                   f"{name}: evaluator did not select both new FMs")
            print(f"E6 only: selected {[row[0] for row in selected]} "
                  "(Morgan omitted)", flush=True)
            return selected

        setattr(module, function_name, only_new_fms)
        old_argv = sys.argv
        sys.argv = [str(source_path), *options, *(["--resume"] if resume else [])]
        try:
            module.main()
        finally:
            sys.argv = old_argv
    else:
        cmd = [sys.executable, str(source_path), *options]
        if resume:
            cmd.append("--resume")
        print("EXEC:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=REPO)


def run_task(name: str, *, dry_run: bool, qc_only: bool) -> None:
    spec = TASKS[name]
    protocol = reference_protocol(name, spec)
    options = cli_options(name, protocol, spec)
    verify_features(name, spec)
    print(f"\n{name}: original={spec['old']} | E6={spec['new']}", flush=True)
    print(f"  original models={protocol['original_models']}; "
          f"training sizes={protocol['sizes']}; modes={protocol['modes']}", flush=True)
    print("  new models only:", ", ".join(NEW_FMS), flush=True)
    print("  original evaluator:", spec["source"], flush=True)
    print("  arguments:", " ".join(options), flush=True)
    if dry_run:
        print("  DRY RUN: preflight passed; no training", flush=True)
        return
    if qc_only:
        do_qc(name, spec)
        return

    output = REPO / spec["new"]
    results_path = output / spec["runs"]
    if results_path.exists():
        if spec.get("override") in ("build_feature_specs",):
            # KEGG & PMN do not provide a --resume option: never overwrite output.
            do_qc(name, spec)
            print("  Already complete; skipped rerun", flush=True)
            return
        try:
            do_qc(name, spec)
            print("  Already complete; skipped rerun", flush=True)
            return
        except (RuntimeError, FileNotFoundError):
            print("  Partial result: requesting evaluator's --resume mode", flush=True)
            resume = True
    else:
        if output.exists() and any(output.iterdir()):
            # The three native resumable evaluators may have written sample QC
            # or config before their first run row was saved.
            resume_configs = {
                "np_classification": "npclassification_label_efficiency_config.json",
                "np_likeness": "np_likeness_label_efficiency_config.json",
                "plantcyc": "plantcyc_reaction_neighbor_data_efficiency_config.json",
            }
            if name in resume_configs and (output / resume_configs[name]).is_file():
                resume = True
                print("  Existing partial output with config: requesting --resume", flush=True)
            else:
                raise RuntimeError(
                    f"{name}: nonempty E6 output directory with no runs; "
                    f"inspect before rerun: {output}")
        else:
            resume = False

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    print("  Running original evaluator with only E6 FM features...", flush=True)
    run_original_evaluator(name, spec, options, resume)
    do_qc(name, spec)

    # Record the original evaluator source hash and the exact E6-only invocation
    # because in-memory filtering is intentionally not a modification to it.
    source_bytes = (REPO / spec["source"]).read_bytes()
    manifest = {
        "source_evaluator": spec["source"],
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "original_result_directory": spec["old"],
        "e6_result_directory": spec["new"],
        "features_evaluated": list(NEW_FMS),
        "feature_selection": "in_memory_filter_of_native_specifications"
            if spec.get("override") else "native_features_argument",
        "cli_options": options,
        "reference_protocol": protocol,
        "no_morgan_reruns": True,
    }
    (output / "e6_runner_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["all", *TASKS], required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="Read original results/config and check embeddings, without modeling")
    ap.add_argument("--qc-only", action="store_true",
                    help="Compare finished E6 result files against originals")
    args = ap.parse_args()
    ensure(not (args.dry_run and args.qc_only),
           "Select only one of --dry-run or --qc-only")
    ensure((REPO / "src/modeling").is_dir(),
           "Run from FM_bench repository root")
    selected = list(TASKS) if args.task == "all" else [args.task]
    for name in selected:
        run_task(name, dry_run=args.dry_run, qc_only=args.qc_only)
    print("\nPASS: all requested tasks finished/preflighted successfully", flush=True)


if __name__ == "__main__":
    main()

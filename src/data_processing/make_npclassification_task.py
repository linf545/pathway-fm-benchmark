#!/usr/bin/env python

from pathlib import Path
import argparse
import ast
import re

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedShuffleSplit


def safe_name(x):
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x[:80] if x else "NA"


def find_smiles_col(df):
    for c in ["canonical_smiles", "smiles", "SMILES", "original_smiles"]:
        if c in df.columns:
            return c

    for c in df.columns:
        if "smiles" in c.lower():
            return c

    raise ValueError(f"Could not find SMILES column. Columns: {list(df.columns)}")


def find_label_col(df, level):
    if level == "label" and "Label" in df.columns:
        return "Label"

    if level == "superclass":
        hits = [c for c in df.columns if "superclass" in c.lower()]
    elif level == "class":
        hits = [
            c for c in df.columns
            if "class" in c.lower()
            and "superclass" not in c.lower()
            and "pathway_class" not in c.lower()
        ]
    elif level == "pathway":
        hits = [c for c in df.columns if "pathway" in c.lower()]
    elif level == "label":
        hits = [c for c in df.columns if c.lower() == "label"]
    else:
        raise ValueError(f"Unknown level: {level}")

    if hits:
        return hits[0]

    raise ValueError(f"Could not find label column for level={level}. Columns: {list(df.columns)}")


def parse_labels(x):
    if pd.isna(x):
        return []

    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "na"}:
        return []

    if s.startswith("[") and s.endswith("]"):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple, set)):
                labels = []
                for item in obj:
                    labels.extend(parse_labels(item))
                return sorted(set(labels))
        except Exception:
            pass

    # Supports scalar labels and multi-label text.
    s = s.replace(";", "|").replace(",", "|")
    labels = []
    for part in s.split("|"):
        part = part.strip().strip("'").strip('"')
        if part and part.lower() not in {"nan", "none", "null", "na", "[]"}:
            labels.append(part)

    return sorted(set(labels))


def mol_info(smiles):
    """
    Parse a SMILES string and return canonical SMILES, InChIKey, and Murcko scaffold.

    Some NPClassifier SMILES contain problematic double-bond stereo that can trigger
    RDKit pre-condition violations during canonical SMILES generation. Those rows
    are treated as invalid for this task rather than crashing the full task build.
    """
    smi = str(smiles).strip()

    if not smi or smi.lower() in {"nan", "none", "null", "na"}:
        return None

    try:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smi)
    except Exception:
        return None

    if mol is None:
        return None

    try:
        with rdBase.BlockLogs():
            inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        return None

    if not inchikey:
        return None

    try:
        with rdBase.BlockLogs():
            canonical_smiles = Chem.MolToSmiles(
                mol,
                canonical=True,
                isomericSmiles=True,
            )
    except Exception:
        # Some molecules fail only because of problematic stereo annotation.
        # Try a non-isomeric canonical SMILES before discarding the molecule.
        try:
            with rdBase.BlockLogs():
                canonical_smiles = Chem.MolToSmiles(
                    mol,
                    canonical=True,
                    isomericSmiles=False,
                )
        except Exception:
            return None

    try:
        with rdBase.BlockLogs():
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaffold = ""

    return canonical_smiles, inchikey, scaffold


def make_random_split(y, seed=42):
    """Create an approximately 80/10/10 stratified single-label split."""
    y = np.asarray(y, dtype=int)
    idx = np.arange(len(y))

    s1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.10,
        random_state=seed,
    )
    trainval_idx, test_idx = next(s1.split(idx, y))

    # Validation is 1/9 of the remaining 90%, or approximately 10% overall.
    s2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1.0 / 9.0,
        random_state=seed,
    )
    train_rel, val_rel = next(s2.split(trainval_idx, y[trainval_idx]))

    train_idx = trainval_idx[train_rel]
    val_idx = trainval_idx[val_rel]

    split = np.array([""] * len(y), dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"

    if np.any(split == ""):
        raise RuntimeError("Internal error: unassigned rows in random split.")

    return split, "stratified_single_label"


def make_scaffold_split(task, seed=42):
    rng = np.random.RandomState(seed)

    groups = task["murcko_scaffold"].fillna("").astype(str).to_numpy()
    fallback = task["inchikey"].astype(str).to_numpy()
    groups = np.where(groups == "", fallback, groups)

    group_to_idx = {}
    for i, g in enumerate(groups):
        group_to_idx.setdefault(g, []).append(i)

    group_names = list(group_to_idx)
    rng.shuffle(group_names)

    n = len(task)
    target_train = int(round(0.80 * n))
    target_val = int(round(0.10 * n))

    split = np.array([""] * n, dtype=object)
    n_train = 0
    n_val = 0

    for g in group_names:
        idxs = group_to_idx[g]

        if n_train < target_train:
            name = "scaffold_train"
            n_train += len(idxs)
        elif n_val < target_val:
            name = "scaffold_val"
            n_val += len(idxs)
        else:
            name = "scaffold_test"

        split[idxs] = name

    return split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="raw_data/bao/bao_dataset_D_npclassifier.csv")
    parser.add_argument("--level", default="label", choices=["label", "pathway", "superclass", "class"])
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--min-count", type=int, default=50)
    parser.add_argument(
        "--split-seeds",
        nargs="+",
        type=int,
        default=[2026, 2027, 2028, 2029, 2030],
        help="Shared seeds used to generate paired random and scaffold splits.",
    )
    parser.add_argument("--out-prefix", default="data/tasks/np_classification/npclassifier_label")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    smiles_col = find_smiles_col(df)

    if args.label_col is not None:
        if args.label_col not in df.columns:
            raise ValueError(f"--label-col {args.label_col} not found. Columns: {list(df.columns)}")
        label_col = args.label_col
    else:
        label_col = find_label_col(df, args.level)

    print("Input:", input_path)
    print("Input rows:", len(df))
    print("SMILES column:", smiles_col)
    print("Label column:", label_col)

    label_field = f"npclassifier_{args.level}_labels"

    rows = []
    invalid_smiles = 0
    empty_label = 0

    for i, row in df.iterrows():
        labels = parse_labels(row[label_col])
        if not labels:
            empty_label += 1
            continue

        info = mol_info(row[smiles_col])
        if info is None:
            invalid_smiles += 1
            continue

        canonical_smiles, inchikey, scaffold = info

        rows.append({
            "source_row": i,
            "canonical_smiles": canonical_smiles,
            "inchikey": inchikey,
            "murcko_scaffold": scaffold,
            label_field: "|".join(labels),
        })

    if not rows:
        raise ValueError("No valid labeled molecules were parsed.")

    raw_task = pd.DataFrame(rows)

    # Deduplicate by InChIKey and union labels.
    dedup_rows = []
    for inchikey, sub in raw_task.groupby("inchikey", sort=False):
        label_set = set()
        for x in sub[label_field]:
            label_set.update(parse_labels(x))

        first = sub.iloc[0]
        dedup_rows.append({
            "np_task_id": f"NPCLS_{len(dedup_rows) + 1:06d}",
            "canonical_smiles": first["canonical_smiles"],
            "inchikey": inchikey,
            "murcko_scaffold": first["murcko_scaffold"],
            label_field: "|".join(sorted(label_set)),
        })

    task = pd.DataFrame(dedup_rows)

    # This benchmark is explicitly single-label multiclass. Exclude compounds
    # whose standardized chemical identity is associated with zero or multiple
    # labels after deduplication; do not collapse them with argmax.
    label_cardinality = task[label_field].map(lambda x: len(parse_labels(x)))
    single_label_mask = label_cardinality.eq(1)
    excluded_non_single = task.loc[~single_label_mask].copy()
    excluded_non_single["label_cardinality"] = label_cardinality.loc[~single_label_mask]
    task = task.loc[single_label_mask].reset_index(drop=True)

    # Count labels only after ambiguous compounds have been removed.
    label_counts = {}
    for x in task[label_field]:
        for lab in parse_labels(x):
            label_counts[lab] = label_counts.get(lab, 0) + 1

    label_count_df = (
        pd.DataFrame([{"label": k, "count": v} for k, v in label_counts.items()])
        .sort_values(["count", "label"], ascending=[False, True])
        .reset_index(drop=True)
    )

    keep_labels = label_count_df.loc[label_count_df["count"] >= args.min_count, "label"].astype(str).tolist()

    if len(keep_labels) < 2:
        raise ValueError(
            f"Only {len(keep_labels)} labels retained with min_count={args.min_count}. "
            "Lower --min-count or check the input labels."
        )

    label_cols = []
    for lab in keep_labels:
        col = f"label_{safe_name(lab)}"
        label_cols.append(col)
        task[col] = task[label_field].map(lambda x, lab=lab: int(lab in parse_labels(x)))

    Y = task[label_cols].to_numpy(dtype=int)

    # At this point rows can have cardinality one (retained class) or zero
    # (their sole class did not meet min_count). Keep only retained classes.
    keep_rows = Y.sum(axis=1) == 1
    excluded_below_min_count = int((~keep_rows).sum())
    task = task.loc[keep_rows].reset_index(drop=True)
    Y = task[label_cols].to_numpy(dtype=int)

    final_cardinality = Y.sum(axis=1)
    if not np.all(final_cardinality == 1):
        raise RuntimeError(
            "Internal error: final task contains non-single-label rows."
        )

    # Stable multiclass target: one integer class index per compound.
    ordered_label_cols = sorted(
        label_cols,
        key=lambda x: int(x.replace("label_", "")),
    )
    y = task[ordered_label_cols].to_numpy(dtype=int).argmax(axis=1)

    split_seeds = list(dict.fromkeys(args.split_seeds))
    if not split_seeds:
        raise ValueError("At least one --split-seeds value is required.")

    split_qc_rows = []
    n_classes = len(ordered_label_cols)

    for seed in split_seeds:
        random_col = f"random_split_{seed}"
        scaffold_col = f"scaffold_split_{seed}"

        task[random_col], split_method = make_random_split(y, seed=seed)
        task[scaffold_col] = make_scaffold_split(task, seed=seed)

        for split_type, col, partitions in [
            ("random", random_col, ["train", "val", "test"]),
            (
                "scaffold",
                scaffold_col,
                ["scaffold_train", "scaffold_val", "scaffold_test"],
            ),
        ]:
            for partition in partitions:
                mask = task[col].astype(str).eq(partition).to_numpy()
                if not mask.any():
                    raise RuntimeError(
                        f"Empty partition: split_type={split_type}, seed={seed}, "
                        f"partition={partition}"
                    )

                class_counts = np.bincount(y[mask], minlength=n_classes)
                if np.any(class_counts == 0):
                    missing = [
                        ordered_label_cols[i].replace("label_", "")
                        for i in np.flatnonzero(class_counts == 0)
                    ]
                    raise RuntimeError(
                        f"Missing classes in split_type={split_type}, seed={seed}, "
                        f"partition={partition}: {missing}"
                    )

                for class_index, count in enumerate(class_counts):
                    split_qc_rows.append({
                        "split_type": split_type,
                        "split_seed": seed,
                        "partition": partition,
                        "class_index": class_index,
                        "label": ordered_label_cols[class_index].replace("label_", ""),
                        "count": int(count),
                        "partition_size": int(mask.sum()),
                    })

    task_path = Path(str(out_prefix) + "_task.tsv")
    qc_path = Path(str(out_prefix) + "_qc.txt")
    label_counts_path = Path(str(out_prefix) + "_label_counts.tsv")
    excluded_path = Path(str(out_prefix) + "_excluded_non_single_label.tsv")
    split_qc_path = Path(str(out_prefix) + "_split_label_counts.tsv")

    task.to_csv(task_path, sep="\t", index=False)
    label_count_df.to_csv(label_counts_path, sep="\t", index=False)
    excluded_non_single.to_csv(excluded_path, sep="\t", index=False)
    pd.DataFrame(split_qc_rows).to_csv(split_qc_path, sep="\t", index=False)

    with open(qc_path, "w") as f:
        f.write("NPClassifier single-label multiclass task QC\n")
        f.write("============================================\n\n")
        f.write(f"input: {input_path}\n")
        f.write(f"input_rows: {len(df)}\n")
        f.write(f"smiles_col: {smiles_col}\n")
        f.write(f"label_col: {label_col}\n")
        f.write(f"level: {args.level}\n")
        f.write(f"min_count: {args.min_count}\n")
        f.write(f"empty_label_rows: {empty_label}\n")
        f.write(f"invalid_smiles: {invalid_smiles}\n")
        f.write(f"valid_labeled_rows_before_dedup: {len(raw_task)}\n")
        f.write(f"deduplicated_compounds_before_label_filter: {len(dedup_rows)}\n")
        f.write(f"excluded_non_single_label_compounds: {len(excluded_non_single)}\n")
        f.write(f"excluded_below_min_count: {excluded_below_min_count}\n")
        f.write(f"task_type: single_label_multiclass\n")
        f.write(f"n_compounds: {len(task)}\n")
        f.write(f"n_labels: {len(label_cols)}\n")
        f.write(f"total_positive_labels: {int(Y.sum())}\n")
        f.write(f"mean_label_cardinality: {float(final_cardinality.mean())}\n")
        f.write(f"non_single_label_compounds_final: {int((final_cardinality != 1).sum())}\n")
        f.write(f"random_split_method: {split_method}\n")
        f.write("scaffold_split_method: randomized_scaffold_group\n")
        f.write("split_seeds: " + ",".join(map(str, split_seeds)) + "\n\n")

        for seed in split_seeds:
            random_col = f"random_split_{seed}"
            scaffold_col = f"scaffold_split_{seed}"
            f.write(f"random split counts, seed={seed}:\n")
            f.write(str(task[random_col].value_counts()) + "\n\n")
            f.write(f"scaffold split counts, seed={seed}:\n")
            f.write(str(task[scaffold_col].value_counts()) + "\n\n")

        f.write("retained labels:\n")
        for lab, col in zip(keep_labels, label_cols):
            f.write(f"{col}\t{lab}\t{int(task[col].sum())}\n")

    print("Valid labeled rows before dedup:", len(raw_task))
    print("Deduplicated compounds before label filter:", len(dedup_rows))
    print("Excluded non-single-label compounds:", len(excluded_non_single))
    print("Excluded below min_count:", excluded_below_min_count)
    print("Final compounds:", len(task))
    print("Final labels:", len(label_cols))
    print("Total positive labels:", int(Y.sum()))
    print("Mean label cardinality:", float(final_cardinality.mean()))
    print("Non-single-label compounds final:", int((final_cardinality != 1).sum()))
    print("Random split method:", split_method)
    print("Split seeds:", split_seeds)
    for seed in split_seeds:
        random_col = f"random_split_{seed}"
        scaffold_col = f"scaffold_split_{seed}"
        print(f"Random split counts, seed={seed}:")
        print(task[random_col].value_counts())
        print(f"Scaffold split counts, seed={seed}:")
        print(task[scaffold_col].value_counts())
    print("Wrote:", task_path)
    print("Wrote:", qc_path)
    print("Wrote:", label_counts_path)
    print("Wrote:", excluded_path)
    print("Wrote:", split_qc_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from pathlib import Path
import re
import numpy as np
import pandas as pd

RANDOM_SEED = 32
TASK_FILE = Path("data/tasks/bao_plant_pathway/bao_plant_pathway_task.tsv")
OUT_DIR = Path("data/splits/bao_plant_pathway")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_label_columns(df):
    pattern = re.compile(r"^label_\d+_")
    return [c for c in df.columns if pattern.match(c)]


def make_scaffold_group(row):
    scaffold = str(row.get("murcko_scaffold", "")).strip()
    if scaffold and scaffold.lower() != "nan":
        return scaffold

    # Important: do not put all acyclic compounds into one empty-scaffold group.
    # Use canonical SMILES as fallback group.
    return str(row["canonical_smiles"])


def main():
    df = pd.read_csv(TASK_FILE, sep="\t", dtype=str)
    label_cols = get_label_columns(df)

    if len(label_cols) != 18:
        raise ValueError(f"Expected 18 label columns, found {len(label_cols)}")

    # Only split Bao A. Bao B remains external test.
    a = df[df["dataset_name"] == "Bao_A_trainval"].copy()
    b = df[df["dataset_name"] == "Bao_B_external_test"].copy()

    a[label_cols] = a[label_cols].astype(int)
    a["scaffold_group"] = a.apply(make_scaffold_group, axis=1)

    # Group by scaffold.
    group_rows = []
    for scaffold, sub in a.groupby("scaffold_group"):
        y_sum = sub[label_cols].sum().values
        group_rows.append({
            "scaffold_group": scaffold,
            "n": len(sub),
            "label_sum": y_sum,
            "compound_ids": list(sub["task_compound_id"]),
        })

    # Greedy assignment to validation set.
    # Goal: about 20% compounds in validation, while preserving label coverage reasonably.
    rng = np.random.default_rng(RANDOM_SEED)

    # Sort large / label-rich scaffold groups first.
    group_rows = sorted(
        group_rows,
        key=lambda x: (x["n"], x["label_sum"].sum()),
        reverse=True,
    )

    total_n = len(a)
    target_val_n = int(round(total_n * 0.20))

    total_label_counts = a[label_cols].sum().values
    target_val_labels = total_label_counts * 0.20

    val_groups = []
    train_groups = []

    current_val_n = 0
    current_val_labels = np.zeros(len(label_cols), dtype=float)

    for g in group_rows:
        # Score if assigned to validation.
        new_val_n = current_val_n + g["n"]
        new_val_labels = current_val_labels + g["label_sum"]

        size_penalty = abs(new_val_n - target_val_n) / max(target_val_n, 1)
        label_penalty = np.mean(
            np.abs(new_val_labels - target_val_labels) / np.maximum(target_val_labels, 1)
        )

        score_val = size_penalty + label_penalty

        # Score if not assigned to validation.
        size_penalty_keep = abs(current_val_n - target_val_n) / max(target_val_n, 1)
        label_penalty_keep = np.mean(
            np.abs(current_val_labels - target_val_labels) / np.maximum(target_val_labels, 1)
        )

        score_keep = size_penalty_keep + label_penalty_keep

        if new_val_n <= target_val_n * 1.10 and score_val <= score_keep:
            val_groups.append(g["scaffold_group"])
            current_val_n = new_val_n
            current_val_labels = new_val_labels
        else:
            train_groups.append(g["scaffold_group"])

    val_groups = set(val_groups)

    a["scaffold_split"] = np.where(
        a["scaffold_group"].isin(val_groups),
        "scaffold_val",
        "scaffold_train",
    )

    b["scaffold_group"] = b.apply(make_scaffold_group, axis=1)
    b["scaffold_split"] = "external_test"

    split_df = pd.concat([a, b], ignore_index=True)

    out_cols = [
        "task_compound_id",
        "dataset_name",
        "canonical_smiles",
        "inchikey",
        "murcko_scaffold",
        "scaffold_group",
        "scaffold_split",
    ]
    split_df[out_cols].to_csv(
        OUT_DIR / "scaffold_split.tsv",
        sep="\t",
        index=False,
    )

    # QC report.
    with open(OUT_DIR / "scaffold_split_qc.txt", "w") as f:
        f.write("Bao plant pathway scaffold split QC\n")
        f.write("====================================\n\n")
        f.write(f"Dataset A compounds: {len(a)}\n")
        f.write(f"Dataset B external compounds: {len(b)}\n")
        f.write(f"Number of scaffold groups in A: {a['scaffold_group'].nunique()}\n\n")

        f.write("Split counts:\n")
        f.write(str(split_df["scaffold_split"].value_counts()) + "\n\n")

        f.write("Label counts by split:\n")
        for split in ["scaffold_train", "scaffold_val", "external_test"]:
            sub = split_df[split_df["scaffold_split"] == split]
            if len(sub) == 0:
                continue

            sub_labels = sub[label_cols].astype(int)
            f.write(f"\n[{split}] n={len(sub)}\n")
            f.write(str(sub_labels.sum().sort_values(ascending=False)) + "\n")

        # Check scaffold leakage between scaffold_train and scaffold_val.
        train_scaffolds = set(a.loc[a["scaffold_split"] == "scaffold_train", "scaffold_group"])
        val_scaffolds = set(a.loc[a["scaffold_split"] == "scaffold_val", "scaffold_group"])
        overlap = train_scaffolds & val_scaffolds

        f.write("\nScaffold leakage check between scaffold_train and scaffold_val:\n")
        f.write(f"overlapping scaffold groups: {len(overlap)}\n")

    print("Done.")
    print(split_df["scaffold_split"].value_counts())
    print(f"Wrote: {OUT_DIR / 'scaffold_split.tsv'}")
    print(f"Wrote: {OUT_DIR / 'scaffold_split_qc.txt'}")


if __name__ == "__main__":
    main()
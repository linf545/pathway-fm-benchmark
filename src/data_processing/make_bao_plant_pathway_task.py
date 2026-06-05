#!/usr/bin/env python3

from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 23

#python src/make_bao_plant_pathway_task.py

def safe_name(x):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_")


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str)


def load_dataset(folder, dataset_name):
    folder = Path(folder)
    comp = read_tsv(folder / "compounds_master.tsv")
    lab = read_tsv(folder / "bao_compound_labels_long.tsv")
    mapping = read_tsv(folder / "bao_label_mapping.tsv")

    comp["dataset_name"] = dataset_name
    lab["dataset_name"] = dataset_name

    comp["dedup_key"] = comp["inchikey"].where(
        comp["inchikey"].fillna("").ne(""),
        comp["canonical_smiles"],
    )

    return comp, lab, mapping


def make_label_matrix(comp, labels, mapping):
    mapping = mapping[["label_class", "pathway_name"]].astype(str).drop_duplicates()
    label_ids = sorted(mapping["label_class"].unique(), key=lambda x: int(x))
    name_map = dict(zip(mapping["label_class"], mapping["pathway_name"]))

    label_cols = {
        lid: f"label_{lid}_{safe_name(name_map[lid])}"
        for lid in label_ids
    }

    y = pd.DataFrame({"task_compound_id": comp["task_compound_id"]})

    for lid, col in label_cols.items():
        y[col] = 0

    id_map = dict(zip(comp["internal_compound_id"], comp["task_compound_id"]))

    labels = labels.copy()
    labels["task_compound_id"] = labels["internal_compound_id"].map(id_map)
    labels = labels.dropna(subset=["task_compound_id"])

    for lid, col in label_cols.items():
        pos = labels.loc[labels["label_class"].astype(str) == str(lid), "task_compound_id"]
        y.loc[y["task_compound_id"].isin(pos), col] = 1

    return y, label_cols


def split_A(comp, label_matrix):
    label_cols = [c for c in label_matrix.columns if c != "task_compound_id"]

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=RANDOM_SEED,
        )

        idx = comp.index.to_numpy().reshape(-1, 1)
        y = label_matrix[label_cols].to_numpy()

        train_idx, val_idx = next(splitter.split(idx, y))
        comp.loc[comp.index[train_idx], "split"] = "train"
        comp.loc[comp.index[val_idx], "split"] = "val"
        return comp, "iterative_multilabel"

    except Exception as e:
        print(f"Warning: iterative multilabel split failed: {e}")
        print("Using fallback split based on first label.")

        tmp = comp.copy()
        tmp["primary_label"] = tmp["label_classes"].fillna("").str.split("|").str[0]

        counts = tmp["primary_label"].value_counts()
        rare = set(counts[counts < 2].index)
        tmp["stratify_label"] = tmp["primary_label"].where(
            ~tmp["primary_label"].isin(rare),
            "RARE",
        )

        stratify = tmp["stratify_label"]
        if stratify.value_counts().min() < 2:
            stratify = None

        train, val = train_test_split(
            tmp,
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=stratify,
        )

        comp["split"] = "unassigned"
        comp.loc[train.index, "split"] = "train"
        comp.loc[val.index, "split"] = "val"

        return comp, "primary_label_fallback"


def main():
    out_dir = Path("data/tasks/bao_plant_pathway")
    out_dir.mkdir(parents=True, exist_ok=True)

    a_comp, a_lab, a_map = load_dataset(
        "data/processed_bao/A_trainval",
        "Bao_A_trainval",
    )
    b_comp, b_lab, b_map = load_dataset(
        "data/processed_bao/B_external_test",
        "Bao_B_external_test",
    )

    # Exclude external-test compounds already present in A.
    a_keys = set(a_comp["dedup_key"])
    b_comp["overlap_with_A"] = b_comp["dedup_key"].isin(a_keys).astype(int)
    b_overlap = b_comp[b_comp["overlap_with_A"] == 1].copy()

    if len(b_overlap) > 0:
        b_overlap.to_csv(
            out_dir / "B_external_overlaps_with_A_excluded.tsv",
            sep="\t",
            index=False,
        )

    b_comp = b_comp[b_comp["overlap_with_A"] == 0].copy()

    a_comp = a_comp.reset_index(drop=True)
    b_comp = b_comp.reset_index(drop=True)

    a_comp["task_compound_id"] = [f"BAO_A_{i:06d}" for i in range(1, len(a_comp) + 1)]
    b_comp["task_compound_id"] = [f"BAO_B_{i:06d}" for i in range(1, len(b_comp) + 1)]

    a_y, label_cols = make_label_matrix(a_comp, a_lab, a_map)
    a_comp, split_method = split_A(a_comp, a_y)

    b_comp["split"] = "external_test"

    task_comp = pd.concat([a_comp, b_comp], ignore_index=True)
    all_lab = pd.concat([a_lab, b_lab], ignore_index=True)

    y_all, label_cols = make_label_matrix(task_comp, all_lab, a_map)
    task = task_comp.merge(y_all, on="task_compound_id", how="left")

    label_columns = list(label_cols.values())

    keep_cols = [
        "task_compound_id",
        "dataset_name",
        "split",
        "source_id",
        "source_database",
        "canonical_smiles",
        "inchikey",
        "dedup_key",
        "murcko_scaffold",
        "label_classes",
        "pathway_names",
        "molecular_weight",
        "num_atoms",
        "num_heavy_atoms",
        "num_rings",
        "num_aromatic_rings",
        "num_hbd",
        "num_hba",
        "tpsa",
        "logp",
    ] + label_columns

    keep_cols = [c for c in keep_cols if c in task.columns]
    task = task[keep_cols]

    mapping_out = a_map[["label_class", "pathway_name"]].astype(str).drop_duplicates()
    mapping_out["label_column"] = mapping_out["label_class"].map(label_cols)

    task.to_csv(out_dir / "bao_plant_pathway_task.tsv", sep="\t", index=False)
    mapping_out.to_csv(out_dir / "bao_plant_pathway_labels.tsv", sep="\t", index=False)

    split_df = task[
        [
            "task_compound_id",
            "dataset_name",
            "split",
            "canonical_smiles",
            "inchikey",
            "dedup_key",
            "murcko_scaffold",
        ]
    ]
    split_df.to_csv(out_dir / "bao_plant_pathway_splits.tsv", sep="\t", index=False)

    with open(out_dir / "bao_plant_pathway_qc.txt", "w") as f:
        f.write("Bao plant pathway task QC\n")
        f.write("==========================\n\n")
        f.write(f"A compounds: {len(a_comp)}\n")
        f.write(f"B compounds after excluding A overlap: {len(b_comp)}\n")
        f.write(f"B compounds excluded due to A overlap: {len(b_overlap)}\n")
        f.write(f"Split method: {split_method}\n\n")

        f.write("Split counts:\n")
        f.write(str(task["split"].value_counts()) + "\n\n")

        f.write("Label counts by split:\n")
        for split in ["train", "val", "external_test"]:
            sub = task[task["split"] == split]
            f.write(f"\n[{split}] n={len(sub)}\n")
            f.write(str(sub[label_columns].astype(int).sum().sort_values(ascending=False)) + "\n")

    print("Done.")
    print(task["split"].value_counts())
    print(f"Wrote: {out_dir / 'bao_plant_pathway_task.tsv'}")
    print(f"Wrote: {out_dir / 'bao_plant_pathway_qc.txt'}")


if __name__ == "__main__":
    main()
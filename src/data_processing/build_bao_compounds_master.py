#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

DATASETS = {
    "A_trainval": "data/processed_bao/A_trainval/compounds_master.tsv",
    "B_external_test": "data/processed_bao/B_external_test/compounds_master.tsv",
    "C_kegg": "data/processed_bao/C_kegg/compounds_master.tsv",
    "D_npclassifier": "data/processed_bao/D_npclassifier/compounds_master.tsv",
}

OUT_MASTER = Path("data/master/compounds_master_bao.tsv")
OUT_CROSSWALK = Path("data/master/bao_compound_source_crosswalk.tsv")
OUT_QC = Path("qc/data_processing/bao_compounds_master_qc.txt")

OUT_MASTER.parent.mkdir(parents=True, exist_ok=True)
OUT_QC.parent.mkdir(parents=True, exist_ok=True)


def load_dataset(name, path):
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["bao_dataset"] = name

    df["dedup_key"] = df["inchikey"].where(
        df["inchikey"].fillna("").ne(""),
        df["canonical_smiles"]
    )

    return df


def main():
    frames = []
    for name, path in DATASETS.items():
        if not Path(path).exists():
            raise FileNotFoundError(path)
        frames.append(load_dataset(name, path))

    all_df = pd.concat(frames, ignore_index=True)

    # Crosswalk: one row per dataset-specific compound.
    crosswalk_cols = [
        "bao_dataset",
        "internal_compound_id",
        "source_id",
        "source_database",
        "dedup_key",
        "canonical_smiles",
        "inchikey",
        "label_classes",
        "pathway_names",
    ]
    crosswalk_cols = [c for c in crosswalk_cols if c in all_df.columns]
    crosswalk = all_df[crosswalk_cols].copy()

    # Master: one row per unique compound across A/B/C/D.
    def join_unique(x):
        vals = sorted(set(str(v) for v in x.dropna() if str(v) != ""))
        return "|".join(vals)

    agg = {
        "canonical_smiles": "first",
        "inchikey": "first",
        "murcko_scaffold": "first",
        "molecular_weight": "first",
        "num_atoms": "first",
        "num_heavy_atoms": "first",
        "num_rings": "first",
        "num_aromatic_rings": "first",
        "num_hbd": "first",
        "num_hba": "first",
        "tpsa": "first",
        "logp": "first",
        "bao_dataset": join_unique,
        "source_database": join_unique,
        "source_id": join_unique,
        "label_classes": join_unique,
        "pathway_names": join_unique,
    }

    agg = {k: v for k, v in agg.items() if k in all_df.columns}

    master = all_df.groupby("dedup_key", as_index=False).agg(agg)
    master = master.sort_values("dedup_key").reset_index(drop=True)
    master.insert(0, "global_bao_compound_id", [f"BAO_CMPD_{i:07d}" for i in range(1, len(master) + 1)])

    for ds in DATASETS:
        master[f"is_{ds}"] = master["bao_dataset"].str.split("|").apply(lambda xs: int(ds in xs))

    key_to_global = dict(zip(master["dedup_key"], master["global_bao_compound_id"]))
    crosswalk["global_bao_compound_id"] = crosswalk["dedup_key"].map(key_to_global)

    # Save.
    master.to_csv(OUT_MASTER, sep="\t", index=False)
    crosswalk.to_csv(OUT_CROSSWALK, sep="\t", index=False)

    with open(OUT_QC, "w") as f:
        f.write("Bao compound master QC\n")
        f.write("======================\n\n")
        f.write(f"Total dataset-specific rows: {len(all_df):,}\n")
        f.write(f"Unique compounds across A/B/C/D: {len(master):,}\n\n")

        f.write("Rows by dataset:\n")
        f.write(str(all_df["bao_dataset"].value_counts()) + "\n\n")

        f.write("Unique compounds by dataset:\n")
        for ds in DATASETS:
            n = master[f"is_{ds}"].sum()
            f.write(f"{ds}: {n:,}\n")

        f.write("\nDataset-overlap patterns:\n")
        f.write(str(master["bao_dataset"].value_counts().head(50)) + "\n")

    print("Done.")
    print(f"Dataset-specific rows: {len(all_df):,}")
    print(f"Unique compounds across A/B/C/D: {len(master):,}")
    print(f"Wrote: {OUT_MASTER}")
    print(f"Wrote: {OUT_CROSSWALK}")
    print(f"Wrote: {OUT_QC}")


if __name__ == "__main__":
    main()
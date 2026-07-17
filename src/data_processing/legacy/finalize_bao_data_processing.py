#!/usr/bin/env python3

from pathlib import Path
import json
import hashlib
import pandas as pd
import numpy as np
from scipy import sparse


FILES = [
    "data/tasks/bao_plant_pathway/bao_plant_pathway_task.tsv",
    "data/tasks/bao_plant_pathway/bao_plant_pathway_labels.tsv",
    "data/tasks/bao_plant_pathway/bao_plant_pathway_splits.tsv",
    "data/splits/bao_plant_pathway/scaffold_split.tsv",
    "data/master/compounds_master_bao.tsv",
    "data/master/bao_compound_source_crosswalk.tsv",
    "data/features/bao_plant_pathway/metadata.tsv",
    "data/features/bao_plant_pathway/labels.npy",
    "data/features/bao_plant_pathway/label_columns.txt",
    "data/features/bao_plant_pathway/morgan_r2_2048.npz",
    "data/features/bao_plant_pathway/morgan_r3_2048.npz",
    "data/features/bao_plant_pathway/rdkit_descriptors.tsv",
]


OUT_DIR = Path("qc/data_processing")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    task = pd.read_csv("data/tasks/bao_plant_pathway/bao_plant_pathway_task.tsv", sep="\t")
    meta = pd.read_csv("data/features/bao_plant_pathway/metadata.tsv", sep="\t")
    y = np.load("data/features/bao_plant_pathway/labels.npy")
    x2 = sparse.load_npz("data/features/bao_plant_pathway/morgan_r2_2048.npz")
    x3 = sparse.load_npz("data/features/bao_plant_pathway/morgan_r3_2048.npz")
    desc = pd.read_csv("data/features/bao_plant_pathway/rdkit_descriptors.tsv", sep="\t")
    scaffold = pd.read_csv("data/splits/bao_plant_pathway/scaffold_split.tsv", sep="\t")
    master = pd.read_csv("data/master/compounds_master_bao.tsv", sep="\t")

    with open(OUT_DIR / "bao_data_processing_final_qc.txt", "w") as f:
        f.write("Bao data-processing final QC\n")
        f.write("============================\n\n")

        f.write("Task table split counts:\n")
        f.write(str(task["split"].value_counts()) + "\n\n")

        f.write("Scaffold split counts:\n")
        f.write(str(scaffold["scaffold_split"].value_counts()) + "\n\n")

        f.write(f"Task rows: {len(task)}\n")
        f.write(f"Metadata rows: {len(meta)}\n")
        f.write(f"Labels shape: {y.shape}\n")
        f.write(f"Morgan r2 shape: {x2.shape}\n")
        f.write(f"Morgan r3 shape: {x3.shape}\n")
        f.write(f"RDKit descriptor shape: {desc.shape}\n")
        f.write(f"Bao master unique compounds: {len(master)}\n\n")

        f.write("File checksums:\n")
        for file in FILES:
            p = Path(file)
            if p.exists():
                f.write(f"{file}\t{p.stat().st_size}\t{md5sum(p)}\n")
            else:
                f.write(f"{file}\tMISSING\tMISSING\n")

    manifest = {
        "task": "Bao plant pathway multilabel prediction",
        "input": "canonical_smiles",
        "labels": "18 plant pathway labels",
        "random_split": "train/val from Bao Dataset A",
        "external_test": "Bao Dataset B after removing A-overlapping compounds",
        "scaffold_split": "scaffold_train/scaffold_val from Bao Dataset A",
        "n_task_rows": int(len(task)),
        "label_shape": list(y.shape),
        "morgan_r2_shape": list(x2.shape),
        "morgan_r3_shape": list(x3.shape),
        "rdkit_descriptor_shape": list(desc.shape),
    }

    with open(OUT_DIR / "bao_data_processing_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    print(f"Wrote: {OUT_DIR / 'bao_data_processing_final_qc.txt'}")
    print(f"Wrote: {OUT_DIR / 'bao_data_processing_manifest.json'}")


if __name__ == "__main__":
    main()
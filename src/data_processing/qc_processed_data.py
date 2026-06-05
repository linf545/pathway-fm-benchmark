#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


def read_tsv(path):
    path = Path(path)

    if not path.exists():
        print(f"[MISSING] {path}")
        return None

    if path.stat().st_size == 0:
        print(f"[EMPTY] {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t", dtype=str)
    except pd.errors.EmptyDataError:
        print(f"[EMPTY] {path}")
        return pd.DataFrame()

def summarize_bao_folder(name, folder):
    folder = Path(folder)
    print("\n" + "=" * 80)
    print(f"Bao dataset: {name}")
    print("=" * 80)

    comp = read_tsv(folder / "compounds_master.tsv")
    lab = read_tsv(folder / "bao_compound_labels_long.tsv")
    bad = read_tsv(folder / "bao_invalid_smiles.tsv")

    if comp is not None:
        print(f"valid compounds: {len(comp):,}")
        print(f"unique InChIKeys: {comp['inchikey'].nunique(dropna=True):,}")
        print(f"unique canonical SMILES: {comp['canonical_smiles'].nunique(dropna=True):,}")
        if "label_classes" in comp.columns:
            n_labeled = comp["label_classes"].fillna("").ne("").sum()
            print(f"compounds with labels: {n_labeled:,}")

    if lab is not None and len(lab) > 0:
        print(f"long label rows: {len(lab):,}")
        print("label counts:")
        print(lab["label_class"].value_counts().sort_index())

        if "pathway_name" in lab.columns:
            print("\npathway counts:")
            print(lab["pathway_name"].value_counts().head(30))

    if bad is not None:
        print(f"invalid SMILES rows: {len(bad):,}")


def summarize_plantcyc(folder):
    folder = Path(folder)
    print("\n" + "=" * 80)
    print("PlantCyc")
    print("=" * 80)

    comp = read_tsv(folder / "plant_compounds.tsv")
    pairs = read_tsv(folder / "plant_reaction_pairs.tsv")
    pathways = read_tsv(folder / "plant_pathways.tsv")
    rxns = read_tsv(folder / "plant_reactions.tsv")

    if comp is not None:
        print(f"plant compounds: {len(comp):,}")
        if "has_smiles" in comp.columns:
            print(f"compounds with SMILES: {comp['has_smiles'].astype(str).eq('1').sum():,}")
        if "pathway_ids" in comp.columns:
            print(f"compounds with pathway_ids: {comp['pathway_ids'].fillna('').ne('').sum():,}")

        print("\nTop pathway names from compounds:")
        if "pathway_names" in comp.columns:
            top = (
                comp["pathway_names"]
                .fillna("")
                .str.split("|")
                .explode()
            )
            top = top[top != ""].value_counts().head(30)
            print(top)

    if pairs is not None:
        print(f"\nreaction pairs: {len(pairs):,}")
        if "has_both_smiles" in pairs.columns:
            print(f"reaction pairs with both SMILES: {pairs['has_both_smiles'].astype(str).eq('1').sum():,}")

        print("\nTop high-degree compounds in reaction pairs:")
        if "substrate_id" in pairs.columns and "product_id" in pairs.columns:
            all_ids = pd.concat([pairs["substrate_id"], pairs["product_id"]]).dropna()
            print(all_ids.value_counts().head(50))

    if pathways is not None:
        print(f"\npathways: {len(pathways):,}")

    if rxns is not None:
        print(f"reactions: {len(rxns):,}")


def main():
    summarize_bao_folder("A_trainval", "data/processed_bao/A_trainval")
    summarize_bao_folder("B_external_test", "data/processed_bao/B_external_test")
    summarize_bao_folder("C_kegg", "data/processed_bao/C_kegg")
    summarize_bao_folder("D_npclassifier", "data/processed_bao/D_npclassifier")

    summarize_plantcyc("data/processed_plantcyc")


if __name__ == "__main__":
    main()
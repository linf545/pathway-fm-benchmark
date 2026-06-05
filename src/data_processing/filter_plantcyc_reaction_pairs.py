#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import re
import pandas as pd


IN_PAIRS = Path("data/processed_plantcyc/plant_reaction_pairs.tsv")
OUT_DIR = Path("data/tasks/plantcyc_reaction_neighbor")
QC_DIR = Path("qc/data_processing")

OUT_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

DEGREE_THRESHOLD = 150


DEFAULT_EXCLUDE_IDS = {
    # generic placeholders / redox carriers
    "Red-NADPH-Hemoprotein-Reductases",
    "Ox-NADPH-Hemoprotein-Reductases",
    "Donor-H2",
    "Acceptor",
    "Reduced-ferredoxins",
    "Oxidized-ferredoxins",
    "FERRICYTOCHROME-B5",
    "FERROCYTOCHROME-B5",
    "NADH-P-OR-NOP",
    "NAD-P-OR-NOP",

    # common currency / primary metabolism hubs
    "WATER",
    "PROTON",
    "ATP",
    "ADP",
    "AMP",
    "NAD",
    "NADH",
    "NADP",
    "NADPH",
    "CO-A",
    "COA",
    "CARBON-DIOXIDE",
    "CO2",
    "OXYGEN-MOLECULE",
    "OXYGEN",
    "PHOSPHATE",
    "PPI",
    "AMMONIUM",
    "2-KETOGLUTARATE",
    "ADENOSYL-HOMO-CYS",
    "GLT",
    "SUC",
    "PYRUVATE",
    "ACETYL-COA",
    "MALONYL-COA",
    "GLUTATHIONE",
    "ACP",
    "PAPS",
    "3-5-ADP",

    # generic compound classes
    "Amino-Acids-20",
    "2-Oxo-carboxylates",
    "Demethylated-methyl-acceptors",
}


GENERIC_PATTERNS = [
    r"^Red-",
    r"^Ox-",
    r"ferredoxin",
    r"cytochrome",
    r"Acceptor",
    r"Donor",
    r"generic",
    r"unknown",
]


def get_reason(cid, degree):
    reasons = []

    if cid in DEFAULT_EXCLUDE_IDS:
        reasons.append("manual_currency_or_generic")

    if degree >= DEGREE_THRESHOLD:
        reasons.append(f"high_degree_ge_{DEGREE_THRESHOLD}")

    for pat in GENERIC_PATTERNS:
        if re.search(pat, cid, flags=re.IGNORECASE):
            reasons.append(f"pattern:{pat}")

    return "|".join(sorted(set(reasons)))


def main():
    pairs = pd.read_csv(IN_PAIRS, sep="\t", dtype=str)

    print(f"Input reaction pairs: {len(pairs):,}")

    # Keep only pairs where both sides have SMILES.
    pairs = pairs[pairs["has_both_smiles"].astype(str) == "1"].copy()
    print(f"Pairs with both SMILES: {len(pairs):,}")

    # Remove self pairs.
    pairs = pairs[pairs["substrate_id"] != pairs["product_id"]].copy()
    print(f"After removing self pairs: {len(pairs):,}")

    # Count pair-degree by compound ID.
    all_ids = list(pairs["substrate_id"]) + list(pairs["product_id"])
    degree = Counter(all_ids)

    high_degree = pd.DataFrame(
        [{"compound_id": k, "degree": v} for k, v in degree.items()]
    ).sort_values("degree", ascending=False)

    high_degree.to_csv(
        OUT_DIR / "plantcyc_reaction_pair_high_degree_compounds.tsv",
        sep="\t",
        index=False,
    )

    # Build exclusion table.
    exclude_rows = []
    for cid, deg in degree.items():
        reason = get_reason(cid, deg)
        if reason:
            exclude_rows.append({
                "compound_id": cid,
                "degree": deg,
                "exclude_reason": reason,
            })

    exclude_df = pd.DataFrame(exclude_rows).sort_values(
        ["degree", "compound_id"],
        ascending=[False, True],
    )

    exclude_df.to_csv(
        OUT_DIR / "plantcyc_reaction_pair_excluded_compounds.tsv",
        sep="\t",
        index=False,
    )

    exclude_ids = set(exclude_df["compound_id"]) if len(exclude_df) else set()

    pairs["substrate_exclude"] = pairs["substrate_id"].isin(exclude_ids).astype(int)
    pairs["product_exclude"] = pairs["product_id"].isin(exclude_ids).astype(int)
    pairs["remove_pair"] = (
        (pairs["substrate_exclude"] == 1) |
        (pairs["product_exclude"] == 1)
    ).astype(int)

    removed = pairs[pairs["remove_pair"] == 1].copy()
    kept = pairs[pairs["remove_pair"] == 0].copy()

    # Deduplicate exact directed substrate-product-reaction entries.
    before_dedup = len(kept)
    kept = kept.drop_duplicates(
        subset=[
            "reaction_id",
            "substrate_id",
            "product_id",
            "substrate_smiles",
            "product_smiles",
        ]
    ).copy()
    after_dedup = len(kept)

    kept.to_csv(
        OUT_DIR / "plantcyc_positive_pairs_filtered.tsv",
        sep="\t",
        index=False,
    )

    removed.to_csv(
        OUT_DIR / "plantcyc_positive_pairs_removed.tsv",
        sep="\t",
        index=False,
    )

    with open(QC_DIR / "plantcyc_reaction_pair_filter_qc.txt", "w") as f:
        f.write("PlantCyc reaction-pair filtering QC\n")
        f.write("===================================\n\n")
        f.write(f"Input pairs with both SMILES: {len(pairs):,}\n")
        f.write(f"Excluded compounds: {len(exclude_ids):,}\n")
        f.write(f"Removed pairs: {len(removed):,}\n")
        f.write(f"Kept pairs before dedup: {before_dedup:,}\n")
        f.write(f"Kept pairs after dedup: {after_dedup:,}\n\n")

        f.write("Top high-degree compounds:\n")
        f.write(str(high_degree.head(50)) + "\n\n")

        f.write("Top excluded compounds:\n")
        if len(exclude_df):
            f.write(str(exclude_df.head(100)) + "\n")
        else:
            f.write("None\n")

    print("Done.")
    print(f"Excluded compounds: {len(exclude_ids):,}")
    print(f"Removed pairs: {len(removed):,}")
    print(f"Kept positive pairs after dedup: {len(kept):,}")
    print(f"Wrote: {OUT_DIR / 'plantcyc_positive_pairs_filtered.tsv'}")
    print(f"Wrote: {QC_DIR / 'plantcyc_reaction_pair_filter_qc.txt'}")


if __name__ == "__main__":
    main()
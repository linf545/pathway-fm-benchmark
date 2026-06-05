#!/usr/bin/env python3

import argparse
import csv
import html
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

# python src/process_plantcyc.py --data_dir raw_data/plantcyc17 --out_dir data/processed_plantcyc

CURRENCY_COMPOUNDS = {
    "WATER",
    "PROTON",
    "PROTONS",
    "ATP",
    "ADP",
    "AMP",
    "GTP",
    "GDP",
    "GMP",
    "UTP",
    "UDP",
    "UMP",
    "CTP",
    "CDP",
    "CMP",
    "NAD",
    "NADH",
    "NADP",
    "NADPH",
    "FAD",
    "FADH2",
    "FMN",
    "CO-A",
    "COA",
    "CARBON-DIOXIDE",
    "CO2",
    "OXYGEN-MOLECULE",
    "OXYGEN",
    "PHOSPHATE",
    "PPI",
    "Pi",
    "PPi",
    "AMMONIA",
    "HYDROGEN-PEROXIDE",
    "S-ADENOSYLMETHIONINE",
    "S-ADENOSYL-L-HOMOCYSTEINE",
}


def clean_text(x):
    if x is None:
        return ""
    x = html.unescape(str(x))
    x = re.sub(r"<[^>]+>", "", x)
    x = x.replace("\n", " ").replace("\r", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def parse_biocyc_dat(path):
    """
    Parse BioCyc / Pathway Tools .dat frame files.

    Returns:
        list[dict[str, list[str]]]

    Handles:
    - KEY - VALUE fields
    - repeated fields
    - // record delimiters
    - ^ annotation lines are stored as ANNOTATION::<key>
    - basic continuation lines are appended to the previous field
    """
    records = []
    record = {}
    last_key = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if not line:
                continue

            if line.startswith("#"):
                continue

            if line == "//":
                if record:
                    records.append(record)
                record = {}
                last_key = None
                continue

            # Annotation line, e.g. ^COEFFICIENT - 2
            if line.startswith("^"):
                ann = line[1:]
                if " - " in ann:
                    key, value = ann.split(" - ", 1)
                    ann_key = f"ANNOTATION::{key}"
                    record.setdefault(ann_key, []).append(value)
                    last_key = ann_key
                continue

            # Standard key-value line.
            if " - " in line:
                key, value = line.split(" - ", 1)
                record.setdefault(key, []).append(value)
                last_key = key
                continue

            # Continuation line. BioCyc flat files rarely need this, but keep it.
            if last_key is not None and record.get(last_key):
                record[last_key][-1] += " " + line.strip()

    if record:
        records.append(record)

    return records


def first(record, key, default=""):
    values = record.get(key, [])
    if not values:
        return default
    return values[0]


def joined(record, key, sep="|"):
    values = record.get(key, [])
    values = [clean_text(v) for v in values if clean_text(v)]
    return sep.join(values)


def normalize_id(x):
    if x is None:
        return ""
    return str(x).strip()


def get_compounds_from_reaction_side(record, side):
    """
    side should be LEFT or RIGHT.
    Returns compound IDs from that side.
    """
    values = record.get(side, [])
    return [normalize_id(v) for v in values if normalize_id(v)]


def infer_direction(record):
    """
    Return one of:
    - LEFT_TO_RIGHT
    - RIGHT_TO_LEFT
    - REVERSIBLE
    - UNKNOWN

    BioCyc direction strings can vary. This function is intentionally tolerant.
    """
    raw = first(record, "REACTION-DIRECTION", "").upper()

    if not raw:
        return "UNKNOWN"

    if "REVERSIBLE" in raw:
        return "REVERSIBLE"

    if "LEFT-TO-RIGHT" in raw:
        return "LEFT_TO_RIGHT"

    if "RIGHT-TO-LEFT" in raw:
        return "RIGHT_TO_LEFT"

    return "UNKNOWN"


def canonical_pair_key(a, b):
    """
    For de-duplicating undirected neighbor rows if needed.
    """
    return tuple(sorted([a, b]))


def extract_pathway_reactions(pathway_records):
    """
    pathway_id -> set(reaction_id)
    reaction_id -> set(pathway_id)
    """
    pathway_to_reactions = defaultdict(set)
    reaction_to_pathways = defaultdict(set)

    for p in pathway_records:
        pathway_id = first(p, "UNIQUE-ID", "")
        if not pathway_id:
            continue

        for rxn_id in p.get("REACTION-LIST", []):
            rxn_id = normalize_id(rxn_id)
            if not rxn_id:
                continue
            pathway_to_reactions[pathway_id].add(rxn_id)
            reaction_to_pathways[rxn_id].add(pathway_id)

    return pathway_to_reactions, reaction_to_pathways


def build_class_lookup(class_records):
    """
    class_id -> class common name/types
    """
    class_lookup = {}

    for c in class_records:
        cid = first(c, "UNIQUE-ID", "")
        if not cid:
            continue

        class_lookup[cid] = {
            "class_id": cid,
            "class_name": clean_text(first(c, "COMMON-NAME", "")),
            "class_types": joined(c, "TYPES"),
        }

    return class_lookup


def build_pathway_lookup(pathway_records, class_lookup):
    """
    pathway_id -> pathway metadata
    """
    pathway_lookup = {}

    for p in pathway_records:
        pid = first(p, "UNIQUE-ID", "")
        if not pid:
            continue

        pathway_types = [normalize_id(x) for x in p.get("TYPES", []) if normalize_id(x)]
        pathway_type_names = []
        for t in pathway_types:
            if t in class_lookup:
                pathway_type_names.append(class_lookup[t]["class_name"])
            else:
                pathway_type_names.append(t)

        pathway_lookup[pid] = {
            "pathway_id": pid,
            "pathway_name": clean_text(first(p, "COMMON-NAME", "")),
            "pathway_types": "|".join(pathway_types),
            "pathway_type_names": "|".join([x for x in pathway_type_names if x]),
            "reaction_list": "|".join([normalize_id(x) for x in p.get("REACTION-LIST", [])]),
            "citations": joined(p, "CITATIONS"),
        }

    return pathway_lookup


def build_reaction_lookup(reaction_records):
    """
    reaction_id -> reaction metadata
    """
    reaction_lookup = {}

    for r in reaction_records:
        rid = first(r, "UNIQUE-ID", "")
        if not rid:
            continue

        left = get_compounds_from_reaction_side(r, "LEFT")
        right = get_compounds_from_reaction_side(r, "RIGHT")

        reaction_lookup[rid] = {
            "reaction_id": rid,
            "reaction_name": clean_text(first(r, "COMMON-NAME", "")),
            "left_compounds": left,
            "right_compounds": right,
            "reaction_direction_raw": first(r, "REACTION-DIRECTION", ""),
            "reaction_direction_inferred": infer_direction(r),
            "reaction_types": joined(r, "TYPES"),
            "ec_numbers": joined(r, "EC-NUMBER"),
            "citations": joined(r, "CITATIONS"),
        }

    return reaction_lookup


def build_enzrxn_lookup(enzrxn_records):
    """
    reaction_id -> enzrxn IDs and enzyme-related metadata.

    This is optional metadata. The output tables do not depend on this.
    """
    reaction_to_enzrxns = defaultdict(list)

    for e in enzrxn_records:
        enzrxn_id = first(e, "UNIQUE-ID", "")
        reaction_id = first(e, "REACTION", "")

        if not enzrxn_id or not reaction_id:
            continue

        reaction_to_enzrxns[reaction_id].append(enzrxn_id)

    return reaction_to_enzrxns


def build_compound_table(compound_records, reaction_lookup, reaction_to_pathways, pathway_lookup):
    """
    Create plant_compounds.tsv.

    Adds pathway membership indirectly:
    compound -> reactions containing compound -> pathways containing those reactions.
    """
    compound_to_reactions = defaultdict(set)
    compound_to_pathways = defaultdict(set)

    for rid, rxn in reaction_lookup.items():
        all_compounds = set(rxn["left_compounds"]) | set(rxn["right_compounds"])
        for cid in all_compounds:
            compound_to_reactions[cid].add(rid)
            for pid in reaction_to_pathways.get(rid, set()):
                compound_to_pathways[cid].add(pid)

    rows = []

    for c in compound_records:
        cid = first(c, "UNIQUE-ID", "")
        if not cid:
            continue

        pathway_ids = sorted(compound_to_pathways.get(cid, set()))
        pathway_names = []
        pathway_types = []

        for pid in pathway_ids:
            meta = pathway_lookup.get(pid, {})
            if meta.get("pathway_name"):
                pathway_names.append(meta["pathway_name"])
            if meta.get("pathway_types"):
                pathway_types.extend(meta["pathway_types"].split("|"))

        pathway_types = sorted(set([x for x in pathway_types if x]))

        smiles = first(c, "SMILES", "")
        inchi = first(c, "INCHI", "")

        rows.append({
            "compound_id": cid,
            "compound_name": clean_text(first(c, "COMMON-NAME", "")),
            "synonyms": joined(c, "SYNONYMS"),
            "types": joined(c, "TYPES"),
            "smiles": smiles,
            "inchi": inchi,
            "dblinks": joined(c, "DBLINKS"),
            "molecular_weight": first(c, "MOLECULAR-WEIGHT", ""),
            "chemical_formula": first(c, "CHEMICAL-FORMULA", ""),
            "citations": joined(c, "CITATIONS"),
            "reaction_ids": "|".join(sorted(compound_to_reactions.get(cid, set()))),
            "pathway_ids": "|".join(pathway_ids),
            "pathway_names": "|".join(sorted(set(pathway_names))),
            "pathway_types": "|".join(pathway_types),
            "has_smiles": int(bool(smiles)),
            "has_inchi": int(bool(inchi)),
            "is_currency_compound": int(cid in CURRENCY_COMPOUNDS),
        })

    return pd.DataFrame(rows)


def make_positive_pairs_for_reaction(rxn):
    """
    Generate directed substrate-product positive pairs.

    For REVERSIBLE, create both directions.
    For UNKNOWN, use LEFT -> RIGHT but mark as UNKNOWN.
    """
    left = rxn["left_compounds"]
    right = rxn["right_compounds"]
    direction = rxn["reaction_direction_inferred"]

    pairs = []

    def add_pairs(substrates, products, pair_direction):
        for s in substrates:
            for p in products:
                if not s or not p:
                    continue
                if s == p:
                    continue
                pairs.append((s, p, pair_direction))

    if direction == "LEFT_TO_RIGHT":
        add_pairs(left, right, "LEFT_TO_RIGHT")
    elif direction == "RIGHT_TO_LEFT":
        add_pairs(right, left, "RIGHT_TO_LEFT")
    elif direction == "REVERSIBLE":
        add_pairs(left, right, "REVERSIBLE_LEFT_TO_RIGHT")
        add_pairs(right, left, "REVERSIBLE_RIGHT_TO_LEFT")
    else:
        add_pairs(left, right, "UNKNOWN_LEFT_TO_RIGHT_ASSUMED")

    return pairs


def build_reaction_pair_table(
    reaction_lookup,
    reaction_to_pathways,
    pathway_lookup,
    compound_lookup,
    reaction_to_enzrxns,
    remove_currency=True,
):
    """
    Create plant_reaction_pairs.tsv with positive substrate-product pairs.

    Negative sampling should usually be done later, after train/test split,
    to avoid leakage. This file therefore only creates positive pairs.
    """
    rows = []
    seen = set()

    for rid, rxn in reaction_lookup.items():
        pids = sorted(reaction_to_pathways.get(rid, set()))

        if pids:
            pathway_names = [
                pathway_lookup.get(pid, {}).get("pathway_name", "")
                for pid in pids
            ]
            pathway_names = sorted(set([x for x in pathway_names if x]))
        else:
            pathway_names = []

        enzrxn_ids = reaction_to_enzrxns.get(rid, [])

        for substrate_id, product_id, pair_direction in make_positive_pairs_for_reaction(rxn):
            if remove_currency and (
                substrate_id in CURRENCY_COMPOUNDS or product_id in CURRENCY_COMPOUNDS
            ):
                continue

            substrate = compound_lookup.get(substrate_id, {})
            product = compound_lookup.get(product_id, {})

            # Keep only compound pairs with at least one structure field.
            # For ML training you may later filter to both have SMILES.
            substrate_smiles = substrate.get("smiles", "")
            product_smiles = product.get("smiles", "")
            substrate_inchi = substrate.get("inchi", "")
            product_inchi = product.get("inchi", "")

            unique_key = (
                rid,
                substrate_id,
                product_id,
                pair_direction,
                "|".join(pids),
            )

            if unique_key in seen:
                continue
            seen.add(unique_key)

            rows.append({
                "reaction_id": rid,
                "reaction_name": rxn["reaction_name"],
                "reaction_direction_raw": rxn["reaction_direction_raw"],
                "reaction_direction_inferred": rxn["reaction_direction_inferred"],
                "pair_direction": pair_direction,
                "substrate_id": substrate_id,
                "substrate_name": substrate.get("compound_name", ""),
                "substrate_smiles": substrate_smiles,
                "substrate_inchi": substrate_inchi,
                "product_id": product_id,
                "product_name": product.get("compound_name", ""),
                "product_smiles": product_smiles,
                "product_inchi": product_inchi,
                "pathway_ids": "|".join(pids),
                "pathway_names": "|".join(pathway_names),
                "reaction_types": rxn["reaction_types"],
                "ec_numbers": rxn["ec_numbers"],
                "enzrxn_ids": "|".join(enzrxn_ids),
                "citations": rxn["citations"],
                "label": 1,
                "is_positive_pair": 1,
                "has_both_smiles": int(bool(substrate_smiles) and bool(product_smiles)),
            })

    return pd.DataFrame(rows)


def make_compound_lookup(compounds_df):
    lookup = {}

    for _, row in compounds_df.iterrows():
        cid = row["compound_id"]
        lookup[cid] = {
            "compound_id": cid,
            "compound_name": row.get("compound_name", ""),
            "smiles": row.get("smiles", ""),
            "inchi": row.get("inchi", ""),
            "types": row.get("types", ""),
        }

    return lookup


def read_pathways_col(path):
    """
    Optional convenience parser for pathways.col.

    Not required for main output, because pathways.dat has richer information.
    This function is included only so the script can save a quick inspection file.
    """
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, sep="\t", comment="#", dtype=str)
        return df
    except Exception:
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing compounds.dat, reactions.dat, pathways.dat, classes.dat, enzrxns.dat, pathways.col",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to write processed TSV files",
    )
    parser.add_argument(
        "--keep_currency",
        action="store_true",
        help="Keep currency metabolites in reaction-pair output. Default removes them.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "compounds.dat",
        "reactions.dat",
        "pathways.dat",
        "classes.dat",
        "enzrxns.dat",
    ]

    for fname in required_files:
        fpath = data_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing required file: {fpath}")

    print("Parsing BioCyc flat files...")

    compound_records = parse_biocyc_dat(data_dir / "compounds.dat")
    reaction_records = parse_biocyc_dat(data_dir / "reactions.dat")
    pathway_records = parse_biocyc_dat(data_dir / "pathways.dat")
    class_records = parse_biocyc_dat(data_dir / "classes.dat")
    enzrxn_records = parse_biocyc_dat(data_dir / "enzrxns.dat")

    print(f"compound records: {len(compound_records):,}")
    print(f"reaction records: {len(reaction_records):,}")
    print(f"pathway records: {len(pathway_records):,}")
    print(f"class records: {len(class_records):,}")
    print(f"enzrxn records: {len(enzrxn_records):,}")

    class_lookup = build_class_lookup(class_records)
    pathway_lookup = build_pathway_lookup(pathway_records, class_lookup)
    pathway_to_reactions, reaction_to_pathways = extract_pathway_reactions(pathway_records)
    reaction_lookup = build_reaction_lookup(reaction_records)
    reaction_to_enzrxns = build_enzrxn_lookup(enzrxn_records)

    print("Building plant_compounds.tsv...")
    compounds_df = build_compound_table(
        compound_records=compound_records,
        reaction_lookup=reaction_lookup,
        reaction_to_pathways=reaction_to_pathways,
        pathway_lookup=pathway_lookup,
    )

    compound_lookup = make_compound_lookup(compounds_df)

    print("Building plant_reaction_pairs.tsv...")
    reaction_pairs_df = build_reaction_pair_table(
        reaction_lookup=reaction_lookup,
        reaction_to_pathways=reaction_to_pathways,
        pathway_lookup=pathway_lookup,
        compound_lookup=compound_lookup,
        reaction_to_enzrxns=reaction_to_enzrxns,
        remove_currency=not args.keep_currency,
    )

    # Save main outputs.
    compounds_out = out_dir / "plant_compounds.tsv"
    pairs_out = out_dir / "plant_reaction_pairs.tsv"

    compounds_df.to_csv(compounds_out, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    reaction_pairs_df.to_csv(pairs_out, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)

    # Save useful auxiliary files.
    pathway_rows = list(pathway_lookup.values())
    pd.DataFrame(pathway_rows).to_csv(
        out_dir / "plant_pathways.tsv", sep="\t", index=False
    )

    reaction_rows = list(reaction_lookup.values())
    reaction_export = []
    for r in reaction_rows:
        rr = dict(r)
        rr["left_compounds"] = "|".join(rr["left_compounds"])
        rr["right_compounds"] = "|".join(rr["right_compounds"])
        rr["pathway_ids"] = "|".join(sorted(reaction_to_pathways.get(rr["reaction_id"], set())))
        reaction_export.append(rr)

    pd.DataFrame(reaction_export).to_csv(
        out_dir / "plant_reactions.tsv", sep="\t", index=False
    )

    pathways_col = read_pathways_col(data_dir / "pathways.col")
    if not pathways_col.empty:
        pathways_col.to_csv(out_dir / "pathways_col_parsed.tsv", sep="\t", index=False)

    # Summary.
    print("\nDone.")
    print(f"Wrote: {compounds_out}")
    print(f"Wrote: {pairs_out}")

    print("\nSummary:")
    print(f"plant_compounds rows: {len(compounds_df):,}")
    print(f"compounds with SMILES: {int(compounds_df['has_smiles'].sum()):,}")
    print(f"compounds with pathway IDs: {(compounds_df['pathway_ids'].astype(str) != '').sum():,}")
    print(f"positive reaction pairs: {len(reaction_pairs_df):,}")
    if len(reaction_pairs_df) > 0:
        print(f"reaction pairs with both SMILES: {int(reaction_pairs_df['has_both_smiles'].sum()):,}")


if __name__ == "__main__":
    main()
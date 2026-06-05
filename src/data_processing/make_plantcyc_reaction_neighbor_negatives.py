#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import random
import pandas as pd


RANDOM_SEED = 63
random.seed(RANDOM_SEED)

IN_FILE = Path("data/tasks/plantcyc_reaction_neighbor/plantcyc_positive_pairs_with_splits.tsv")
OUT_DIR = Path("data/tasks/plantcyc_reaction_neighbor")
QC_DIR = Path("qc/data_processing")

OUT_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

# Total negatives per positive = random_neg_per_pos + hard_neg_per_pos
RANDOM_NEG_PER_POS = 1
HARD_NEG_PER_POS = 2
MAX_ATTEMPTS_FACTOR = 200


def split_pathways(x):
    if pd.isna(x):
        return set()
    vals = [v.strip() for v in str(x).split("|") if v.strip()]
    return set(vals)


def undirected_key(a, b):
    return "||".join(sorted([str(a), str(b)]))


def directed_key(a, b):
    return f"{a}>>{b}"


def add_compound(pool, cid, name, smiles, scaffold, pathway_names):
    if not smiles or pd.isna(smiles):
        return

    key = str(smiles)

    if key not in pool:
        pool[key] = {
            "compound_keys": set(),
            "compound_ids": set(),
            "compound_names": set(),
            "canonical_smiles": key,
            "scaffolds": set(),
            "pathways": set(),
        }

    pool[key]["compound_keys"].add(key)

    if cid and not pd.isna(cid):
        pool[key]["compound_ids"].add(str(cid))

    if name and not pd.isna(name):
        pool[key]["compound_names"].add(str(name))

    if scaffold and not pd.isna(scaffold):
        pool[key]["scaffolds"].add(str(scaffold))

    pool[key]["pathways"].update(split_pathways(pathway_names))


def build_compound_pool(pos_df):
    pool = {}

    for _, r in pos_df.iterrows():
        add_compound(
            pool,
            r.get("substrate_id", ""),
            r.get("substrate_name", ""),
            r.get("substrate_canonical_smiles", ""),
            r.get("substrate_scaffold", ""),
            r.get("pathway_names", ""),
        )

        add_compound(
            pool,
            r.get("product_id", ""),
            r.get("product_name", ""),
            r.get("product_canonical_smiles", ""),
            r.get("product_scaffold", ""),
            r.get("pathway_names", ""),
        )

    return list(pool.values())


def compact_set(x):
    vals = sorted(v for v in x if v and str(v).lower() != "nan")
    return "|".join(vals)


def make_positive_rows(pos_df, split_col, split_scheme):
    rows = []

    for _, r in pos_df.iterrows():
        rows.append({
            "pair_id": r["positive_pair_id"],
            "split_scheme": split_scheme,
            "split": r[split_col],
            "label": 1,
            "pair_type": "positive",
            "negative_type": "positive",
            "negative_subtype": "positive",
            "reaction_id": r.get("reaction_id", ""),
            "reaction_name": r.get("reaction_name", ""),
            "substrate_id": r.get("substrate_id", ""),
            "substrate_name": r.get("substrate_name", ""),
            "substrate_smiles": r.get("substrate_canonical_smiles", ""),
            "substrate_scaffold": r.get("substrate_scaffold", ""),
            "product_id": r.get("product_id", ""),
            "product_name": r.get("product_name", ""),
            "product_smiles": r.get("product_canonical_smiles", ""),
            "product_scaffold": r.get("product_scaffold", ""),
            "pathway_names": r.get("pathway_names", ""),
            "source_positive_pair_id": r.get("positive_pair_id", ""),
        })

    return rows


def index_by_pathway(compounds):
    pathway_to_compounds = defaultdict(list)

    for c in compounds:
        for p in c["pathways"]:
            pathway_to_compounds[p].append(c)

    return {
        p: xs
        for p, xs in pathway_to_compounds.items()
        if len(xs) >= 2
    }


def index_by_scaffold(compounds):
    scaffold_to_compounds = defaultdict(list)

    for c in compounds:
        for s in c["scaffolds"]:
            if s:
                scaffold_to_compounds[s].append(c)

    return {
        s: xs
        for s, xs in scaffold_to_compounds.items()
        if len(xs) >= 2
    }


def sample_random_negative(
    compounds,
    known_positive_undirected,
    used_negative_directed,
):
    if len(compounds) < 2:
        return None

    max_attempts = max(1000, len(compounds) * MAX_ATTEMPTS_FACTOR)

    for _ in range(max_attempts):
        a, b = random.sample(compounds, 2)

        a_smiles = a["canonical_smiles"]
        b_smiles = b["canonical_smiles"]

        if a_smiles == b_smiles:
            continue

        # Prefer random negatives from different pathway contexts.
        if a["pathways"] and b["pathways"] and (a["pathways"] & b["pathways"]):
            continue

        ukey = undirected_key(a_smiles, b_smiles)
        dkey = directed_key(a_smiles, b_smiles)

        if ukey in known_positive_undirected:
            continue
        if dkey in used_negative_directed:
            continue

        return a, b, "random_different_pathway"

    # Fallback: any non-positive pair.
    for _ in range(max_attempts):
        a, b = random.sample(compounds, 2)

        a_smiles = a["canonical_smiles"]
        b_smiles = b["canonical_smiles"]

        if a_smiles == b_smiles:
            continue

        ukey = undirected_key(a_smiles, b_smiles)
        dkey = directed_key(a_smiles, b_smiles)

        if ukey in known_positive_undirected:
            continue
        if dkey in used_negative_directed:
            continue

        return a, b, "random_any_nonpositive"

    return None


def sample_hard_negative(
    compounds,
    pathway_index,
    scaffold_index,
    known_positive_undirected,
    used_negative_directed,
):
    max_attempts = max(1000, len(compounds) * MAX_ATTEMPTS_FACTOR)

    # First choice: same pathway, not a known reaction neighbor.
    pathways = list(pathway_index.keys())
    if pathways:
        for _ in range(max_attempts):
            p = random.choice(pathways)
            group = pathway_index[p]

            if len(group) < 2:
                continue

            a, b = random.sample(group, 2)
            a_smiles = a["canonical_smiles"]
            b_smiles = b["canonical_smiles"]

            if a_smiles == b_smiles:
                continue

            ukey = undirected_key(a_smiles, b_smiles)
            dkey = directed_key(a_smiles, b_smiles)

            if ukey in known_positive_undirected:
                continue
            if dkey in used_negative_directed:
                continue

            return a, b, "hard_same_pathway", p

    # Fallback: same scaffold, not a known reaction neighbor.
    scaffolds = list(scaffold_index.keys())
    if scaffolds:
        for _ in range(max_attempts):
            s = random.choice(scaffolds)
            group = scaffold_index[s]

            if len(group) < 2:
                continue

            a, b = random.sample(group, 2)
            a_smiles = a["canonical_smiles"]
            b_smiles = b["canonical_smiles"]

            if a_smiles == b_smiles:
                continue

            ukey = undirected_key(a_smiles, b_smiles)
            dkey = directed_key(a_smiles, b_smiles)

            if ukey in known_positive_undirected:
                continue
            if dkey in used_negative_directed:
                continue

            return a, b, "hard_same_scaffold", s

    return None


def negative_row(pair_id, split_scheme, split, a, b, negative_type, negative_subtype, context):
    return {
        "pair_id": pair_id,
        "split_scheme": split_scheme,
        "split": split,
        "label": 0,
        "pair_type": "negative",
        "negative_type": negative_type,
        "negative_subtype": negative_subtype,
        "reaction_id": "",
        "reaction_name": "",
        "substrate_id": compact_set(a["compound_ids"]),
        "substrate_name": compact_set(a["compound_names"]),
        "substrate_smiles": a["canonical_smiles"],
        "substrate_scaffold": compact_set(a["scaffolds"]),
        "product_id": compact_set(b["compound_ids"]),
        "product_name": compact_set(b["compound_names"]),
        "product_smiles": b["canonical_smiles"],
        "product_scaffold": compact_set(b["scaffolds"]),
        "pathway_names": compact_set(a["pathways"] & b["pathways"]),
        "negative_context": context,
        "source_positive_pair_id": "",
    }


def generate_for_split_scheme(pos, split_col, split_scheme):
    # Global known positives, so negatives are not any known positive pair anywhere.
    known_positive_undirected = set(
        undirected_key(a, b)
        for a, b in zip(pos["substrate_canonical_smiles"], pos["product_canonical_smiles"])
    )

    all_rows = []
    all_rows.extend(make_positive_rows(pos, split_col, split_scheme))

    used_negative_directed = set()
    neg_counter = 1
    qc_rows = []

    for split in sorted(pos[split_col].dropna().unique()):
        sub = pos[pos[split_col] == split].copy()
        n_pos = len(sub)

        compounds = build_compound_pool(sub)
        pathway_index = index_by_pathway(compounds)
        scaffold_index = index_by_scaffold(compounds)

        target_random = n_pos * RANDOM_NEG_PER_POS
        target_hard = n_pos * HARD_NEG_PER_POS

        made_random = 0
        made_hard = 0

        # Random negatives.
        for _ in range(target_random):
            sampled = sample_random_negative(
                compounds,
                known_positive_undirected,
                used_negative_directed,
            )
            if sampled is None:
                break

            a, b, subtype = sampled
            dkey = directed_key(a["canonical_smiles"], b["canonical_smiles"])
            used_negative_directed.add(dkey)

            pair_id = f"PCYC_NEG_{split_scheme}_R_{neg_counter:08d}"
            neg_counter += 1

            all_rows.append(
                negative_row(
                    pair_id=pair_id,
                    split_scheme=split_scheme,
                    split=split,
                    a=a,
                    b=b,
                    negative_type="random_negative",
                    negative_subtype=subtype,
                    context="",
                )
            )
            made_random += 1

        # Hard negatives.
        for _ in range(target_hard):
            sampled = sample_hard_negative(
                compounds,
                pathway_index,
                scaffold_index,
                known_positive_undirected,
                used_negative_directed,
            )
            if sampled is None:
                break

            a, b, subtype, context = sampled
            dkey = directed_key(a["canonical_smiles"], b["canonical_smiles"])
            used_negative_directed.add(dkey)

            pair_id = f"PCYC_NEG_{split_scheme}_H_{neg_counter:08d}"
            neg_counter += 1

            all_rows.append(
                negative_row(
                    pair_id=pair_id,
                    split_scheme=split_scheme,
                    split=split,
                    a=a,
                    b=b,
                    negative_type="hard_negative",
                    negative_subtype=subtype,
                    context=context,
                )
            )
            made_hard += 1

        qc_rows.append({
            "split_scheme": split_scheme,
            "split": split,
            "positive_pairs": n_pos,
            "unique_compounds_in_split": len(compounds),
            "target_random_negatives": target_random,
            "made_random_negatives": made_random,
            "target_hard_negatives": target_hard,
            "made_hard_negatives": made_hard,
            "total_rows_expected_if_full": n_pos + target_random + target_hard,
        })

    out = pd.DataFrame(all_rows)
    qc = pd.DataFrame(qc_rows)

    return out, qc


def write_qc(out, qc, split_scheme):
    qc_file = QC_DIR / f"plantcyc_reaction_neighbor_{split_scheme}_negative_sampling_qc.txt"

    with open(qc_file, "w") as f:
        f.write(f"PlantCyc reaction-neighbor negative sampling QC: {split_scheme}\n")
        f.write("=" * 70 + "\n\n")

        f.write("Sampling settings:\n")
        f.write(f"RANDOM_NEG_PER_POS: {RANDOM_NEG_PER_POS}\n")
        f.write(f"HARD_NEG_PER_POS: {HARD_NEG_PER_POS}\n\n")

        f.write("Rows by label:\n")
        f.write(str(out["label"].value_counts()) + "\n\n")

        f.write("Rows by split and label:\n")
        f.write(str(out.groupby(["split", "label"]).size()) + "\n\n")

        f.write("Rows by negative_type:\n")
        f.write(str(out["negative_type"].value_counts()) + "\n\n")

        f.write("Rows by negative_subtype:\n")
        f.write(str(out["negative_subtype"].value_counts()) + "\n\n")

        f.write("Per-split sampling QC:\n")
        f.write(str(qc) + "\n")

    return qc_file


def main():
    pos = pd.read_csv(IN_FILE, sep="\t", dtype=str)

    print(f"Input positive pairs: {len(pos):,}")

    outputs = [
        ("random_split", "random"),
        ("scaffold_pair_split", "scaffold_pair"),
    ]

    for split_col, split_scheme in outputs:
        print(f"\nGenerating negatives for split scheme: {split_scheme}")

        out, qc = generate_for_split_scheme(
            pos=pos,
            split_col=split_col,
            split_scheme=split_scheme,
        )

        out_file = OUT_DIR / f"plantcyc_reaction_neighbor_pairs_{split_scheme}.tsv"
        qc_table = QC_DIR / f"plantcyc_reaction_neighbor_{split_scheme}_negative_sampling_qc.tsv"

        out.to_csv(out_file, sep="\t", index=False)
        qc.to_csv(qc_table, sep="\t", index=False)

        qc_txt = write_qc(out, qc, split_scheme)

        print("Done.")
        print(out["label"].value_counts())
        print(out["negative_type"].value_counts())
        print(f"Wrote: {out_file}")
        print(f"Wrote: {qc_table}")
        print(f"Wrote: {qc_txt}")


if __name__ == "__main__":
    main()
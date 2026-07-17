#!/usr/bin/env python3
"""Generate balanced PlantCyc reaction-neighbor benchmark datasets.

For each split scheme (random and scaffold-pair), this script creates three
separate 1:1 binary classification datasets with identical positive examples:

1. easy:  one easy negative per positive
2. hard:  one hard negative per positive
3. mixed: one negative per positive, with an approximately 50:50 mixture of
          easy and hard negatives (exactly balanced globally whenever the
          number of positives is even)

Negative definitions
--------------------
Easy negatives:
    two compounds with no known shared PlantCyc pathway annotation that are
    not a known positive reaction-neighbor pair. Compounds lacking pathway
    annotation are allowed, and this limitation is made explicit in the
    negative subtype definition.

Hard negatives:
    first preference: same pathway but not a known positive pair;
    fallback: same Murcko scaffold but not a known positive pair.

The easy and hard negative pools are generated once per split scheme and are
mutually exclusive. The mixed dataset is then constructed as a deterministic
subset of these two pools, which makes comparisons among the three benchmark
variants easier to interpret.

Integrity guarantees
--------------------
- one row per undirected canonical-SMILES pair within every output dataset;
- no self-pairs;
- no sampled negative matches any known positive pair;
- no undirected pair crosses train/validation/test splits;
- no positive/negative label conflicts;
- exact 1:1 positive:negative ratio within every split;
- for scaffold-pair outputs, no undirected scaffold-pair group crosses splits.

The obsolete 1:3 output files and their QC files are deleted automatically.
All six balanced output files are also cleared before regeneration so old and
new benchmark files cannot be confused.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random

import pandas as pd


RANDOM_SEED = 63
MAX_ATTEMPTS_FACTOR = 300

IN_FILE = Path(
    "data/tasks/plantcyc_reaction_neighbor/plantcyc_positive_pairs_with_splits.tsv"
)
OUT_DIR = Path("data/tasks/plantcyc_reaction_neighbor")
QC_DIR = Path("qc/data_processing")

OUT_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_SCHEMES = {
    "random": {
        "split_col": "random_split",
        "split_order": ["train", "val", "test"],
        "seed_offset": 0,
    },
    "scaffold_pair": {
        "split_col": "scaffold_pair_split",
        "split_order": ["scaffold_train", "scaffold_val", "scaffold_test"],
        "seed_offset": 1000,
    },
}

VARIANTS = ("easy", "hard", "mixed")


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def split_tokens(value) -> set[str]:
    if pd.isna(value):
        return set()
    return {
        token.strip()
        for token in str(value).split("|")
        if token.strip() and token.strip().lower() != "nan"
    }


def compact_set(values) -> str:
    return "|".join(sorted(v for v in values if v and str(v).lower() != "nan"))


def undirected_key(a: str, b: str) -> str:
    return "||".join(sorted([str(a), str(b)]))


def scaffold_pair_key(a: dict, b: dict) -> str:
    return undirected_key(a["scaffold"], b["scaffold"])


def canonical_order(a: dict, b: dict) -> tuple[dict, dict]:
    return (a, b) if a["canonical_smiles"] <= b["canonical_smiles"] else (b, a)


# -----------------------------------------------------------------------------
# Compound pool construction
# -----------------------------------------------------------------------------

def add_compound(pool: dict, cid, name, smiles, scaffold, pathway_names) -> None:
    if pd.isna(smiles) or not str(smiles).strip():
        return

    smiles = str(smiles).strip()
    scaffold = "" if pd.isna(scaffold) else str(scaffold).strip()

    entry = pool.setdefault(
        smiles,
        {
            "canonical_smiles": smiles,
            "compound_ids": set(),
            "compound_names": set(),
            "scaffold": scaffold,
            "pathways": set(),
        },
    )

    entry["compound_ids"].update(split_tokens(cid))
    entry["compound_names"].update(split_tokens(name))
    entry["pathways"].update(split_tokens(pathway_names))

    if scaffold:
        if entry["scaffold"] and entry["scaffold"] != scaffold:
            raise AssertionError(
                f"One canonical SMILES has inconsistent scaffolds: {smiles}"
            )
        entry["scaffold"] = scaffold


def build_compound_pool(pos_df: pd.DataFrame) -> list[dict]:
    pool: dict[str, dict] = {}

    for _, row in pos_df.iterrows():
        add_compound(
            pool,
            row.get("substrate_id", ""),
            row.get("substrate_name", ""),
            row.get("substrate_canonical_smiles", ""),
            row.get("substrate_scaffold", ""),
            row.get("pathway_names", ""),
        )
        add_compound(
            pool,
            row.get("product_id", ""),
            row.get("product_name", ""),
            row.get("product_canonical_smiles", ""),
            row.get("product_scaffold", ""),
            row.get("pathway_names", ""),
        )

    compounds = [pool[key] for key in sorted(pool)]
    missing_scaffold = [c["canonical_smiles"] for c in compounds if not c["scaffold"]]
    if missing_scaffold:
        raise AssertionError(
            f"Missing scaffold for {len(missing_scaffold)} compounds; first: "
            f"{missing_scaffold[:3]}"
        )

    return compounds


def index_by_pathway(compounds: list[dict]) -> dict[str, list[dict]]:
    index = defaultdict(list)
    for compound in compounds:
        for pathway in compound["pathways"]:
            index[pathway].append(compound)

    return {
        key: sorted(values, key=lambda x: x["canonical_smiles"])
        for key, values in index.items()
        if len(values) >= 2
    }


def index_by_scaffold(compounds: list[dict]) -> dict[str, list[dict]]:
    index = defaultdict(list)
    for compound in compounds:
        index[compound["scaffold"]].append(compound)

    return {
        key: sorted(values, key=lambda x: x["canonical_smiles"])
        for key, values in index.items()
        if key and len(values) >= 2
    }


# -----------------------------------------------------------------------------
# Candidate validation and sampling
# -----------------------------------------------------------------------------

def candidate_is_allowed(
    a: dict,
    b: dict,
    split: str,
    split_scheme: str,
    known_positive_undirected: set[str],
    used_negative_undirected: set[str],
    scaffold_group_owner: dict[str, str],
) -> bool:
    a_smiles = a["canonical_smiles"]
    b_smiles = b["canonical_smiles"]

    if a_smiles == b_smiles:
        return False

    pair_key = undirected_key(a_smiles, b_smiles)
    if pair_key in known_positive_undirected:
        return False
    if pair_key in used_negative_undirected:
        return False

    if split_scheme == "scaffold_pair":
        group = scaffold_pair_key(a, b)
        owner = scaffold_group_owner.get(group)
        if owner is not None and owner != split:
            return False

    return True


def accept_candidate(
    a: dict,
    b: dict,
    split: str,
    split_scheme: str,
    used_negative_undirected: set[str],
    scaffold_group_owner: dict[str, str],
) -> None:
    pair_key = undirected_key(a["canonical_smiles"], b["canonical_smiles"])
    used_negative_undirected.add(pair_key)

    if split_scheme == "scaffold_pair":
        group = scaffold_pair_key(a, b)
        previous = scaffold_group_owner.setdefault(group, split)
        if previous != split:
            raise AssertionError("A scaffold-pair group was assigned to two splits.")


def sample_easy_negative(
    rng: random.Random,
    compounds: list[dict],
    split: str,
    split_scheme: str,
    known_positive_undirected: set[str],
    used_negative_undirected: set[str],
    scaffold_group_owner: dict[str, str],
):
    """Sample an easy negative with no known shared pathway annotation."""
    if len(compounds) < 2:
        return None

    max_attempts = max(2000, len(compounds) * MAX_ATTEMPTS_FACTOR)

    for _ in range(max_attempts):
        a, b = rng.sample(compounds, 2)

        # Reject pairs with a known shared pathway. If one or both compounds
        # lack pathway annotation, they are retained as having no known shared
        # pathway rather than being treated as confirmed different-pathway.
        if a["pathways"] and b["pathways"] and (a["pathways"] & b["pathways"]):
            continue

        if candidate_is_allowed(
            a,
            b,
            split,
            split_scheme,
            known_positive_undirected,
            used_negative_undirected,
            scaffold_group_owner,
        ):
            missing_context = not a["pathways"] or not b["pathways"]
            subtype = (
                "easy_no_known_shared_pathway_missing_annotation"
                if missing_context
                else "easy_different_annotated_pathway"
            )
            return a, b, subtype, ""

    return None


def sample_hard_negative(
    rng: random.Random,
    compounds: list[dict],
    pathway_index: dict[str, list[dict]],
    scaffold_index: dict[str, list[dict]],
    split: str,
    split_scheme: str,
    known_positive_undirected: set[str],
    used_negative_undirected: set[str],
    scaffold_group_owner: dict[str, str],
):
    """Sample same-pathway negatives, falling back to same-scaffold negatives."""
    max_attempts = max(2000, len(compounds) * MAX_ATTEMPTS_FACTOR)

    pathways = sorted(pathway_index)
    if pathways:
        for _ in range(max_attempts):
            pathway = rng.choice(pathways)
            group = pathway_index[pathway]
            a, b = rng.sample(group, 2)

            if candidate_is_allowed(
                a,
                b,
                split,
                split_scheme,
                known_positive_undirected,
                used_negative_undirected,
                scaffold_group_owner,
            ):
                return a, b, "hard_same_pathway", pathway

    scaffolds = sorted(scaffold_index)
    if scaffolds:
        for _ in range(max_attempts):
            scaffold = rng.choice(scaffolds)
            group = scaffold_index[scaffold]
            a, b = rng.sample(group, 2)

            if candidate_is_allowed(
                a,
                b,
                split,
                split_scheme,
                known_positive_undirected,
                used_negative_undirected,
                scaffold_group_owner,
            ):
                return a, b, "hard_same_scaffold", scaffold

    return None


# -----------------------------------------------------------------------------
# Output row construction
# -----------------------------------------------------------------------------

def make_positive_rows(
    pos: pd.DataFrame,
    split_col: str,
    split_scheme: str,
    dataset_variant: str,
) -> list[dict]:
    rows = []

    for _, row in pos.iterrows():
        pair_key = undirected_key(
            row["substrate_canonical_smiles"],
            row["product_canonical_smiles"],
        )
        scaffold_key = undirected_key(
            row["substrate_scaffold"],
            row["product_scaffold"],
        )

        rows.append(
            {
                "pair_id": row["positive_pair_id"],
                "dataset_variant": dataset_variant,
                "split_scheme": split_scheme,
                "split": row[split_col],
                "label": 1,
                "pair_type": "positive",
                "negative_type": "positive",
                "negative_subtype": "positive",
                "reaction_id": row.get("reaction_id", ""),
                "reaction_name": row.get("reaction_name", ""),
                "substrate_id": row.get("substrate_id", ""),
                "substrate_name": row.get("substrate_name", ""),
                "substrate_smiles": row["substrate_canonical_smiles"],
                "substrate_scaffold": row.get("substrate_scaffold", ""),
                "product_id": row.get("product_id", ""),
                "product_name": row.get("product_name", ""),
                "product_smiles": row["product_canonical_smiles"],
                "product_scaffold": row.get("product_scaffold", ""),
                "pathway_names": row.get("pathway_names", ""),
                "source_positive_pair_id": row["positive_pair_id"],
                "negative_context": "",
                "undirected_pair_key": pair_key,
                "undirected_scaffold_pair_key": scaffold_key,
            }
        )

    return rows


def make_negative_row(
    pair_id: str,
    split_scheme: str,
    split: str,
    a: dict,
    b: dict,
    negative_type: str,
    negative_subtype: str,
    context: str,
) -> dict:
    a, b = canonical_order(a, b)
    pair_key = undirected_key(a["canonical_smiles"], b["canonical_smiles"])
    scaffold_key = scaffold_pair_key(a, b)

    return {
        "pair_id": pair_id,
        "dataset_variant": "POOL",
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
        "substrate_scaffold": a["scaffold"],
        "product_id": compact_set(b["compound_ids"]),
        "product_name": compact_set(b["compound_names"]),
        "product_smiles": b["canonical_smiles"],
        "product_scaffold": b["scaffold"],
        "pathway_names": compact_set(a["pathways"] & b["pathways"]),
        "source_positive_pair_id": "",
        "negative_context": context,
        "undirected_pair_key": pair_key,
        "undirected_scaffold_pair_key": scaffold_key,
    }


# -----------------------------------------------------------------------------
# Split/scaffold ownership and negative-pool generation
# -----------------------------------------------------------------------------

def initialize_scaffold_group_owner(
    pos: pd.DataFrame,
    split_col: str,
    split_scheme: str,
) -> dict[str, str]:
    if split_scheme != "scaffold_pair":
        return {}

    owner: dict[str, str] = {}
    for _, row in pos.iterrows():
        group = undirected_key(row["substrate_scaffold"], row["product_scaffold"])
        split = row[split_col]
        previous = owner.setdefault(group, split)
        if previous != split:
            raise AssertionError(
                f"Positive scaffold-pair group crosses splits: {group}"
            )

    return owner


def observed_split_order(
    pos: pd.DataFrame,
    split_col: str,
    preferred_order: list[str],
) -> list[str]:
    observed = set(pos[split_col].dropna().unique())
    order = [split for split in preferred_order if split in observed]
    order += sorted(observed - set(order))
    return order


def generate_negative_pools(
    pos: pd.DataFrame,
    split_col: str,
    split_scheme: str,
    split_order: list[str],
    seed_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one easy and one hard negative per positive.

    A single global undirected registry is shared by both pools, making the easy
    and hard pools mutually exclusive within a split scheme.
    """
    known_positive_undirected = {
        undirected_key(a, b)
        for a, b in zip(
            pos["substrate_canonical_smiles"],
            pos["product_canonical_smiles"],
        )
    }
    if len(known_positive_undirected) != len(pos):
        raise AssertionError("The positive input is not unique by undirected pair.")

    scaffold_group_owner = initialize_scaffold_group_owner(
        pos,
        split_col,
        split_scheme,
    )
    used_negative_undirected: set[str] = set()
    negative_rows: list[dict] = []
    qc_rows: list[dict] = []

    rng = random.Random(RANDOM_SEED + seed_offset)
    easy_counter = 1
    hard_counter = 1

    for split in split_order:
        sub = pos.loc[pos[split_col] == split].copy()
        n_pos = len(sub)
        compounds = build_compound_pool(sub)
        pathway_index = index_by_pathway(compounds)
        scaffold_index = index_by_scaffold(compounds)

        made_easy = 0
        made_hard = 0

        # Generate easy negatives first, matching the previously successful
        # split-local sampling order; one hard negative per positive is then
        # sampled from the remaining undirected candidate pairs.
        for _ in range(n_pos):
            sampled = sample_easy_negative(
                rng,
                compounds,
                split,
                split_scheme,
                known_positive_undirected,
                used_negative_undirected,
                scaffold_group_owner,
            )
            if sampled is None:
                raise RuntimeError(
                    f"Could not generate all easy negatives for "
                    f"{split_scheme}/{split}: made {made_easy} of {n_pos}."
                )

            a, b, subtype, context = sampled
            accept_candidate(
                a,
                b,
                split,
                split_scheme,
                used_negative_undirected,
                scaffold_group_owner,
            )
            negative_rows.append(
                make_negative_row(
                    pair_id=(
                        f"PCYC_NEG_{split_scheme}_E_{easy_counter:08d}"
                    ),
                    split_scheme=split_scheme,
                    split=split,
                    a=a,
                    b=b,
                    negative_type="easy_negative",
                    negative_subtype=subtype,
                    context=context,
                )
            )
            easy_counter += 1
            made_easy += 1

        for _ in range(n_pos):
            sampled = sample_hard_negative(
                rng,
                compounds,
                pathway_index,
                scaffold_index,
                split,
                split_scheme,
                known_positive_undirected,
                used_negative_undirected,
                scaffold_group_owner,
            )
            if sampled is None:
                raise RuntimeError(
                    f"Could not generate all hard negatives for "
                    f"{split_scheme}/{split}: made {made_hard} of {n_pos}."
                )

            a, b, subtype, context = sampled
            accept_candidate(
                a,
                b,
                split,
                split_scheme,
                used_negative_undirected,
                scaffold_group_owner,
            )
            negative_rows.append(
                make_negative_row(
                    pair_id=(
                        f"PCYC_NEG_{split_scheme}_H_{hard_counter:08d}"
                    ),
                    split_scheme=split_scheme,
                    split=split,
                    a=a,
                    b=b,
                    negative_type="hard_negative",
                    negative_subtype=subtype,
                    context=context,
                )
            )
            hard_counter += 1
            made_hard += 1

        qc_rows.append(
            {
                "split_scheme": split_scheme,
                "split": split,
                "positive_pairs": n_pos,
                "unique_compounds_in_split": len(compounds),
                "made_easy_negatives": made_easy,
                "made_hard_negatives": made_hard,
                "total_negative_pool_rows": made_easy + made_hard,
            }
        )

    pools = pd.DataFrame(negative_rows)
    qc = pd.DataFrame(qc_rows)

    if pools["undirected_pair_key"].duplicated().any():
        raise AssertionError("Easy and hard negative pools are not mutually unique.")

    return pools, qc


# -----------------------------------------------------------------------------
# Balanced variant construction
# -----------------------------------------------------------------------------

def allocate_mixed_targets(
    pos: pd.DataFrame,
    split_col: str,
    split_order: list[str],
) -> dict[str, dict[str, int]]:
    """Allocate an exact global 50:50 easy/hard mixture when possible.

    Every split receives exactly one negative per positive. Odd-sized splits
    differ by at most one between easy and hard negatives.
    """
    n_by_split = {
        split: int((pos[split_col] == split).sum())
        for split in split_order
    }
    total = sum(n_by_split.values())
    target_easy_total = total // 2

    targets = {
        split: {
            "easy": n // 2,
            "hard": n - (n // 2),
        }
        for split, n in n_by_split.items()
    }

    current_easy = sum(x["easy"] for x in targets.values())
    need_extra_easy = target_easy_total - current_easy

    # Move one item from hard to easy in deterministic split order as needed.
    for split in split_order:
        if need_extra_easy <= 0:
            break
        n = n_by_split[split]
        if n % 2 == 1:
            targets[split]["easy"] += 1
            targets[split]["hard"] -= 1
            need_extra_easy -= 1

    if need_extra_easy != 0:
        raise AssertionError("Could not allocate mixed easy/hard targets.")

    if sum(x["easy"] + x["hard"] for x in targets.values()) != total:
        raise AssertionError("Mixed-target allocation does not match total positives.")

    return targets


def deterministic_subset(
    df: pd.DataFrame,
    n: int,
    seed: int,
) -> pd.DataFrame:
    if n < 0 or n > len(df):
        raise ValueError(f"Invalid subset size {n} for {len(df)} available rows.")
    if n == len(df):
        return df.copy()
    if n == 0:
        return df.iloc[0:0].copy()
    return df.sample(n=n, random_state=seed, replace=False)


def build_variant_dataset(
    pos: pd.DataFrame,
    negative_pools: pd.DataFrame,
    split_col: str,
    split_scheme: str,
    split_order: list[str],
    variant: str,
    seed_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]] | None]:
    positive_rows = pd.DataFrame(
        make_positive_rows(pos, split_col, split_scheme, variant)
    )

    mixed_targets = None
    selected_negative_parts = []
    summary_rows = []

    if variant == "mixed":
        mixed_targets = allocate_mixed_targets(pos, split_col, split_order)

    for split_index, split in enumerate(split_order):
        n_pos = int((pos[split_col] == split).sum())
        easy_pool = negative_pools.loc[
            (negative_pools["split"] == split)
            & (negative_pools["negative_type"] == "easy_negative")
        ].copy()
        hard_pool = negative_pools.loc[
            (negative_pools["split"] == split)
            & (negative_pools["negative_type"] == "hard_negative")
        ].copy()

        if len(easy_pool) != n_pos or len(hard_pool) != n_pos:
            raise AssertionError(
                f"Unexpected pool sizes for {split_scheme}/{split}: "
                f"positive={n_pos}, easy={len(easy_pool)}, hard={len(hard_pool)}"
            )

        if variant == "easy":
            selected_easy = easy_pool
            selected_hard = hard_pool.iloc[0:0]
        elif variant == "hard":
            selected_easy = easy_pool.iloc[0:0]
            selected_hard = hard_pool
        elif variant == "mixed":
            assert mixed_targets is not None
            n_easy = mixed_targets[split]["easy"]
            n_hard = mixed_targets[split]["hard"]
            selected_easy = deterministic_subset(
                easy_pool,
                n_easy,
                RANDOM_SEED + seed_offset + 10_000 + split_index,
            )
            selected_hard = deterministic_subset(
                hard_pool,
                n_hard,
                RANDOM_SEED + seed_offset + 20_000 + split_index,
            )
        else:
            raise ValueError(f"Unknown dataset variant: {variant}")

        selected_negative_parts.extend([selected_easy, selected_hard])
        summary_rows.append(
            {
                "dataset_variant": variant,
                "split_scheme": split_scheme,
                "split": split,
                "positive_pairs": n_pos,
                "easy_negatives": len(selected_easy),
                "hard_negatives": len(selected_hard),
                "total_negatives": len(selected_easy) + len(selected_hard),
                "total_rows": n_pos + len(selected_easy) + len(selected_hard),
            }
        )

    negatives = pd.concat(selected_negative_parts, ignore_index=True)
    negatives["dataset_variant"] = variant

    out = pd.concat([positive_rows, negatives], ignore_index=True)
    qc = pd.DataFrame(summary_rows)
    return out, qc, mixed_targets


# -----------------------------------------------------------------------------
# Integrity checks and output
# -----------------------------------------------------------------------------

def assert_final_integrity(
    out: pd.DataFrame,
    split_scheme: str,
    variant: str,
    mixed_targets: dict[str, dict[str, int]] | None,
) -> dict[str, int]:
    if out["undirected_pair_key"].duplicated().any():
        repeated = int(out["undirected_pair_key"].duplicated(keep=False).sum())
        raise AssertionError(f"Repeated undirected pairs remain: {repeated} rows.")

    if (out["substrate_smiles"] == out["product_smiles"]).any():
        raise AssertionError("Self-pairs remain in the final table.")

    if out.groupby("undirected_pair_key")["label"].nunique().max() != 1:
        raise AssertionError("At least one pair has conflicting labels.")

    if out.groupby("undirected_pair_key")["split"].nunique().max() != 1:
        raise AssertionError("At least one undirected pair crosses splits.")

    positives = set(out.loc[out["label"] == 1, "undirected_pair_key"])
    negatives = set(out.loc[out["label"] == 0, "undirected_pair_key"])
    if positives & negatives:
        raise AssertionError("A sampled negative matches a known positive pair.")

    if set(out["dataset_variant"].unique()) != {variant}:
        raise AssertionError("dataset_variant column is inconsistent.")

    per_split = out.groupby(["split", "label"]).size().unstack(fill_value=0)
    for split, row in per_split.iterrows():
        n_pos = int(row.get(1, 0))
        n_neg = int(row.get(0, 0))
        if n_neg != n_pos:
            raise AssertionError(
                f"Dataset is not 1:1 in {split}: positives={n_pos}, negatives={n_neg}."
            )

        neg = out.loc[(out["split"] == split) & (out["label"] == 0)]
        n_easy = int((neg["negative_type"] == "easy_negative").sum())
        n_hard = int((neg["negative_type"] == "hard_negative").sum())

        if variant == "easy" and (n_easy != n_pos or n_hard != 0):
            raise AssertionError(f"Easy dataset composition is wrong in {split}.")
        if variant == "hard" and (n_hard != n_pos or n_easy != 0):
            raise AssertionError(f"Hard dataset composition is wrong in {split}.")
        if variant == "mixed":
            if mixed_targets is None:
                raise AssertionError("Missing mixed target allocation.")
            if n_easy != mixed_targets[split]["easy"]:
                raise AssertionError(f"Mixed easy count is wrong in {split}.")
            if n_hard != mixed_targets[split]["hard"]:
                raise AssertionError(f"Mixed hard count is wrong in {split}.")

    scaffold_cross = 0
    if split_scheme == "scaffold_pair":
        scaffold_cross = int(
            out.groupby("undirected_scaffold_pair_key")["split"]
            .nunique()
            .gt(1)
            .sum()
        )
        if scaffold_cross:
            raise AssertionError(
                f"{scaffold_cross} scaffold-pair groups cross scaffold splits."
            )

    negative_rows = out.loc[out["label"] == 0]
    return {
        "n_rows": len(out),
        "n_positive": int((out["label"] == 1).sum()),
        "n_negative": int((out["label"] == 0).sum()),
        "n_easy_negative": int(
            (negative_rows["negative_type"] == "easy_negative").sum()
        ),
        "n_hard_negative": int(
            (negative_rows["negative_type"] == "hard_negative").sum()
        ),
        "n_unique_pairs": out["undirected_pair_key"].nunique(),
        "n_cross_split_pairs": int(
            out.groupby("undirected_pair_key")["split"].nunique().gt(1).sum()
        ),
        "n_mixed_label_pairs": int(
            out.groupby("undirected_pair_key")["label"].nunique().gt(1).sum()
        ),
        "n_cross_split_scaffold_groups": scaffold_cross,
    }


def sort_output(out: pd.DataFrame, split_order: list[str]) -> pd.DataFrame:
    out = out.copy()
    out["_split_order"] = pd.Categorical(
        out["split"],
        categories=split_order,
        ordered=True,
    )
    return (
        out.sort_values(
            ["_split_order", "label", "negative_type", "pair_id"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .drop(columns="_split_order")
        .reset_index(drop=True)
    )


def write_qc(
    out: pd.DataFrame,
    qc: pd.DataFrame,
    audit: dict[str, int],
    split_scheme: str,
    variant: str,
) -> tuple[Path, Path]:
    qc_table = QC_DIR / (
        f"plantcyc_reaction_neighbor_{variant}_{split_scheme}_qc.tsv"
    )
    qc_text = QC_DIR / (
        f"plantcyc_reaction_neighbor_{variant}_{split_scheme}_qc.txt"
    )

    qc.to_csv(qc_table, sep="\t", index=False)

    with qc_text.open("w") as handle:
        handle.write(
            "PlantCyc reaction-neighbor balanced-dataset QC\n"
        )
        handle.write("=" * 72 + "\n\n")
        handle.write(f"dataset_variant: {variant}\n")
        handle.write(f"split_scheme: {split_scheme}\n")
        handle.write("positive:negative ratio: 1:1 within every split\n\n")
        handle.write("Rows by split and label:\n")
        handle.write(out.groupby(["split", "label"]).size().to_string() + "\n\n")
        handle.write("Rows by negative type:\n")
        handle.write(out["negative_type"].value_counts().to_string() + "\n\n")
        handle.write("Rows by negative subtype:\n")
        handle.write(out["negative_subtype"].value_counts().to_string() + "\n\n")
        handle.write("Per-split composition:\n")
        handle.write(qc.to_string(index=False) + "\n\n")
        handle.write("Final integrity audit:\n")
        for key, value in audit.items():
            handle.write(f"{key}: {value}\n")

    return qc_table, qc_text


def managed_output_paths() -> list[Path]:
    paths: list[Path] = []

    # Obsolete 1:3 outputs and QC files.
    for split_scheme in SPLIT_SCHEMES:
        paths.extend(
            [
                OUT_DIR / f"plantcyc_reaction_neighbor_pairs_{split_scheme}.tsv",
                QC_DIR
                / f"plantcyc_reaction_neighbor_{split_scheme}_negative_sampling_qc.tsv",
                QC_DIR
                / f"plantcyc_reaction_neighbor_{split_scheme}_negative_sampling_qc.txt",
            ]
        )

    # Balanced outputs managed by this script.
    for split_scheme in SPLIT_SCHEMES:
        for variant in VARIANTS:
            paths.extend(
                [
                    OUT_DIR
                    / f"plantcyc_reaction_neighbor_{variant}_{split_scheme}.tsv",
                    QC_DIR
                    / f"plantcyc_reaction_neighbor_{variant}_{split_scheme}_qc.tsv",
                    QC_DIR
                    / f"plantcyc_reaction_neighbor_{variant}_{split_scheme}_qc.txt",
                ]
            )

    return paths


def cleanup_managed_outputs() -> None:
    for path in managed_output_paths():
        if path.exists():
            path.unlink()
            print(f"Removed stale/obsolete file: {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    cleanup_managed_outputs()

    pos = pd.read_csv(IN_FILE, sep="\t", dtype=str)
    print(f"Input unique positive pairs: {len(pos):,}")

    required = {
        "positive_pair_id",
        "substrate_canonical_smiles",
        "product_canonical_smiles",
        "substrate_scaffold",
        "product_scaffold",
        "random_split",
        "scaffold_pair_split",
    }
    missing = sorted(required - set(pos.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    for split_scheme, config in SPLIT_SCHEMES.items():
        split_col = config["split_col"]
        split_order = observed_split_order(
            pos,
            split_col,
            config["split_order"],
        )

        print(f"\nGenerating easy and hard pools: {split_scheme}")
        negative_pools, pool_qc = generate_negative_pools(
            pos=pos,
            split_col=split_col,
            split_scheme=split_scheme,
            split_order=split_order,
            seed_offset=config["seed_offset"],
        )
        print(pool_qc.to_string(index=False))

        for variant in VARIANTS:
            out, qc, mixed_targets = build_variant_dataset(
                pos=pos,
                negative_pools=negative_pools,
                split_col=split_col,
                split_scheme=split_scheme,
                split_order=split_order,
                variant=variant,
                seed_offset=config["seed_offset"],
            )
            audit = assert_final_integrity(
                out=out,
                split_scheme=split_scheme,
                variant=variant,
                mixed_targets=mixed_targets,
            )
            out = sort_output(out, split_order)

            out_file = OUT_DIR / (
                f"plantcyc_reaction_neighbor_{variant}_{split_scheme}.tsv"
            )
            out.to_csv(out_file, sep="\t", index=False)
            qc_table, qc_text = write_qc(
                out=out,
                qc=qc,
                audit=audit,
                split_scheme=split_scheme,
                variant=variant,
            )

            print(f"\n{split_scheme}/{variant}")
            print(out.groupby(["split", "label"]).size())
            print("Negative composition:")
            print(
                out.loc[out["label"] == 0]
                .groupby(["split", "negative_type"])
                .size()
            )
            print("Integrity audit:", audit)
            print(f"Wrote: {out_file}")
            print(f"Wrote: {qc_table}")
            print(f"Wrote: {qc_text}")


if __name__ == "__main__":
    main()

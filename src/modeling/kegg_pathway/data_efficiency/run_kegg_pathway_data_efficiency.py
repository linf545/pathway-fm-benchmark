#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.warning")

SCRIPT_VERSION = "kegg_pathway_data_efficiency_repeated_splits_v4_val_tuned"

DEFAULT_FM_PATTERNS = [
    "chemberta_77m_mlm",
    "chemberta_100m_mlm",
    "molformer_c3_1_1b",
    "nafm_graph_mean",
    "molclr_gcn",
    "molclr_gin",
    "graphmvp_gin",
    "molebert_gin",
]

PRIMARY_METRICS = [
    "macro_auprc",
    "macro_auroc",
    "macro_f1",
    "micro_f1",
    "sample_f1",
]

ALL_METRICS = PRIMARY_METRICS + [
    "macro_precision",
    "macro_recall",
    "exact_match_accuracy",
    "hamming_loss",
    "mean_true_labels",
    "mean_predicted_labels",
]


def default_threshold_grid() -> List[float]:
    return [round(x, 2) for x in np.arange(0.2, 1.00, 0.2)]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Repeated-split data-efficiency analysis for the Bao C/KEGG pathway dataset "
            "(Bao Dataset C / KEGG, 11-label multilabel classification). "
            "Hyperparameter C and decision threshold are tuned on validation only; "
            "test metrics are never used for tuning."
        )
    )
    parser.add_argument(
        "--compound-file",
        default="data/processed_bao/C_kegg/compounds_master.tsv",
    )
    parser.add_argument(
        "--label-file",
        default=(
            "data/processed_bao/C_kegg/"
            "bao_compound_labels_multilabel.tsv"
        ),
    )
    parser.add_argument(
        "--embedding-dir",
        default="data/features/bao_plant_pathway/full",
    )
    parser.add_argument(
        "--split-dir",
        default="data/splits/kegg_pathway",
    )
    parser.add_argument(
        "--output-dir",
        default="results/kegg_pathway/data_efficiency",
    )
    parser.add_argument(
        "--split-mode",
        choices=["scaffold", "random", "both"],
        default="scaffold",
    )
    parser.add_argument(
        "--split-seeds",
        nargs="+",
        type=int,
        default=[2026, 2027, 2028, 2029, 2030],
    )
    parser.add_argument(
        "--sampling-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--n-per-label",
        nargs="+",
        type=int,
        default=[4, 8, 16, 24, 32, 64],
    )
    parser.add_argument(
        "--fm-patterns",
        nargs="+",
        default=DEFAULT_FM_PATTERNS,
    )
    parser.add_argument(
        "--skip-missing-fm",
        action="store_true",
        help=(
            "Skip FM patterns whose embedding/metadata files are absent. "
            "By default, missing FM files raise an error."
        ),
    )
    parser.add_argument(
        "--C-grid",
        nargs="+",
        type=float,
        default=[0.01, 0.1, 1.0],
        help="Logistic-regression C candidates. Selected using validation macro-AUPRC.",
    )
    parser.add_argument(
        "--threshold-grid",
        nargs="+",
        type=float,
        default=default_threshold_grid(),
        help=(
            "Candidate probability thresholds. The selected C is chosen first by "
            "validation macro-AUPRC; the threshold is then chosen by validation macro-F1."
        ),
    )
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--scaffold-candidates", type=int, default=200)
    parser.add_argument("--scaffold-size-weight", type=float, default=1.0)
    parser.add_argument("--scaffold-label-weight", type=float, default=0.25)
    parser.add_argument("--overwrite-splits", action="store_true")
    return parser.parse_args()


def requested_modes(split_mode: str) -> List[str]:
    if split_mode == "both":
        return ["scaffold", "random"]
    return [split_mode]


def label_number(column: str) -> int:
    match = re.match(r"label_(\d+)_", column)
    if not match:
        raise ValueError(f"Cannot parse label number from: {column}")
    return int(match.group(1))


def load_task(compound_file: str, label_file: str):
    compounds = pd.read_csv(compound_file, sep="\t", low_memory=False)
    labels = pd.read_csv(label_file, sep="\t", low_memory=False)

    required_compound_cols = {
        "internal_compound_id",
        "canonical_smiles",
        "inchikey",
        "murcko_scaffold",
    }
    missing = required_compound_cols - set(compounds.columns)
    if missing:
        raise ValueError(f"Missing compound columns: {sorted(missing)}")

    label_cols = sorted(
        [column for column in labels.columns if column.startswith("label_")],
        key=label_number,
    )
    if len(label_cols) != 11:
        raise ValueError(f"Expected 11 pathway labels, found {len(label_cols)}")

    compounds = compounds.copy()
    labels = labels.copy()
    compounds["internal_compound_id"] = compounds["internal_compound_id"].astype(str)
    labels["internal_compound_id"] = labels["internal_compound_id"].astype(str)
    compounds["inchikey"] = compounds["inchikey"].astype(str)

    task = compounds.merge(
        labels[["internal_compound_id"] + label_cols],
        on="internal_compound_id",
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    if len(task) != len(compounds):
        raise ValueError(
            f"Compound/label merge retained {len(task)} of {len(compounds)} compounds"
        )
    if task["internal_compound_id"].duplicated().any():
        raise ValueError("Duplicated internal_compound_id values")
    if task["inchikey"].duplicated().any():
        raise ValueError("Duplicated InChIKeys in Task 2 data")

    Y = task[label_cols].to_numpy(dtype=int)
    cardinality = Y.sum(axis=1)
    if np.any(cardinality == 0):
        raise ValueError(f"{int((cardinality == 0).sum())} compounds have no labels")

    print("SCRIPT_VERSION:", SCRIPT_VERSION)
    print("Task compounds:", len(task))
    print("Pathway labels:", len(label_cols))
    print("Label cardinality:")
    print(pd.Series(cardinality).value_counts().sort_index().to_string())
    print("\nOverall positive counts:")
    for column, count in zip(label_cols, Y.sum(axis=0)):
        print(f"{column}: {int(count)}")

    return task, label_cols, Y


def scaffold_groups(task: pd.DataFrame) -> pd.Series:
    scaffold = task["murcko_scaffold"].fillna("").astype(str).str.strip().copy()
    empty = scaffold.eq("") | scaffold.str.lower().eq("nan")
    scaffold.loc[empty] = "NO_SCAFFOLD__" + task.loc[empty, "internal_compound_id"].astype(str)
    return scaffold


def split_file_path(split_dir: str, split_mode: str, split_seed: int) -> Path:
    return Path(split_dir) / f"kegg_pathway_{split_mode}_split_seed{split_seed}.tsv"


def assign_saved_split(task: pd.DataFrame, split_file: Path, split_mode: str, split_seed: int) -> pd.DataFrame:
    saved = pd.read_csv(split_file, sep="\t", low_memory=False)
    required = {"internal_compound_id", "split"}
    missing = required - set(saved.columns)
    if missing:
        raise ValueError(f"Existing {split_mode} split file lacks: {sorted(missing)}")

    saved = saved.copy()
    saved["internal_compound_id"] = saved["internal_compound_id"].astype(str)
    if saved["internal_compound_id"].duplicated().any():
        raise ValueError(f"Existing {split_mode} split contains duplicated compound IDs")

    result = task.copy()
    result["split"] = result["internal_compound_id"].map(
        saved.set_index("internal_compound_id")["split"]
    )
    if result["split"].isna().any():
        raise ValueError(
            f"Existing {split_mode} split does not cover "
            f"{int(result['split'].isna().sum())} Task 2 compounds"
        )

    allowed = {"train", "validation", "test"}
    observed = set(result["split"].astype(str).unique())
    unexpected = observed - allowed
    if unexpected:
        raise ValueError(f"Unexpected values in {split_mode} split: {sorted(unexpected)}")

    result["split_mode"] = split_mode
    result["split_seed"] = split_seed
    return result


def save_split(task: pd.DataFrame, split_file: Path, split_mode: str, split_seed: int):
    split_file.parent.mkdir(parents=True, exist_ok=True)
    out = task.copy()
    out["split_mode"] = split_mode
    out["split_seed"] = split_seed
    if "scaffold_group" not in out.columns:
        out["scaffold_group"] = scaffold_groups(out)
    cols = [
        "internal_compound_id",
        "inchikey",
        "canonical_smiles",
        "murcko_scaffold",
        "scaffold_group",
        "split",
        "split_mode",
        "split_seed",
    ]
    out[cols].to_csv(split_file, sep="\t", index=False)
    print("Wrote split:", split_file)


def split_qc_record(task: pd.DataFrame, Y: np.ndarray, label_cols: List[str], split_mode: str, split_seed: int):
    rows = []
    for split_name in ["train", "validation", "test"]:
        mask = task["split"].eq(split_name).to_numpy()
        counts = Y[mask].sum(axis=0)
        row = {
            "split_mode": split_mode,
            "split_seed": split_seed,
            "split": split_name,
            "n_compounds": int(mask.sum()),
            "mean_label_cardinality": float(Y[mask].sum(axis=1).mean()),
        }
        for col, count in zip(label_cols, counts):
            row[f"positive_count__{col}"] = int(count)
        rows.append(row)
    return rows


def validate_split(task: pd.DataFrame, Y: np.ndarray, label_cols: List[str], split_mode: str, split_seed: int, required_train_positives: int):
    print(f"\n{split_mode.capitalize()} split seed {split_seed} counts:")
    print(task["split"].value_counts().to_string())

    for split_name in ["train", "validation", "test"]:
        mask = task["split"].eq(split_name).to_numpy()
        positive_counts = Y[mask].sum(axis=0)
        zero_support = [label_cols[i] for i in range(len(label_cols)) if positive_counts[i] == 0]
        if zero_support:
            raise ValueError(
                f"{split_mode} seed {split_seed} {split_name} has labels with no positives: "
                + ", ".join(zero_support)
            )

    train_mask = task["split"].eq("train").to_numpy()
    train_counts = Y[train_mask].sum(axis=0)
    insufficient = [
        f"{label_cols[i]}={int(train_counts[i])}"
        for i in range(len(label_cols))
        if train_counts[i] < required_train_positives
    ]
    if insufficient:
        raise ValueError(
            f"{split_mode} seed {split_seed} training pool lacks enough positives for maximum N: "
            + ", ".join(insufficient)
        )


def create_random_split(task: pd.DataFrame, Y: np.ndarray, split_seed: int) -> pd.DataFrame:
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError as exc:
        raise ImportError(
            "Random split mode requires iterative-stratification. Install with: pip install iterative-stratification"
        ) from exc

    dummy_X = np.zeros((len(task), 1), dtype=np.uint8)
    first = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=split_seed)
    train_idx, holdout_idx = next(first.split(dummy_X, Y))

    second = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=split_seed + 1)
    validation_rel, test_rel = next(second.split(dummy_X[holdout_idx], Y[holdout_idx]))
    validation_idx = holdout_idx[validation_rel]
    test_idx = holdout_idx[test_rel]

    split = np.full(len(task), "", dtype=object)
    split[train_idx] = "train"
    split[validation_idx] = "validation"
    split[test_idx] = "test"
    if np.any(split == ""):
        raise RuntimeError("Some compounds were not assigned to a random split")

    result = task.copy()
    result["split"] = split
    result["scaffold_group"] = scaffold_groups(result)
    return result


def score_split(Y: np.ndarray, split: np.ndarray, target_fracs: Dict[str, float], size_weight: float, label_weight: float) -> Tuple[float, Dict[str, float]]:
    n_total = len(split)
    overall_prev = Y.mean(axis=0)
    size_penalty = 0.0
    label_penalty = 0.0
    for split_name in ["train", "validation", "test"]:
        mask = split == split_name
        frac = mask.sum() / n_total
        size_penalty += abs(frac - target_fracs[split_name])
        if mask.sum() == 0:
            label_penalty += 999.0
        else:
            prev = Y[mask].mean(axis=0)
            label_penalty += float(np.mean(np.abs(prev - overall_prev)))
    score = size_weight * size_penalty + label_weight * label_penalty
    return score, {"size_penalty": size_penalty, "label_penalty": label_penalty}


def valid_candidate(Y: np.ndarray, split: np.ndarray, required_train_positives: int) -> bool:
    for split_name in ["train", "validation", "test"]:
        mask = split == split_name
        if mask.sum() == 0:
            return False
        if np.any(Y[mask].sum(axis=0) == 0):
            return False
    train_mask = split == "train"
    if np.any(Y[train_mask].sum(axis=0) < required_train_positives):
        return False
    return True


def create_scaffold_split(
    task: pd.DataFrame,
    Y: np.ndarray,
    split_seed: int,
    required_train_positives: int,
    n_candidates: int,
    size_weight: float,
    label_weight: float,
) -> pd.DataFrame:
    scaffold = scaffold_groups(task)
    indices = np.arange(len(task))
    target_fracs = {"train": 0.80, "validation": 0.10, "test": 0.10}

    best = None
    best_score = np.inf
    best_details = None

    # Candidate search: use GroupShuffleSplit to generate scaffold-disjoint candidates,
    # then select the candidate closest to 80/10/10 with label prevalence as a weak penalty.
    first = GroupShuffleSplit(n_splits=n_candidates, test_size=0.20, random_state=split_seed)
    for candidate_id, (train_idx, holdout_idx) in enumerate(first.split(indices, groups=scaffold)):
        second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=split_seed * 100000 + candidate_id)
        validation_rel, test_rel = next(second.split(holdout_idx, groups=scaffold.iloc[holdout_idx]))
        validation_idx = holdout_idx[validation_rel]
        test_idx = holdout_idx[test_rel]

        split = np.full(len(task), "", dtype=object)
        split[train_idx] = "train"
        split[validation_idx] = "validation"
        split[test_idx] = "test"
        if np.any(split == ""):
            continue
        if not valid_candidate(Y, split, required_train_positives):
            continue

        score, details = score_split(Y, split, target_fracs, size_weight, label_weight)
        if score < best_score:
            best = split.copy()
            best_score = score
            best_details = details

    if best is None:
        raise RuntimeError(
            f"No valid scaffold split found for seed={split_seed}. Increase --scaffold-candidates."
        )

    result = task.copy()
    result["split"] = best
    result["scaffold_group"] = scaffold

    # Explicit scaffold leakage check.
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        groups_a = set(result.loc[result["split"].eq(a), "scaffold_group"])
        groups_b = set(result.loc[result["split"].eq(b), "scaffold_group"])
        overlap = groups_a & groups_b
        if overlap:
            raise RuntimeError(f"Scaffold leakage between {a} and {b}: {len(overlap)} groups")

    print(
        f"Selected scaffold split seed {split_seed}: score={best_score:.6f}, "
        f"size_penalty={best_details['size_penalty']:.6f}, "
        f"label_penalty={best_details['label_penalty']:.6f}"
    )
    return result


def create_or_load_split(
    task: pd.DataFrame,
    Y: np.ndarray,
    label_cols: List[str],
    split_mode: str,
    split_seed: int,
    args,
) -> pd.DataFrame:
    path = split_file_path(args.split_dir, split_mode, split_seed)
    if path.exists() and not args.overwrite_splits:
        print(f"\nLoading {split_mode} split seed {split_seed}: {path}")
        result = assign_saved_split(task, path, split_mode, split_seed)
    else:
        print(f"\nCreating {split_mode} split seed {split_seed}")
        if split_mode == "random":
            result = create_random_split(task, Y, split_seed)
        elif split_mode == "scaffold":
            result = create_scaffold_split(
                task=task,
                Y=Y,
                split_seed=split_seed,
                required_train_positives=max(args.n_per_label),
                n_candidates=args.scaffold_candidates,
                size_weight=args.scaffold_size_weight,
                label_weight=args.scaffold_label_weight,
            )
        else:
            raise ValueError(split_mode)
        save_split(result, path, split_mode, split_seed)

    result["split_mode"] = split_mode
    result["split_seed"] = split_seed
    if "scaffold_group" not in result.columns:
        result["scaffold_group"] = scaffold_groups(result)
    validate_split(result, Y, label_cols, split_mode, split_seed, max(args.n_per_label))
    return result


def make_nested_samples(
    task: pd.DataFrame,
    Y: np.ndarray,
    label_cols: List[str],
    n_values: Iterable[int],
    sampling_seeds: Iterable[int],
    split_mode: str,
    split_seed: int,
):
    train_pool = np.flatnonzero(task["split"].eq("train").to_numpy())
    samples = {}
    sample_rows = []

    for sampling_seed in sampling_seeds:
        rng = np.random.default_rng(sampling_seed)
        positive_orders = {}
        for label_idx in range(Y.shape[1]):
            candidates = train_pool[Y[train_pool, label_idx] == 1]
            positive_orders[label_idx] = rng.permutation(candidates)

        for n in sorted(n_values):
            selected = set()
            for label_idx in range(Y.shape[1]):
                selected.update(positive_orders[label_idx][:n].tolist())
            selected = np.asarray(sorted(selected), dtype=int)

            selected_Y = Y[selected]
            positive_counts = selected_Y.sum(axis=0)
            negative_counts = len(selected) - positive_counts
            if np.any(positive_counts < n):
                raise RuntimeError(
                    f"Sampling failed for {split_mode} split_seed={split_seed}, "
                    f"sampling_seed={sampling_seed}, N={n}"
                )
            if np.any(negative_counts == 0):
                failed = [label_cols[i] for i in range(len(label_cols)) if negative_counts[i] == 0]
                raise RuntimeError(
                    f"No negatives for {split_mode} split_seed={split_seed}, "
                    f"sampling_seed={sampling_seed}, N={n}: "
                    + ", ".join(failed)
                )

            samples[(sampling_seed, n)] = selected
            for row_idx in selected:
                active = [
                    str(label_number(label_cols[label_idx]))
                    for label_idx in range(len(label_cols))
                    if Y[row_idx, label_idx] == 1
                ]
                sample_rows.append(
                    {
                        "split_mode": split_mode,
                        "split_seed": split_seed,
                        "sampling_seed": sampling_seed,
                        "requested_n_per_label": n,
                        "task_row": int(row_idx),
                        "internal_compound_id": task.iloc[row_idx]["internal_compound_id"],
                        "inchikey": task.iloc[row_idx]["inchikey"],
                        "murcko_scaffold": task.iloc[row_idx]["murcko_scaffold"],
                        "label_classes": "|".join(active),
                    }
                )

            print(
                f"split={split_mode} split_seed={split_seed} sampling_seed={sampling_seed} N={n}: "
                f"n_train={len(selected)}, positive range={int(positive_counts.min())}-{int(positive_counts.max())}"
            )

    return samples, pd.DataFrame(sample_rows)


def make_morgan(task: pd.DataFrame, radius: int, n_bits: int):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    X = np.zeros((len(task), n_bits), dtype=np.uint8)
    for index, smiles in enumerate(task["canonical_smiles"]):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"Invalid SMILES at task row {index}: {smiles}")
        fingerprint = generator.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fingerprint, X[index])
        if (index + 1) % 1000 == 0:
            print(f"Morgan r{radius}: {index + 1}/{len(task)}")
    return X


def find_embedding_files(pattern: str, embedding_dir: str):
    embedding_dir = Path(embedding_dir)
    exact = embedding_dir / f"{pattern}.embeddings.npy"
    if exact.exists():
        embedding_path = exact
    else:
        matches = sorted(embedding_dir.glob(f"{pattern}*.embeddings.npy"))
        if not matches:
            raise FileNotFoundError(f"No embedding file for pattern: {pattern}")
        if len(matches) > 1:
            raise ValueError(f"Multiple embedding files for {pattern}: {matches}")
        embedding_path = matches[0]
    stem = embedding_path.name.replace(".embeddings.npy", "")
    metadata_path = embedding_dir / f"{stem}.metadata.tsv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    return stem, embedding_path, metadata_path


def load_aligned_embedding(task: pd.DataFrame, pattern: str, embedding_dir: str):
    name, embedding_path, metadata_path = find_embedding_files(pattern, embedding_dir)
    X_all = np.load(embedding_path, mmap_mode="r")
    metadata = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    if len(metadata) != X_all.shape[0]:
        raise ValueError(f"{name}: metadata and embedding row counts differ")
    metadata = metadata.copy()
    metadata["inchikey"] = metadata["inchikey"].astype(str)
    if metadata["inchikey"].duplicated().any():
        raise ValueError(f"{name}: duplicated metadata InChIKeys")
    metadata = metadata.reset_index().rename(columns={"index": "embedding_row"})
    merged = task[["inchikey"]].merge(
        metadata[["inchikey", "embedding_row"]],
        on="inchikey",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    missing = int(merged["embedding_row"].isna().sum())
    if missing:
        raise ValueError(f"{name}: missing embeddings for {missing} compounds")
    rows = merged["embedding_row"].to_numpy(dtype=int)
    X = np.asarray(X_all[rows], dtype=np.float32)
    print(f"{name}: aligned shape {X.shape}")
    return name, X


def build_model(feature_type: str, seed: int, C: float, max_iter: int):
    base = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
    )
    classifier = OneVsRestClassifier(base, n_jobs=-1)
    if feature_type == "dense_embedding":
        return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])
    if feature_type == "fingerprint":
        return classifier
    raise ValueError(f"Unknown feature type: {feature_type}")


def probability_metrics(y_true: np.ndarray, y_probability: np.ndarray):
    auprc_values = []
    auroc_values = []
    for label_idx in range(y_true.shape[1]):
        y_label = y_true[:, label_idx]
        p_label = y_probability[:, label_idx]
        valid = np.isfinite(p_label)
        y_label = y_label[valid]
        p_label = p_label[valid]
        if len(np.unique(y_label)) < 2:
            continue
        auprc_values.append(average_precision_score(y_label, p_label))
        auroc_values.append(roc_auc_score(y_label, p_label))
    return {
        "macro_auprc": float(np.mean(auprc_values)) if auprc_values else np.nan,
        "macro_auroc": float(np.mean(auroc_values)) if auroc_values else np.nan,
        "n_valid_auprc_labels": int(len(auprc_values)),
        "n_valid_auroc_labels": int(len(auroc_values)),
    }


def threshold_metrics(y_true: np.ndarray, y_probability: np.ndarray, threshold: float):
    y_pred = (y_probability >= threshold).astype(int)
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "sample_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "exact_match_accuracy": accuracy_score(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "mean_true_labels": float(y_true.sum(axis=1).mean()),
        "mean_predicted_labels": float(y_pred.sum(axis=1).mean()),
    }


def compute_metrics(y_true: np.ndarray, y_probability: np.ndarray, threshold: float):
    metrics = threshold_metrics(y_true, y_probability, threshold)
    metrics.update(probability_metrics(y_true, y_probability))
    return metrics


def tune_threshold(y_true: np.ndarray, y_probability: np.ndarray, threshold_grid: Iterable[float]):
    best_threshold = None
    best_metrics = None
    best_score = -np.inf
    best_cardinality_gap = np.inf
    true_cardinality = float(y_true.sum(axis=1).mean())

    for threshold in threshold_grid:
        metrics = threshold_metrics(y_true, y_probability, threshold)
        score = metrics["macro_f1"]
        cardinality_gap = abs(metrics["mean_predicted_labels"] - true_cardinality)
        if (score > best_score) or (np.isclose(score, best_score) and cardinality_gap < best_cardinality_gap):
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
            best_cardinality_gap = cardinality_gap

    return best_threshold, best_metrics


def fit_select_c_and_threshold(
    X: np.ndarray,
    Y: np.ndarray,
    train_idx: np.ndarray,
    validation_mask: np.ndarray,
    feature_type: str,
    seed: int,
    C_grid: Iterable[float],
    max_iter: int,
    threshold_grid: Iterable[float],
):
    y_val = Y[validation_mask]
    best = None
    best_score = -np.inf
    best_auroc = -np.inf

    for C in C_grid:
        model = build_model(feature_type, seed=seed, C=float(C), max_iter=max_iter)
        model.fit(X[train_idx], Y[train_idx])
        val_probability = np.asarray(model.predict_proba(X[validation_mask]), dtype=float)
        pmetrics = probability_metrics(y_val, val_probability)
        score = pmetrics["macro_auprc"]
        auroc = pmetrics["macro_auroc"]
        if (score > best_score) or (np.isclose(score, best_score) and auroc > best_auroc):
            best = {
                "model": model,
                "selected_C": float(C),
                "validation_probability": val_probability,
                "validation_macro_auprc_for_C": score,
                "validation_macro_auroc_for_C": auroc,
            }
            best_score = score
            best_auroc = auroc

    tuned_threshold, threshold_val_metrics = tune_threshold(
        y_val, best["validation_probability"], threshold_grid
    )
    best["tuned_threshold"] = tuned_threshold
    best["validation_best_macro_f1_at_threshold"] = threshold_val_metrics["macro_f1"]
    return best


def evaluate_feature(
    task: pd.DataFrame,
    Y: np.ndarray,
    X: np.ndarray,
    feature_name: str,
    feature_type: str,
    samples: Dict[Tuple[int, int], np.ndarray],
    n_values: Iterable[int],
    sampling_seeds: Iterable[int],
    split_mode: str,
    split_seed: int,
    C_grid: Iterable[float],
    max_iter: int,
    threshold_grid: Iterable[float],
):
    validation_mask = task["split"].eq("validation").to_numpy()
    eval_masks = {
        "validation": validation_mask,
        "test": task["split"].eq("test").to_numpy(),
    }

    rows = []
    for sampling_seed in sampling_seeds:
        for n in sorted(n_values):
            train_idx = samples[(sampling_seed, n)]
            selected = fit_select_c_and_threshold(
                X=X,
                Y=Y,
                train_idx=train_idx,
                validation_mask=validation_mask,
                feature_type=feature_type,
                seed=sampling_seed,
                C_grid=C_grid,
                max_iter=max_iter,
                threshold_grid=threshold_grid,
            )
            model = selected["model"]
            tuned_threshold = selected["tuned_threshold"]
            train_positive_counts = Y[train_idx].sum(axis=0)

            for eval_set, eval_mask in eval_masks.items():
                if eval_set == "validation":
                    probabilities = selected["validation_probability"]
                else:
                    probabilities = np.asarray(model.predict_proba(X[eval_mask]), dtype=float)

                metrics = compute_metrics(Y[eval_mask], probabilities, tuned_threshold)
                row = {
                    "task": "kegg_pathway_11label",
                    "feature": feature_name,
                    "feature_type": feature_type,
                    "split_mode": split_mode,
                    "split_seed": split_seed,
                    "eval_set": eval_set,
                    "sampling_seed": sampling_seed,
                    "requested_n_per_label": n,
                    "n_train": int(len(train_idx)),
                    "n_eval": int(eval_mask.sum()),
                    "n_labels": int(Y.shape[1]),
                    "min_train_positive_count": int(train_positive_counts.min()),
                    "max_train_positive_count": int(train_positive_counts.max()),
                    "classifier": "one_vs_rest_logistic_regression",
                    "C_grid": "|".join(str(float(c)) for c in C_grid),
                    "selected_C": selected["selected_C"],
                    "C_selection_metric": "validation_macro_auprc",
                    "validation_macro_auprc_for_selected_C": selected["validation_macro_auprc_for_C"],
                    "validation_macro_auroc_for_selected_C": selected["validation_macro_auroc_for_C"],
                    "threshold_strategy": "global_threshold_tuned_on_validation_macro_f1",
                    "threshold_grid": "|".join(str(float(t)) for t in threshold_grid),
                    "tuned_threshold": tuned_threshold,
                    "validation_best_macro_f1_at_tuned_threshold": selected[
                        "validation_best_macro_f1_at_threshold"
                    ],
                }
                row.update(metrics)
                rows.append(row)
                print(
                    f"{split_mode:8s} split_seed={split_seed} {feature_name:24s} "
                    f"N={n:2d} sampling_seed={sampling_seed} {eval_set:10s} "
                    f"C={selected['selected_C']} thr={tuned_threshold:.2f} "
                    f"AUPRC={metrics['macro_auprc']:.4f} AUROC={metrics['macro_auroc']:.4f} "
                    f"macro-F1={metrics['macro_f1']:.4f}"
                )
    return rows


def aggregate_per_split_means(runs: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "task",
        "feature",
        "feature_type",
        "split_mode",
        "split_seed",
        "eval_set",
        "requested_n_per_label",
        "n_labels",
        "classifier",
        "C_selection_metric",
        "threshold_strategy",
    ]
    value_cols = ALL_METRICS + [
        "n_train",
        "n_eval",
        "selected_C",
        "tuned_threshold",
        "validation_macro_auprc_for_selected_C",
        "validation_macro_auroc_for_selected_C",
        "validation_best_macro_f1_at_tuned_threshold",
        "n_valid_auprc_labels",
        "n_valid_auroc_labels",
    ]
    agg = {col: "mean" for col in value_cols if col in runs.columns}
    agg["sampling_seed"] = "nunique"
    per_split = runs.groupby(group_cols, dropna=False).agg(agg).reset_index()
    per_split = per_split.rename(columns={"sampling_seed": "n_sampling_replicates"})
    return per_split


def summarize_across_splits(per_split: pd.DataFrame, test_only: bool = True) -> pd.DataFrame:
    df = per_split.copy()
    if test_only:
        df = df[df["eval_set"] == "test"].copy()

    group_cols = [
        "task",
        "feature",
        "feature_type",
        "split_mode",
        "eval_set",
        "requested_n_per_label",
        "n_labels",
        "classifier",
        "C_selection_metric",
        "threshold_strategy",
    ]
    metric_cols = [col for col in PRIMARY_METRICS if col in df.columns]

    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_split_replicates"] = int(sub["split_seed"].nunique())
        for metric in metric_cols:
            values = sub[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_sd"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_se"] = row[f"{metric}_sd"] / np.sqrt(len(values)) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_specs(args, task: pd.DataFrame):
    specs = [
        (f"morgan_r2_{args.n_bits}", "fingerprint", ("morgan", 2)),
        (f"morgan_r3_{args.n_bits}", "fingerprint", ("morgan", 3)),
    ]
    for pattern in args.fm_patterns:
        try:
            name, _, _ = find_embedding_files(pattern, args.embedding_dir)
            specs.append((name, "dense_embedding", ("embedding", pattern)))
        except FileNotFoundError as exc:
            if args.skip_missing_fm:
                print(f"Skipping missing FM pattern: {pattern}")
                continue
            raise exc
    return specs


def main():
    args = parse_args()
    if any(n <= 0 for n in args.n_per_label):
        raise ValueError("All --n-per-label values must be positive")
    if any(C <= 0 for C in args.C_grid):
        raise ValueError("All --C-grid values must be positive")
    if not all(0.0 < t < 1.0 for t in args.threshold_grid):
        raise ValueError("All --threshold-grid values must be between 0 and 1")

    task, label_cols, Y = load_task(args.compound_file, args.label_file)

    experiment_sets = []
    split_qc_rows = []
    all_sample_rows = []
    for mode in requested_modes(args.split_mode):
        for split_seed in args.split_seeds:
            split_task = create_or_load_split(task, Y, label_cols, mode, split_seed, args)
            split_qc_rows.extend(split_qc_record(split_task, Y, label_cols, mode, split_seed))
            samples, sample_table = make_nested_samples(
                task=split_task,
                Y=Y,
                label_cols=label_cols,
                n_values=args.n_per_label,
                sampling_seeds=args.sampling_seeds,
                split_mode=mode,
                split_seed=split_seed,
            )
            all_sample_rows.append(sample_table)
            experiment_sets.append(
                {
                    "split_mode": mode,
                    "split_seed": split_seed,
                    "task": split_task,
                    "samples": samples,
                    "rows": [],
                }
            )

    feature_specs = build_feature_specs(args, task)

    for requested_name, feature_type, loader in feature_specs:
        print(f"\n=== {requested_name} ===")
        if loader[0] == "morgan":
            feature_name = requested_name
            X = make_morgan(task, loader[1], args.n_bits)
        else:
            feature_name, X = load_aligned_embedding(task, loader[1], args.embedding_dir)

        for exp in experiment_sets:
            rows = evaluate_feature(
                task=exp["task"],
                Y=Y,
                X=X,
                feature_name=feature_name,
                feature_type=feature_type,
                samples=exp["samples"],
                n_values=args.n_per_label,
                sampling_seeds=args.sampling_seeds,
                split_mode=exp["split_mode"],
                split_seed=exp["split_seed"],
                C_grid=args.C_grid,
                max_iter=args.max_iter,
                threshold_grid=args.threshold_grid,
            )
            exp["rows"].extend(rows)

        del X
        gc.collect()

    all_rows = []
    for exp in experiment_sets:
        all_rows.extend(exp["rows"])
    runs = pd.DataFrame(all_rows)
    per_split = aggregate_per_split_means(runs)
    main_stats = summarize_across_splits(per_split, test_only=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_out = output_dir / "kegg_pathway_data_efficiency_repeated_runs.tsv"
    per_split_out = output_dir / "kegg_pathway_data_efficiency_per_split_means.tsv"
    main_out = output_dir / "kegg_pathway_data_efficiency_main_statistics.tsv"
    samples_out = output_dir / "kegg_pathway_data_efficiency_repeated_samples.tsv"
    split_qc_out = output_dir / "kegg_pathway_data_efficiency_split_qc.tsv"

    runs.to_csv(runs_out, sep="\t", index=False)
    per_split.to_csv(per_split_out, sep="\t", index=False)
    main_stats.to_csv(main_out, sep="\t", index=False)
    pd.concat(all_sample_rows, ignore_index=True).to_csv(samples_out, sep="\t", index=False)
    pd.DataFrame(split_qc_rows).to_csv(split_qc_out, sep="\t", index=False)

    print("\nWrote:")
    print(runs_out)
    print(per_split_out)
    print(main_out)
    print(samples_out)
    print(split_qc_out)

    if not main_stats.empty:
        print("\n=== Main test statistics ===")
        display_cols = [
            "split_mode",
            "feature",
            "requested_n_per_label",
            "n_split_replicates",
            "macro_auprc_mean",
            "macro_auprc_sd",
            "macro_auprc_se",
            "macro_auroc_mean",
            "macro_f1_mean",
            "macro_f1_se",
        ]
        display_cols = [c for c in display_cols if c in main_stats.columns]
        print(main_stats.sort_values(["split_mode", "requested_n_per_label", "macro_auprc_mean"], ascending=[True, True, False])[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()

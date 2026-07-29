#!/usr/bin/env python

"""
Repeated-split frozen-representation benchmark for the Bao A/B plant pathway task.

Task definition
---------------
- Bao Dataset A: repeatedly split into training and validation partitions.
- Bao Dataset B: fixed external test set for every split replicate.
- 18-label multilabel plant secondary-metabolic pathway prediction.

Representations
---------------
- Precomputed Morgan radius-2 and radius-3 fingerprints aligned to the A/B task.
- Precomputed molecular foundation-model embeddings aligned by InChIKey.

Model selection
---------------
- Logistic-regression C is selected using validation macro-AUPRC.
- A single global decision threshold is selected using validation macro-F1.
- Bao B is never used for model or threshold selection.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

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


SCRIPT_VERSION = "bao_ab_plant_pathway_full_ensemble_repeated_splits_v1"
TASK_NAME = "bao_ab_plant_pathway_18label"
DATASET_A_NAME = "Bao_A_trainval"
DATASET_B_NAME = "Bao_B_external_test"
ID_COL = "task_compound_id"

DEFAULT_FULL_FM_PATTERNS = [
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


def default_threshold_grid() -> List[float]:
    # Reduced default for faster runs; users can provide a denser grid by CLI.
    return [0.2, 0.4, 0.6, 0.8]


def requested_modes(split_mode: str) -> List[str]:
    if split_mode == "both":
        return ["scaffold", "random"]
    return [split_mode]


def label_number(column: str) -> int:
    match = re.match(r"label_(\d+)_", column)
    if not match:
        raise ValueError(f"Cannot parse label number from: {column}")
    return int(match.group(1))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Repeated-split full-training analysis for the Bao A/B 18-label plant "
            "pathway task. Bao A is split into train/validation partitions and Bao B "
            "is retained as a fixed external test set."
        )
    )
    parser.add_argument(
        "--metadata-file",
        default="data/features/bao_plant_pathway/metadata.tsv",
        help="A/B task metadata aligned row-for-row with labels and precomputed fingerprints.",
    )
    parser.add_argument(
        "--labels-file",
        default="data/features/bao_plant_pathway/labels.npy",
    )
    parser.add_argument(
        "--label-columns-file",
        default="data/features/bao_plant_pathway/label_columns.txt",
    )
    parser.add_argument(
        "--morgan-r2-file",
        default="data/features/bao_plant_pathway/morgan_r2_2048.npz",
    )
    parser.add_argument(
        "--morgan-r3-file",
        default="data/features/bao_plant_pathway/morgan_r3_2048.npz",
    )
    parser.add_argument(
        "--embedding-dir",
        default="data/features/bao_plant_pathway/full",
    )
    parser.add_argument(
        "--split-dir",
        default="data/splits/pmn_secondary_pathway",
    )
    parser.add_argument(
        "--output-dir",
        default="results/pmn_secondary_pathway/full",
    )
    parser.add_argument(
        "--split-mode",
        choices=["scaffold", "random", "both"],
        default="both",
    )
    parser.add_argument(
        "--split-seeds",
        nargs="+",
        type=int,
        default=[2026, 2027, 2028, 2029, 2030],
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Fraction of Bao A assigned to validation for each replicate.",
    )
    parser.add_argument(
        "--fm-patterns",
        nargs="+",
        default=DEFAULT_FULL_FM_PATTERNS,
    )
    parser.add_argument(
        "--skip-missing-fm",
        action="store_true",
        help="Skip FM patterns whose embedding or metadata file is missing.",
    )
    parser.add_argument(
        "--C-grid",
        nargs="+",
        type=float,
        default=[0.01, 0.1,1],
        help="Logistic-regression C candidates selected by validation macro-AUPRC.",
    )
    parser.add_argument(
        "--threshold-grid",
        nargs="+",
        type=float,
        default=default_threshold_grid(),
        help="Candidate global thresholds selected by validation macro-F1.",
    )
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Parallel one-vs-rest label fits. Avoid -1 when other runs are active.",
    )
    parser.add_argument("--scaffold-candidates", type=int, default=200)
    parser.add_argument("--scaffold-size-weight", type=float, default=1.0)
    parser.add_argument("--scaffold-label-weight", type=float, default=0.25)
    parser.add_argument("--overwrite-splits", action="store_true")
    return parser.parse_args()


def load_task(metadata_file: str, labels_file: str, label_columns_file: str):
    metadata = pd.read_csv(metadata_file, sep="\t", low_memory=False)
    labels = np.load(labels_file).astype(int)
    with open(label_columns_file) as handle:
        label_cols = [line.strip() for line in handle if line.strip()]

    required = {
        ID_COL,
        "dataset_name",
        "canonical_smiles",
        "inchikey",
        "murcko_scaffold",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata lacks required columns: {sorted(missing)}")
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D label matrix, found shape {labels.shape}")
    if len(metadata) != labels.shape[0]:
        raise ValueError(
            f"Metadata/label row mismatch: metadata={len(metadata)}, labels={labels.shape[0]}"
        )
    if labels.shape[1] != len(label_cols):
        raise ValueError(
            f"Label matrix/name mismatch: matrix={labels.shape[1]}, names={len(label_cols)}"
        )
    if len(label_cols) != 18:
        raise ValueError(f"Expected 18 Bao A/B pathway labels, found {len(label_cols)}")

    parsed_order = sorted(label_cols, key=label_number)
    if parsed_order != label_cols:
        raise ValueError("label_columns.txt is not ordered by numeric label index")

    task = metadata.copy()
    task[ID_COL] = task[ID_COL].astype(str)
    task["dataset_name"] = task["dataset_name"].astype(str)
    task["inchikey"] = task["inchikey"].astype(str)

    if task[ID_COL].duplicated().any():
        raise ValueError(f"Duplicated {ID_COL} values")
    if task["inchikey"].duplicated().any():
        raise ValueError("Duplicated InChIKeys in the final A/B task")

    observed_datasets = set(task["dataset_name"].unique())
    expected_datasets = {DATASET_A_NAME, DATASET_B_NAME}
    if observed_datasets != expected_datasets:
        raise ValueError(
            f"Expected datasets {sorted(expected_datasets)}, found {sorted(observed_datasets)}"
        )

    cardinality = labels.sum(axis=1)
    if np.any(cardinality == 0):
        raise ValueError(f"{int((cardinality == 0).sum())} compounds have no pathway labels")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("Label matrix contains values other than 0 and 1")

    a_mask = task["dataset_name"].eq(DATASET_A_NAME).to_numpy()
    b_mask = task["dataset_name"].eq(DATASET_B_NAME).to_numpy()
    if not a_mask.any() or not b_mask.any():
        raise ValueError("Both Bao A and Bao B must be present")

    print("SCRIPT_VERSION:", SCRIPT_VERSION)
    print("Task:", TASK_NAME)
    print("Bao A compounds:", int(a_mask.sum()))
    print("Bao B external compounds:", int(b_mask.sum()))
    print("Pathway labels:", len(label_cols))
    print("Overall label cardinality:")
    print(pd.Series(cardinality).value_counts().sort_index().to_string())
    print("\nPositive counts in Bao A:")
    for column, count in zip(label_cols, labels[a_mask].sum(axis=0)):
        print(f"{column}: {int(count)}")
    print("\nPositive counts in Bao B external test:")
    for column, count in zip(label_cols, labels[b_mask].sum(axis=0)):
        print(f"{column}: {int(count)}")

    return task, label_cols, labels


def scaffold_groups(task: pd.DataFrame) -> pd.Series:
    scaffold = task["murcko_scaffold"].fillna("").astype(str).str.strip().copy()
    empty = scaffold.eq("") | scaffold.str.lower().isin({"nan", "none"})
    scaffold.loc[empty] = "NO_SCAFFOLD__" + task.loc[empty, ID_COL].astype(str)
    return scaffold


def split_file_path(split_dir: str, split_mode: str, split_seed: int) -> Path:
    return Path(split_dir) / f"pmn_secondary_pathway_{split_mode}_split_seed{split_seed}.tsv"


def assign_saved_split(
    task: pd.DataFrame,
    split_file: Path,
    split_mode: str,
    split_seed: int,
) -> pd.DataFrame:
    saved = pd.read_csv(split_file, sep="\t", low_memory=False)
    required = {ID_COL, "split"}
    missing = required - set(saved.columns)
    if missing:
        raise ValueError(f"Existing split file lacks columns: {sorted(missing)}")

    saved = saved.copy()
    saved[ID_COL] = saved[ID_COL].astype(str)
    if saved[ID_COL].duplicated().any():
        raise ValueError(f"Existing split contains duplicated {ID_COL} values")

    result = task.copy()
    result["split"] = result[ID_COL].map(saved.set_index(ID_COL)["split"])
    if result["split"].isna().any():
        raise ValueError(
            f"Existing split does not cover {int(result['split'].isna().sum())} task compounds"
        )

    allowed = {"train", "validation", "external_test"}
    observed = set(result["split"].astype(str).unique())
    unexpected = observed - allowed
    if unexpected:
        raise ValueError(f"Unexpected split values: {sorted(unexpected)}")

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
        ID_COL,
        "dataset_name",
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


def validate_split(
    task: pd.DataFrame,
    Y: np.ndarray,
    label_cols: List[str],
    split_mode: str,
    split_seed: int,
):
    a_mask = task["dataset_name"].eq(DATASET_A_NAME).to_numpy()
    b_mask = task["dataset_name"].eq(DATASET_B_NAME).to_numpy()

    if not np.all(task.loc[b_mask, "split"].eq("external_test")):
        raise ValueError("Every Bao B compound must remain in external_test")
    if task.loc[a_mask, "split"].eq("external_test").any():
        raise ValueError("Bao A compounds cannot be assigned to external_test")

    print(f"\n{split_mode.capitalize()} split seed {split_seed} counts:")
    print(task["split"].value_counts().to_string())

    for split_name in ["train", "validation", "external_test"]:
        mask = task["split"].eq(split_name).to_numpy()
        if not mask.any():
            raise ValueError(f"{split_mode} seed {split_seed}: empty {split_name} split")
        positive_counts = Y[mask].sum(axis=0)
        negative_counts = mask.sum() - positive_counts
        zero_support = [
            label_cols[i] for i in range(len(label_cols)) if positive_counts[i] == 0
        ]
        no_negatives = [
            label_cols[i] for i in range(len(label_cols)) if negative_counts[i] == 0
        ]

        # Train and validation must contain both classes for every label because they
        # are used for model fitting and hyperparameter/threshold selection. Bao B is
        # fixed by the source dataset and is allowed to lack support for a rare label;
        # threshold-independent metrics skip labels that contain only one class.
        if split_name in {"train", "validation"}:
            if zero_support:
                raise ValueError(
                    f"{split_mode} seed {split_seed} {split_name} has labels with no positives: "
                    + ", ".join(zero_support)
                )
            if no_negatives:
                raise ValueError(
                    f"{split_mode} seed {split_seed} {split_name} has labels with no negatives: "
                    + ", ".join(no_negatives)
                )
        elif zero_support or no_negatives:
            print(
                f"Warning: fixed Bao B external test has single-class labels; "
                f"zero-positive={zero_support}, zero-negative={no_negatives}. "
                "AUROC/AUPRC will exclude these labels."
            )


def split_qc_record(
    task: pd.DataFrame,
    Y: np.ndarray,
    label_cols: List[str],
    split_mode: str,
    split_seed: int,
):
    rows = []
    for split_name in ["train", "validation", "external_test"]:
        mask = task["split"].eq(split_name).to_numpy()
        counts = Y[mask].sum(axis=0)
        row = {
            "task": TASK_NAME,
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


def create_random_split(
    task: pd.DataFrame,
    Y: np.ndarray,
    split_seed: int,
    validation_fraction: float,
) -> pd.DataFrame:
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError as exc:
        raise ImportError(
            "Random split mode requires iterative-stratification. "
            "Install with: pip install iterative-stratification"
        ) from exc

    a_idx = np.flatnonzero(task["dataset_name"].eq(DATASET_A_NAME).to_numpy())
    b_idx = np.flatnonzero(task["dataset_name"].eq(DATASET_B_NAME).to_numpy())
    dummy_X = np.zeros((len(a_idx), 1), dtype=np.uint8)

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=split_seed,
    )
    train_rel, validation_rel = next(splitter.split(dummy_X, Y[a_idx]))

    split = np.full(len(task), "", dtype=object)
    split[a_idx[train_rel]] = "train"
    split[a_idx[validation_rel]] = "validation"
    split[b_idx] = "external_test"
    if np.any(split == ""):
        raise RuntimeError("Some compounds were not assigned to a random split")

    result = task.copy()
    result["split"] = split
    result["scaffold_group"] = scaffold_groups(result)
    return result


def candidate_is_valid(Y_a: np.ndarray, candidate: np.ndarray) -> bool:
    for split_name in ["train", "validation"]:
        mask = candidate == split_name
        if not mask.any():
            return False
        positive = Y_a[mask].sum(axis=0)
        negative = mask.sum() - positive
        if np.any(positive == 0) or np.any(negative == 0):
            return False
    return True


def score_ab_split(
    Y_a: np.ndarray,
    candidate: np.ndarray,
    validation_fraction: float,
    size_weight: float,
    label_weight: float,
) -> Tuple[float, Dict[str, float]]:
    validation_mask = candidate == "validation"
    train_mask = candidate == "train"

    observed_validation_fraction = float(validation_mask.mean())
    size_penalty = abs(observed_validation_fraction - validation_fraction)

    overall_prev = Y_a.mean(axis=0)
    train_prev = Y_a[train_mask].mean(axis=0)
    validation_prev = Y_a[validation_mask].mean(axis=0)
    label_penalty = float(
        np.mean(np.abs(train_prev - overall_prev))
        + np.mean(np.abs(validation_prev - overall_prev))
    )

    score = size_weight * size_penalty + label_weight * label_penalty
    return score, {
        "size_penalty": size_penalty,
        "label_penalty": label_penalty,
    }


def create_scaffold_split(
    task: pd.DataFrame,
    Y: np.ndarray,
    split_seed: int,
    validation_fraction: float,
    n_candidates: int,
    size_weight: float,
    label_weight: float,
) -> pd.DataFrame:
    a_idx = np.flatnonzero(task["dataset_name"].eq(DATASET_A_NAME).to_numpy())
    b_idx = np.flatnonzero(task["dataset_name"].eq(DATASET_B_NAME).to_numpy())
    all_scaffolds = scaffold_groups(task)
    a_scaffolds = all_scaffolds.iloc[a_idx].reset_index(drop=True)
    Y_a = Y[a_idx]
    relative_indices = np.arange(len(a_idx))

    best_candidate = None
    best_score = np.inf
    best_details = None

    splitter = GroupShuffleSplit(
        n_splits=n_candidates,
        test_size=validation_fraction,
        random_state=split_seed,
    )
    for train_rel, validation_rel in splitter.split(
        relative_indices,
        groups=a_scaffolds,
    ):
        candidate = np.full(len(a_idx), "", dtype=object)
        candidate[train_rel] = "train"
        candidate[validation_rel] = "validation"
        if np.any(candidate == "") or not candidate_is_valid(Y_a, candidate):
            continue

        score, details = score_ab_split(
            Y_a=Y_a,
            candidate=candidate,
            validation_fraction=validation_fraction,
            size_weight=size_weight,
            label_weight=label_weight,
        )
        if score < best_score:
            best_candidate = candidate.copy()
            best_score = score
            best_details = details

    if best_candidate is None:
        raise RuntimeError(
            f"No valid Bao A scaffold split found for seed={split_seed}. "
            "Increase --scaffold-candidates."
        )

    split = np.full(len(task), "", dtype=object)
    split[a_idx] = best_candidate
    split[b_idx] = "external_test"

    result = task.copy()
    result["split"] = split
    result["scaffold_group"] = all_scaffolds

    train_groups = set(result.loc[result["split"].eq("train"), "scaffold_group"])
    validation_groups = set(
        result.loc[result["split"].eq("validation"), "scaffold_group"]
    )
    overlap = train_groups & validation_groups
    if overlap:
        raise RuntimeError(
            f"Scaffold leakage between Bao A train and validation: {len(overlap)} groups"
        )

    print(
        f"Selected Bao A scaffold split seed {split_seed}: score={best_score:.6f}, "
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
            result = create_random_split(
                task=task,
                Y=Y,
                split_seed=split_seed,
                validation_fraction=args.validation_fraction,
            )
        elif split_mode == "scaffold":
            result = create_scaffold_split(
                task=task,
                Y=Y,
                split_seed=split_seed,
                validation_fraction=args.validation_fraction,
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
    validate_split(result, Y, label_cols, split_mode, split_seed)
    return result


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


def fm_pattern_available(pattern: str, embedding_dir: str) -> Tuple[bool, str]:
    try:
        find_embedding_files(pattern, embedding_dir)
    except (FileNotFoundError, ValueError) as exc:
        return False, str(exc)
    return True, "ok"


def load_aligned_embedding(task: pd.DataFrame, pattern: str, embedding_dir: str):
    name, embedding_path, metadata_path = find_embedding_files(pattern, embedding_dir)
    X_all = np.load(embedding_path, mmap_mode="r")
    metadata = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    if len(metadata) != X_all.shape[0]:
        raise ValueError(f"{name}: metadata and embedding row counts differ")
    if "inchikey" not in metadata.columns:
        raise ValueError(f"{name}: metadata lacks inchikey")

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
        raise ValueError(f"{name}: missing embeddings for {missing} task compounds")

    rows = merged["embedding_row"].to_numpy(dtype=int)
    X = np.asarray(X_all[rows], dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError(f"{name}: aligned embedding contains non-finite values")
    print(f"{name}: aligned shape {X.shape}")
    return name, X


def load_fingerprint(path: str, expected_rows: int, feature_name: str):
    feature_path = Path(path)
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    X = sparse.load_npz(feature_path).tocsr()
    if X.shape[0] != expected_rows:
        raise ValueError(
            f"{feature_name}: expected {expected_rows} rows, found {X.shape[0]}"
        )
    if X.shape[1] <= 0:
        raise ValueError(f"{feature_name}: invalid shape {X.shape}")
    print(f"{feature_name}: loaded shape {X.shape}")
    return X


def build_feature_specs(args):
    specs = [
        ("morgan_r2_2048", "fingerprint", ("fingerprint", args.morgan_r2_file)),
        ("morgan_r3_2048", "fingerprint", ("fingerprint", args.morgan_r3_file)),
    ]

    for pattern in args.fm_patterns:
        if args.skip_missing_fm:
            available, reason = fm_pattern_available(pattern, args.embedding_dir)
            if not available:
                print(f"Skipping FM pattern {pattern}: {reason}")
                continue
        specs.append((pattern, "dense_embedding", ("embedding", pattern)))

    if not specs:
        raise ValueError("No features selected")
    return specs


def load_feature_matrix(
    task: pd.DataFrame,
    args,
    requested_name: str,
    loader: Tuple[str, str],
):
    if loader[0] == "fingerprint":
        return requested_name, load_fingerprint(
            path=loader[1],
            expected_rows=len(task),
            feature_name=requested_name,
        )
    return load_aligned_embedding(task, loader[1], args.embedding_dir)


def build_model(
    feature_type: str,
    seed: int,
    C: float,
    max_iter: int,
    n_jobs: int,
):
    base = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
    )
    classifier = OneVsRestClassifier(base, n_jobs=n_jobs)
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
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
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


def tune_threshold(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold_grid: Iterable[float],
):
    best_threshold = None
    best_metrics = None
    best_score = -np.inf
    best_cardinality_gap = np.inf
    true_cardinality = float(y_true.sum(axis=1).mean())

    for threshold in threshold_grid:
        metrics = threshold_metrics(y_true, y_probability, threshold)
        score = metrics["macro_f1"]
        cardinality_gap = abs(metrics["mean_predicted_labels"] - true_cardinality)
        if (score > best_score) or (
            np.isclose(score, best_score) and cardinality_gap < best_cardinality_gap
        ):
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
            best_cardinality_gap = cardinality_gap

    return best_threshold, best_metrics


def fit_select_c_and_threshold(
    X,
    Y: np.ndarray,
    train_idx: np.ndarray,
    validation_mask: np.ndarray,
    feature_type: str,
    seed: int,
    C_grid: Iterable[float],
    max_iter: int,
    threshold_grid: Iterable[float],
    n_jobs: int,
):
    y_val = Y[validation_mask]
    best = None
    best_score = -np.inf
    best_auroc = -np.inf

    for C in C_grid:
        model = build_model(
            feature_type,
            seed=seed,
            C=float(C),
            max_iter=max_iter,
            n_jobs=n_jobs,
        )
        model.fit(X[train_idx], Y[train_idx])
        val_probability = np.asarray(
            model.predict_proba(X[validation_mask]),
            dtype=float,
        )
        pmetrics = probability_metrics(y_val, val_probability)
        score = pmetrics["macro_auprc"]
        auroc = pmetrics["macro_auroc"]
        if (score > best_score) or (
            np.isclose(score, best_score) and auroc > best_auroc
        ):
            best = {
                "model": model,
                "selected_C": float(C),
                "validation_probability": val_probability,
                "validation_macro_auprc_for_C": score,
                "validation_macro_auroc_for_C": auroc,
            }
            best_score = score
            best_auroc = auroc

    if best is None:
        raise RuntimeError("No valid model was selected from the C grid")

    tuned_threshold, threshold_val_metrics = tune_threshold(
        y_val,
        best["validation_probability"],
        threshold_grid,
    )
    best["tuned_threshold"] = tuned_threshold
    best["validation_best_macro_f1_at_threshold"] = threshold_val_metrics[
        "macro_f1"
    ]
    return best


def evaluate_probability_row(
    feature: str,
    feature_type: str,
    model_name: str,
    split_mode: str,
    split_seed: int,
    eval_set: str,
    n_train: int,
    n_eval: int,
    n_labels: int,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    tuned_threshold: float,
    metadata: dict,
):
    row = {
        "task": TASK_NAME,
        "feature": feature,
        "feature_type": feature_type,
        "model": model_name,
        "split_mode": split_mode,
        "split_seed": split_seed,
        "eval_set": eval_set,
        "n_train": int(n_train),
        "n_eval": int(n_eval),
        "n_labels": int(n_labels),
        "threshold_strategy": metadata.get(
            "threshold_strategy",
            "global_threshold_tuned_on_validation_macro_f1",
        ),
        "tuned_threshold": float(tuned_threshold),
    }
    row.update(metadata)
    row.update(compute_metrics(y_true, probabilities, tuned_threshold))
    return row


def add_gain_columns(runs: pd.DataFrame, individual_features: List[str]) -> pd.DataFrame:
    result = runs.copy()
    for metric in ["macro_auprc", "macro_auroc", "macro_f1"]:
        result[f"gain_vs_morgan_r2_{metric}"] = np.nan
        result[f"gain_vs_best_individual_{metric}"] = np.nan

    for (split_mode, split_seed, eval_set), sub in result.groupby(
        ["split_mode", "split_seed", "eval_set"],
        dropna=False,
    ):
        for metric in ["macro_auprc", "macro_auroc", "macro_f1"]:
            morgan_rows = sub[sub["feature"] == "morgan_r2_2048"]
            if morgan_rows.empty:
                continue
            morgan_value = float(morgan_rows[metric].iloc[0])
            individual_values = sub.loc[
                sub["feature"].isin(individual_features), metric
            ].astype(float)
            best_individual = float(individual_values.max())
            mask = (
                result["split_mode"].eq(split_mode)
                & result["split_seed"].eq(split_seed)
                & result["eval_set"].eq(eval_set)
            )
            result.loc[mask, f"gain_vs_morgan_r2_{metric}"] = (
                result.loc[mask, metric] - morgan_value
            )
            result.loc[mask, f"gain_vs_best_individual_{metric}"] = (
                result.loc[mask, metric] - best_individual
            )
    return result


def aggregate_main_statistics(runs: pd.DataFrame) -> pd.DataFrame:
    test = runs[runs["eval_set"] == "bao_B_external_test"].copy()
    group_cols = [
        "task",
        "feature",
        "feature_type",
        "model",
        "split_mode",
        "eval_set",
        "n_labels",
        "threshold_strategy",
    ]
    metric_cols = [column for column in PRIMARY_METRICS if column in test.columns]
    metric_cols += [column for column in test.columns if column.startswith("gain_vs_")]

    rows = []
    for keys, sub in test.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_split_replicates"] = int(sub["split_seed"].nunique())
        for metric in metric_cols:
            values = sub[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_sd"] = (
                float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan
            )
            row[f"{metric}_se"] = (
                row[f"{metric}_sd"] / np.sqrt(len(values))
                if len(values) > 1
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")
    if any(C <= 0 for C in args.C_grid):
        raise ValueError("All --C-grid values must be positive")
    if not all(0.0 < threshold < 1.0 for threshold in args.threshold_grid):
        raise ValueError("All --threshold-grid values must be between 0 and 1")
    if args.scaffold_candidates <= 0:
        raise ValueError("--scaffold-candidates must be positive")

    task, label_cols, Y = load_task(
        args.metadata_file,
        args.labels_file,
        args.label_columns_file,
    )

    split_qc_rows = []
    experiments = []
    for split_mode in requested_modes(args.split_mode):
        for split_seed in args.split_seeds:
            split_task = create_or_load_split(
                task=task,
                Y=Y,
                label_cols=label_cols,
                split_mode=split_mode,
                split_seed=split_seed,
                args=args,
            )
            split_qc_rows.extend(
                split_qc_record(
                    split_task,
                    Y,
                    label_cols,
                    split_mode,
                    split_seed,
                )
            )
            experiments.append(
                {
                    "split_mode": split_mode,
                    "split_seed": split_seed,
                    "task": split_task,
                }
            )

    feature_specs = build_feature_specs(args)
    feature_matrices = {}
    feature_types = {}
    for requested_name, feature_type, loader in feature_specs:
        print(f"\n=== Loading feature: {requested_name} ===")
        feature_name, X = load_feature_matrix(task, args, requested_name, loader)
        if feature_name in feature_matrices:
            raise ValueError(f"Duplicated resolved feature name: {feature_name}")
        feature_matrices[feature_name] = X
        feature_types[feature_name] = feature_type

    individual_features = list(feature_matrices.keys())
    fm_features = [
        feature
        for feature in individual_features
        if feature not in {"morgan_r2_2048", "morgan_r3_2048"}
    ]

    rows = []
    prediction_payload = {
        "script_version": np.asarray([SCRIPT_VERSION], dtype=object),
        "task": np.asarray([TASK_NAME], dtype=object),
        "label_columns": np.asarray(label_cols, dtype=object),
        "feature_order": np.asarray(individual_features, dtype=object),
    }

    for experiment in experiments:
        split_mode = experiment["split_mode"]
        split_seed = experiment["split_seed"]
        split_task = experiment["task"]

        train_mask = split_task["split"].eq("train").to_numpy()
        validation_mask = split_task["split"].eq("validation").to_numpy()
        external_mask = split_task["split"].eq("external_test").to_numpy()
        train_idx = np.flatnonzero(train_mask)
        eval_masks = {
            "validation": validation_mask,
            "bao_B_external_test": external_mask,
        }

        print(
            f"\n=== Evaluating split_mode={split_mode} "
            f"split_seed={split_seed} ==="
        )
        probability_store = {eval_set: {} for eval_set in eval_masks}

        for feature_name in individual_features:
            X = feature_matrices[feature_name]
            feature_type = feature_types[feature_name]
            selected = fit_select_c_and_threshold(
                X=X,
                Y=Y,
                train_idx=train_idx,
                validation_mask=validation_mask,
                feature_type=feature_type,
                seed=split_seed,
                C_grid=args.C_grid,
                max_iter=args.max_iter,
                threshold_grid=args.threshold_grid,
                n_jobs=args.n_jobs,
            )
            model = selected["model"]
            tuned_threshold = selected["tuned_threshold"]
            metadata = {
                "C_grid": "|".join(str(float(C)) for C in args.C_grid),
                "selected_C": selected["selected_C"],
                "C_selection_metric": "validation_macro_auprc",
                "validation_macro_auprc_for_selected_C": selected[
                    "validation_macro_auprc_for_C"
                ],
                "validation_macro_auroc_for_selected_C": selected[
                    "validation_macro_auroc_for_C"
                ],
                "threshold_strategy": (
                    "global_threshold_tuned_on_validation_macro_f1"
                ),
                "threshold_grid": "|".join(
                    str(float(threshold)) for threshold in args.threshold_grid
                ),
                "validation_best_macro_f1_at_tuned_threshold": selected[
                    "validation_best_macro_f1_at_threshold"
                ],
                "external_test_source": DATASET_B_NAME,
            }

            for eval_set, eval_mask in eval_masks.items():
                if eval_set == "validation":
                    probabilities = selected["validation_probability"]
                else:
                    probabilities = np.asarray(
                        model.predict_proba(X[eval_mask]),
                        dtype=float,
                    )
                probability_store[eval_set][feature_name] = probabilities
                row = evaluate_probability_row(
                    feature=feature_name,
                    feature_type=feature_type,
                    model_name="one_vs_rest_logistic_regression",
                    split_mode=split_mode,
                    split_seed=split_seed,
                    eval_set=eval_set,
                    n_train=int(train_mask.sum()),
                    n_eval=int(eval_mask.sum()),
                    n_labels=Y.shape[1],
                    y_true=Y[eval_mask],
                    probabilities=probabilities,
                    tuned_threshold=tuned_threshold,
                    metadata=metadata,
                )
                rows.append(row)
                print(
                    f"{feature_name:28s} {eval_set:20s} "
                    f"C={selected['selected_C']} thr={tuned_threshold:.2f} "
                    f"AUPRC={row['macro_auprc']:.4f} "
                    f"AUROC={row['macro_auroc']:.4f} "
                    f"macro-F1={row['macro_f1']:.4f}"
                )

        ensemble_specs = [
            (f"ensemble_morgan_r2_plus_{fm}", ["morgan_r2_2048", fm])
            for fm in fm_features
        ]
        ensemble_specs.append(
            ("ensemble_all_equal_probability", individual_features)
        )

        for ensemble_name, members in ensemble_specs:
            val_probability = np.mean(
                np.stack(
                    [probability_store["validation"][member] for member in members],
                    axis=0,
                ),
                axis=0,
            )
            tuned_threshold, threshold_val_metrics = tune_threshold(
                Y[validation_mask],
                val_probability,
                args.threshold_grid,
            )
            metadata = {
                "C_grid": "base_models_independently_tuned",
                "selected_C": np.nan,
                "C_selection_metric": "base_models_validation_macro_auprc",
                "validation_macro_auprc_for_selected_C": np.nan,
                "validation_macro_auroc_for_selected_C": np.nan,
                "threshold_strategy": (
                    "global_threshold_tuned_on_validation_macro_f1"
                ),
                "threshold_grid": "|".join(
                    str(float(threshold)) for threshold in args.threshold_grid
                ),
                "validation_best_macro_f1_at_tuned_threshold": (
                    threshold_val_metrics["macro_f1"]
                ),
                "ensemble_members": "|".join(members),
                "external_test_source": DATASET_B_NAME,
            }

            for eval_set, eval_mask in eval_masks.items():
                ensemble_probability = np.mean(
                    np.stack(
                        [probability_store[eval_set][member] for member in members],
                        axis=0,
                    ),
                    axis=0,
                )
                row = evaluate_probability_row(
                    feature=ensemble_name,
                    feature_type="probability_ensemble",
                    model_name="equal_weight_probability_average",
                    split_mode=split_mode,
                    split_seed=split_seed,
                    eval_set=eval_set,
                    n_train=int(train_mask.sum()),
                    n_eval=int(eval_mask.sum()),
                    n_labels=Y.shape[1],
                    y_true=Y[eval_mask],
                    probabilities=ensemble_probability,
                    tuned_threshold=tuned_threshold,
                    metadata=metadata,
                )
                rows.append(row)
                probability_store[eval_set][ensemble_name] = ensemble_probability
                print(
                    f"{ensemble_name:48s} {eval_set:20s} "
                    f"thr={tuned_threshold:.2f} "
                    f"AUPRC={row['macro_auprc']:.4f} "
                    f"AUROC={row['macro_auroc']:.4f} "
                    f"macro-F1={row['macro_f1']:.4f}"
                )

        split_key = f"{split_mode}_seed{split_seed}"
        for eval_set, eval_mask in eval_masks.items():
            prediction_payload[f"{split_key}_{eval_set}_ids"] = (
                split_task.loc[eval_mask, ID_COL].astype(str).to_numpy()
            )
            prediction_payload[f"{split_key}_{eval_set}_y_true"] = Y[eval_mask]
            for feature_name, probabilities in probability_store[eval_set].items():
                prediction_payload[
                    f"{split_key}_{eval_set}_{feature_name}_probability"
                ] = probabilities.astype(np.float32)

        gc.collect()

    runs = pd.DataFrame(rows)
    runs = add_gain_columns(runs, individual_features)
    main_stats = aggregate_main_statistics(runs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_out = output_dir / "pmn_secondary_pathway_full_ensemble_repeated_runs.tsv"
    main_out = output_dir / "pmn_secondary_pathway_full_ensemble_main_statistics.tsv"
    pred_out = output_dir / "pmn_secondary_pathway_full_ensemble_predictions.npz"
    split_qc_out = output_dir / "pmn_secondary_pathway_full_ensemble_split_qc.tsv"

    runs.to_csv(runs_out, sep="\t", index=False)
    main_stats.to_csv(main_out, sep="\t", index=False)
    np.savez_compressed(pred_out, **prediction_payload)
    pd.DataFrame(split_qc_rows).to_csv(split_qc_out, sep="\t", index=False)

    print("\nWrote:")
    print(runs_out)
    print(main_out)
    print(pred_out)
    print(split_qc_out)

    if not main_stats.empty:
        print("\n=== Main Bao B external-test statistics ===")
        display_cols = [
            "split_mode",
            "feature",
            "n_split_replicates",
            "macro_auprc_mean",
            "macro_auprc_sd",
            "macro_auprc_se",
            "macro_auroc_mean",
            "macro_f1_mean",
            "macro_f1_se",
        ]
        display_cols = [
            column for column in display_cols if column in main_stats.columns
        ]
        print(
            main_stats.sort_values(
                ["split_mode", "macro_auprc_mean"],
                ascending=[True, False],
            )[display_cols].to_string(index=False)
        )

    for X in feature_matrices.values():
        del X
    gc.collect()


if __name__ == "__main__":
    main()

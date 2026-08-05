#!/usr/bin/env python3
"""
Repeated-split LoRA adaptation for NP-likeness prediction.

Task
----
Balanced binary classification:
- 1: COCONUT natural product
- 0: ZINC synthetic decoy

Design
------
- Reuses the authoritative NP-likeness task table.
- Reuses the five predefined random and scaffold split files.
- Supports LoRA adaptation and an optional frozen-encoder linear-head control.
- Uses the same mean pooling over non-padding tokens as the frozen-embedding
  benchmark.
- Selects the best epoch using validation AUPRC, with validation AUROC as a
  tie-breaker.
- Selects a global classification threshold using validation F1.
- Evaluates the untouched test set after model and threshold selection.

This script uses a direct PyTorch training loop rather than Hugging Face Trainer,
so it does not require Accelerate for training.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

SCRIPT_VERSION = "np_likeness_lora_repeated_splits_v2"
TASK_NAME = "np_likeness"
ID_COL = "compound_id"
LABEL_COL = "is_np"
DEFAULT_SPLIT_SEEDS = [2026, 2027, 2028, 2029, 2030]
DEFAULT_THRESHOLD_GRID = [0.2, 0.4, 0.6, 0.8]


class SmilesDataset(Dataset):
    def __init__(
        self,
        compound_ids: Sequence[str],
        smiles: Sequence[str],
        labels: Sequence[int],
    ) -> None:
        self.compound_ids = np.asarray(compound_ids, dtype=str)
        self.smiles = np.asarray(smiles, dtype=object)
        self.labels = np.asarray(labels, dtype=np.int64)
        if not (
            len(self.compound_ids) == len(self.smiles) == len(self.labels)
        ):
            raise ValueError("Dataset arrays have inconsistent lengths")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {
            "compound_id": str(self.compound_ids[index]),
            "smiles": str(self.smiles[index]),
            "label": int(self.labels[index]),
        }


class SmilesBatchCollator:
    def __init__(
        self,
        tokenizer,
        max_length: int,
        pad_to_multiple_of: int | None = 8,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.padding_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    def __call__(self, examples: list[dict]) -> dict:
        encoded = [
            self.tokenizer(
                example["smiles"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_attention_mask=True,
            )
            for example in examples
        ]
        batch = self.padding_collator(encoded)
        batch["labels"] = torch.tensor(
            [example["label"] for example in examples],
            dtype=torch.float32,
        )
        batch["compound_ids"] = [
            example["compound_id"] for example in examples
        ]
        return batch


class MeanPooledBinaryClassifier(nn.Module):
    """Mean-pooled transformer encoder followed by one linear output unit."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(float(head_dropout))
        self.classifier = nn.Linear(int(hidden_size), 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **model_inputs,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **model_inputs,
        )
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(self.dropout(pooled)).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--task-file",
        default="data/tasks/np_likeness/np_likeness_task.tsv",
    )
    parser.add_argument(
        "--split-dir",
        default="data/splits/np_likeness",
    )
    parser.add_argument(
        "--output-root",
        default="results/np_likeness/lora",
    )

    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument(
        "--feature-name",
        required=True,
        help=(
            "Stable benchmark name, e.g. chemberta_77m_mlm, "
            "chemberta_100m_mlm, or molformer_c3_1_1b."
        ),
    )
    parser.add_argument(
        "--adaptation-modes",
        nargs="+",
        choices=["lora", "frozen_head"],
        default=["lora"],
        help=(
            "Use 'lora' for adapter training. Add 'frozen_head' to train only "
            "the same linear head on a completely frozen encoder."
        ),
    )

    parser.add_argument(
        "--split-mode",
        choices=["random", "scaffold", "both"],
        default="both",
    )
    parser.add_argument(
        "--split-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SPLIT_SEEDS,
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)

    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
    )
    parser.add_argument(
        "--threshold-grid",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLD_GRID,
    )

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["query", "value"],
    )

    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-predictions", action="store_true")

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed split/mode/seed runs listed in the result TSV.",
    )
    parser.add_argument(
        "--overwrite-feature-results",
        action="store_true",
        help=(
            "Delete the result directory for --feature-name before running. "
            "Cannot be combined with --resume."
        ),
    )
    parser.add_argument(
        "--list-linear-modules-and-exit",
        action="store_true",
        help=(
            "Load the base encoder, print linear module names, and exit. "
            "Useful if a model needs different LoRA target-module names."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.train_batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.lora_r <= 0 or args.lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if args.overwrite_feature_results and args.resume:
        raise ValueError(
            "--overwrite-feature-results and --resume cannot be combined"
        )
    grid = np.asarray(args.threshold_grid, dtype=float)
    if len(grid) == 0 or np.any((grid <= 0.0) | (grid >= 1.0)):
        raise ValueError("Thresholds must lie strictly between 0 and 1")
    if args.precision != "fp32" and not torch.cuda.is_available():
        raise RuntimeError(
            f"--precision {args.precision} requires a CUDA device"
        )
    if args.precision == "bf16" and torch.cuda.is_available():
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA device does not support bfloat16")


def requested_modes(split_mode: str) -> list[str]:
    return ["random", "scaffold"] if split_mode == "both" else [split_mode]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_task(path: str) -> pd.DataFrame:
    task_path = Path(path)
    if not task_path.exists():
        raise FileNotFoundError(task_path)
    task = pd.read_csv(task_path, sep="\t", low_memory=False)
    required = {
        ID_COL,
        "canonical_smiles",
        "inchikey",
        "scaffold_split_key",
        LABEL_COL,
        "source",
    }
    missing = required - set(task.columns)
    if missing:
        raise ValueError(f"{task_path} lacks columns: {sorted(missing)}")

    task = task.copy()
    task[ID_COL] = task[ID_COL].astype(str)
    task["inchikey"] = task["inchikey"].astype(str)
    task["canonical_smiles"] = task["canonical_smiles"].astype(str)
    task[LABEL_COL] = task[LABEL_COL].astype(int)

    if task[ID_COL].duplicated().any():
        raise ValueError(f"Duplicated {ID_COL} values")
    if task["inchikey"].duplicated().any():
        raise ValueError("Duplicated InChIKeys")
    if task["canonical_smiles"].isna().any():
        raise ValueError("Missing canonical SMILES")
    if set(task[LABEL_COL].unique()) != {0, 1}:
        raise ValueError(f"{LABEL_COL} must contain exactly classes 0 and 1")
    return task


def split_file_path(
    split_dir: str,
    split_mode: str,
    split_seed: int,
) -> Path:
    return (
        Path(split_dir)
        / f"np_likeness_{split_mode}_split_seed{split_seed}.tsv"
    )


def load_split(
    task: pd.DataFrame,
    split_dir: str,
    split_mode: str,
    split_seed: int,
) -> pd.DataFrame:
    path = split_file_path(split_dir, split_mode, split_seed)
    if not path.exists():
        raise FileNotFoundError(path)
    saved = pd.read_csv(path, sep="\t", low_memory=False)
    required = {ID_COL, "split"}
    missing = required - set(saved.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")

    saved = saved.copy()
    saved[ID_COL] = saved[ID_COL].astype(str)
    if saved[ID_COL].duplicated().any():
        raise ValueError(f"{path}: duplicated {ID_COL}")

    keep = [ID_COL, "split"]
    for column in ["inchikey", LABEL_COL, "source"]:
        if column in saved.columns:
            keep.append(column)
    rename = {
        column: f"saved_{column}"
        for column in keep
        if column not in {ID_COL, "split"}
    }
    saved = saved[keep].rename(columns=rename)

    merged = task.merge(
        saved,
        on=ID_COL,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged["split"].isna().any():
        raise ValueError(
            f"{path}: missing assignments for "
            f"{int(merged['split'].isna().sum())} compounds"
        )
    observed = set(merged["split"].astype(str).unique())
    if observed != {"train", "validation", "test"}:
        raise ValueError(
            f"{path}: unexpected split values: {sorted(observed)}"
        )

    if "saved_inchikey" in merged.columns:
        mismatch = (
            merged["inchikey"].astype(str)
            != merged["saved_inchikey"].astype(str)
        )
        if mismatch.any():
            raise ValueError(
                f"{path}: {int(mismatch.sum())} InChIKey mismatches"
            )
    if f"saved_{LABEL_COL}" in merged.columns:
        mismatch = (
            merged[LABEL_COL].astype(int)
            != merged[f"saved_{LABEL_COL}"].astype(int)
        )
        if mismatch.any():
            raise ValueError(
                f"{path}: {int(mismatch.sum())} label mismatches"
            )

    if split_mode == "scaffold":
        groups = {
            name: set(
                merged.loc[
                    merged["split"].eq(name),
                    "scaffold_split_key",
                ].astype(str)
            )
            for name in ["train", "validation", "test"]
        }
        for left, right in [
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise ValueError(
                    f"{path}: scaffold leakage for {left}/{right}: "
                    f"{len(overlap)} keys"
                )

    return merged


def infer_hidden_size(encoder: nn.Module) -> int:
    config = getattr(encoder, "config", None)
    for name in ["hidden_size", "d_model", "n_embd", "embed_dim"]:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)

    embeddings = encoder.get_input_embeddings()
    for name in ["embedding_dim", "hidden_size"]:
        value = getattr(embeddings, name, None)
        if value is not None:
            return int(value)
    weight = getattr(embeddings, "weight", None)
    if weight is not None and weight.ndim == 2:
        return int(weight.shape[1])
    raise ValueError("Could not infer encoder hidden size")


def linear_module_names(model: nn.Module) -> list[str]:
    return sorted(
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    )


def matched_target_modules(
    model: nn.Module,
    targets: Sequence[str],
) -> list[str]:
    names = linear_module_names(model)
    return sorted(
        name
        for name in names
        if any(
            name == target
            or name.endswith(f".{target}")
            or target in name
            for target in targets
        )
    )


def ensure_peft_dtensor_compatibility() -> str:
    """Expose a DTensor namespace expected by some PEFT/PyTorch combinations.

    Recent PEFT releases inspect ``torch.distributed.tensor.DTensor`` while
    injecting adapters. In some PyTorch releases the module is importable but
    is not exposed as ``torch.distributed.tensor`` until explicitly imported;
    older releases expose the same API under ``torch.distributed._tensor``.
    This function normalizes those layouts without modifying model behavior.
    """
    distributed = getattr(torch, "distributed", None)
    if distributed is None:
        raise RuntimeError("This PyTorch build does not provide torch.distributed")

    existing = getattr(distributed, "tensor", None)
    if existing is not None and hasattr(existing, "DTensor"):
        return "torch.distributed.tensor (already available)"

    errors: list[str] = []
    for module_name in ("torch.distributed.tensor", "torch.distributed._tensor"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # report exact environment incompatibility
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        if hasattr(module, "DTensor"):
            setattr(distributed, "tensor", module)
            return module_name
        errors.append(f"{module_name}: imported but DTensor was absent")

    try:
        import peft
        peft_version = peft.__version__
    except Exception:
        peft_version = "unknown"
    raise RuntimeError(
        "The installed PEFT/PyTorch versions are incompatible for LoRA "
        "adapter injection. PEFT expects a DTensor class, but neither "
        "torch.distributed.tensor nor torch.distributed._tensor provides it. "
        f"torch={torch.__version__}, peft={peft_version}. Details: "
        + " | ".join(errors)
    )


def configure_encoder(
    model_name_or_path: str,
    adaptation_mode: str,
    args: argparse.Namespace,
) -> tuple[nn.Module, int, list[str]]:
    encoder = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    hidden_size = infer_hidden_size(encoder)

    if hasattr(encoder.config, "use_cache"):
        encoder.config.use_cache = False

    if adaptation_mode == "frozen_head":
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        return encoder, hidden_size, []

    dtensor_backend = ensure_peft_dtensor_compatibility()
    print("PEFT DTensor compatibility backend:", dtensor_backend)
    from peft import LoraConfig, TaskType, get_peft_model

    matches = matched_target_modules(
        encoder,
        args.lora_target_modules,
    )
    if not matches:
        available = "\n".join(linear_module_names(encoder)[:300])
        raise ValueError(
            "No linear modules matched --lora-target-modules "
            f"{args.lora_target_modules}. Available linear modules include:\n"
            f"{available}"
        )

    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(args.lora_target_modules),
        bias="none",
    )
    encoder = get_peft_model(encoder, config)

    if args.gradient_checkpointing:
        if hasattr(encoder, "gradient_checkpointing_enable"):
            encoder.gradient_checkpointing_enable()
        if hasattr(encoder, "enable_input_require_grads"):
            encoder.enable_input_require_grads()

    return encoder, hidden_size, matches


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return int(total), int(trainable)


def class_pos_weight(
    labels: np.ndarray,
    mode: str,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    n_positive = int((labels == 1).sum())
    n_negative = int((labels == 0).sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError("Training split must contain both classes")
    return torch.tensor(
        [n_negative / n_positive],
        dtype=torch.float32,
    )


def probability_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    return {
        "auprc": float(average_precision_score(y_true, probability)),
        "auroc": float(roc_auc_score(y_true, probability)),
    }


def threshold_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    prediction = (probability >= float(threshold)).astype(np.int8)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, prediction)
        ),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(
            precision_score(y_true, prediction, zero_division=0)
        ),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "predicted_positive_fraction": float(prediction.mean()),
    }


def tune_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold_grid: Iterable[float],
) -> tuple[float, float]:
    true_prevalence = float(np.mean(y_true))
    best_threshold = None
    best_f1 = -np.inf
    best_prevalence_gap = np.inf

    for threshold in threshold_grid:
        metrics = threshold_metrics(y_true, probability, float(threshold))
        prevalence_gap = abs(
            metrics["predicted_positive_fraction"] - true_prevalence
        )
        if (
            metrics["f1"] > best_f1
            or (
                np.isclose(metrics["f1"], best_f1)
                and prevalence_gap < best_prevalence_gap
            )
        ):
            best_threshold = float(threshold)
            best_f1 = float(metrics["f1"])
            best_prevalence_gap = float(prevalence_gap)

    if best_threshold is None:
        raise RuntimeError("Threshold selection failed")
    return best_threshold, best_f1


def autocast_context(precision: str):
    if not torch.cuda.is_available() or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key not in {"labels", "compound_ids"}
    }


def snapshot_trainable_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_parameters(
    model: nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    named = dict(model.named_parameters())
    missing = [name for name in state if name not in named]
    if missing:
        raise KeyError(f"Missing trainable parameters during restore: {missing}")
    with torch.no_grad():
        for name, value in state.items():
            named[name].copy_(value.to(named[name].device))


def build_optimizer(
    model: MeanPooledBinaryClassifier,
    adaptation_mode: str,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    head_parameters = [
        parameter
        for parameter in model.classifier.parameters()
        if parameter.requires_grad
    ]
    groups = [
        {
            "params": head_parameters,
            "lr": args.head_learning_rate,
            "weight_decay": args.weight_decay,
        }
    ]
    if adaptation_mode == "lora":
        adapter_parameters = [
            parameter
            for parameter in model.encoder.parameters()
            if parameter.requires_grad
        ]
        if not adapter_parameters:
            raise RuntimeError("LoRA mode has no trainable encoder parameters")
        groups.insert(
            0,
            {
                "params": adapter_parameters,
                "lr": args.lora_learning_rate,
                "weight_decay": 0.0,
            },
        )
    return torch.optim.AdamW(groups)


def make_loader(
    dataset: SmilesDataset,
    collator: SmilesBatchCollator,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        collate_fn=collator,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def predict(
    model: MeanPooledBinaryClassifier,
    loader: DataLoader,
    device: torch.device,
    precision: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    probabilities = []
    labels = []
    compound_ids: list[str] = []
    total_loss = 0.0
    total_n = 0

    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    for batch in loader:
        y = batch["labels"].to(device, non_blocking=True)
        inputs = move_batch_to_device(batch, device)
        with autocast_context(precision):
            logits = model(**inputs)
            loss = criterion(logits.float(), y)
        probability = torch.sigmoid(logits.float())
        probabilities.append(probability.cpu().numpy())
        labels.append(y.cpu().numpy())
        compound_ids.extend(batch["compound_ids"])
        total_loss += float(loss.item())
        total_n += int(len(y))

    return (
        np.concatenate(labels).astype(np.int8),
        np.concatenate(probabilities).astype(np.float64),
        np.asarray(compound_ids, dtype=str),
        total_loss / max(total_n, 1),
    )


def train_one_run(
    split_task: pd.DataFrame,
    tokenizer,
    adaptation_mode: str,
    split_mode: str,
    split_seed: int,
    run_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    set_seed(split_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, hidden_size, matched_modules = configure_encoder(
        args.model_name_or_path,
        adaptation_mode,
        args,
    )
    model = MeanPooledBinaryClassifier(
        encoder=encoder,
        hidden_size=hidden_size,
        head_dropout=args.head_dropout,
    )
    model.to(device)

    total_parameters, trainable_parameters = count_parameters(model)
    if adaptation_mode == "frozen_head":
        model.encoder.eval()

    subsets = {}
    for split_name in ["train", "validation", "test"]:
        part = split_task[split_task["split"].eq(split_name)].copy()
        subsets[split_name] = SmilesDataset(
            compound_ids=part[ID_COL].astype(str).to_numpy(),
            smiles=part["canonical_smiles"].astype(str).to_numpy(),
            labels=part[LABEL_COL].astype(int).to_numpy(),
        )

    collator = SmilesBatchCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        pad_to_multiple_of=8 if args.precision != "fp32" else None,
    )
    train_loader = make_loader(
        subsets["train"],
        collator,
        args.train_batch_size,
        True,
        split_seed,
        args.num_workers,
    )
    validation_loader = make_loader(
        subsets["validation"],
        collator,
        args.eval_batch_size,
        False,
        split_seed,
        args.num_workers,
    )
    test_loader = make_loader(
        subsets["test"],
        collator,
        args.eval_batch_size,
        False,
        split_seed,
        args.num_workers,
    )

    train_labels = subsets["train"].labels
    pos_weight = class_pos_weight(train_labels, args.class_weight)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device) if pos_weight is not None else None
    )

    optimizer = build_optimizer(model, adaptation_mode, args)
    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(round(total_updates * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(torch.cuda.is_available() and args.precision == "fp16")
    )

    best_state = None
    best_epoch = None
    best_auprc = -np.inf
    best_auroc = -np.inf
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        if adaptation_mode == "frozen_head":
            model.encoder.eval()

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_examples = 0

        for step, batch in enumerate(train_loader, start=1):
            y = batch["labels"].to(device, non_blocking=True)
            inputs = move_batch_to_device(batch, device)

            with autocast_context(args.precision):
                logits = model(**inputs)
                loss = criterion(logits.float(), y)
                scaled_loss = loss / args.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_update = (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_loader)
            )
            if should_update:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    args.max_grad_norm,
                )
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * len(y)
            total_examples += int(len(y))

        (
            y_validation,
            validation_probability,
            _,
            validation_loss,
        ) = predict(
            model,
            validation_loader,
            device,
            args.precision,
        )
        validation_probability_metrics = probability_metrics(
            y_validation,
            validation_probability,
        )
        validation_at_half = threshold_metrics(
            y_validation,
            validation_probability,
            0.5,
        )

        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_examples, 1),
            "validation_loss": validation_loss,
            "validation_auprc": validation_probability_metrics["auprc"],
            "validation_auroc": validation_probability_metrics["auroc"],
            "validation_f1_at_0.5": validation_at_half["f1"],
            "seconds": time.time() - epoch_start,
        }
        history.append(row)
        print(
            f"{args.feature_name} | {adaptation_mode} | {split_mode} "
            f"| seed={split_seed} | epoch={epoch} "
            f"| train_loss={row['train_loss']:.6f} "
            f"| val_AUPRC={row['validation_auprc']:.6f} "
            f"| val_AUROC={row['validation_auroc']:.6f}"
        )

        improved = (
            row["validation_auprc"] > best_auprc
            or (
                np.isclose(row["validation_auprc"], best_auprc)
                and row["validation_auroc"] > best_auroc
            )
        )
        if improved:
            best_auprc = row["validation_auprc"]
            best_auroc = row["validation_auroc"]
            best_epoch = epoch
            best_state = snapshot_trainable_parameters(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            args.early_stopping_patience >= 0
            and epochs_without_improvement > args.early_stopping_patience
        ):
            print(
                f"Early stopping after epoch {epoch}; "
                f"best epoch was {best_epoch}"
            )
            break

    if best_state is None or best_epoch is None:
        raise RuntimeError("No best epoch was selected")
    restore_trainable_parameters(model, best_state)

    (
        y_validation,
        validation_probability,
        validation_ids,
        validation_loss,
    ) = predict(
        model,
        validation_loader,
        device,
        args.precision,
    )
    threshold, validation_best_f1 = tune_threshold(
        y_validation,
        validation_probability,
        args.threshold_grid,
    )
    (
        y_test,
        test_probability,
        test_ids,
        test_loss,
    ) = predict(
        model,
        test_loader,
        device,
        args.precision,
    )

    validation_metrics = {
        **probability_metrics(y_validation, validation_probability),
        **threshold_metrics(y_validation, validation_probability, threshold),
    }
    test_metrics = {
        **probability_metrics(y_test, test_probability),
        **threshold_metrics(y_test, test_probability, threshold),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(
        run_dir / "epoch_history.tsv",
        sep="\t",
        index=False,
    )

    best_model_dir = run_dir / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    if adaptation_mode == "lora":
        model.encoder.save_pretrained(best_model_dir / "adapter")
    torch.save(
        model.classifier.state_dict(),
        best_model_dir / "classifier_head.pt",
    )
    tokenizer.save_pretrained(best_model_dir / "tokenizer")

    run_metadata = {
        "script_version": SCRIPT_VERSION,
        "task": TASK_NAME,
        "feature": args.feature_name,
        "model_name_or_path": args.model_name_or_path,
        "adaptation_mode": adaptation_mode,
        "split_mode": split_mode,
        "split_seed": split_seed,
        "best_epoch": best_epoch,
        "checkpoint_selection_metric": "validation_auprc",
        "validation_auprc_at_best_epoch": best_auprc,
        "validation_auroc_at_best_epoch": best_auroc,
        "tuned_threshold": threshold,
        "validation_best_f1_at_tuned_threshold": validation_best_f1,
        "hidden_size": hidden_size,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_fraction": (
            trainable_parameters / total_parameters
            if total_parameters
            else float("nan")
        ),
        "lora_target_modules_requested": (
            list(args.lora_target_modules)
            if adaptation_mode == "lora"
            else []
        ),
        "lora_linear_modules_matched": matched_modules,
        "n_train": len(subsets["train"]),
        "n_validation": len(subsets["validation"]),
        "n_test": len(subsets["test"]),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2)
    )

    if not args.skip_predictions:
        np.savez_compressed(
            run_dir / "predictions.npz",
            validation_ids=validation_ids,
            validation_y_true=y_validation,
            validation_probability=validation_probability,
            test_ids=test_ids,
            test_y_true=y_test,
            test_probability=test_probability,
            tuned_threshold=np.asarray([threshold], dtype=np.float64),
        )

    rows = []
    for eval_set, metrics, loss in [
        ("validation", validation_metrics, validation_loss),
        ("test", test_metrics, test_loss),
    ]:
        rows.append(
            {
                "script_version": SCRIPT_VERSION,
                "task": TASK_NAME,
                "feature": args.feature_name,
                "model_name_or_path": args.model_name_or_path,
                "adaptation_mode": adaptation_mode,
                "split_mode": split_mode,
                "split_seed": split_seed,
                "eval_set": eval_set,
                "n_train": len(subsets["train"]),
                "n_validation": len(subsets["validation"]),
                "n_test": len(subsets["test"]),
                "n_eval": (
                    len(subsets["validation"])
                    if eval_set == "validation"
                    else len(subsets["test"])
                ),
                "loss": loss,
                "selected_epoch": best_epoch,
                "selected_by": "validation_auprc_then_auroc",
                "tuned_threshold": threshold,
                "auprc": metrics["auprc"],
                "auroc": metrics["auroc"],
                "f1": metrics["f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "predicted_positive_fraction": metrics[
                    "predicted_positive_fraction"
                ],
                "class_weight": args.class_weight,
                "pos_weight": (
                    float(pos_weight.item())
                    if pos_weight is not None
                    else np.nan
                ),
                "max_length": args.max_length,
                "precision_mode": args.precision,
                "epochs_requested": args.epochs,
                "train_batch_size": args.train_batch_size,
                "eval_batch_size": args.eval_batch_size,
                "gradient_accumulation_steps": (
                    args.gradient_accumulation_steps
                ),
                "lora_learning_rate": (
                    args.lora_learning_rate
                    if adaptation_mode == "lora"
                    else np.nan
                ),
                "head_learning_rate": args.head_learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "lora_r": (
                    args.lora_r if adaptation_mode == "lora" else np.nan
                ),
                "lora_alpha": (
                    args.lora_alpha if adaptation_mode == "lora" else np.nan
                ),
                "lora_dropout": (
                    args.lora_dropout
                    if adaptation_mode == "lora"
                    else np.nan
                ),
                "head_dropout": args.head_dropout,
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "trainable_fraction": (
                    trainable_parameters / total_parameters
                    if total_parameters
                    else np.nan
                ),
            }
        )

    del model
    del optimizer
    del scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run_metadata, rows


def completed_keys(metrics_path: Path) -> set[tuple[str, str, int]]:
    if not metrics_path.exists():
        return set()
    frame = pd.read_csv(metrics_path, sep="\t", low_memory=False)
    required = {
        "adaptation_mode",
        "split_mode",
        "split_seed",
        "eval_set",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"{metrics_path} lacks columns required for resume"
        )
    test = frame[frame["eval_set"].astype(str).eq("test")]
    return set(
        zip(
            test["adaptation_mode"].astype(str),
            test["split_mode"].astype(str),
            test["split_seed"].astype(int),
        )
    )


def append_metric_rows(metrics_path: Path, rows: list[dict]) -> None:
    incoming = pd.DataFrame(rows)
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path, sep="\t", low_memory=False)
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming
    key_columns = [
        "feature",
        "adaptation_mode",
        "split_mode",
        "split_seed",
        "eval_set",
    ]
    if combined.duplicated(key_columns).any():
        duplicates = combined.loc[
            combined.duplicated(key_columns, keep=False),
            key_columns,
        ]
        raise ValueError(
            "Duplicate result keys detected:\n"
            + duplicates.to_string(index=False)
        )
    combined = combined.sort_values(
        ["adaptation_mode", "split_mode", "split_seed", "eval_set"]
    )
    combined.to_csv(metrics_path, sep="\t", index=False)


def split_qc_row(
    split_task: pd.DataFrame,
    split_mode: str,
    split_seed: int,
    split_name: str,
) -> dict:
    subset = split_task[split_task["split"].eq(split_name)]
    return {
        "split_mode": split_mode,
        "split_seed": split_seed,
        "split": split_name,
        "n_compounds": int(len(subset)),
        "n_negative": int((subset[LABEL_COL] == 0).sum()),
        "n_positive": int((subset[LABEL_COL] == 1).sum()),
        "positive_fraction": float(subset[LABEL_COL].mean()),
        "n_scaffold_keys": int(
            subset["scaffold_split_key"].astype(str).nunique()
        ),
    }


def environment_info() -> dict:
    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None
    try:
        import peft
        peft_version = peft.__version__
    except Exception:
        peft_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "peft": peft_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    feature_dir = Path(args.output_root) / args.feature_name
    if args.overwrite_feature_results and feature_dir.exists():
        shutil.rmtree(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = (
        feature_dir / "np_likeness_adaptation_repeated_runs.tsv"
    )
    split_qc_path = feature_dir / "np_likeness_adaptation_split_qc.tsv"
    config_path = feature_dir / "np_likeness_adaptation_config.json"

    task = load_task(args.task_file)
    print("SCRIPT_VERSION:", SCRIPT_VERSION)
    print("Task:", TASK_NAME)
    print("Compounds:", f"{len(task):,}")
    print(
        "Class counts:",
        task[LABEL_COL].value_counts().sort_index().to_dict(),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        fallback = tokenizer.eos_token or tokenizer.sep_token or tokenizer.cls_token
        if fallback is None:
            raise ValueError("Tokenizer has no usable padding token")
        tokenizer.pad_token = fallback

    if args.list_linear_modules_and_exit:
        encoder = AutoModel.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        names = linear_module_names(encoder)
        print("\n".join(names))
        print(f"\nTotal linear modules: {len(names)}")
        matches = matched_target_modules(
            encoder,
            args.lora_target_modules,
        )
        print(
            "\nMatched requested targets "
            f"{args.lora_target_modules}: {len(matches)}"
        )
        print("\n".join(matches))
        return

    config = {
        "script_version": SCRIPT_VERSION,
        "task_file": args.task_file,
        "split_dir": args.split_dir,
        "output_root": args.output_root,
        "feature_name": args.feature_name,
        "model_name_or_path": args.model_name_or_path,
        "arguments": vars(args),
        "environment": environment_info(),
    }
    if args.resume and config_path.exists():
        previous = json.loads(config_path.read_text())
        for field in [
            "task_file",
            "split_dir",
            "feature_name",
            "model_name_or_path",
        ]:
            if str(previous.get(field)) != str(config.get(field)):
                raise ValueError(
                    f"Resume configuration mismatch for {field}: "
                    f"previous={previous.get(field)!r}, current={config.get(field)!r}"
                )
    else:
        config_path.write_text(json.dumps(config, indent=2, default=str))

    done = completed_keys(metrics_path) if args.resume else set()
    if metrics_path.exists() and not args.resume:
        raise FileExistsError(
            f"{metrics_path} already exists. Use --resume or "
            "--overwrite-feature-results."
        )

    all_qc_rows: list[dict] = []
    for split_mode in requested_modes(args.split_mode):
        for split_seed in args.split_seeds:
            split_task = load_split(
                task,
                args.split_dir,
                split_mode,
                split_seed,
            )
            for split_name in ["train", "validation", "test"]:
                all_qc_rows.append(
                    split_qc_row(
                        split_task,
                        split_mode,
                        split_seed,
                        split_name,
                    )
                )

            for adaptation_mode in args.adaptation_modes:
                key = (adaptation_mode, split_mode, int(split_seed))
                if key in done:
                    print("Skipping completed run:", key)
                    continue

                run_dir = (
                    feature_dir
                    / adaptation_mode
                    / split_mode
                    / f"seed{split_seed}"
                )
                _, rows = train_one_run(
                    split_task=split_task,
                    tokenizer=tokenizer,
                    adaptation_mode=adaptation_mode,
                    split_mode=split_mode,
                    split_seed=int(split_seed),
                    run_dir=run_dir,
                    args=args,
                )
                append_metric_rows(metrics_path, rows)
                done.add(key)

    qc = pd.DataFrame(all_qc_rows).drop_duplicates(
        ["split_mode", "split_seed", "split"]
    )
    qc = qc.sort_values(["split_mode", "split_seed", "split"])
    qc.to_csv(split_qc_path, sep="\t", index=False)

    results = pd.read_csv(metrics_path, sep="\t", low_memory=False)
    expected = {
        (mode, split_mode, int(seed))
        for mode in args.adaptation_modes
        for split_mode in requested_modes(args.split_mode)
        for seed in args.split_seeds
    }
    observed = completed_keys(metrics_path)
    missing = expected - observed
    if missing:
        raise RuntimeError(
            f"Run ended with {len(missing)} missing test result keys: "
            f"{sorted(missing)}"
        )

    print("\nWrote:")
    print(metrics_path)
    print(split_qc_path)
    print(config_path)
    print("\nTest summary:")
    display = results[results["eval_set"].eq("test")][
        [
            "feature",
            "adaptation_mode",
            "split_mode",
            "split_seed",
            "auprc",
            "auroc",
            "f1",
            "selected_epoch",
            "trainable_parameters",
        ]
    ]
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()

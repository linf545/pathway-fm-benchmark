#!/usr/bin/env python3
"""
Repeated-split LoRA adaptation for NP chemical-class prediction.

Task
----
Single-label, seven-class NPClassifier prediction.

Design
------
- Reuses the authoritative NP-classification task table.
- Reuses random_split_{seed} and scaffold_split_{seed} columns for seeds
  2026--2030.
- Supports LoRA adaptation and an optional frozen-encoder linear-head control.
- Uses mean pooling over non-padding tokens, matching the frozen SMILES-model
  embedding benchmark.
- Selects the best epoch by validation macro one-vs-rest AUROC, with validation
  macro one-vs-rest AUPRC as a tie-breaker.
- Evaluates the untouched test set only after checkpoint selection.
- Reports overall and per-class metrics for direct comparison with the frozen
  representation benchmark.

A direct PyTorch loop is used instead of Hugging Face Trainer, avoiding the
Accelerate integration problem encountered in the older script.
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
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

SCRIPT_VERSION = "npclassification_lora_repeated_splits_v2"
TASK_NAME = "np_classification"
DEFAULT_SPLIT_SEEDS = [2026, 2027, 2028, 2029, 2030]

NP_CLASS_NAMES = {
    1: "Alkaloids",
    2: "Amino acids and peptides",
    3: "Carbohydrates",
    4: "Fatty acids",
    5: "Polyketides",
    6: "Shikimates and phenylpropanoids",
    7: "Terpenoids",
}


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
            dtype=torch.long,
        )
        batch["compound_ids"] = [
            example["compound_id"] for example in examples
        ]
        return batch


class MeanPooledMulticlassClassifier(nn.Module):
    """Mean-pooled transformer encoder plus one linear classification layer."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        n_classes: int,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(float(head_dropout))
        self.classifier = nn.Linear(int(hidden_size), int(n_classes))

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
        return self.classifier(self.dropout(pooled))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--task-file",
        default="data/tasks/np_classification/npclassifier_label_task.tsv",
    )
    parser.add_argument(
        "--output-root",
        default="results/np_classification/lora",
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
            "Use lora for adapter training. Add frozen_head to train only "
            "the identical linear head with a completely frozen encoder."
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
        "--class-weight-mode",
        choices=["none", "balanced", "sqrt_balanced"],
        default="balanced",
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
        help="Skip completed adaptation/split/seed runs.",
    )
    parser.add_argument(
        "--overwrite-feature-results",
        action="store_true",
        help=(
            "Delete the existing result directory for --feature-name before "
            "running. Cannot be combined with --resume."
        ),
    )
    parser.add_argument(
        "--list-linear-modules-and-exit",
        action="store_true",
        help=(
            "Print encoder linear-module names and their matches to "
            "--lora-target-modules, then exit."
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
    if len(set(args.split_seeds)) != len(args.split_seeds):
        raise ValueError("--split-seeds contains duplicates")
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


def choose_id_column(task: pd.DataFrame) -> str:
    for candidate in ["np_task_id", "compound_id", "inchikey"]:
        if candidate in task.columns:
            return candidate
    raise ValueError("No compound identifier column was found")


def load_task(
    path: str,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[int], list[str], str]:
    task_path = Path(path)
    if not task_path.exists():
        raise FileNotFoundError(task_path)
    task = pd.read_csv(task_path, sep="\t", low_memory=False)

    required = {"canonical_smiles", "inchikey"}
    missing = required - set(task.columns)
    if missing:
        raise ValueError(f"{task_path} lacks columns: {sorted(missing)}")
    if task["canonical_smiles"].isna().any():
        raise ValueError("Task contains missing canonical SMILES")
    if task["inchikey"].isna().any() or not task["inchikey"].is_unique:
        raise ValueError("Task InChIKeys must be nonmissing and unique")

    id_column = choose_id_column(task)
    task = task.copy()
    task[id_column] = task[id_column].astype(str)
    task["canonical_smiles"] = task["canonical_smiles"].astype(str)
    task["inchikey"] = task["inchikey"].astype(str)
    if task[id_column].duplicated().any():
        raise ValueError(f"Task {id_column} values must be unique")

    label_columns = sorted(
        [column for column in task.columns if column.startswith("label_")],
        key=lambda column: int(column.replace("label_", "")),
    )
    if len(label_columns) != 7:
        raise ValueError(
            f"Expected seven label columns; found {label_columns}"
        )

    label_ids = [int(column.replace("label_", "")) for column in label_columns]
    label_matrix = task[label_columns].to_numpy(dtype=np.int8)
    cardinality = label_matrix.sum(axis=1)
    if np.any(cardinality != 1):
        bad = np.flatnonzero(cardinality != 1)
        examples = task.iloc[bad[:10]][[id_column, *label_columns]]
        raise ValueError(
            f"{len(bad)} task rows are not single-label:\n"
            + examples.to_string(index=False)
        )

    y = label_matrix.argmax(axis=1).astype(np.int64)
    expected_classes = np.arange(len(label_columns))
    if not np.array_equal(np.unique(y), expected_classes):
        raise ValueError("Observed classes do not match the label columns")

    label_names = [
        NP_CLASS_NAMES.get(label_id, f"Class {label_id}")
        for label_id in label_ids
    ]
    return task, y, label_columns, label_ids, label_names, id_column


def split_definition(
    split_mode: str,
    split_seed: int,
) -> tuple[str, str, str, str]:
    if split_mode == "random":
        return f"random_split_{split_seed}", "train", "val", "test"
    if split_mode == "scaffold":
        return (
            f"scaffold_split_{split_seed}",
            "scaffold_train",
            "scaffold_val",
            "scaffold_test",
        )
    raise ValueError(f"Unknown split mode: {split_mode}")


def load_split_indices(
    task: pd.DataFrame,
    y: np.ndarray,
    split_mode: str,
    split_seed: int,
    n_classes: int,
) -> tuple[dict[str, np.ndarray], str]:
    split_column, train_value, validation_value, test_value = split_definition(
        split_mode,
        split_seed,
    )
    if split_column not in task.columns:
        raise ValueError(f"Task file is missing split column: {split_column}")

    values = task[split_column].astype(str)
    allowed = {train_value, validation_value, test_value}
    unexpected = set(values.unique()) - allowed
    if unexpected:
        raise ValueError(
            f"{split_column} contains unexpected values: {sorted(unexpected)}"
        )

    indices = {
        "train": np.flatnonzero(values.eq(train_value).to_numpy()),
        "validation": np.flatnonzero(
            values.eq(validation_value).to_numpy()
        ),
        "test": np.flatnonzero(values.eq(test_value).to_numpy()),
    }
    combined = np.concatenate(list(indices.values()))
    if len(combined) != len(task) or len(np.unique(combined)) != len(task):
        raise ValueError(
            f"{split_column} does not form a complete disjoint partition"
        )

    for eval_set, row_indices in indices.items():
        counts = np.bincount(y[row_indices], minlength=n_classes)
        if np.any(counts == 0):
            missing_classes = np.flatnonzero(counts == 0).tolist()
            raise ValueError(
                f"{split_column} {eval_set} lacks classes {missing_classes}"
            )

    if split_mode == "scaffold" and "murcko_scaffold" in task.columns:
        scaffold_sets = {
            name: {
                value
                for value in task.iloc[row_indices]["murcko_scaffold"]
                .fillna("")
                .astype(str)
                .tolist()
                if value
            }
            for name, row_indices in indices.items()
        }
        for left, right in [
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ]:
            overlap = scaffold_sets[left] & scaffold_sets[right]
            if overlap:
                raise ValueError(
                    f"{split_column}: {len(overlap)} scaffolds are shared by "
                    f"{left} and {right}"
                )

    return indices, split_column


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
    """Expose the DTensor namespace expected by some PEFT/PyTorch pairs."""
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
        except Exception as exc:
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
        "adapter injection. "
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

    matches = matched_target_modules(encoder, args.lora_target_modules)
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


def class_weights_from_train(
    y_train: np.ndarray,
    n_classes: int,
    mode: str,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    if np.any(counts == 0):
        raise ValueError("Training split lacks one or more classes")
    weights = len(y_train) / (n_classes * counts)
    if mode == "sqrt_balanced":
        weights = np.sqrt(weights)
    elif mode != "balanced":
        raise ValueError(mode)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    label_ids: Sequence[int],
    label_names: Sequence[str],
) -> tuple[dict[str, float], list[dict]]:
    n_classes = len(label_ids)
    classes = np.arange(n_classes)
    if probabilities.shape != (len(y_true), n_classes):
        raise ValueError(
            f"Probability shape {probabilities.shape} does not match "
            f"({len(y_true)}, {n_classes})"
        )
    prediction = probabilities.argmax(axis=1)
    y_binary = label_binarize(y_true, classes=classes)

    overall = {
        "auroc_ovr_macro": float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
        ),
        "auprc_ovr_macro": float(
            average_precision_score(
                y_binary,
                probabilities,
                average="macro",
            )
        ),
        "macro_f1": float(
            f1_score(y_true, prediction, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                prediction,
                average="weighted",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, prediction)
        ),
    }

    per_class: list[dict] = []
    for class_index, (label_id, label_name) in enumerate(
        zip(label_ids, label_names)
    ):
        truth = y_binary[:, class_index]
        class_prediction = (prediction == class_index).astype(np.int8)
        per_class.append(
            {
                "class_index": int(class_index),
                "label_id": int(label_id),
                "label": str(label_name),
                "support": int(truth.sum()),
                "auroc": float(
                    roc_auc_score(truth, probabilities[:, class_index])
                ),
                "auprc": float(
                    average_precision_score(
                        truth,
                        probabilities[:, class_index],
                    )
                ),
                "f1": float(
                    f1_score(
                        truth,
                        class_prediction,
                        zero_division=0,
                    )
                ),
            }
        )
    return overall, per_class


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
    model: MeanPooledMulticlassClassifier,
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
    model: MeanPooledMulticlassClassifier,
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

    criterion = nn.CrossEntropyLoss(reduction="sum")
    for batch in loader:
        y = batch["labels"].to(device, non_blocking=True)
        inputs = move_batch_to_device(batch, device)
        with autocast_context(precision):
            logits = model(**inputs)
            loss = criterion(logits.float(), y)
        probability = torch.softmax(logits.float(), dim=1)
        probabilities.append(probability.cpu().numpy())
        labels.append(y.cpu().numpy())
        compound_ids.extend(batch["compound_ids"])
        total_loss += float(loss.item())
        total_n += int(len(y))

    return (
        np.concatenate(labels).astype(np.int64),
        np.concatenate(probabilities).astype(np.float64),
        np.asarray(compound_ids, dtype=str),
        total_loss / max(total_n, 1),
    )


def split_qc_rows(
    task: pd.DataFrame,
    y: np.ndarray,
    indices: dict[str, np.ndarray],
    split_mode: str,
    split_seed: int,
    split_column: str,
    n_classes: int,
) -> list[dict]:
    rows = []
    for eval_set, row_indices in indices.items():
        counts = np.bincount(y[row_indices], minlength=n_classes)
        row = {
            "task": TASK_NAME,
            "split_mode": split_mode,
            "split_seed": split_seed,
            "split_column": split_column,
            "eval_set": eval_set,
            "n_rows": int(len(row_indices)),
            "class_count_min": int(counts.min()),
            "class_count_max": int(counts.max()),
            "class_counts": "|".join(str(int(value)) for value in counts),
        }
        if "murcko_scaffold" in task.columns:
            row["n_scaffolds"] = int(
                task.iloc[row_indices]["murcko_scaffold"]
                .fillna("")
                .astype(str)
                .nunique()
            )
        rows.append(row)
    return rows


def train_one_run(
    task: pd.DataFrame,
    y_all: np.ndarray,
    indices: dict[str, np.ndarray],
    split_column: str,
    tokenizer,
    adaptation_mode: str,
    split_mode: str,
    split_seed: int,
    run_dir: Path,
    label_ids: Sequence[int],
    label_names: Sequence[str],
    id_column: str,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    set_seed(split_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(label_ids)

    encoder, hidden_size, matched_modules = configure_encoder(
        args.model_name_or_path,
        adaptation_mode,
        args,
    )
    model = MeanPooledMulticlassClassifier(
        encoder=encoder,
        hidden_size=hidden_size,
        n_classes=n_classes,
        head_dropout=args.head_dropout,
    )
    model.to(device)
    total_parameters, trainable_parameters = count_parameters(model)

    datasets = {}
    for eval_set, row_indices in indices.items():
        part = task.iloc[row_indices]
        datasets[eval_set] = SmilesDataset(
            compound_ids=part[id_column].astype(str).to_numpy(),
            smiles=part["canonical_smiles"].astype(str).to_numpy(),
            labels=y_all[row_indices],
        )

    collator = SmilesBatchCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        pad_to_multiple_of=8 if args.precision != "fp32" else None,
    )
    train_loader = make_loader(
        datasets["train"],
        collator,
        args.train_batch_size,
        True,
        split_seed,
        args.num_workers,
    )
    validation_loader = make_loader(
        datasets["validation"],
        collator,
        args.eval_batch_size,
        False,
        split_seed,
        args.num_workers,
    )
    test_loader = make_loader(
        datasets["test"],
        collator,
        args.eval_batch_size,
        False,
        split_seed,
        args.num_workers,
    )

    class_weights = class_weights_from_train(
        datasets["train"].labels,
        n_classes,
        args.class_weight_mode,
    )
    criterion = nn.CrossEntropyLoss(
        weight=(class_weights.to(device) if class_weights is not None else None)
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
    best_macro_auroc = -np.inf
    best_macro_auprc = -np.inf
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
            labels = batch["labels"].to(device, non_blocking=True)
            inputs = move_batch_to_device(batch, device)

            with autocast_context(args.precision):
                logits = model(**inputs)
                loss = criterion(logits.float(), labels)
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

            total_loss += float(loss.item()) * len(labels)
            total_examples += int(len(labels))

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
        validation_metrics, _ = evaluate_probabilities(
            y_validation,
            validation_probability,
            label_ids,
            label_names,
        )

        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_examples, 1),
            "validation_loss": validation_loss,
            "validation_auroc_ovr_macro": validation_metrics[
                "auroc_ovr_macro"
            ],
            "validation_auprc_ovr_macro": validation_metrics[
                "auprc_ovr_macro"
            ],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "seconds": time.time() - epoch_start,
        }
        history.append(row)
        print(
            f"{args.feature_name} | {adaptation_mode} | {split_mode} "
            f"| seed={split_seed} | epoch={epoch} "
            f"| train_loss={row['train_loss']:.6f} "
            f"| val_macro_AUROC={row['validation_auroc_ovr_macro']:.6f} "
            f"| val_macro_AUPRC={row['validation_auprc_ovr_macro']:.6f}"
        )

        improved = (
            row["validation_auroc_ovr_macro"] > best_macro_auroc
            or (
                np.isclose(
                    row["validation_auroc_ovr_macro"],
                    best_macro_auroc,
                )
                and row["validation_auprc_ovr_macro"] > best_macro_auprc
            )
        )
        if improved:
            best_macro_auroc = row["validation_auroc_ovr_macro"]
            best_macro_auprc = row["validation_auprc_ovr_macro"]
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

    evaluation_outputs = {}
    metric_rows: list[dict] = []
    per_class_rows: list[dict] = []
    for eval_set, loader in [
        ("validation", validation_loader),
        ("test", test_loader),
    ]:
        y_true, probabilities, compound_ids, loss = predict(
            model,
            loader,
            device,
            args.precision,
        )
        overall, per_class = evaluate_probabilities(
            y_true,
            probabilities,
            label_ids,
            label_names,
        )
        evaluation_outputs[eval_set] = {
            "ids": compound_ids,
            "y_true": y_true,
            "probabilities": probabilities,
        }

        base = {
            "script_version": SCRIPT_VERSION,
            "task": TASK_NAME,
            "feature": args.feature_name,
            "model_name_or_path": args.model_name_or_path,
            "adaptation_mode": adaptation_mode,
            "model": (
                "mean_pooled_encoder_lora_linear_head"
                if adaptation_mode == "lora"
                else "mean_pooled_frozen_encoder_linear_head"
            ),
            "split": split_mode,
            "split_mode": split_mode,
            "split_column": split_column,
            "split_seed": split_seed,
            "eval_set": eval_set,
            "n_train": len(datasets["train"]),
            "n_validation": len(datasets["validation"]),
            "n_test": len(datasets["test"]),
            "n_eval": len(y_true),
            "n_classes": n_classes,
            "loss": loss,
            "selected_epoch": best_epoch,
            "selected_by": "validation_macro_auroc_then_macro_auprc",
            "class_weight_mode": args.class_weight_mode,
            "max_length": args.max_length,
            "precision_mode": args.precision,
            "epochs_requested": args.epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "lora_learning_rate": (
                args.lora_learning_rate
                if adaptation_mode == "lora"
                else np.nan
            ),
            "head_learning_rate": args.head_learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "lora_r": args.lora_r if adaptation_mode == "lora" else np.nan,
            "lora_alpha": (
                args.lora_alpha if adaptation_mode == "lora" else np.nan
            ),
            "lora_dropout": (
                args.lora_dropout if adaptation_mode == "lora" else np.nan
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
        metric_rows.append({**base, **overall})
        for class_row in per_class:
            per_class_rows.append({**base, **class_row})

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
    (best_model_dir / "labels.json").write_text(
        json.dumps(
            [
                {
                    "class_index": index,
                    "label_id": int(label_id),
                    "label": label_name,
                }
                for index, (label_id, label_name) in enumerate(
                    zip(label_ids, label_names)
                )
            ],
            indent=2,
        )
    )

    run_metadata = {
        "script_version": SCRIPT_VERSION,
        "task": TASK_NAME,
        "feature": args.feature_name,
        "model_name_or_path": args.model_name_or_path,
        "adaptation_mode": adaptation_mode,
        "split_mode": split_mode,
        "split_seed": split_seed,
        "split_column": split_column,
        "best_epoch": best_epoch,
        "checkpoint_selection_metric": "validation_macro_auroc",
        "validation_macro_auroc_at_best_epoch": best_macro_auroc,
        "validation_macro_auprc_at_best_epoch": best_macro_auprc,
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
        "class_weight_mode": args.class_weight_mode,
        "class_weights": (
            class_weights.tolist() if class_weights is not None else None
        ),
        "n_train": len(datasets["train"]),
        "n_validation": len(datasets["validation"]),
        "n_test": len(datasets["test"]),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2)
    )

    if not args.skip_predictions:
        np.savez_compressed(
            run_dir / "predictions.npz",
            validation_ids=evaluation_outputs["validation"]["ids"],
            validation_y_true=evaluation_outputs["validation"]["y_true"],
            validation_probability=evaluation_outputs["validation"][
                "probabilities"
            ],
            test_ids=evaluation_outputs["test"]["ids"],
            test_y_true=evaluation_outputs["test"]["y_true"],
            test_probability=evaluation_outputs["test"]["probabilities"],
            label_ids=np.asarray(label_ids, dtype=np.int64),
            label_names=np.asarray(label_names, dtype=object),
        )

    del model
    del encoder
    del optimizer
    del scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metric_rows, per_class_rows


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


def append_rows(
    path: Path,
    rows: list[dict],
    key_columns: list[str],
    sort_columns: list[str],
) -> None:
    incoming = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path, sep="\t", low_memory=False)
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming
    if combined.duplicated(key_columns).any():
        duplicates = combined.loc[
            combined.duplicated(key_columns, keep=False),
            key_columns,
        ]
        raise ValueError(
            f"Duplicate result keys detected in {path}:\n"
            + duplicates.to_string(index=False)
        )
    combined = combined.sort_values(sort_columns)
    combined.to_csv(path, sep="\t", index=False)


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

    metrics_path = feature_dir / "npclassification_adaptation_repeated_runs.tsv"
    per_class_path = (
        feature_dir / "npclassification_adaptation_per_class_repeated_runs.tsv"
    )
    split_qc_path = feature_dir / "npclassification_adaptation_split_qc.tsv"
    config_path = feature_dir / "npclassification_adaptation_config.json"

    (
        task,
        y_all,
        label_columns,
        label_ids,
        label_names,
        id_column,
    ) = load_task(args.task_file)

    print("SCRIPT_VERSION:", SCRIPT_VERSION)
    print("Task:", TASK_NAME)
    print("Compounds:", f"{len(task):,}")
    print("ID column:", id_column)
    print("Label columns:", label_columns)
    print("Label mapping:")
    for class_index, (label_id, label_name) in enumerate(
        zip(label_ids, label_names)
    ):
        count = int((y_all == class_index).sum())
        print(
            f"  index={class_index} | label_{label_id} | "
            f"{label_name} | n={count:,}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        fallback = (
            tokenizer.eos_token
            or tokenizer.sep_token
            or tokenizer.cls_token
        )
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
        "output_root": args.output_root,
        "feature_name": args.feature_name,
        "model_name_or_path": args.model_name_or_path,
        "id_column": id_column,
        "label_columns": label_columns,
        "label_ids": label_ids,
        "label_names": label_names,
        "arguments": vars(args),
        "environment": environment_info(),
    }
    config_path.write_text(json.dumps(config, indent=2, default=str))

    done = completed_keys(metrics_path) if args.resume else set()
    if metrics_path.exists() and not args.resume:
        raise FileExistsError(
            f"{metrics_path} already exists. Use --resume or "
            "--overwrite-feature-results."
        )

    qc_rows: list[dict] = []
    for split_mode in requested_modes(args.split_mode):
        for split_seed in args.split_seeds:
            indices, split_column = load_split_indices(
                task=task,
                y=y_all,
                split_mode=split_mode,
                split_seed=int(split_seed),
                n_classes=len(label_ids),
            )
            qc_rows.extend(
                split_qc_rows(
                    task,
                    y_all,
                    indices,
                    split_mode,
                    int(split_seed),
                    split_column,
                    len(label_ids),
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
                metric_rows, per_class_rows = train_one_run(
                    task=task,
                    y_all=y_all,
                    indices=indices,
                    split_column=split_column,
                    tokenizer=tokenizer,
                    adaptation_mode=adaptation_mode,
                    split_mode=split_mode,
                    split_seed=int(split_seed),
                    run_dir=run_dir,
                    label_ids=label_ids,
                    label_names=label_names,
                    id_column=id_column,
                    args=args,
                )
                append_rows(
                    metrics_path,
                    metric_rows,
                    key_columns=[
                        "feature",
                        "adaptation_mode",
                        "split_mode",
                        "split_seed",
                        "eval_set",
                    ],
                    sort_columns=[
                        "adaptation_mode",
                        "split_mode",
                        "split_seed",
                        "eval_set",
                    ],
                )
                append_rows(
                    per_class_path,
                    per_class_rows,
                    key_columns=[
                        "feature",
                        "adaptation_mode",
                        "split_mode",
                        "split_seed",
                        "eval_set",
                        "class_index",
                    ],
                    sort_columns=[
                        "adaptation_mode",
                        "split_mode",
                        "split_seed",
                        "eval_set",
                        "class_index",
                    ],
                )
                done.add(key)

    qc = pd.DataFrame(qc_rows).drop_duplicates(
        ["split_mode", "split_seed", "eval_set"]
    )
    qc = qc.sort_values(["split_mode", "split_seed", "eval_set"])
    qc.to_csv(split_qc_path, sep="\t", index=False)

    expected = {
        (adaptation_mode, split_mode, int(split_seed))
        for adaptation_mode in args.adaptation_modes
        for split_mode in requested_modes(args.split_mode)
        for split_seed in args.split_seeds
    }
    observed = completed_keys(metrics_path)
    missing = expected - observed
    if missing:
        raise RuntimeError(
            f"Run ended with {len(missing)} missing test result keys: "
            f"{sorted(missing)}"
        )

    results = pd.read_csv(metrics_path, sep="\t", low_memory=False)
    print("\nWrote:")
    print(metrics_path)
    print(per_class_path)
    print(split_qc_path)
    print(config_path)
    print("\nTest summary:")
    display = results[results["eval_set"].eq("test")][
        [
            "feature",
            "adaptation_mode",
            "split_mode",
            "split_seed",
            "auroc_ovr_macro",
            "auprc_ovr_macro",
            "macro_f1",
            "selected_epoch",
            "trainable_parameters",
        ]
    ]
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()

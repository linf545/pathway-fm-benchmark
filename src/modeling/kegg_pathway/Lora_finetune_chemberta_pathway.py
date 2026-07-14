#!/usr/bin/env python

"""
Fine-tune ChemBERTa for Bao plant pathway prediction.

Task:
    Multilabel Bao plant natural-product pathway prediction.

Input:
    data/features/bao_plant_pathway/metadata.tsv
    data/features/bao_plant_pathway/labels.npy
    data/features/bao_plant_pathway/label_columns.txt
    data/tasks/bao_plant_pathway/bao_plant_pathway_splits.tsv
    data/splits/bao_plant_pathway/scaffold_split.tsv

Output:
    results/bao_pathway/finetune/<run_name>/
"""

from __future__ import annotations

from pathlib import Path
import argparse
import inspect
import json
import os
import random

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, TaskType, get_peft_model


ROOT = Path("/private/vina/FM_bench")

FEATURE_DIR = ROOT / "data/features/bao_plant_pathway"
TASK_DIR = ROOT / "data/tasks/bao_plant_pathway"
SPLIT_DIR = ROOT / "data/splits/bao_plant_pathway"
OUT_ROOT = ROOT / "results/bao_pathway/finetune"

ID_COL = "task_compound_id"


class SmilesMultilabelDataset(torch.utils.data.Dataset):
    def __init__(self, smiles, labels, tokenizer, max_length: int):
        self.smiles = list(smiles)
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.smiles[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )
        enc["labels"] = self.labels[idx].astype(np.float32).tolist()
        return enc


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_trainable_parameters(model):
    trainable = 0
    total = 0

    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    pct = 100 * trainable / total if total > 0 else 0
    print(f"trainable parameters: {trainable:,} / {total:,} ({pct:.4f}%)")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class WeightedMultilabelTrainer(Trainer):
    """
    Trainer using class-weighted BCE loss for imbalanced multilabel prediction.
    """

    def __init__(self, pos_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits

        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(logits.device)

        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def load_core():
    metadata = pd.read_csv(FEATURE_DIR / "metadata.tsv", sep="\t")
    labels = np.load(FEATURE_DIR / "labels.npy").astype(int)

    with open(FEATURE_DIR / "label_columns.txt") as f:
        label_names = [line.strip() for line in f if line.strip()]

    if ID_COL not in metadata.columns:
        raise ValueError(f"metadata.tsv lacks {ID_COL}")

    if "canonical_smiles" not in metadata.columns:
        raise ValueError("metadata.tsv lacks canonical_smiles")

    if labels.shape[0] != metadata.shape[0]:
        raise ValueError("labels.npy row count does not match metadata.tsv")

    if labels.shape[1] != len(label_names):
        raise ValueError("labels.npy column count does not match label_columns.txt")

    return metadata, labels, label_names


def load_split_table(metadata):
    split_table = metadata[[ID_COL]].copy()

    task_split_file = TASK_DIR / "bao_plant_pathway_splits.tsv"
    if task_split_file.exists():
        s = pd.read_csv(task_split_file, sep="\t").drop_duplicates(ID_COL)
        split_table = split_table.merge(s, on=ID_COL, how="left")

    scaffold_file = SPLIT_DIR / "scaffold_split.tsv"
    if scaffold_file.exists():
        s = pd.read_csv(scaffold_file, sep="\t").drop_duplicates(ID_COL)
        keep = [c for c in [ID_COL, "scaffold_split"] if c in s.columns]
        split_table = split_table.merge(s[keep], on=ID_COL, how="left")

    if "dataset_name" in metadata.columns and "dataset_name" not in split_table.columns:
        split_table["dataset_name"] = metadata["dataset_name"].values

    if "split" in metadata.columns and "split" not in split_table.columns:
        split_table["split"] = metadata["split"].values

    return split_table


def choose_split_column(split_table, split_name):
    if split_name == "random":
        for c in ["split", "random_split", "data_split"]:
            if c in split_table.columns:
                return c
        candidates = [c for c in split_table.columns if "random" in c.lower() and "split" in c.lower()]
    elif split_name == "scaffold":
        for c in ["scaffold_split"]:
            if c in split_table.columns:
                return c
        candidates = [c for c in split_table.columns if "scaffold" in c.lower() and "split" in c.lower()]
    else:
        raise ValueError(split_name)

    if not candidates:
        raise ValueError(f"Cannot find split column for {split_name}. Columns: {list(split_table.columns)}")

    return candidates[0]


def normalize_split_values(split_table, split_col):
    raw = split_table[split_col].astype(str).str.lower()

    if "dataset_name" in split_table.columns:
        dataset = split_table["dataset_name"].astype(str).str.lower()
    else:
        dataset = pd.Series([""] * len(split_table), index=split_table.index)

    split = pd.Series(index=split_table.index, dtype=object)

    external = (
        dataset.str.contains("external", na=False)
        | dataset.str.contains("bao_b", na=False)
        | dataset.str.contains("b_external", na=False)
        | raw.str.contains("external", na=False)
        | raw.str.contains("test", na=False)
    )

    split[external] = "test"
    split[raw.str.contains("train", na=False) & ~raw.str.contains("val", na=False) & ~external] = "train"
    split[raw.str.contains("val", na=False) & ~external] = "val"

    return split


def metric_dict(y_true, y_pred):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "sample_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "exact_match_accuracy": accuracy_score(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
    }


def make_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = sigmoid(logits)
        preds = (probs >= 0.5).astype(int)
        return metric_dict(labels.astype(int), preds.astype(int))
    return compute_metrics


def training_args_compatible(**kwargs):
    """
    Transformers changed evaluation_strategy to eval_strategy in some versions.
    This keeps the script compatible with both.
    """
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    if "eval_strategy" in params and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    if "evaluation_strategy" in params and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    allowed = set(params.keys())
    clean = {k: v for k, v in kwargs.items() if k in allowed}
    return TrainingArguments(**clean)


def predict_and_save(
    trainer,
    dataset,
    y_true,
    metadata_subset,
    out_dir,
    split_name,
    eval_set,
    label_names,
):
    pred = trainer.predict(dataset)
    logits = pred.predictions
    probs = sigmoid(logits)
    y_pred = (probs >= 0.5).astype(int)

    metrics = metric_dict(y_true.astype(int), y_pred.astype(int))
    metrics.update({
        "split": split_name,
        "eval_set": eval_set,
        "n_eval": int(y_true.shape[0]),
    })

    np.savez_compressed(
        out_dir / f"predictions.{split_name}.{eval_set}.npz",
        compound_ids=metadata_subset[ID_COL].astype(str).to_numpy(),
        y_true=y_true.astype(int),
        y_pred=y_pred.astype(int),
        y_score=probs.astype(float),
        label_names=np.asarray(label_names).astype(str),
    )

    return metrics


def run_one_split(args, split_name, metadata, labels, label_names):
    split_table = load_split_table(metadata)
    split_col = choose_split_column(split_table, split_name)
    split_values = normalize_split_values(split_table, split_col)

    train_idx = np.where(split_values.to_numpy() == "train")[0]
    val_idx = np.where(split_values.to_numpy() == "val")[0]
    test_idx = np.where(split_values.to_numpy() == "test")[0]

    print()
    print(f"=== Split: {split_name} ===")
    print(f"split column: {split_col}")
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(f"{split_name}: missing train or validation rows")

    out_dir = OUT_ROOT / args.run_name / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=labels.shape[1],
        problem_type="multi_label_classification",
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        ignore_mismatched_sizes=True,
    )

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
            modules_to_save=["classifier"],
        )

        model = get_peft_model(model, lora_config)
        print("Using LoRA fine-tuning")
        print("LoRA target modules:", args.lora_target_modules)
        print_trainable_parameters(model)
    else:
        print("Using full fine-tuning")
        print_trainable_parameters(model)

    train_ds = SmilesMultilabelDataset(
        metadata.iloc[train_idx]["canonical_smiles"].tolist(),
        labels[train_idx],
        tokenizer,
        args.max_length,
    )

    val_ds = SmilesMultilabelDataset(
        metadata.iloc[val_idx]["canonical_smiles"].tolist(),
        labels[val_idx],
        tokenizer,
        args.max_length,
    )

    training_args = training_args_compatible(
        output_dir=str(out_dir / "checkpoints"),
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=20,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available() and args.fp16,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
    )

    pos_counts = labels[train_idx].sum(axis=0).astype(float)
    neg_counts = len(train_idx) - pos_counts

    if args.pos_weight_mode == "none":
        pos_weight = np.ones_like(pos_counts, dtype=float)
    elif args.pos_weight_mode == "sqrt":
        pos_weight = np.sqrt(neg_counts / np.maximum(pos_counts, 1.0))
    elif args.pos_weight_mode == "ratio":
        pos_weight = neg_counts / np.maximum(pos_counts, 1.0)
    else:
        raise ValueError(args.pos_weight_mode)

    pos_weight = np.clip(pos_weight, 1.0, args.max_pos_weight)

    print("positive label counts:", pos_counts.astype(int).tolist())
    print("positive loss weights:", np.round(pos_weight, 2).tolist())
    print("positive weight mode:", args.pos_weight_mode)

    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)

    trainer = WeightedMultilabelTrainer(
        pos_weight=pos_weight_tensor,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics(),
    )

    trainer.train()

    all_metrics = []

    val_metrics = predict_and_save(
        trainer=trainer,
        dataset=val_ds,
        y_true=labels[val_idx],
        metadata_subset=metadata.iloc[val_idx],
        out_dir=out_dir,
        split_name=split_name,
        eval_set="validation",
        label_names=label_names,
    )
    all_metrics.append(val_metrics)

    print(
        f"validation: macro_f1={val_metrics['macro_f1']:.4f} "
        f"micro_f1={val_metrics['micro_f1']:.4f} "
        f"sample_f1={val_metrics['sample_f1']:.4f}"
    )

    if len(test_idx) > 0:
        test_ds = SmilesMultilabelDataset(
            metadata.iloc[test_idx]["canonical_smiles"].tolist(),
            labels[test_idx],
            tokenizer,
            args.max_length,
        )

        test_metrics = predict_and_save(
            trainer=trainer,
            dataset=test_ds,
            y_true=labels[test_idx],
            metadata_subset=metadata.iloc[test_idx],
            out_dir=out_dir,
            split_name=split_name,
            eval_set="bao_B_external_test",
            label_names=label_names,
        )
        all_metrics.append(test_metrics)

        print(
            f"Bao B external test: macro_f1={test_metrics['macro_f1']:.4f} "
            f"micro_f1={test_metrics['micro_f1']:.4f} "
            f"sample_f1={test_metrics['sample_f1']:.4f}"
        )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.insert(0, "model_name_or_path", args.model_name_or_path)
    metrics_df.insert(0, "run_name", args.run_name)
    metrics_df.insert(0, "task", "bao_pathway")
    metrics_df.insert(0, "n_train", len(train_idx))
    metrics_df.insert(0, "split_column", split_col)

    metrics_df.to_csv(out_dir / f"metrics.{split_name}.tsv", sep="\t", index=False)

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args) | {"split": split_name, "split_column": split_col}, f, indent=2)

    trainer.save_model(out_dir / "best_model")
    tokenizer.save_pretrained(out_dir / "best_model")

    return metrics_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name-or-path",
        default="DeepChem/ChemBERTa-77M-MLM",
        help="HF model path or local model directory.",
    )
    parser.add_argument(
        "--run-name",
        default="chemberta_77m_mlm",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["random", "scaffold"],
        choices=["random", "scaffold"],
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-pos-weight", type=float, default=50.0)
    parser.add_argument(
        "--pos-weight-mode",
        choices=["ratio", "sqrt", "none"],
        default="ratio",
        help="How to compute positive-class weights for multilabel BCE.",
    )

    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["query", "value"],
        help="LoRA target modules for Roberta attention layers.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(args.seed)

    metadata, labels, label_names = load_core()

    all_results = []
    for split_name in args.splits:
        result = run_one_split(args, split_name, metadata, labels, label_names)
        all_results.append(result)

    final = pd.concat(all_results, ignore_index=True)

    out_dir = OUT_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_dir / "metrics_all_splits.tsv", sep="\t", index=False)

    print()
    print("Saved:")
    print(out_dir / "metrics_all_splits.tsv")
    print(out_dir)


if __name__ == "__main__":
    main()

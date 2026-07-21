#!/usr/bin/env python

from pathlib import Path
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


class SmilesClassDataset(torch.utils.data.Dataset):
    def __init__(self, smiles, y, tokenizer, max_length):
        self.smiles = list(smiles)
        self.y = np.asarray(y, dtype=np.int64)
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
        enc["labels"] = int(self.y[idx])
        return enc


class WeightedCETrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
        else:
            weights = None

        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def softmax_np(x):
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)


def make_label_mapping(task):
    label_cols = [c for c in task.columns if c.startswith("label_")]
    if len(label_cols) < 2:
        raise ValueError("Too few label columns found.")

    Y = task[label_cols].to_numpy(dtype=int)
    cardinality = Y.sum(axis=1)

    keep = cardinality == 1
    dropped = int((~keep).sum())

    task2 = task.loc[keep].reset_index(drop=True)
    Y2 = Y[keep]

    y = Y2.argmax(axis=1).astype(int)

    # label_4 -> 4, etc.
    label_names = [c.replace("label_", "") for c in label_cols]

    return task2, y, label_cols, label_names, dropped


def get_split_masks(task, split_name):
    if split_name == "random":
        split_col = "split"
        train_name = "train"
        val_name = "val"
        test_name = "test"
    elif split_name == "scaffold":
        split_col = "scaffold_split"
        train_name = "scaffold_train"
        val_name = "scaffold_val"
        test_name = "scaffold_test"
    else:
        raise ValueError(split_name)

    train_mask = task[split_col].astype(str).eq(train_name).to_numpy()
    val_mask = task[split_col].astype(str).eq(val_name).to_numpy()
    test_mask = task[split_col].astype(str).eq(test_name).to_numpy()

    return split_col, train_mask, val_mask, test_mask


def class_weights_from_train(y_train, n_classes, mode):
    counts = np.bincount(y_train, minlength=n_classes).astype(float)

    if mode == "none":
        return None

    weights = len(y_train) / (n_classes * np.maximum(counts, 1.0))

    if mode == "sqrt_balanced":
        weights = np.sqrt(weights)
    elif mode == "balanced":
        pass
    else:
        raise ValueError(mode)

    # Normalize mean weight to 1 for numerical stability.
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics_fn(label_names):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred = np.asarray(logits).argmax(axis=1)
        labels = np.asarray(labels)

        return {
            "accuracy": accuracy_score(labels, pred),
            "balanced_accuracy": balanced_accuracy_score(labels, pred),
            "macro_f1": f1_score(labels, pred, average="macro", zero_division=0),
            "micro_f1": f1_score(labels, pred, average="micro", zero_division=0),
            "weighted_f1": f1_score(labels, pred, average="weighted", zero_division=0),
            "macro_precision": precision_score(labels, pred, average="macro", zero_division=0),
            "macro_recall": recall_score(labels, pred, average="macro", zero_division=0),
        }

    return compute_metrics


def save_predictions(out_dir, name, y_true, logits, label_names):
    probs = softmax_np(logits)
    y_pred = probs.argmax(axis=1)

    np.savez_compressed(
        out_dir / f"predictions.{name}.npz",
        y_true=y_true,
        y_pred=y_pred,
        y_score=probs,
        logits=logits,
        label_names=np.array(label_names, dtype=object),
    )

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names)))
    pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(
        out_dir / f"confusion_matrix.{name}.tsv",
        sep="\t",
    )

    rows = []
    for i, lab in enumerate(label_names):
        mask_true = y_true == i
        mask_pred = y_pred == i

        tp = int(np.logical_and(mask_true, mask_pred).sum())
        fp = int(np.logical_and(~mask_true, mask_pred).sum())
        fn = int(np.logical_and(mask_true, ~mask_pred).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        rows.append({
            "label": lab,
            "support": int(mask_true.sum()),
            "predicted": int(mask_pred.sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    pd.DataFrame(rows).to_csv(out_dir / f"per_class_metrics.{name}.tsv", sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--task", default="data/tasks/np_classification/npclassifier_label_task.tsv")
    ap.add_argument("--model-name-or-path", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--splits", nargs="+", default=["random", "scaffold"], choices=["random", "scaffold"])

    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--class-weight-mode", default="balanced", choices=["none", "balanced", "sqrt_balanced"])

    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--lora-target-modules", nargs="+", default=["query", "value"])

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--local-files-only", action="store_true")

    ap.add_argument("--output-root", default="results/np_classification/finetune")

    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    task = pd.read_csv(args.task, sep="\t")
    task, y, label_cols, label_names, dropped_multilabel = make_label_mapping(task)

    n_classes = len(label_names)

    print("Task:", args.task)
    print("Rows after single-label filter:", len(task))
    print("Dropped non-single-label rows:", dropped_multilabel)
    print("Classes:", n_classes)
    print("Label names:", label_names)
    print("Class counts:")
    print(pd.Series(y).value_counts().sort_index().rename(index=dict(enumerate(label_names))))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.cls_token

    all_metric_rows = []

    for split_name in args.splits:
        print("\n==============================")
        print("Split:", split_name)
        print("==============================")

        split_col, train_mask, val_mask, test_mask = get_split_masks(task, split_name)

        out_dir = Path(args.output_root) / args.run_name / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=n_classes,
            id2label={i: lab for i, lab in enumerate(label_names)},
            label2id={lab: i for i, lab in enumerate(label_names)},
            problem_type="single_label_classification",
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )

        if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

        if args.use_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules,
                modules_to_save=["classifier", "score"],
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        y_train = y[train_mask]
        y_val = y[val_mask]
        y_test = y[test_mask]

        class_weights = class_weights_from_train(
            y_train,
            n_classes=n_classes,
            mode=args.class_weight_mode,
        )

        print("n_train:", int(train_mask.sum()))
        print("n_val:", int(val_mask.sum()))
        print("n_test:", int(test_mask.sum()))
        print("class_weight_mode:", args.class_weight_mode)
        if class_weights is not None:
            print("class_weights:", class_weights.numpy().round(4).tolist())

        train_ds = SmilesClassDataset(
            task.loc[train_mask, "canonical_smiles"].tolist(),
            y_train,
            tokenizer,
            args.max_length,
        )
        val_ds = SmilesClassDataset(
            task.loc[val_mask, "canonical_smiles"].tolist(),
            y_val,
            tokenizer,
            args.max_length,
        )
        test_ds = SmilesClassDataset(
            task.loc[test_mask, "canonical_smiles"].tolist(),
            y_test,
            tokenizer,
            args.max_length,
        )

        training_args = TrainingArguments(
            output_dir=str(out_dir / "checkpoints"),
            overwrite_output_dir=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=2,
            fp16=args.fp16,
            bf16=args.bf16,
            report_to=[],
            seed=args.seed,
            dataloader_num_workers=2,
        )

        trainer = WeightedCETrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_fn(label_names),
            class_weights=class_weights,
        )

        trainer.train()

        val_metrics = trainer.evaluate(val_ds)
        test_pred = trainer.predict(test_ds)
        val_pred = trainer.predict(val_ds)

        test_logits = np.asarray(test_pred.predictions)
        val_logits = np.asarray(val_pred.predictions)

        y_test_pred = test_logits.argmax(axis=1)
        y_val_pred = val_logits.argmax(axis=1)

        test_metrics = {
            "eval_accuracy": accuracy_score(y_test, y_test_pred),
            "eval_balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
            "eval_macro_f1": f1_score(y_test, y_test_pred, average="macro", zero_division=0),
            "eval_micro_f1": f1_score(y_test, y_test_pred, average="micro", zero_division=0),
            "eval_weighted_f1": f1_score(y_test, y_test_pred, average="weighted", zero_division=0),
            "eval_macro_precision": precision_score(y_test, y_test_pred, average="macro", zero_division=0),
            "eval_macro_recall": recall_score(y_test, y_test_pred, average="macro", zero_division=0),
        }

        save_predictions(out_dir, "validation", y_val, val_logits, label_names)
        save_predictions(out_dir, "test", y_test, test_logits, label_names)

        trainer.save_model(str(out_dir / "best_model"))
        tokenizer.save_pretrained(str(out_dir / "best_model"))

        with open(out_dir / "label_names.json", "w") as f:
            json.dump(label_names, f, indent=2)

        with open(out_dir / "train_args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

        for eval_set, metrics in [("validation", val_metrics), ("test", test_metrics)]:
            row = {
                "task": "np_classification",
                "run_name": args.run_name,
                "model_name_or_path": args.model_name_or_path,
                "split": split_name,
                "split_column": split_col,
                "eval_set": eval_set,
                "n_train": int(train_mask.sum()),
                "n_eval": int(val_mask.sum() if eval_set == "validation" else test_mask.sum()),
                "n_classes": n_classes,
                "class_weight_mode": args.class_weight_mode,
                "use_lora": args.use_lora,
                "lora_r": args.lora_r if args.use_lora else None,
                "lora_alpha": args.lora_alpha if args.use_lora else None,
                "lora_dropout": args.lora_dropout if args.use_lora else None,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
            }

            for k, v in metrics.items():
                if k.startswith("eval_"):
                    row[k.replace("eval_", "")] = float(v)

            all_metric_rows.append(row)

        pd.DataFrame(all_metric_rows).to_csv(
            Path(args.output_root) / args.run_name / "metrics.tsv",
            sep="\t",
            index=False,
        )

        print("\nValidation metrics:")
        print(val_metrics)
        print("\nTest metrics:")
        print(test_metrics)

    metrics_path = Path(args.output_root) / args.run_name / "metrics.tsv"
    print("\nWrote:", metrics_path)
    print(pd.read_csv(metrics_path, sep="\t").to_string(index=False))


if __name__ == "__main__":
    main()

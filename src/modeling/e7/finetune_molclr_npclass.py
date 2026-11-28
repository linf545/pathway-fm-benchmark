#!/usr/bin/env python3
"""E7 MolCLR-GIN full fine-tuning on the existing NP-classification splits."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch_geometric.loader import DataLoader

from models.ginet_molclr import GINet
from src.modeling.extract_molclr_embeddings import smiles_to_data


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    loss_sum = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss_sum += nn.functional.cross_entropy(
                logits, batch.y, reduction="sum"
            ).item()
            count += batch.y.numel()
            probabilities = logits.softmax(dim=1)
            ys.append(batch.y.cpu().numpy())
            ps.append(probabilities.cpu().numpy())

    y = np.concatenate(ys)
    p = np.concatenate(ps)

    return {
        "cross_entropy_loss": float(loss_sum / count),
        "macro_auroc": float(
            roc_auc_score(y, p, multi_class="ovr", average="macro")
        ),
        "macro_auprc": float(np.mean([
            average_precision_score((y == k).astype(int), p[:, k])
            for k in range(7)
        ])),
        "macro_f1": float(
            f1_score(y, p.argmax(axis=1), average="macro", zero_division=0)
        ),
    }


class MolCLRClassifier(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()

        self.encoder = GINet(
            num_layer=5,
            emb_dim=300,
            feat_dim=512,
            drop_ratio=0.0,
            pool="mean",
        )

        state = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
        self.encoder.load_state_dict(state, strict=True)

        # The contrastive projection head is unused for classification.
        for p in self.encoder.out_lin.parameters():
            p.requires_grad_(False)

        self.head = nn.Linear(512, 7)

    def forward(self, graph):
        embedding, _ = self.encoder(graph)
        return self.head(embedding)



def _e7_balanced_ce_from_original_split(args, model):
    """Sklearn-style balanced class weights, computed on training data only."""
    import numpy as np
    import pandas as pd
    import torch

    task = pd.read_csv(
        "data/tasks/np_classification/npclassifier_label_task.tsv",
        sep="\t",
    )
    columns = [f"label_{i}" for i in range(1, 8)]
    labels = task[columns].to_numpy(dtype=int)

    assert (labels.sum(axis=1) == 1).all()

    mode = str(args.split_mode)
    seed = int(args.seed)
    split_column = f"{mode}_split_{seed}"
    train_value = "train" if mode == "random" else "scaffold_train"

    assert split_column in task.columns
    y_train = labels[
        task[split_column].astype(str).eq(train_value).to_numpy()
    ].argmax(axis=1)

    counts = np.bincount(y_train, minlength=7)
    assert len(y_train) > 0 and (counts > 0).all()

    weights = len(y_train) / (7 * counts)

    print(
        "E7_BALANCED_CLASS_WEIGHTS:",
        weights.tolist(),
        "training_counts:",
        counts.tolist(),
        flush=True,
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--split-mode", choices=["random", "scaffold"], default="random"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument(
        "--checkpoint",
        default="/private/vina/model_repos/MolCLR/"
                "ckpt/pretrained_gin/checkpoints/model.pth",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    assert torch.cuda.is_available(), "GPU required for this run"
    device = torch.device("cuda")

    out = Path("results/e7")
    if args.smoke:
        out = out / "smoke"
    out = out / "np_classification" / "molclr_gin" / (
        f"{args.split_mode}_seed{args.seed}"
    )
    out.mkdir(parents=True, exist_ok=True)

    task = pd.read_csv(
        "data/tasks/np_classification/npclassifier_label_task.tsv",
        sep="\t",
    )
    label_cols = [f"label_{i}" for i in range(1, 8)]
    labels = task[label_cols].to_numpy(dtype=int)
    assert np.isin(labels, [0, 1]).all()
    assert (labels.sum(axis=1) == 1).all()
    task["class_id"] = labels.argmax(axis=1)

    if args.split_mode == "random":
        col = f"random_split_{args.seed}"
        split_values = {
            "train": "train", "val": "val", "test": "test"
        }
    else:
        col = f"scaffold_split_{args.seed}"
        split_values = {
            "train": "scaffold_train",
            "val": "scaffold_val",
            "test": "scaffold_test",
        }

    assert set(task[col].dropna().unique()) == set(split_values.values())
    assert task[col].notna().all()

    loaders = {}

    for split, value in split_values.items():
        part = task.loc[task[col] == value]

        if args.smoke:
            n = 140 if split == "train" else 30
            part = pd.concat([
                group.sample(
                    n=min(n, len(group)), random_state=args.seed
                )
                for _, group in part.groupby("class_id")
            ]).sort_index()

        graphs = []
        for smiles, label in zip(
            part["canonical_smiles"], part["class_id"]
        ):
            graph = smiles_to_data(str(smiles))
            if graph is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            graph.y = torch.tensor([int(label)], dtype=torch.long)
            graphs.append(graph)

        loaders[split] = DataLoader(
            graphs,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )
        print(f"{split}: {len(graphs)} molecules", flush=True)

    model = MolCLRClassifier(args.checkpoint).to(device)
    encoder_parameters = [
        p for p in model.encoder.parameters() if p.requires_grad
    ]

    print("Pretrained MolCLR-GIN loaded", flush=True)
    print(
        "Trainable encoder parameters:",
        sum(p.numel() for p in encoder_parameters),
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss(weight=_e7_balanced_ce_from_original_split(args, model))
    best_val_loss = float("inf")
    best_val_auroc = float("-inf")
    best_val_auprc = float("-inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_n = 0

        for step, batch in enumerate(loaders["train"]):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()

            if epoch == 0 and step == 0:
                gradient = model.encoder.feat_lin.weight.grad
                assert gradient is not None
                assert gradient.abs().sum().item() > 0
                print("Encoder gradient check: PASS", flush=True)

            torch.nn.utils.clip_grad_norm_(
                encoder_parameters + list(model.head.parameters()), 1.0
            )
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            total_n += batch.num_graphs

        val = evaluate(model, loaders["val"], device)
        print(
            f"Epoch {epoch + 1}: train_loss={total_loss / total_n:.4f} "
            f"validation={val}",
            flush=True,
        )

        if (
            val["macro_auroc"] > best_val_auroc
            or (
                np.isclose(val["macro_auroc"], best_val_auroc,
                           rtol=0, atol=1e-12)
                and val["macro_auprc"] > best_val_auprc
            )
        ):
            best_val_loss = val["cross_entropy_loss"]
            best_val_auroc = float(val["macro_auroc"])
            best_val_auprc = float(val["macro_auprc"])
            torch.save(model.state_dict(), out / "best.pt")

    model.load_state_dict(
        torch.load(out / "best.pt", map_location=device, weights_only=True)
    )
    test = evaluate(model, loaders["test"], device)

    summary = {
        "model": "molclr_gin",
        "adaptation": "full",
        "split_mode": args.split_mode,
        "seed": args.seed,
        "smoke": args.smoke,
        "epochs": args.epochs,
        "best_validation_macro_auroc": best_val_auroc,
        "best_validation_loss": best_val_loss,
        "test": test,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("TEST:", test, flush=True)
    print("Saved:", out / "best.pt", flush=True)
    print("Saved:", out / "summary.json", flush=True)


if __name__ == "__main__":
    main()

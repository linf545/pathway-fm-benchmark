#!/usr/bin/env python3

import argparse
import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool


EXPECTED_EMBEDDING_DIM = 300


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_molebert(repo_dir: Path) -> tuple[Any, Any]:
    repo_dir = repo_dir.resolve()

    if not (repo_dir / "model.py").exists():
        raise FileNotFoundError(
            f"Mole-BERT model.py was not found in {repo_dir}"
        )

    if not (repo_dir / "loader.py").exists():
        raise FileNotFoundError(
            f"Mole-BERT loader.py was not found in {repo_dir}"
        )

    repo_string = str(repo_dir)

    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)

    model_module = importlib.import_module("model")
    loader_module = importlib.import_module("loader")

    return model_module.GNN, loader_module.mol_to_graph_data_obj_simple


def load_checkpoint(checkpoint: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        state = torch.load(
            checkpoint,
            map_location="cpu",
        )

    if not isinstance(state, dict):
        raise TypeError(
            "Expected the Mole-BERT checkpoint to contain a state dictionary, "
            f"but received {type(state)}"
        )

    return state


def resolve_device(device_argument: str) -> torch.device:
    if device_argument == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(device_argument)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {device_argument}, but CUDA is unavailable."
        )

    return device


def check_output_paths(
    paths: list[Path],
    overwrite: bool,
) -> None:
    existing = [path for path in paths if path.exists()]

    if existing and not overwrite:
        formatted = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace them:\n"
            + formatted
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen Mole-BERT graph embeddings using the official "
            "pretrained GIN encoder."
        )
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--id-col", default="inchikey")
    parser.add_argument("--smiles-col", default="canonical_smiles")

    parser.add_argument(
        "--repo-dir",
        default="/private/vina/model_repos/Mole-BERT",
    )
    parser.add_argument(
        "--ckpt",
        default=(
            "/private/vina/model_repos/Mole-BERT/"
            "model_gin/Mole-BERT.pth"
        ),
    )

    parser.add_argument(
        "--model-name",
        default="molebert_gin",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a device such as cuda:0",
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    repo_dir = Path(args.repo_dir).resolve()
    checkpoint_path = Path(args.ckpt).resolve()
    output_dir = Path(args.out_dir).resolve()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_path = (
        output_dir / f"{args.model_name}.embeddings.npy"
    )
    metadata_path = (
        output_dir / f"{args.model_name}.metadata.tsv"
    )
    invalid_path = (
        output_dir / f"{args.model_name}.invalid.tsv"
    )

    check_output_paths(
        [embedding_path, metadata_path, invalid_path],
        overwrite=args.overwrite,
    )

    df = pd.read_csv(input_path, sep="\t")

    required_columns = [args.id_col, args.smiles_col]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Input is missing columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    selected = (
        df[required_columns]
        .dropna(subset=required_columns)
        .drop_duplicates(subset=required_columns)
        .reset_index(drop=True)
    )

    if selected.empty:
        raise ValueError("No compounds remained after filtering.")

    GNN, mol_to_graph_data_obj_simple = import_molebert(
        repo_dir
    )

    device = resolve_device(args.device)

    print("input:", input_path)
    print("input rows:", len(df))
    print("unique nonmissing compound rows:", len(selected))
    print("repository:", repo_dir)
    print("checkpoint:", checkpoint_path)
    print("checkpoint sha256:", sha256sum(checkpoint_path))
    print("model:", args.model_name)
    print("device:", device)
    print("batch size:", args.batch_size)

    model = GNN(
        num_layer=5,
        emb_dim=EXPECTED_EMBEDDING_DIM,
        JK="last",
        drop_ratio=0.0,
        gnn_type="gin",
    )

    state = load_checkpoint(checkpoint_path)
    load_result = model.load_state_dict(state, strict=True)

    print("checkpoint loading:", load_result)

    model.to(device)
    model.eval()

    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    embedding_batches: list[np.ndarray] = []

    current_graphs = []
    current_rows: list[dict[str, Any]] = []

    def process_batch() -> None:
        if not current_graphs:
            return

        batch = Batch.from_data_list(current_graphs).to(device)

        with torch.inference_mode():
            node_embeddings = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
            )

            graph_embeddings = global_mean_pool(
                node_embeddings,
                batch.batch,
            )

        if graph_embeddings.ndim != 2:
            raise RuntimeError(
                "Expected a two-dimensional embedding matrix, "
                f"received shape {tuple(graph_embeddings.shape)}"
            )

        if graph_embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
            raise RuntimeError(
                "Unexpected Mole-BERT embedding dimension: "
                f"{graph_embeddings.shape[1]}"
            )

        if not torch.isfinite(graph_embeddings).all():
            raise RuntimeError(
                "Mole-BERT produced NaN or infinite embeddings."
            )

        embedding_batches.append(
            graph_embeddings.detach().cpu().numpy()
        )
        valid_rows.extend(current_rows)

        current_graphs.clear()
        current_rows.clear()

    for position, row in selected.iterrows():
        row_dict = row.to_dict()
        smiles = str(row_dict[args.smiles_col])

        try:
            molecule = Chem.MolFromSmiles(smiles)

            if molecule is None:
                raise ValueError("RDKit MolFromSmiles returned None")

            graph = mol_to_graph_data_obj_simple(molecule)

            current_graphs.append(graph)
            current_rows.append(row_dict)

            if len(current_graphs) >= args.batch_size:
                process_batch()

        except Exception as exc:
            invalid_record = dict(row_dict)
            invalid_record["error_type"] = type(exc).__name__
            invalid_record["error_message"] = str(exc)
            invalid_rows.append(invalid_record)

        if (position + 1) % 5000 == 0:
            print(
                f"processed {position + 1:,}/{len(selected):,}; "
                f"valid={len(valid_rows) + len(current_rows):,}; "
                f"invalid={len(invalid_rows):,}"
            )

    process_batch()

    if not embedding_batches:
        raise RuntimeError(
            "No valid Mole-BERT embeddings were generated."
        )

    embeddings = np.concatenate(
        embedding_batches,
        axis=0,
    ).astype(np.float32, copy=False)

    valid_df = pd.DataFrame(
        valid_rows,
        columns=required_columns,
    )

    invalid_columns = (
        required_columns + ["error_type", "error_message"]
    )
    invalid_df = pd.DataFrame(
        invalid_rows,
        columns=invalid_columns,
    )

    if len(valid_df) != embeddings.shape[0]:
        raise RuntimeError(
            "Metadata/embedding row mismatch: "
            f"metadata={len(valid_df)}, "
            f"embeddings={embeddings.shape[0]}"
        )

    if embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise RuntimeError(
            f"Expected 300 columns, found {embeddings.shape[1]}"
        )

    valid_df["model_name"] = args.model_name
    valid_df["embedding_dim"] = embeddings.shape[1]

    np.save(embedding_path, embeddings)
    valid_df.to_csv(
        metadata_path,
        sep="\t",
        index=False,
    )
    invalid_df.to_csv(
        invalid_path,
        sep="\t",
        index=False,
    )

    print("\n=== Mole-BERT extraction summary ===")
    print("input compounds:", len(selected))
    print("valid compounds:", len(valid_df))
    print("invalid compounds:", len(invalid_df))
    print("embedding shape:", embeddings.shape)
    print("embedding dtype:", embeddings.dtype)
    print("all finite:", bool(np.isfinite(embeddings).all()))
    print("wrote:", embedding_path)
    print("wrote:", metadata_path)
    print("wrote:", invalid_path)
    print("Mole-BERT embedding extraction passed")


if __name__ == "__main__":
    main()

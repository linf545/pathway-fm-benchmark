#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
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


def load_source_module(module_name: str, path: Path) -> Any:
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def import_graphmvp(repo_dir: Path) -> tuple[Any, Any]:
    source_dir = repo_dir.resolve() / "src_classification"

    model_module = load_source_module(
        "graphmvp_molecule_gnn_model",
        source_dir / "models" / "molecule_gnn_model.py",
    )

    dataset_module = load_source_module(
        "graphmvp_molecule_datasets",
        source_dir / "datasets" / "molecule_datasets.py",
    )

    return (
        model_module.GNN,
        dataset_module.mol_to_graph_data_obj_simple,
    )


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        state = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(state, dict):
        raise TypeError(
            "Expected a state dictionary, but checkpoint contains "
            f"{type(state)}"
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
            f"CUDA device requested but unavailable: {device_argument}"
        )

    return device


def ensure_outputs_available(
    paths: list[Path],
    overwrite: bool,
) -> None:
    existing = [path for path in paths if path.exists()]

    if existing and not overwrite:
        existing_text = "\n".join(str(path) for path in existing)

        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace:\n"
            + existing_text
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen graph embeddings from the official standard "
            "GraphMVP simple-feature classification GIN encoder."
        )
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--id-col", default="inchikey")
    parser.add_argument("--smiles-col", default="canonical_smiles")

    parser.add_argument(
        "--repo-dir",
        default="/private/vina/model_repos/GraphMVP",
    )

    parser.add_argument(
        "--ckpt",
        default=(
            "/private/vina/model_repos/GraphMVP/pretrained/"
            "graphmvp_classification_simple_m015_t02/model.pth"
        ),
    )

    parser.add_argument(
        "--model-name",
        default="graphmvp_gin",
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

    ensure_outputs_available(
        [embedding_path, metadata_path, invalid_path],
        overwrite=args.overwrite,
    )

    input_df = pd.read_csv(input_path, sep="\t")

    required_columns = [args.id_col, args.smiles_col]
    missing_columns = [
        column
        for column in required_columns
        if column not in input_df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Input is missing columns {missing_columns}. "
            f"Available columns: {list(input_df.columns)}"
        )

    compounds = (
        input_df[required_columns]
        .dropna(subset=required_columns)
        .drop_duplicates(subset=required_columns)
        .reset_index(drop=True)
    )

    if compounds.empty:
        raise ValueError("No valid input rows remain after filtering.")

    duplicated_ids = compounds[args.id_col].duplicated(
        keep=False
    )

    if duplicated_ids.any():
        examples = compounds.loc[
            duplicated_ids,
            required_columns,
        ].head(20)

        raise ValueError(
            f"Duplicate {args.id_col} values map to multiple rows:\n"
            + examples.to_string(index=False)
        )

    GNN, mol_to_graph_data_obj_simple = import_graphmvp(
        repo_dir
    )

    device = resolve_device(args.device)

    print("input:", input_path)
    print("input rows:", len(input_df))
    print("unique nonmissing compounds:", len(compounds))
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

    load_result = model.load_state_dict(
        state,
        strict=True,
    )

    print("checkpoint loading:", load_result)

    model.to(device)
    model.eval()

    embedding_batches: list[np.ndarray] = []
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []

    batch_graphs = []
    batch_rows: list[dict[str, Any]] = []

    def process_batch() -> None:
        if not batch_graphs:
            return

        graph_batch = Batch.from_data_list(
            batch_graphs
        ).to(device)

        with torch.inference_mode():
            node_embeddings = model(
                graph_batch.x,
                graph_batch.edge_index,
                graph_batch.edge_attr,
            )

            graph_embeddings = global_mean_pool(
                node_embeddings,
                graph_batch.batch,
            )

        if graph_embeddings.ndim != 2:
            raise RuntimeError(
                "Expected a two-dimensional embedding matrix, "
                f"received {tuple(graph_embeddings.shape)}"
            )

        if graph_embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
            raise RuntimeError(
                "Unexpected embedding dimension: "
                f"{graph_embeddings.shape[1]}"
            )

        if not torch.isfinite(graph_embeddings).all():
            raise RuntimeError(
                "GraphMVP produced NaN or infinite embeddings."
            )

        embedding_batches.append(
            graph_embeddings.detach().cpu().numpy()
        )

        valid_rows.extend(batch_rows)

        batch_graphs.clear()
        batch_rows.clear()

    for position, row in compounds.iterrows():
        row_dict = row.to_dict()
        smiles = str(row_dict[args.smiles_col])

        try:
            molecule = Chem.MolFromSmiles(smiles)

            if molecule is None:
                raise ValueError(
                    "RDKit MolFromSmiles returned None"
                )

            graph = mol_to_graph_data_obj_simple(
                molecule
            )

            batch_graphs.append(graph)
            batch_rows.append(row_dict)

            if len(batch_graphs) >= args.batch_size:
                process_batch()

        except Exception as exc:
            invalid_record = dict(row_dict)
            invalid_record["error_type"] = type(exc).__name__
            invalid_record["error_message"] = str(exc)

            invalid_rows.append(invalid_record)

        if (position + 1) % 5000 == 0:
            print(
                f"processed {position + 1:,}/{len(compounds):,}; "
                f"valid={len(valid_rows) + len(batch_rows):,}; "
                f"invalid={len(invalid_rows):,}"
            )

    process_batch()

    if not embedding_batches:
        raise RuntimeError(
            "No GraphMVP embeddings were generated."
        )

    embeddings = np.concatenate(
        embedding_batches,
        axis=0,
    ).astype(np.float32, copy=False)

    valid_df = pd.DataFrame(
        valid_rows,
        columns=required_columns,
    )

    invalid_df = pd.DataFrame(
        invalid_rows,
        columns=(
            required_columns
            + ["error_type", "error_message"]
        ),
    )

    if embeddings.shape[0] != len(valid_df):
        raise RuntimeError(
            "Embedding/metadata mismatch: "
            f"embeddings={embeddings.shape[0]}, "
            f"metadata={len(valid_df)}"
        )

    if embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise RuntimeError(
            f"Expected 300 dimensions, found {embeddings.shape[1]}"
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

    print("\n=== GraphMVP extraction summary ===")
    print("input compounds:", len(compounds))
    print("valid compounds:", len(valid_df))
    print("invalid compounds:", len(invalid_df))
    print("embedding shape:", embeddings.shape)
    print("embedding dtype:", embeddings.dtype)
    print("all finite:", bool(np.isfinite(embeddings).all()))
    print("wrote:", embedding_path)
    print("wrote:", metadata_path)
    print("wrote:", invalid_path)
    print("GraphMVP embedding extraction passed")


if __name__ == "__main__":
    main()

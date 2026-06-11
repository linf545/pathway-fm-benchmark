import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, Batch
from torch_geometric.nn.pool import global_mean_pool

from gnn.models.gnn import PretrainedGNN


# NaFM expects:
# x columns: atom_type, chirality_tag, formal_charge
# edge_attr columns: bond_type, bond_direction

ATOM_LIST = list(range(1, 10))  # NaFM embedding size is 10: indices 0..9
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
CHARGE_LIST = [-2, -1, 0, 1, 2]

BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
    Chem.rdchem.BondDir.EITHERDOUBLE,
]


def safe_index_with_unknown(lst, value):
    try:
        return lst.index(value)
    except ValueError:
        return len(lst)


def safe_index_clamped(lst, value):
    try:
        return lst.index(value)
    except ValueError:
        return 0


def smiles_to_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        atom_idx = atomic_num if 1 <= atomic_num <= 9 else 0

        chirality_idx = safe_index_with_unknown(CHIRALITY_LIST, atom.GetChiralTag())
        if chirality_idx > 4:
            chirality_idx = 0

        charge_idx = safe_index_with_unknown(CHARGE_LIST, atom.GetFormalCharge())
        if charge_idx > 5:
            charge_idx = 0

        atom_features.append([atom_idx, chirality_idx, charge_idx])

    x = torch.tensor(atom_features, dtype=torch.long)

    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_type_idx = safe_index_with_unknown(BOND_LIST, bond.GetBondType())
        if bond_type_idx > 4:
            bond_type_idx = 0

        bond_dir_idx = safe_index_with_unknown(BONDDIR_LIST, bond.GetBondDir())
        if bond_dir_idx > 4:
            bond_dir_idx = 0

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [[bond_type_idx, bond_dir_idx], [bond_type_idx, bond_dir_idx]]

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def load_nafm(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt["hyper_parameters"]

    model = PretrainedGNN(
        num_layer=hp["num_layer"],
        emb_dim=hp["emb_dim"],
        drop_ratio=hp["drop_ratio"],
        gnn_type=hp["gnn_type"],
        feat_dim=hp["feat_dim"],
    )

    prefix = "model.pretrain_gnn."
    state = {
        k[len(prefix):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith(prefix)
    }

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing:", len(missing), missing[:20])
    print("unexpected:", len(unexpected), unexpected[:20])

    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--id-col", default="inchikey")
    parser.add_argument("--smiles-col", default="canonical_smiles")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model-name", default="nafm")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--embedding-type",
        choices=["contrast", "graph_mean"],
        default="contrast",
        help="contrast gives PretrainedGNN output dim 256; graph_mean gives pooled representation_model dim 1024.",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, sep="\t")
    df = df[[args.id_col, args.smiles_col]].dropna().drop_duplicates()

    valid_rows = []
    data_list = []
    invalid_rows = []

    for _, row in df.iterrows():
        smi = str(row[args.smiles_col])
        data = smiles_to_data(smi)
        if data is None:
            invalid_rows.append(row)
            continue
        valid_rows.append(row)
        data_list.append(data)

    valid_df = pd.DataFrame(valid_rows)
    invalid_df = pd.DataFrame(invalid_rows)

    print("input:", inp)
    print("model:", args.model_name)
    print("embedding_type:", args.embedding_type)
    print("valid compounds:", len(data_list))
    print("invalid compounds:", len(invalid_df))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = load_nafm(args.ckpt, device)

    all_emb = []

    for start in range(0, len(data_list), args.batch_size):
        batch_data = Batch.from_data_list(data_list[start:start + args.batch_size]).to(device)

        with torch.no_grad():
            if args.embedding_type == "contrast":
                out = model(batch_data, masked=False)
            else:
                node_emb = model.representation_model(batch_data)
                out = global_mean_pool(node_emb, batch_data.batch)

        all_emb.append(out.detach().cpu().numpy())

    X = np.vstack(all_emb)

    emb_path = out_dir / f"{args.model_name}.embeddings.npy"
    meta_path = out_dir / f"{args.model_name}.metadata.tsv"
    invalid_path = out_dir / f"{args.model_name}.invalid.tsv"

    np.save(emb_path, X)

    meta = valid_df.copy()
    meta["model_name"] = args.model_name
    meta["embedding_dim"] = X.shape[1]
    meta["embedding_type"] = args.embedding_type
    meta.to_csv(meta_path, sep="\t", index=False)

    invalid_df.to_csv(invalid_path, sep="\t", index=False)

    print("embedding shape:", X.shape)
    print("wrote:", emb_path)
    print("wrote:", meta_path)
    print("wrote:", invalid_path)
    print("NaFM embedding extraction passed")


if __name__ == "__main__":
    main()

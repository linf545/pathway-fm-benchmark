import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, Batch

from models.gcn_molclr import GCN
from models.ginet_molclr import GINet


ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
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
]


def safe_index(lst, value):
    try:
        return lst.index(value)
    except ValueError:
        return len(lst) - 1


def smiles_to_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        atom_type = safe_index(ATOM_LIST, atom.GetAtomicNum())
        chirality = safe_index(CHIRALITY_LIST, atom.GetChiralTag())
        atom_features.append([atom_type, chirality])

    x = torch.tensor(atom_features, dtype=torch.long)

    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_type = safe_index(BOND_LIST, bond.GetBondType())
        bond_dir = safe_index(BONDDIR_LIST, bond.GetBondDir())

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [[bond_type, bond_dir], [bond_type, bond_dir]]

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def load_molclr_model(model_type, ckpt_path, device):
    if model_type == "gcn":
        model = GCN(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0.0, pool="mean")
    elif model_type == "gin":
        model = GINet(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0.0, pool="mean")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    state = torch.load(ckpt_path, map_location="cpu")
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
    parser.add_argument("--model-type", choices=["gcn", "gin"], required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
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
    print("valid compounds:", len(data_list))
    print("invalid compounds:", len(invalid_df))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = load_molclr_model(args.model_type, args.ckpt, device)

    all_emb = []

    for start in range(0, len(data_list), args.batch_size):
        batch_data = Batch.from_data_list(data_list[start:start + args.batch_size]).to(device)

        with torch.no_grad():
            out = model(batch_data)

        # MolCLR models may return a tuple, e.g.:
        #   (graph_embedding, projection_output)
        # For benchmark features, use the graph_embedding.
        if isinstance(out, (tuple, list)):
            print("model output tuple shapes:", [
                tuple(x.shape) if hasattr(x, "shape") else type(x)
                for x in out
            ])
            out = out[0]

        all_emb.append(out.detach().cpu().numpy())

    X = np.vstack(all_emb)

    emb_path = out_dir / f"{args.model_name}.embeddings.npy"
    meta_path = out_dir / f"{args.model_name}.metadata.tsv"
    invalid_path = out_dir / f"{args.model_name}.invalid.tsv"

    np.save(emb_path, X)

    meta = valid_df.copy()
    meta["model_name"] = args.model_name
    meta["embedding_dim"] = X.shape[1]
    meta.to_csv(meta_path, sep="\t", index=False)

    invalid_df.to_csv(invalid_path, sep="\t", index=False)

    print("embedding shape:", X.shape)
    print("wrote:", emb_path)
    print("wrote:", meta_path)
    print("wrote:", invalid_path)
    print("MolCLR embedding extraction passed")


if __name__ == "__main__":
    main()

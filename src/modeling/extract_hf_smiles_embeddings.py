import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def load_model(model_dir, model_class, trust_remote_code, local_files_only):
    if model_class == "auto":
        return AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    if model_class == "mlm":
        return AutoModelForMaskedLM.from_pretrained(
            model_dir,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    raise ValueError(f"Unknown model_class: {model_class}")


def get_hidden(outputs):
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states[-1]
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        return outputs[0]
    raise RuntimeError("Could not find hidden states in model output.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--id-col", default="inchikey")
    parser.add_argument("--smiles-col", default="canonical_smiles")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--model-class", choices=["auto", "mlm"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    inp = Path(args.input)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, sep="\t")
    df = df[[args.id_col, args.smiles_col]].dropna().drop_duplicates()

    ids = df[args.id_col].astype(str).tolist()
    smiles = df[args.smiles_col].astype(str).tolist()

    print("input:", inp)
    print("model:", args.model_name)
    print("model_dir:", model_dir)
    print("n_compounds:", len(smiles))

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model = load_model(
        model_dir,
        args.model_class,
        args.trust_remote_code,
        args.local_files_only,
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("device:", device)

    all_emb = []

    for start in range(0, len(smiles), args.batch_size):
        batch_smiles = smiles[start:start + args.batch_size]

        enc = tokenizer(
            batch_smiles,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)

        hidden = get_hidden(outputs)
        emb = mean_pool(hidden, enc["attention_mask"])
        all_emb.append(emb.detach().cpu().numpy())

    X = np.vstack(all_emb)

    emb_path = out_dir / f"{args.model_name}.embeddings.npy"
    meta_path = out_dir / f"{args.model_name}.metadata.tsv"

    np.save(emb_path, X)

    meta = pd.DataFrame({
        args.id_col: ids,
        args.smiles_col: smiles,
        "model_name": args.model_name,
        "embedding_dim": X.shape[1],
    })
    meta.to_csv(meta_path, sep="\t", index=False)

    print("embedding shape:", X.shape)
    print("wrote:", emb_path)
    print("wrote:", meta_path)
    print("HF embedding extraction passed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/private/vina/model_cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/private/vina/model_cache/huggingface/transformers")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM


MODEL_PATHS = {
    "chemberta_77m_mlm": "/private/vina/model_repos/ChemBERTa-77M-MLM",

    # Add these later only if the actual model checkpoint folders exist.
    #"c3_molformer": "/private/vina/model_repos/MoLFormer-c3-1.1B",
    "ibm_molformer_hf": "/private/vina/model_repos/molformer",
}

SMILES = [
    "CCO",
    "COc1ccc(C[C@H]2c3cc(OC)c(OC)cc3CC[NH+]2C)cc1O",
    "CCCCCCCCCCCC(=O)[O-]",
]


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def load_model(path):
    try:
        model = AutoModel.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=True,
        )
        model_kind = "AutoModel"
        return model, model_kind
    except Exception as e1:
        print(f"AutoModel failed: {repr(e1)}")
        print("Trying AutoModelForMaskedLM...")

        model = AutoModelForMaskedLM.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=True,
        )
        model_kind = "AutoModelForMaskedLM"
        return model, model_kind


def get_hidden_state(output):
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state

    if hasattr(output, "hidden_states") and output.hidden_states is not None:
        return output.hidden_states[-1]

    raise RuntimeError("Could not find hidden states in model output.")


def main():
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for name, path in MODEL_PATHS.items():
        print("\n" + "=" * 80)
        print(f"Testing: {name}")
        print(f"Path: {path}")

        p = Path(path)
        if not p.exists():
            print("SKIP: path does not exist")
            continue

        if not (p / "config.json").exists():
            print("SKIP: config.json not found. This is not a local HF model folder.")
            continue

        tokenizer = AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=True,
        )

        model, model_kind = load_model(path)
        model = model.to(device)
        model.eval()

        batch = tokenizer(
            SMILES,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            output = model(**batch, output_hidden_states=True)

        hidden = get_hidden_state(output)
        emb = mean_pool(hidden, batch["attention_mask"])

        n_param = sum(x.numel() for x in model.parameters())

        print("model_kind:", model_kind)
        print(f"parameters: {n_param:,} ({n_param / 1e6:.1f}M)")
        print("hidden:", tuple(hidden.shape))
        print("embedding:", tuple(emb.shape))
        print("OK")


if __name__ == "__main__":
    main()
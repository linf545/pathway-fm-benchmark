#!/usr/bin/env python3
"""E6: Extract frozen CheMeleon or Mol-JEPA representations with resumable chunks.

Outputs match existing benchmark conventions:
    <model>.embeddings.npy  (float32; same row order as metadata)
    <model>.metadata.tsv    (inchikey, canonical_smiles, model_name, embedding_dim)
    <model>.invalid.tsv     (molecules the extractor could not represent)
    <model>.qc.json         (model, input hash, dimensions, counts)

Keep .<model>.chunks while running; completed chunks resume automatically.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SPECS = {"chemeleon": 2048, "moljepa_cls": 512}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def infer_or_split(featurizer, rows, id_col, smiles_col, dim):
    """Preserve input order, isolate failures, and never pad missing features."""
    if not rows:
        return [], []
    try:
        x = np.asarray(featurizer([r[smiles_col] for r in rows]), dtype=np.float32)
        if x.shape != (len(rows), dim):
            raise ValueError(f"Unexpected embedding shape {x.shape}, expected {(len(rows), dim)}")
        if not np.isfinite(x).all():
            raise ValueError("Non-finite embedding")
        return list(zip(rows, x)), []
    except Exception as exc:
        if len(rows) == 1:
            bad = dict(rows[0])
            bad.update(error_type=type(exc).__name__, error_message=str(exc)[:400])
            return [], [bad]
        mid = len(rows) // 2
        l_good, l_bad = infer_or_split(featurizer, rows[:mid], id_col, smiles_col, dim)
        r_good, r_bad = infer_or_split(featurizer, rows[mid:], id_col, smiles_col, dim)
        return l_good + r_good, l_bad + r_bad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--model", choices=SPECS, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--id-col", default="inchikey")
    p.add_argument("--smiles-col", default="canonical_smiles")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--limit", type=int, default=None, help="Pilot: first N unique molecules")
    p.add_argument("--accept-moljepa-license", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or args.chunk_size < 1:
        p.error("Batch and chunk sizes must be positive")
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    if args.model == "moljepa_cls" and not args.accept_moljepa_license:
        p.error("Mol-JEPA uses CC BY-NC 4.0; pass --accept-moljepa-license after checking terms")

    input_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.model
    dim = SPECS[stem]
    outputs = {suffix: out_dir / f"{stem}.{suffix}" for suffix in
               ("embeddings.npy", "metadata.tsv", "invalid.tsv", "qc.json")}
    work = out_dir / f".{stem}.chunks"
    if args.overwrite:
        for path in outputs.values():
            path.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError(f"Output already exists for {stem}; use --overwrite to replace")

    df = pd.read_csv(input_path, sep="\t", usecols=[args.id_col, args.smiles_col])
    df = df.dropna(subset=[args.id_col, args.smiles_col]).drop_duplicates().reset_index(drop=True)
    if df[args.id_col].duplicated().any():
        raise ValueError(f"Multiple SMILES associated with {args.id_col}; resolve before embedding")
    if args.limit is not None:
        df = df.head(args.limit).copy()
    if df.empty:
        raise ValueError("No molecules remain")
    df[args.id_col] = df[args.id_col].astype(str)
    df[args.smiles_col] = df[args.smiles_col].astype(str)
    n = len(df)
    manifest = {"input": str(input_path), "input_sha256": sha256(input_path),
                "n_molecules": n, "limit": args.limit, "model_name": stem,
                "embedding_dim": dim, "id_col": args.id_col,
                "smiles_col": args.smiles_col, "chunk_size": args.chunk_size}
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = work / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old != manifest:
            raise ValueError("Input/options differ from unfinished extraction; use --overwrite")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if stem == "chemeleon":
        from molfeat.trans.pretrained import CheMeleonTransformer
        featurizer = CheMeleonTransformer(device=device)
        provenance = {"adapter": "CheMeleonTransformer", "checkpoint": featurizer.checkpoint_path}
    else:
        from molfeat.trans.pretrained import MolJEPATransformer
        featurizer = MolJEPATransformer(device=device, output="cls",
                                       trust_remote_code=True,
                                       accept_noncommercial_license=True)
        provenance = {"adapter": "MolJEPATransformer", "repo": featurizer.model_name,
                      "revision": featurizer.revision, "output": "cls",
                      "license": "CC BY-NC 4.0"}
    print(f"Model={stem} device={device} n={n:,} dim={dim}; batch={args.batch_size}, chunk={args.chunk_size}", flush=True)

    keys = [args.id_col, args.smiles_col]
    nchunks = (n + args.chunk_size - 1) // args.chunk_size
    for j in range(nchunks):
        base = work / f"{j:05d}"
        done = Path(str(base) + ".done")
        if done.exists():
            print(f"Chunk {j+1}/{nchunks}: already complete", flush=True)
            continue
        chunk_df = df.iloc[j*args.chunk_size : (j+1)*args.chunk_size]
        records = chunk_df[keys].to_dict("records")
        good, bad = [], []
        for start in range(0, len(records), args.batch_size):
            g, b = infer_or_split(featurizer, records[start:start+args.batch_size],
                                  args.id_col, args.smiles_col, dim)
            good.extend(g)
            bad.extend(b)
            if len(b) >= 8 and not good:
                raise RuntimeError(f"Repeated inference errors in chunk {j}; first: {b[0]['error_message']}")
        x = np.stack([v for _, v in good]).astype(np.float32) if good else np.empty((0, dim), np.float32)
        metadata = pd.DataFrame([r for r, _ in good], columns=keys)
        invalid = pd.DataFrame(bad, columns=keys + ["error_type", "error_message"])
        with open(str(base) + ".npy.tmp", "wb") as f:
            np.save(f, x)
        metadata.to_csv(str(base) + ".meta.tsv.tmp", sep="\t", index=False)
        invalid.to_csv(str(base) + ".invalid.tsv.tmp", sep="\t", index=False)
        for suffix in (".npy", ".meta.tsv", ".invalid.tsv"):
            os.replace(str(base) + suffix + ".tmp", str(base) + suffix)
        done.write_text("ok\n")
        print(f"Chunk {j+1}/{nchunks}: valid={len(good):,} invalid={len(bad):,}", flush=True)

    count = 0
    for j in range(nchunks):
        base = work / f"{j:05d}"
        count += np.load(str(base) + ".npy", mmap_mode="r").shape[0]
    if not count:
        raise RuntimeError("No usable embeddings")

    temp_emb = Path(str(outputs["embeddings.npy"]) + ".tmp")
    X = np.lib.format.open_memmap(temp_emb, mode="w+", dtype=np.float32, shape=(count, dim))
    temp_meta = Path(str(outputs["metadata.tsv"]) + ".tmp")
    temp_bad = Path(str(outputs["invalid.tsv"]) + ".tmp")
    with open(temp_meta, "w") as mh, open(temp_bad, "w") as bh:
        offset = 0
        bad_count = 0
        for j in range(nchunks):
            base = work / f"{j:05d}"
            block = np.load(str(base) + ".npy", mmap_mode="r")
            m = pd.read_csv(str(base) + ".meta.tsv", sep="\t")
            b = pd.read_csv(str(base) + ".invalid.tsv", sep="\t")
            if block.shape != (len(m), dim) or len(block) + len(b) != len(df.iloc[j*args.chunk_size:(j+1)*args.chunk_size]):
                raise RuntimeError(f"Chunk {j} feature/metadata/count mismatch")
            nblock = len(block)
            if nblock:
                X[offset:offset+nblock] = block
                offset += nblock
            m["model_name"] = stem
            m["embedding_dim"] = dim
            m.to_csv(mh, sep="\t", index=False, header=(j == 0))
            b.to_csv(bh, sep="\t", index=False, header=(j == 0))
            bad_count += len(b)
            del block  # Release memory map before cleanup
    X.flush()
    del X
    qc = {**manifest, **provenance, "device": str(device), "n_valid": count,
          "n_invalid": bad_count, "dtype": "float32"}
    temp_qc = Path(str(outputs["qc.json"]) + ".tmp")
    temp_qc.write_text(json.dumps(qc, indent=2) + "\n")
    for label, tmp in (("embeddings.npy", temp_emb), ("metadata.tsv", temp_meta),
                       ("invalid.tsv", temp_bad), ("qc.json", temp_qc)):
        os.replace(tmp, outputs[label])
    try:
        shutil.rmtree(work)
    except OSError as exc:
        print(f"WARNING: outputs saved; chunk cleanup failed: {exc}")
    print(f"PASS: {stem} shape=({count}, {dim}) invalid={bad_count}")
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

"""
Train classifiers on pretrained foundation-model molecular embeddings
for Bao plant pathway prediction.

This script evaluates precomputed embeddings from:
    - ChemBERTa
    - MolCLR
    - MolFormer
    - NaFM

The foundation-model weights are not updated.
Only downstream linear classifier weights are trained.
"""

from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
ENGINE = SCRIPT_DIR / "train_linear_baselines.py"

FEATURES = [
    "chemberta_77m_mlm",
    "chemberta_100m_mlm",
    "molclr_gcn",
    "molclr_gin",
    "molformer_c3_1_1b",
    "nafm_contrast",
    "nafm_graph_mean",
]

cmd = [
    sys.executable,
    str(ENGINE),
    "--features",
    *FEATURES,
    "--splits",
    "random",
    "scaffold",
    "--output-prefix",
    "fm_embedding_classifier",
]

subprocess.run(cmd, check=True)

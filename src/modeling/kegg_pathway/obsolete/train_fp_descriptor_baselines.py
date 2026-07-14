#!/usr/bin/env python

"""
Train classical molecular-feature baselines for Bao plant pathway prediction.

This script evaluates:
    - Morgan fingerprints
    - RDKit molecular descriptors

The foundation-model embeddings are intentionally excluded.
"""

from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
ENGINE = SCRIPT_DIR / "train_linear_baselines.py"

FEATURES = [
    "morgan_r2_2048",
    "morgan_r3_2048",
    "rdkit_descriptors",
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
    "fp_descriptor_baseline",
]

subprocess.run(cmd, check=True)

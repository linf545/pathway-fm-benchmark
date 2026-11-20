#!/usr/bin/env bash
set -euo pipefail
cd /private/vina/FM_bench
# Initialize conda for execution via bash rather than an interactive shell.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate plantfm_data
mkdir -p logs/modeling/np_likeness

# STEP 1: Match NP/ZINC molecules by molecular weight, logP, exact ring count.
# Also create new random/scaffold splits, 3-descriptor embedding features,
# QC plots, and symlinks to the already extracted FM embeddings.
python src/data_processing/make_np_likeness_property_matched.py \
  --mw-bin 25 --logp-bin 0.5 \
  --split-seeds 2026 2027 2028 2029 2030 \
  2>&1 | tee logs/modeling/np_likeness/E1_matching.log

# Inspect QC before proceeding (especially n_matched and balance SMD).
cat results/np_likeness/matched/matching_qc.json
cat results/np_likeness/matched/property_balance_summary.tsv

# STEP 2: Original dataset: the 3-descriptor LR baseline, plus Morgan.
# The original 8-FM benchmark results are already available in results/np_likeness/full.
python src/modeling/np_likeness/run_np_likeness_full_ensemble.py \
  --task-file data/tasks/np_likeness/np_likeness_task.tsv \
  --split-dir data/splits/np_likeness \
  --feature-dir data/features/np_likeness \
  --output-dir results/np_likeness/original_descriptors \
  --fm-patterns mw_logp_rings \
  --split-mode both \
  --split-seeds 2026 2027 2028 2029 2030 \
  --skip-predictions \
  2>&1 | tee logs/modeling/np_likeness/E1_original_descriptors.log

# STEP 3: Full original Morgan + 8 FM + descriptor comparison, on matched data.
python src/modeling/np_likeness/run_np_likeness_full_ensemble.py \
  --task-file data/tasks/np_likeness_matched/np_likeness_task.tsv \
  --split-dir data/splits/np_likeness_matched \
  --feature-dir data/features/np_likeness_matched \
  --output-dir results/np_likeness/matched/full \
  --split-mode both \
  --split-seeds 2026 2027 2028 2029 2030 \
  --fm-patterns \
    chemberta_77m_mlm chemberta_100m_mlm molformer_c3_1_1b \
    nafm_graph_mean molclr_gcn molclr_gin graphmvp_gin molebert_gin \
    mw_logp_rings \
  --skip-predictions \
  2>&1 | tee logs/modeling/np_likeness/E1_matched_full.log

# STEP 4: Assemble original vs matched AUROC/AUPRC table and figure.
python src/analysis/compare_np_likeness_e1.py

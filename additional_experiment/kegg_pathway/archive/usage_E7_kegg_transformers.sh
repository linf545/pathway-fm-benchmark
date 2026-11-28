#!/usr/bin/env bash
# E7 KEGG (Bao Dataset C) — transformer full FT and LoRA.
# Copy to additional_experiment/kegg_pathway/usage_E7_kegg_transformers.sh
# Intended for execution AFTER the NP-classification MolFormer batch completes.
set -euo pipefail
cd /private/vina/FM_bench
mkdir -p logs/e7/kegg_pathway results/e7/kegg_pathway results/e7/computation_budget
{
  date -Is
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
} > "results/e7/computation_budget/kegg_transformer_hardware_$(date +%Y%m%d_%H%M%S).txt"
SCRIPT=src/modeling/e7/finetune_kegg_transformer.py
ENV=/private/vina/conda_envs/plantfm_chemberta_hf

audit() {
  CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -p "$ENV" \
      python "$SCRIPT" \
      --model-name-or-path /private/vina/model_repos/ChemBERTa-77M-MLM \
      --feature-name chemberta_77m_mlm \
      --data-check-only
}

run() {
  local model="$1" path="$2" length="$3" remote="$4" gpu="$5"
  local extra=()
  if [[ "$remote" == true ]]; then extra+=(--trust-remote-code); fi
  CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -p "$ENV" \
    python "$SCRIPT" \
      --model-name-or-path "$path" \
      --feature-name "$model" \
      --adaptation-modes full lora \
      --split-mode both \
      --split-seeds 2026 2027 2028 2029 2030 \
      --output-root results/e7/kegg_pathway \
      --epochs 5 \
      --train-batch-size 16 \
      --eval-batch-size 64 \
      --gradient-accumulation-steps 4 \
      --full-learning-rate 2e-5 \
      --lora-learning-rate 1e-4 \
      --head-learning-rate 1e-3 \
      --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
      --lora-target-modules query value \
      --max-length "$length" \
      --precision bf16 \
      --num-workers 2 \
      --local-files-only \
      --resume "${extra[@]}" \
      2>&1 | tee "logs/e7/kegg_pathway/${model}.log"
}

case "${1:-}" in
  audit) audit ;;
  chemberta) run chemberta_77m_mlm /private/vina/model_repos/ChemBERTa-77M-MLM 256 false "${2:-0}" ;;
  molformer) run molformer_c3_1_1b /private/vina/model_repos/molformer_c3_1_1b_e7 202 true "${2:-1}" ;;
  *)
    echo 'Usage: bash additional_experiment/kegg_pathway/usage_E7_kegg_transformers.sh {audit|chemberta [gpu]|molformer [gpu]}' >&2
    exit 2
    ;;
esac

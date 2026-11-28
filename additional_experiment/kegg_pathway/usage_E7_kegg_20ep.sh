#!/usr/bin/env bash
# E7 KEGG revised adaptation protocol: 20 max epochs, patience 3, min_delta 0.001.
# Separate results from the original completed 5-epoch experiment.
# Install as additional_experiment/kegg_pathway/usage_E7_kegg_20ep.sh
set -euo pipefail
cd /private/vina/FM_bench

MODE="${1:-}"
GPU="${2:-1}"
ROOT=results/e7/kegg_pathway_20ep
SMOKE=results/e7/kegg_pathway_20ep_smoke
LOG=logs/e7/kegg_pathway_20ep
SEEDS=(2026 2027 2028 2029 2030)
MAX_EPOCHS=20
PATIENCE=3
MIN_DELTA=0.001
MIN_EPOCHS=5
TF_SCRIPT=src/modeling/e7/finetune_kegg_transformer.py
MOLCLR_SCRIPT=src/modeling/e7/finetune_kegg_molclr.py
CHEM_SCRIPT=src/modeling/e7/finetune_kegg_chemeleon.py
ENV_TF=/private/vina/conda_envs/plantfm_chemberta_hf
ENV_MOLCLR=/private/vina/conda_envs/plantfm_molclr
ENV_CHEM=/private/vina/conda_envs/plantfm_chemeleon_moljepa
mkdir -p "$ROOT" "$SMOKE" "$LOG"

usage() {
  echo 'Usage: bash additional_experiment/kegg_pathway/usage_E7_kegg_20ep.sh {audit|smoke-chemberta|smoke-molformer|smoke-molclr|smoke-chemeleon|smoke-all|chemberta|molformer|molclr|chemeleon|all} [gpu_id]' >&2
  exit 2
}

check_sources() {
  python -m py_compile src/modeling/e7/kegg_ft_common.py "$TF_SCRIPT" "$MOLCLR_SCRIPT" "$CHEM_SCRIPT"
}

# Arguments: model_name, output_root, split_mode, max_epochs, min_epochs, seeds...
run_transformer() {
  local model="$1" out="$2" split="$3" epochs="$4" min_epochs="$5"; shift 5
  local checkpoint maxlen extra=()
  case "$model" in
    chemberta_77m_mlm)
      checkpoint=/private/vina/model_repos/ChemBERTa-77M-MLM
      maxlen=256
      ;;
    molformer_c3_1_1b)
      checkpoint=/private/vina/model_repos/molformer_c3_1_1b_e7
      maxlen=202
      extra+=(--trust-remote-code)
      ;;
    *) echo "Unknown transformer model $model" >&2; return 2 ;;
  esac
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="/private/vina/FM_bench${PYTHONPATH:+:$PYTHONPATH}" \
  conda run --no-capture-output -p "$ENV_TF" \
    python "$TF_SCRIPT" \
      --model-name-or-path "$checkpoint" --feature-name "$model" \
      --adaptation-modes full lora \
      --split-mode "$split" --split-seeds "$@" \
      --output-root "$out" \
      --epochs "$epochs" --min-epochs "$min_epochs" \
      --early-stopping-patience "$PATIENCE" --min-delta "$MIN_DELTA" \
      --train-batch-size 16 --eval-batch-size 64 --gradient-accumulation-steps 4 \
      --full-learning-rate 2e-5 --lora-learning-rate 1e-4 --head-learning-rate 1e-3 \
      --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
      --lora-target-modules query value --max-length "$maxlen" \
      --precision bf16 --num-workers 2 --local-files-only --resume "${extra[@]}"
}

run_molclr() {
  local out="$1" split="$2" epochs="$3" min_epochs="$4"; shift 4
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="/private/vina/FM_bench:/private/vina/model_repos/MolCLR${PYTHONPATH:+:$PYTHONPATH}" \
  conda run --no-capture-output -p "$ENV_MOLCLR" \
    python "$MOLCLR_SCRIPT" \
      --split-mode "$split" --split-seeds "$@" --output-root "$out" \
      --epochs "$epochs" --min-epochs "$min_epochs" \
      --early-stopping-patience "$PATIENCE" --min-delta "$MIN_DELTA" \
      --batch-size 32 --encoder-lr 2e-5 --head-lr 1e-3 --resume
}

run_chemeleon() {
  local out="$1" split="$2" epochs="$3" min_epochs="$4"; shift 4
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="/private/vina/FM_bench${PYTHONPATH:+:$PYTHONPATH}" \
  conda run --no-capture-output -p "$ENV_CHEM" \
    python "$CHEM_SCRIPT" \
      --split-mode "$split" --split-seeds "$@" --output-root "$out" \
      --epochs "$epochs" --min-epochs "$min_epochs" \
      --early-stopping-patience "$PATIENCE" --min-delta "$MIN_DELTA" \
      --batch-size 16 --encoder-lr 2e-5 --head-lr 1e-3 --resume
}

require_smokes() {
  python - <<'PY'
import json
from pathlib import Path
root=Path('results/e7/kegg_pathway_20ep_smoke')
for model, mode, version in (
    ('chemberta_77m_mlm','full','e7_kegg_transformer_balanced_auc_earlystop_v2'),
    ('chemberta_77m_mlm','lora','e7_kegg_transformer_balanced_auc_earlystop_v2'),
    ('molformer_c3_1_1b','full','e7_kegg_transformer_balanced_auc_earlystop_v2'),
    ('molformer_c3_1_1b','lora','e7_kegg_transformer_balanced_auc_earlystop_v2'),
    ('molclr_gin','full','e7_kegg_gnn_balanced_auc_earlystop_v2'),
    ('chemeleon','full','e7_kegg_gnn_balanced_auc_earlystop_v2'),
):
    path=root/model/mode/'random'/'seed2026'/'run_metadata.json'
    if not path.is_file():
        raise SystemExit(f'Missing revised smoke test: {path}')
    data=json.loads(path.read_text())
    if (data.get('script_version')!=version or data.get('epochs_trained')!=1
       or data.get('early_stopping_patience')!=3
       or data.get('early_stopping_min_delta')!=0.001):
        raise SystemExit(f'Incompatible revised smoke-test metadata: {path}')
print('PASS: all six revised smoke tests')
PY
}

check_sources
case "$MODE" in
  audit)
    run_molclr "$SMOKE" both 20 5 "${SEEDS[@]}" --data-check-only
    ;;
  smoke-chemberta)
    run_transformer chemberta_77m_mlm "$SMOKE" random 1 1 2026 2>&1 | tee "$LOG/smoke_chemberta.log"
    ;;
  smoke-molformer)
    run_transformer molformer_c3_1_1b "$SMOKE" random 1 1 2026 2>&1 | tee "$LOG/smoke_molformer.log"
    ;;
  smoke-molclr)
    run_molclr "$SMOKE" random 1 1 2026 2>&1 | tee "$LOG/smoke_molclr.log"
    ;;
  smoke-chemeleon)
    run_chemeleon "$SMOKE" random 1 1 2026 2>&1 | tee "$LOG/smoke_chemeleon.log"
    ;;
  smoke-all)
    run_transformer chemberta_77m_mlm "$SMOKE" random 1 1 2026 2>&1 | tee "$LOG/smoke_chemberta.log"
    run_transformer molformer_c3_1_1b "$SMOKE" random 1 1 2026 2>&1 | tee "$LOG/smoke_molformer.log"
    run_molclr "$SMOKE" random 1 1 2026 2>&1 | tee "$LOG/smoke_molclr.log"
    run_chemeleon "$SMOKE" random 1 1 2026 2>&1 | tee "$LOG/smoke_chemeleon.log"
    require_smokes
    ;;
  chemberta)
    run_transformer chemberta_77m_mlm "$ROOT" both "$MAX_EPOCHS" "$MIN_EPOCHS" "${SEEDS[@]}" 2>&1 | tee "$LOG/chemberta.log"
    ;;
  molformer)
    run_transformer molformer_c3_1_1b "$ROOT" both "$MAX_EPOCHS" "$MIN_EPOCHS" "${SEEDS[@]}" 2>&1 | tee "$LOG/molformer.log"
    ;;
  molclr)
    run_molclr "$ROOT" both "$MAX_EPOCHS" "$MIN_EPOCHS" "${SEEDS[@]}" 2>&1 | tee "$LOG/molclr.log"
    ;;
  chemeleon)
    run_chemeleon "$ROOT" both "$MAX_EPOCHS" "$MIN_EPOCHS" "${SEEDS[@]}" 2>&1 | tee "$LOG/chemeleon.log"
    ;;
  all)
    require_smokes
    echo "Starting revised 60-run E7 KEGG batch on GPU $GPU"
    run_transformer chemberta_77m_mlm "$ROOT" both "$MAX_EPOCHS" "$MIN_EPOCHS" "${SEEDS[@]}" 2>&1 | tee "$LOG/chemberta.log"
    run_transformer molformer_c3_1_1b "$ROOT" both "$MAX_EPOCHS" "$MIN_EPOCHS" "${SEEDS[@]}" 2>&1 | tee "$LOG/molformer.log"
    run_molclr "$ROOT" both "$MAX_EPOCHS" "$MIN_EPOCHS" "${SEEDS[@]}" 2>&1 | tee "$LOG/molclr.log"
    run_chemeleon "$ROOT" both "$MAX_EPOCHS" "$MIN_EPOCHS" "${SEEDS[@]}" 2>&1 | tee "$LOG/chemeleon.log"
    echo 'PASS: E7 KEGG 20-epoch-cap / patience-3 batch complete'
    ;;
  *) usage ;;
esac

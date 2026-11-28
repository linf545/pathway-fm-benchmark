#!/usr/bin/env bash
# E7 KEGG: first smoke-test all models, then queue all 60 runs sequentially.
# Copy this file to additional_experiment/kegg_pathway/usage_E7_kegg_all.sh
set -euo pipefail
cd /private/vina/FM_bench

ROOT=results/e7/kegg_pathway
SMOKE=results/e7/kegg_pathway_smoke
LOG=logs/e7/kegg_pathway
TRANSFORMER=additional_experiment/kegg_pathway/usage_E7_kegg_transformers.sh
DATAENV=/private/vina/conda_envs/plantfm_data
MOLCLRENV=/private/vina/conda_envs/plantfm_molclr
CHEMENV=/private/vina/conda_envs/plantfm_chemeleon_moljepa
MOLCLRSCRIPT=src/modeling/e7/finetune_kegg_molclr.py
CHEMSCRIPT=src/modeling/e7/finetune_kegg_chemeleon.py
GPU="${2:-1}"
mkdir -p "$ROOT" "$SMOKE" "$LOG"

usage() {
  echo 'Usage: bash additional_experiment/kegg_pathway/usage_E7_kegg_all.sh {audit|smoke-molclr|smoke-chemeleon|molclr|chemeleon|all} [physical_gpu_id]' >&2
  exit 2
}

check_source() {
  test -s src/modeling/e7/kegg_ft_common.py || { echo 'Missing kegg_ft_common.py' >&2; exit 1; }
  test -s "$MOLCLRSCRIPT" || { echo "Missing $MOLCLRSCRIPT" >&2; exit 1; }
  test -s "$CHEMSCRIPT" || { echo "Missing $CHEMSCRIPT" >&2; exit 1; }
  python -m py_compile src/modeling/e7/kegg_ft_common.py "$MOLCLRSCRIPT" "$CHEMSCRIPT"
}

run_molclr() {
  local output="$1" mode="$2" epochs="$3"; shift 3
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="/private/vina/FM_bench:/private/vina/model_repos/MolCLR${PYTHONPATH:+:$PYTHONPATH}" \
  conda run --no-capture-output -p "$MOLCLRENV" \
    python "$MOLCLRSCRIPT" \
       --split-mode "$mode" --split-seeds "$@" \
       --epochs "$epochs" --batch-size 32 --encoder-lr 2e-5 --head-lr 1e-3 \
       --output-root "$output" --resume
}

run_chemeleon() {
  local output="$1" mode="$2" epochs="$3"; shift 3
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="/private/vina/FM_bench${PYTHONPATH:+:$PYTHONPATH}" \
  conda run --no-capture-output -p "$CHEMENV" \
    python "$CHEMSCRIPT" \
       --split-mode "$mode" --split-seeds "$@" \
       --epochs "$epochs" --batch-size 16 --encoder-lr 2e-5 --head-lr 2e-5 \
       --output-root "$output" --resume
}

require_smokes() {
  local item
  for item in \
    'chemberta_77m_mlm/full' 'chemberta_77m_mlm/lora' \
    'molformer_c3_1_1b/full' 'molformer_c3_1_1b/lora' \
    'molclr_gin/full' 'chemeleon/full'; do
    test -s "$SMOKE/$item/random/seed2026/run_metadata.json" || {
      echo "Missing completed smoke test: $SMOKE/$item/random/seed2026/run_metadata.json" >&2
      exit 1
    }
  done
}

case "${1:-}" in
  audit)
    check_source
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONPATH="/private/vina/FM_bench:/private/vina/model_repos/MolCLR${PYTHONPATH:+:$PYTHONPATH}" \
    conda run --no-capture-output -p "$MOLCLRENV" \
      python "$MOLCLRSCRIPT" --data-check-only --split-mode both
    ;;
  smoke-molclr)
    check_source
    run_molclr "$SMOKE" random 1 2026 2>&1 | tee "$LOG/molclr_gin_smoke.log"
    ;;
  smoke-chemeleon)
    check_source
    run_chemeleon "$SMOKE" random 1 2026 2>&1 | tee "$LOG/chemeleon_smoke.log"
    ;;
  molclr)
    check_source
    run_molclr "$ROOT" both 5 2026 2027 2028 2029 2030 \
      2>&1 | tee "$LOG/molclr_gin_all.log"
    ;;
  chemeleon)
    check_source
    run_chemeleon "$ROOT" both 5 2026 2027 2028 2029 2030 \
      2>&1 | tee "$LOG/chemeleon_all.log"
    ;;
  all)
    check_source
    require_smokes
    echo "Starting all E7 KEGG models in sequence on GPU $GPU"
    bash "$TRANSFORMER" chemberta "$GPU"
    bash "$TRANSFORMER" molformer "$GPU"
    run_molclr "$ROOT" both 5 2026 2027 2028 2029 2030 \
      2>&1 | tee "$LOG/molclr_gin_all.log"
    run_chemeleon "$ROOT" both 5 2026 2027 2028 2029 2030 \
      2>&1 | tee "$LOG/chemeleon_all.log"
    echo 'PASS: E7 KEGG all 60 runs completed'
    ;;
  *) usage ;;
esac

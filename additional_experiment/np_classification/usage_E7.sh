#!/usr/bin/env bash
set -euo pipefail

cd /private/vina/FM_bench

ROOT=results/e7/np_classification
LOGDIR=logs/e7
SEEDS=(2026 2027 2028 2029 2030)
MODES=(random scaffold)

mkdir -p "$ROOT" "$LOGDIR"

# Require two visible GPUs.
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( NGPU < 2 )); then
    echo "ERROR: This launcher requires at least 2 GPUs."
    exit 1
fi

# Export existing splits for Chemprop. Never create new split assignments.
conda run --no-capture-output \
    -p /private/vina/conda_envs/plantfm_data \
    python - <<'PY'
from pathlib import Path
import pandas as pd
import numpy as np

df = pd.read_csv(
    "data/tasks/np_classification/npclassifier_label_task.tsv",
    sep="\t",
)
labels = df[[f"label_{i}" for i in range(1, 8)]].to_numpy(dtype=int)
assert np.isin(labels, [0, 1]).all()
assert (labels.sum(axis=1) == 1).all()
df["class_id"] = labels.argmax(axis=1)

for mode in ("random", "scaffold"):
    prefix = "" if mode == "random" else "scaffold_"

    for seed in range(2026, 2031):
        column = f"{mode}_split_{seed}"
        assert column in df.columns

        out = Path(
            f"results/e7/data/np_classification/{mode}_seed{seed}"
        )
        out.mkdir(parents=True, exist_ok=True)

        expected_values = {
            "train": prefix + "train",
            "val": prefix + "val",
            "test": prefix + "test",
        }
        assert set(df[column].unique()) == set(expected_values.values())

        for split, value in expected_values.items():
            part = df[df[column] == value]
            expected = part[
                ["canonical_smiles", "class_id"]
            ].rename(columns={"canonical_smiles": "smiles"}
            ).reset_index(drop=True)

            path = out / f"{split}.csv"
            if path.exists():
                existing = pd.read_csv(path)
                pd.testing.assert_frame_equal(
                    existing, expected, check_dtype=False
                )
            else:
                expected.to_csv(path, index=False)

            print(mode, seed, split, len(expected), flush=True)

print("Original split assignments verified.")
PY

run_hf() {
    local gpu="$1"
    local name="$2"
    local checkpoint="$3"
    local extra=()

    if [[ "$name" == "molformer_c3_1_1b" ]]; then
        extra=(--max-length 202 --trust-remote-code)
    fi

    echo "GPU $gpu: starting $name"

    CUDA_VISIBLE_DEVICES="$gpu" \
    conda run --no-capture-output \
      -p /private/vina/conda_envs/plantfm_chemberta_hf \
      python src/modeling/e7/finetune_np_classification.py \
        --model-name-or-path "$checkpoint" \
        --feature-name "$name" \
        --adaptation-modes full \
        --split-mode both \
        --split-seeds "${SEEDS[@]}" \
        --output-root "$ROOT" \
        --epochs 5 \
        --train-batch-size 16 \
        --eval-batch-size 64 \
        --gradient-accumulation-steps 4 \
        --full-learning-rate 2e-5 \
        --head-learning-rate 1e-3 \
        --class-weight-mode none \
        --early-stopping-patience -1 \
        --precision bf16 \
        --local-files-only \
        --num-workers 2 \
        --resume \
        "${extra[@]}" \
        2>&1 | tee "$LOGDIR/${name}_all.log"

    echo "GPU $gpu: completed $name"
}

run_chemeleon() {
    local gpu="$1"

    for mode in "${MODES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            local data="results/e7/data/np_classification/${mode}_seed${seed}"
            local out="$ROOT/chemeleon/${mode}_seed${seed}"

            if [[ -s "$out/model_0/best.pt" &&
                  -s "$out/model_0/test_predictions.csv" ]]; then
                echo "SKIP CheMeleon $mode $seed"
                continue
            fi

            echo "GPU $gpu: CheMeleon $mode $seed"

            CUDA_VISIBLE_DEVICES="$gpu" \
            conda run --no-capture-output \
              -p /private/vina/conda_envs/plantfm_chemeleon_moljepa \
              chemprop train \
                --data-path \
                  "$data/train.csv" \
                  "$data/val.csv" \
                  "$data/test.csv" \
                --smiles-columns smiles \
                --target-columns class_id \
                --task-type multiclass \
                --multiclass-num-classes 7 \
                --from-foundation chemeleon \
                --epochs 5 \
                --batch-size 16 \
                --num-workers 2 \
                --devices 1 \
                --init-lr 2e-5 \
                --max-lr 2e-5 \
                --final-lr 2e-5 \
                --warmup-epochs 0 \
                --tracking-metric val_loss \
                --pytorch-seed "$seed" \
                --output-dir "$out" \
                2>&1 | tee "$LOGDIR/chemeleon_${mode}_${seed}.log"
        done
    done
}

run_molclr() {
    local gpu="$1"

    for mode in "${MODES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            local out="$ROOT/molclr_gin/${mode}_seed${seed}"

            # Skip only runs using the updated loss-based selection.
            if [[ -s "$out/summary.json" ]] &&
               grep -q '"best_validation_loss"' "$out/summary.json"; then
                echo "SKIP MolCLR-GIN $mode $seed"
                continue
            fi

            echo "GPU $gpu: MolCLR-GIN $mode $seed"

            CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONPATH="/private/vina/FM_bench:/private/vina/model_repos/MolCLR${PYTHONPATH:+:$PYTHONPATH}" \
            conda run --no-capture-output \
              -p /private/vina/conda_envs/plantfm_molclr \
              python src/modeling/e7/finetune_molclr_npclass.py \
                --split-mode "$mode" \
                --seed "$seed" \
                --epochs 5 \
                --batch-size 32 \
                --encoder-lr 2e-5 \
                --head-lr 1e-3 \
                2>&1 | tee "$LOGDIR/molclr_gin_${mode}_${seed}.log"
        done
    done
}

# Two concurrent queues, one per GPU.
(
    run_hf 0 chemberta_77m_mlm \
        /private/vina/model_repos/ChemBERTa-77M-MLM

    run_hf 0 molformer_c3_1_1b \
        /private/vina/model_repos/molformer_c3_1_1b_e7
) &
PID0=$!

(
    run_chemeleon 1
    run_molclr 1
) &
PID1=$!

set +e
wait "$PID0"
STATUS0=$?
wait "$PID1"
STATUS1=$?
set -e

echo "GPU 0 queue exit status: $STATUS0"
echo "GPU 1 queue exit status: $STATUS1"

if (( STATUS0 != 0 || STATUS1 != 0 )); then
    echo "E7 batch incomplete: inspect model logs."
    exit 1
fi

echo "E7 NP-classification batch completed."

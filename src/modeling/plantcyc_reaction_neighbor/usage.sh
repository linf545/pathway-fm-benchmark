## feature prep
conda activate plantfm_graphmvp
cd /private/vina/FM_bench

python src/modeling/extract_graphmvp_embeddings.py \
  --input data/features/plantcyc_reaction_neighbor/metadata.tsv \
  --id-col task_compound_id \
  --smiles-col canonical_smiles \
  --repo-dir /private/vina/model_repos/GraphMVP \
  --ckpt /private/vina/model_repos/GraphMVP/pretrained/graphmvp_classification_simple_m015_t02/model.pth \
  --model-name graphmvp_gin \
  --out-dir data/features/plantcyc_reaction_neighbor/full \
  --batch-size 256 \
  --device auto \
  2>&1 | tee logs/feature_extraction/plantcyc_reaction_neighbor_graphmvp_gin.log

conda activate plantfm_molebert
cd /private/vina/FM_bench

python src/modeling/extract_molebert_embeddings.py \
  --input data/features/plantcyc_reaction_neighbor/metadata.tsv \
  --id-col task_compound_id \
  --smiles-col canonical_smiles \
  --repo-dir /private/vina/model_repos/Mole-BERT \
  --ckpt /private/vina/model_repos/Mole-BERT/model_gin/Mole-BERT.pth \
  --model-name molebert_gin \
  --out-dir data/features/plantcyc_reaction_neighbor/full \
  --batch-size 256 \
  --device auto \
  2>&1 | tee logs/feature_extraction/plantcyc_reaction_neighbor_molebert_gin.log




## run model
cd /private/vina/FM_bench
conda activate plantfm_data

mkdir -p logs/modeling/plantcyc_reaction_neighbor

python src/modeling/plantcyc_reaction_neighbor/run_plantcyc_reaction_neighbor_full.py \
  --split-seeds 2026 2027 2028 2029 2030 \
  --schemes random scaffold_pair \
  --variants easy mixed hard \
  --overwrite-results \
  2>&1 | tee logs/modeling/plantcyc_reaction_neighbor/full_repeated.log




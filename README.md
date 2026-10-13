# Molecular Foundation Models for Natural-Product and Pathway Prediction

This repository contains the code for benchmarking molecular foundation-model (FM) representations against conventional molecular fingerprints on natural-product and metabolism-related prediction tasks.

The benchmark evaluates frozen molecular embeddings with linear classifiers, including full-data performance and sample efficiency under random and scaffold-based splits. Selected experiments also examine probability ensembles and LoRA adaptation.

## Benchmark tasks

| Task | Prediction target |
| --- | --- |
| Natural-product likeness | Natural product vs. non-natural product |
| NP chemical class | Seven-class chemical classification |
| PMN pathway association | Multilabel plant secondary-metabolic pathway prediction |
| KEGG pathway association | Multilabel metabolic pathway prediction |
| PlantCyc reaction neighbors | Whether two compounds are linked by a metabolic reaction |

The benchmark includes SMILES-based and graph-based FM representations, with Morgan fingerprints as a primary baseline.

## Repository structure

- `src/data_processing/` — dataset preparation, splitting, and quality control
- `src/modeling/` — embedding extraction, downstream models, and LoRA experiments
- `src/analysis/` — statistical analysis and figure generation
- `env/` — Conda environments and model-specific dependency records
- `qc/` — data-processing and modeling quality-control reports

## Usage

Create the data-processing environment:

```bash
conda env create -f env/environment_processing.yml
conda activate plantfm_data
```

The workflow is organized by task:

1. **Prepare datasets:** see `src/data_processing/data_processing_README.txt`.
2. **Extract FM embeddings:** see `src/modeling/embedding_extraction_usage.txt`. Different FM architectures may require separate environments and pretrained checkpoints.
3. **Run benchmarks:** see the `usage.txt` or `usage.sh` files in each `src/modeling/<task>/` directory.
4. **Generate figures:** see `src/analysis/make_figures/usage.sh`.

## Data and results

The repository does not include raw datasets, large embedding files, pretrained model checkpoints, or experiment outputs in `results/`. These inputs and outputs must be supplied separately to reproduce the analyses.

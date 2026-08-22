#!/usr/bin/env bash
set -euo pipefail

# Run from /private/vina/FM_bench.
python src/analysis/make_figures/audit_figure_inputs.py --project-root .

# Generate all available main task figures. The driver skips KEGG until both
# results/kegg_pathway/full and results/kegg_pathway/data_efficiency exist.
python src/analysis/make_figures/make_all_main_figures.py \
  --project-root . \
  --formats pdf,png \
  --dpi 600 \
  --curve-uncertainty ci95 \
  --include-pathway-ensemble-figure

# Individual task examples:
# python src/analysis/make_figures/make_np_likeness_figure.py --project-root .
# python src/analysis/make_figures/make_np_classification_figure.py --project-root .
# python src/analysis/make_figures/make_pmn_secondary_pathway_figure.py --project-root .
# python src/analysis/make_figures/make_kegg_pathway_figure.py --project-root .
# python src/analysis/make_figures/make_plantcyc_reaction_neighbor_figure.py --project-root .

# Full-data panels retain individual split-seed points and show mean with a 95%
# Student-t confidence interval. Label-efficiency panels omit replicate lines and
# show mean curves with 95% Student-t confidence ribbons.
# Morgan fingerprints use neutral baseline colors and distinct line styles;
# FM representations use categorical colors.

# PMN now uses a six-panel layout by default:
# individual representations (Dataset A validation and Dataset B external test),
# equal-weight probability ensembles (same two evaluation sets), and two
# label-efficiency panels. Use --exclude-pmn-ensembles to omit the ensemble row.

python src/analysis/make_figures/make_pmn_morgan_graphmvp_error_overlap_figures.py \
  --project-root . \
  --formats pdf,png \
  --dpi 600

python src/analysis/make_figures/make_per_label_auroc_heatmaps.py \
  --project-root . \
  --formats png,pdf \
  --dpi 600 \
  --top-fms 5 \
  --label-order support

python src/analysis/make_figures/plot_plantcyc_reaction_neighbor_sample_efficiency_variants.py



python src/analysis/make_figures/make_per_label_auroc_heatmaps_v3.py \
  --project-root /private/vina/FM_bench \
  --output-dir results/figures/main \
  --tasks pmn_secondary_pathway \
  --top-fms 0 \
  --formats pdf,png \
  --dpi 600 \
  --strict
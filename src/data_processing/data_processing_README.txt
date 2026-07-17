M_BENCH DATA WORKFLOW
======================

This repository has independent task pipelines. Do not run every processing
script as one global sequence.

========================================================================
0. SHARED BAO SOURCE NORMALIZATION AND EMBEDDING MASTER
========================================================================

Raw inputs:
    raw_data/bao/bao_dataset_A_trainval.csv
    raw_data/bao/bao_dataset_B_external_test.csv
    raw_data/bao/bao_dataset_C_kegg.csv
    raw_data/bao/bao_dataset_D_npclassifier.csv

Label mapping inputs:
    A/B -> raw_data/bao/pathway_label.csv
    C   -> raw_data/bao/kegg_pathway_class.csv
    D   -> raw_data/bao/npclassifier_pathway_class.csv

For each dataset, process_bao_dataset.py:
    raw SMILES + numeric labels
        -> RDKit validation
        -> canonical isomeric SMILES
        -> InChIKey
        -> Murcko scaffold
        -> basic molecular properties
        -> deduplication by InChIKey, canonical SMILES fallback
        -> long label table
        -> wide multilabel matrix
        -> invalid-SMILES report

Outputs:
    data/processed_bao/<dataset>/compounds_master.tsv
    data/processed_bao/<dataset>/bao_compound_labels_long.tsv
    data/processed_bao/<dataset>/bao_compound_labels_multilabel.tsv
    data/processed_bao/<dataset>/bao_label_mapping.tsv
    data/processed_bao/<dataset>/bao_invalid_smiles.tsv

Then build_bao_compounds_master.py:
    processed A/B/C/D compound tables
        -> one structure master across all Bao datasets
        -> source crosswalk

Outputs:
    data/master/compounds_master_bao.tsv
    data/master/bao_compound_source_crosswalk.tsv

The shared master is used for one-time FM embedding extraction. Task labels
should remain task-specific and should not be interpreted from the aggregated
master label metadata.

========================================================================
1. TASK 0: NP-LIKENESS
========================================================================

Raw inputs:
    raw_data/NPlikeliness/data/coconut_synthetic.csv
    raw_data/NPlikeliness/data/to_keep_molecules.pkl

make_np_likeness_task.py:
    published row filter
        -> canonicalization and InChIKey generation
        -> invalid structure removal
        -> cross-label identity conflict removal
        -> same-class identity deduplication
        -> class balancing
        -> five stratified random 80/10/10 splits
        -> five scaffold-group 80/10/10 splits
        -> QC

Outputs:
    data/tasks/np_likeness/np_likeness_task.tsv
    data/tasks/np_likeness/np_likeness_qc.txt
    data/splits/np_likeness/np_likeness_random_split_seed*.tsv
    data/splits/np_likeness/np_likeness_scaffold_split_seed*.tsv

Do not use original_npl_score as a model feature.

========================================================================
2. TASK 1: NPCLASSIFIER SINGLE-LABEL CLASSIFICATION
========================================================================

Raw input:
    raw_data/bao/bao_dataset_D_npclassifier.csv

make_npclassification_task.py:
    detect SMILES and labels
        -> canonicalize and compute InChIKey/scaffold
        -> deduplicate by structure and union source labels
        -> remove multi-label/ambiguous compounds
        -> remove classes below min-count
        -> construct the seven-class single-label task
        -> five random split replicates
        -> five scaffold split replicates
        -> class-coverage QC

The current validated mode is:
    --level label

Task 1 reads raw D directly. It does not depend on D pathway_names written by
process_bao_dataset.py.

========================================================================
3. PMN SECONDARY-PATHWAY CLASSIFICATION
========================================================================

Inputs:
    data/processed_bao/A_trainval/
    data/processed_bao/B_external_test/

make_bao_plant_pathway_task.py:
    load processed A/B compounds and long labels
        -> remove B structures overlapping A
        -> construct aligned 18-label matrix
        -> split A into train/validation
        -> keep cleaned B as fixed external test
        -> label-cardinality QC

Outputs:
    data/tasks/bao_plant_pathway/bao_plant_pathway_task.tsv
    data/tasks/bao_plant_pathway/bao_plant_pathway_labels.tsv
    data/tasks/bao_plant_pathway/bao_plant_pathway_splits.tsv
    data/tasks/bao_plant_pathway/bao_plant_pathway_qc.txt

Optional old feature cache:
    make_bao_plant_pathway_features.py
        -> metadata.tsv
        -> labels.npy
        -> Morgan r2/r3
        -> RDKit descriptors

Current repeated random/scaffold split construction and model tuning are handled
by the PMN modeling scripts. The old one-off make_bao_scaffold_split.py is only
needed by old downstream code that explicitly reads scaffold_split.tsv.

========================================================================
4. KEGG 11-LABEL PATHWAY CLASSIFICATION
========================================================================

Raw input:
    raw_data/bao/bao_dataset_C_kegg.csv

Mapping input:
    raw_data/bao/kegg_pathway_class.csv

process_bao_dataset.py:
    raw Bao C
        -> data/processed_bao/C_kegg/compounds_master.tsv
        -> data/processed_bao/C_kegg/bao_compound_labels_multilabel.tsv
        -> other normalized/QC tables

There is no separate make_kegg_pathway_task.py.

The current KEGG modeling scripts perform the remaining task construction:
    run_kegg_pathway_data_efficiency.py
    run_kegg_pathway_full_ensemble.py

They:
    load compounds_master.tsv and bao_compound_labels_multilabel.tsv
        -> align labels by internal_compound_id
        -> require exactly 11 label columns
        -> build or load repeated random/scaffold train/validation/test splits
        -> run models and evaluation

Therefore, KEGG task construction is split across:
    data processing: process_bao_dataset.py
    modeling-time task assembly/splitting: run_kegg_pathway_*.py

This is functional, but a standalone task table/manifest could be added later
for cleaner reproducibility.

========================================================================
5. TASK 3: PLANTCYC REACTION-NEIGHBOR PREDICTION
========================================================================

process_plantcyc.py:
    PlantCyc flat files
        -> normalized compounds, reactions, pathways
        -> all opposite-side compound combinations per reaction

filter_plantcyc_reaction_pairs.py:
    require structures for both compounds
        -> remove self-pairs
        -> remove currency/generic/high-occurrence compounds
        -> remove exact repeated reaction rows

make_plantcyc_positive_pair_splits.py:
    canonicalize endpoints
        -> remove invalid and chemical self-pairs
        -> convert to undirected ordered compound pairs
        -> deduplicate chemical pairs
        -> aggregate reaction/pathway provenance
        -> pair-level random split
        -> scaffold-pair-group split

make_plantcyc_reaction_neighbor_negatives.py:
    easy negatives:
        no known shared pathway annotation
    hard negatives:
        same pathway but not a known positive pair
        same-scaffold fallback
    mixed negatives:
        deterministic mixture of easy and hard
    each split/variant is balanced 1:1 positive:negative

make_plantcyc_reaction_neighbor_compound_master.py:
    six task tables
        -> unique compound master
        -> pair-index tables

make_plantcyc_reaction_neighbor_features.py:
    compound master
        -> Morgan r2/r3
        -> RDKit descriptors

FM embeddings are extracted at compound level. Pair representations are created
during modeling. The agreed symmetric dense representation is:
    [a + b, |a - b|]




usage: 

cd /private/vina/FM_bench
conda activate plantfm_data

python src/data_processing/process_bao_dataset.py \
  --bao_csv raw_data/bao/bao_dataset_A_trainval.csv \
  --pathway_label_csv raw_data/bao/pathway_label.csv \
  --out_dir data/processed_bao/A_trainval \
  --source_name Bao_Dataset_A_trainval


python src/data_processing/process_bao_dataset.py \
  --bao_csv raw_data/bao/bao_dataset_B_external_test.csv \
  --pathway_label_csv raw_data/bao/pathway_label.csv \
  --out_dir data/processed_bao/B_external_test \
  --source_name Bao_Dataset_B_external_test


python src/data_processing/process_bao_dataset.py \
  --bao_csv raw_data/bao/bao_dataset_C_kegg.csv \
  --pathway_label_csv raw_data/bao/pathway_label.csv \
  --out_dir data/processed_bao/C_kegg \
  --source_name Bao_Dataset_C_kegg

python src/data_processing/make_npclassification_task.py \
  --input raw_data/bao/bao_dataset_D_npclassifier.csv \
  --level label \
  --min-count 50 \
  --split-seeds 2026 2027 2028 2029 2030 \
  --out-prefix data/tasks/np_classification/npclassifier_label \
  2>&1 | tee logs/modeling/npclassification_make_task_singlelabel_multiseed.log


python src/data_processing/make_np_likeness_task.py \
  --n-jobs 8 \
  --scaffold-candidates 10 \
  --scaffold-size-weight 1.0 \
  --scaffold-label-weight 0.25 \
  --overwrite
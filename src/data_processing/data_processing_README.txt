run scripts in the following order:
1 process_plantcyc.py
2 process_bao_dataset.py
3 qc_processed_data.py
4 make_bao_plant_pathway_features.py
5 build_bao_compounds_master.py
6 finalize_bao_data_processing.py
7 filter_plantcyc_reaction_pairs.py
8 make_plantcyc_positive_pair_splits.py
9 make_plantcyc_reaction_neighbor_negatives.py

check_bao_ab_overlap.py is a seperate script to confirm the data leakage issue in the original dataset.
#!/usr/bin/env python3
"""E6: consolidate frozen model benchmark, assert per-seed comparability.
Run from the FM_bench repository root with plantfm_data environment.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OLD_FMS = [
    'chemberta_77m_mlm', 'chemberta_100m_mlm', 'molformer_c3_1_1b',
    'nafm_graph_mean', 'molclr_gcn', 'molclr_gin', 'graphmvp_gin',
    'molebert_gin',
]
NEW_FMS = ['chemeleon', 'moljepa_cls']
FPS = ['morgan_r2_2048', 'morgan_r3_2048']
SEEDS = set(range(2026, 2031))
OUT = Path('results/e6/consolidated')

# name, old directory, E6 directory, runs filename, split field, test-set value,
# AUROC field, AUPRC field, split-QC filename, extra condition columns.
SPECS = [
    ('np_likeness', 'results/np_likeness/full',
     'results/e6/np_likeness/full',
     'np_likeness_full_ensemble_repeated_runs.tsv', 'split_mode', 'test',
     'auroc', 'auprc', 'np_likeness_full_ensemble_split_qc.tsv', []),
    ('np_classification', 'results/np_classification/full',
     'results/e6/np_classification/full',
     'npclassification_repeated_runs.tsv', 'split', 'test',
     'auroc_ovr_macro', 'auprc_ovr_macro', 'npclassification_split_qc.tsv', []),
    ('kegg_pathway', 'results/kegg_pathway/full',
     'results/e6/kegg_pathway/full',
     'kegg_pathway_full_ensemble_repeated_runs.tsv', 'split_mode', 'test',
     'macro_auroc', 'macro_auprc', 'kegg_pathway_full_ensemble_split_qc.tsv', []),
    ('pmn_secondary_pathway', 'results/pmn_secondary_pathway/full',
     'results/e6/pmn_secondary_pathway/full',
     'pmn_secondary_pathway_full_ensemble_repeated_runs.tsv', 'split_mode',
     'bao_B_external_test', 'macro_auroc', 'macro_auprc',
     'pmn_secondary_pathway_full_ensemble_split_qc.tsv', []),
    ('plantcyc_reaction_neighbor', 'results/plantcyc_reaction_neighbor/full',
     'results/e6/plantcyc_reaction_neighbor/full',
     'reaction_neighbor_repeated_runs.tsv', 'split_scheme', 'test',
     'auroc', 'auprc', 'reaction_neighbor_split_qc.tsv', ['negative_variant']),
]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def normalized_values(series):
    """Represent numerical grid strings equivalently (1 and 1.0)."""
    result = []
    for value in series:
        value = str(value)
        if '|' in value:
            try:
                result.append(tuple(float(x) for x in value.split('|')))
                continue
            except ValueError:
                pass
        result.append(value)
    return result


def read_table(path):
    require(path.exists(), f'Missing: {path}')
    return pd.read_csv(path, sep='\t', low_memory=False)


def compare_split_qc(oldpath, newpath, task):
    a, b = read_table(oldpath), read_table(newpath)
    shared = sorted(set(a.columns) & set(b.columns) - {'source_file'})
    require(len(shared) >= 3, f'{task}: insufficient shared split-QC columns')
    a = a[shared].fillna('<NA>').astype(str).sort_values(shared).reset_index(drop=True)
    b = b[shared].fillna('<NA>').astype(str).sort_values(shared).reset_index(drop=True)
    require(a.equals(b), f'{task}: original vs E6 split-QC contents differ. '
            f'Inspect {oldpath} and {newpath}')
    return len(a)


def extract(task, folder, filename, split_col, test_set, auc_col, prc_col, extras, allowed_features):
    f = read_table(Path(folder) / filename)
    need = ['feature', split_col, 'split_seed', 'eval_set', auc_col, prc_col]
    require(set(need).issubset(f.columns), f'{task}: missing {set(need)-set(f.columns)}')
    f = f.loc[f['eval_set'].eq(test_set) & f['feature'].isin(allowed_features)].copy()
    f = f.rename(columns={split_col: 'split', auc_col: 'auroc', prc_col: 'auprc'})
    f['split_seed'] = pd.to_numeric(f['split_seed'], errors='raise').astype(int)
    f['condition'] = f[extras].astype(str).agg('/'.join, axis=1) if extras else 'all'
    keys = ['split', 'condition', 'split_seed', 'feature']
    require(not f.duplicated(keys).any(), f'{task}: duplicate model/seed/split results')
    for metric in ['auroc', 'auprc']:
        require(pd.to_numeric(f[metric], errors='coerce').between(0, 1).all(),
                f'{task}: missing or invalid {metric} values')
    f['task'] = task
    return f


def paired_delta(target, ref, group, metric='auroc'):
    keys = ['task', 'split', 'condition', 'split_seed']
    ref_rows = ref[keys+[metric]].rename(columns={metric: 'reference_score'})
    joined = target.merge(ref_rows, on=keys, validate='one_to_one')
    require(len(joined)==len(target), f'{group}: reference missing one or more split seeds')
    return float((joined[metric]-joined['reference_score']).mean())


def check_one_task(spec):
    task, original_dir, e6_dir, filename, split_col, test_set, ac, pc, qc_file, extras = spec
    old = extract(task, original_dir, filename, split_col, test_set, ac, pc, extras, OLD_FMS+FPS)
    new = extract(task, e6_dir, filename, split_col, test_set, ac, pc, extras, NEW_FMS)
    require(set(OLD_FMS).issubset(old['feature']),
            f'{task}: missing original FMs {sorted(set(OLD_FMS)-set(old.feature))}')
    require(set(NEW_FMS).issubset(new['feature']),
            f'{task}: missing E6 models {sorted(set(NEW_FMS)-set(new.feature))}')
    require('morgan_r2_2048' in set(old.feature), f'{task}: original Morgan r2 missing')
    # One model must have all five seeds for every split-condition combination.
    groups = sorted(set(zip(new['split'],new['condition'])))
    require(set(zip(old['split'],old['condition'])) == set(groups),
            f'{task}: original and E6 split conditions differ')
    for d, names in [(old, OLD_FMS+FPS), (new, NEW_FMS)]:
        for sp, cond in groups:
            for model in names:
                if model not in set(d.feature):
                    continue
                have = set(d.loc[(d['split']==sp)&(d['condition']==cond)&
                                 (d['feature']==model),'split_seed'])
                require(have == SEEDS,
                        f'{task}/{sp}/{cond}/{model}: seeds {sorted(have)} != {sorted(SEEDS)}')
    n_qc = compare_split_qc(Path(original_dir)/qc_file, Path(e6_dir)/qc_file, task)

    # Compare identical cohort sizes, classifier hyperparameter grids and settings
    # using an original dense FM and the new dense FM on exactly matching seeds.
    shared_settings = ['n_train','n_eval','n_test','n_validation','n_labels',
                       'n_classes','C_grid','threshold_grid','tol',
                       'class_weight','solver','selection_metric',
                       'C_selection_metric','threshold_strategy',
                       'split_column','source_pair_file', 'max_iter']
    key = ['split','condition','split_seed']
    a = old[old.feature.eq(OLD_FMS[0])][key+ [c for c in shared_settings if c in old.columns]]
    b = new[new.feature.eq(NEW_FMS[0])][key+ [c for c in shared_settings if c in new.columns]]
    ab = a.merge(b,on=key,suffixes=('_old','_e6'),validate='one_to_one')
    require(len(ab)==len(groups)*5, f'{task}: unmatched dense-FM splits')
    checked = []
    for col in shared_settings:
        left, right = f'{col}_old', f'{col}_e6'
        if left not in ab.columns or right not in ab.columns:
            continue
        require(normalized_values(ab[left]) == normalized_values(ab[right]),
                f'{task}: mismatched evaluation setting: {col}')
        checked.append(col)
    require('n_eval' in checked and 'n_train' in checked,
            f'{task}: training/test sizes not comparable')
    # Keep the original models; add only the two new FMs (no duplicate baselines).
    combined = pd.concat([old[old.feature.isin(OLD_FMS+FPS)],
                          new[new.feature.isin(NEW_FMS)]],ignore_index=True)
    return combined, {'task':task,'split_qc_rows':n_qc,
                      'conditions':len(groups),'settings_checked':'|'.join(checked),
                      'status':'PASS'}


def main():
    all_data, qc = [], []
    for spec in SPECS:
        rows, record = check_one_task(spec)
        all_data.append(rows)
        qc.append(record)
        print(f"PASS {record['task']}: {record['conditions']} conditions, "
              f"{record['split_qc_rows']} split-QC rows")
    data = pd.concat(all_data,ignore_index=True)
    data['model_group'] = np.where(data.feature.isin(FPS),'Morgan fingerprint',
                                   np.where(data.feature.isin(NEW_FMS),'E6 FM','original FM'))
    by = ['task','split','condition','feature','model_group']
    summary = (data.groupby(by,as_index=False)
               .agg(n_seeds=('split_seed','nunique'),
                    auroc_mean=('auroc','mean'),auroc_sd=('auroc','std'),
                    auprc_mean=('auprc','mean'),auprc_sd=('auprc','std')))
    grp = ['task','split','condition']
    summary['rank_all'] = summary.groupby(grp)['auroc_mean'].rank(
        method='min',ascending=False).astype(int)
    summary['rank_fm'] = np.nan
    fm_mask = summary.model_group.ne('Morgan fingerprint')
    summary.loc[fm_mask,'rank_fm'] = (
        summary.loc[fm_mask].groupby(grp)['auroc_mean']
               .rank(method='min',ascending=False).astype(int))
    summary['rank_fm'] = summary['rank_fm'].astype('Int64')

    comparisons = []
    for (task, split, condition), sub in summary.groupby(grp):
        old_best = (sub[sub.feature.isin(OLD_FMS)]
                    .sort_values(['auroc_mean','feature'],ascending=[False,True])
                    .iloc[0]['feature'])
        base = data[(data.task==task)&(data.split==split)&
                    (data.condition==condition)]
        for model in NEW_FMS:
            current = base[base.feature.eq(model)]
            morgan = base[base.feature.eq('morgan_r2_2048')]
            best = base[base.feature.eq(old_best)]
            comparisons.append({
                'task':task,'split':split,'condition':condition,'feature':model,
                'best_original_fm':old_best,
                'auroc_mean':float(current.auroc.mean()),
                'rank_all':int(sub.loc[sub.feature.eq(model),'rank_all'].iloc[0]),
                'rank_fm':int(sub.loc[sub.feature.eq(model),'rank_fm'].iloc[0]),
                'delta_auroc_vs_morgan_r2':paired_delta(current,morgan,task),
                'delta_auroc_vs_best_original_fm':paired_delta(current,best,task),
            })
    OUT.mkdir(parents=True,exist_ok=True)
    cols = ['task','split','condition','split_seed','feature','model_group','auroc','auprc']
    data[cols].sort_values(cols[:-2]).to_csv(
        OUT/'e6_all_models_per_seed.tsv',sep='\t',index=False)
    summary.sort_values(grp+['rank_all']).to_csv(
        OUT/'e6_all_models_summary.tsv',sep='\t',index=False)
    pd.DataFrame(comparisons).sort_values(grp+['feature']).to_csv(
        OUT/'e6_new_models_vs_baselines.tsv',sep='\t',index=False)
    pd.DataFrame(qc).to_csv(OUT/'e6_qc.tsv',sep='\t',index=False)
    print('\n=== E6 rankings and comparisons (AUROC) ===')
    print(pd.DataFrame(comparisons).round(4).to_string(index=False))
    print(f'\nPASS: all five tasks consolidated; results saved to {OUT}/')

if __name__ == '__main__':
    main()

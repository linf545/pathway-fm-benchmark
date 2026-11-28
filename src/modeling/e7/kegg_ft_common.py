#!/usr/bin/env python3
"""Shared E7 KEGG data/QC, balanced multilabel loss and metric utilities.

No model architecture dependencies: usable from the MolCLR and Chemprop environments.
"""
from __future__ import annotations
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

VERSION = 'e7_kegg_gnn_balanced_auc_earlystop_v2'
DEFAULT_SEEDS = [2026, 2027, 2028, 2029, 2030]


def seed_all(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def add_data_arguments(parser):
    parser.add_argument('--compound-file', default='data/processed_bao/C_kegg/compounds_master.tsv')
    parser.add_argument('--label-file', default='data/processed_bao/C_kegg/bao_compound_labels_multilabel.tsv')
    parser.add_argument('--split-dir', default='data/splits/kegg_pathway')
    parser.add_argument('--frozen-split-qc', default='results/kegg_pathway/full/kegg_pathway_full_ensemble_split_qc.tsv')
    parser.add_argument('--split-mode', choices=['random','scaffold','both'], default='both')
    parser.add_argument('--split-seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--output-root', default='results/e7/kegg_pathway')
    parser.add_argument('--data-check-only', action='store_true')
    parser.add_argument('--resume', action='store_true')


def load_data(args):
    comp = pd.read_csv(args.compound_file, sep='\t', low_memory=False)
    labels = pd.read_csv(args.label_file, sep='\t', low_memory=False)
    id_col = 'internal_compound_id'
    required = {id_col, 'canonical_smiles', 'inchikey', 'murcko_scaffold'}
    if not required.issubset(comp):
        raise ValueError(f'Missing KEGG compound columns: {sorted(required-set(comp))}')
    if id_col not in labels:
        raise ValueError(f'Missing {id_col} in label table')
    cols = sorted((c for c in labels.columns if c.startswith('label_')), key=lambda c: int(c.split('_')[1]))
    if len(cols) != 11 or any(not c.startswith(f'label_{j}_') for j,c in enumerate(cols)):
        raise ValueError(f'Expected label_0..label_10 in numeric order, found {cols}')
    comp[id_col] = comp[id_col].astype(str)
    labels[id_col] = labels[id_col].astype(str)
    if comp[id_col].duplicated().any() or labels[id_col].duplicated().any():
        raise ValueError('Duplicated KEGG compound ID')
    task = comp.merge(labels[[id_col] + cols], on=id_col, how='inner', validate='one_to_one', sort=False)
    if len(task) != len(comp) or len(task) != len(labels):
        raise ValueError('KEGG compound/label join lost rows')
    if task['inchikey'].isna().any() or task['inchikey'].duplicated().any():
        raise ValueError('KEGG InChIKeys are missing or duplicated')
    if task['canonical_smiles'].isna().any() or task['canonical_smiles'].astype(str).str.strip().eq('').any():
        raise ValueError('Empty KEGG canonical SMILES')
    y = task[cols].to_numpy(dtype=np.float32)
    if not np.isfinite(y).all() or not np.isin(y,[0,1]).all() or (y.sum(axis=1)==0).any():
        raise ValueError('Expected non-empty 11-label binary multilabel target matrix')
    print(f'KEGG compounds= {len(task)} labels= {len(cols)} script= {VERSION}', flush=True)
    return task, y, cols


def load_split(task, y, labels, mode, seed, args):
    path = Path(args.split_dir) / f'kegg_pathway_{mode}_split_seed{seed}.tsv'
    if not path.is_file():
        raise FileNotFoundError(f'Original frozen KEGG split missing: {path}')
    saved = pd.read_csv(path, sep='\t', low_memory=False)
    id_col = 'internal_compound_id'
    if not {id_col, 'split'}.issubset(saved):
        raise ValueError(f'{path} must contain compound ID and split columns')
    saved[id_col] = saved[id_col].astype(str)
    if saved[id_col].duplicated().any() or len(saved)!=len(task):
        raise ValueError(f'{path} contains duplicated/missing IDs')
    partition = task[id_col].map(saved.set_index(id_col)['split'])
    if partition.isna().any() or set(partition.unique()) != {'train','validation','test'}:
        raise ValueError(f'{path} missing IDs/partitions')
    ix={part: np.flatnonzero(partition.eq(part).to_numpy()) for part in ['train','validation','test']}
    for part, ids in ix.items():
        pos = y[ids].sum(axis=0)
        if ((pos<1)|(pos>=len(ids))).any():
            raise ValueError(f'{path}: empty positive/negative label support in {part}')
    if mode=='scaffold':
        groups={part: {g for g in task.iloc[ids]['murcko_scaffold'].fillna('').astype(str) if g not in {'','nan'}} for part,ids in ix.items()}
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            if groups[a]&groups[b]:
                raise ValueError(f'{path}: scaffold leakage {a} vs {b}')
    if not Path(args.frozen_split_qc).is_file():
        raise FileNotFoundError(f'Original frozen QC missing: {args.frozen_split_qc}')
    qc=pd.read_csv(args.frozen_split_qc, sep='\t')
    for part,ids in ix.items():
        match=qc[(qc.split_mode==mode)&(qc.split_seed==int(seed))&(qc.split==part)]
        if len(match)!=1 or int(match.iloc[0].n_compounds)!=len(ids):
            raise ValueError(f'Frozen split size mismatch: {mode} {seed} {part}')
        counts=y[ids].sum(axis=0).astype(int)
        for label,count in zip(labels,counts):
            key=f'positive_count__{label}'
            if key not in match or int(match.iloc[0][key]) != int(count):
                raise ValueError(f'Frozen class count mismatch: {mode} {seed} {part} {label}')
    print('SPLIT-QC PASS',mode,seed,{k:len(v) for k,v in ix.items()},flush=True)
    return ix,path


def experiments(args):
    task,y,labels=load_data(args)
    modes=['random','scaffold'] if args.split_mode=='both' else [args.split_mode]
    conditions=[]
    for mode in modes:
        for seed in args.split_seeds:
            ix,path=load_split(task,y,labels,mode,seed,args)
            conditions.append((mode,int(seed),ix,path))
    return task,y,labels,conditions


def balanced_weights(y_train, device):
    pos=np.asarray(y_train,dtype=np.float32).sum(axis=0)
    n=len(y_train);neg=n-pos
    if np.any(pos<=0) or np.any(neg<=0):
        raise ValueError('Cannot balance label without both classes in training partition')
    wp=n/(2*pos)
    wn=n/(2*neg)
    return (torch.tensor(wp,device=device,dtype=torch.float32),
            torch.tensor(wn,device=device,dtype=torch.float32), pos.astype(int))


def balanced_bce(logits, target, wpos, wneg):
    target=target.float()
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), target, reduction='none')
    weight=target*wpos+(1-target)*wneg
    return (loss*weight).mean()


def macro_scores(y,p):
    y=np.asarray(y);p=np.asarray(p)
    if y.shape!=p.shape or y.ndim!=2 or y.shape[1]!=11:
        raise ValueError(f'Prediction and label shapes must be [N,11], got {y.shape}, {p.shape}')
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite probability prediction')
    auroc=[];auprc=[]
    for i in range(y.shape[1]):
        if np.unique(y[:,i]).size!=2:
            raise ValueError(f'Label {i} absent/present only in evaluation set')
        auroc.append(roc_auc_score(y[:,i],p[:,i]))
        auprc.append(average_precision_score(y[:,i],p[:,i]))
    return dict(macro_auroc=float(np.mean(auroc)),macro_auprc=float(np.mean(auprc)),
                per_label_auroc=[float(x) for x in auroc],
                per_label_auprc=[float(x) for x in auprc])


def is_better(auc,pr,best_auc,best_pr):
    return auc>best_auc+1e-12 or (abs(auc-best_auc)<=1e-12 and pr>best_pr+1e-12)


class MeaningfulEarlyStopper:
    """Stop after patience epochs without >= min_delta macro-AUROC gain.

    The stopping reference is the last *meaningful* validation AUROC, not the
    previous epoch and not the raw best checkpoint. Checkpoint selection must
    separately maximize raw AUROC (with AUPRC tie-breaker).
    """

    def __init__(self, patience=3, min_delta=0.001, min_epochs=5):
        if patience < -1 or patience == 0:
            raise ValueError('Patience must be >= 1 or -1 for disabled')
        if not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError('min_delta must be finite and nonnegative')
        if min_epochs < 1:
            raise ValueError('min_epochs must be >= 1')
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_epochs = int(min_epochs)
        self.reference_auc = float('-inf')
        self.stale_epochs = 0
        self.stopped = False

    def step(self, epoch, auc):
        if self.stopped:
            raise RuntimeError('Early stopper invoked after stopping')
        if not np.isfinite(auc):
            raise ValueError(f'Nonfinite validation AUROC at epoch {epoch}: {auc}')
        meaningful = (
            self.reference_auc == float('-inf')
            or auc >= self.reference_auc + self.min_delta - 1e-12
        )
        if meaningful:
            self.reference_auc = float(auc)
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        self.stopped = (
            self.patience >= 1
            and epoch >= self.min_epochs
            and self.stale_epochs >= self.patience
        )
        return self.stopped, meaningful


def output_dir(args,model,mode,seed):
    return Path(args.output_root)/model/'full'/mode/f'seed{seed}'


def run_is_complete(out, args):
    metadata=out/'run_metadata.json'
    if not metadata.exists():
        return False
    if not args.resume:
        raise FileExistsError(f'{metadata} exists; use --resume or another output root')
    details=json.loads(metadata.read_text())
    expected={'script_version':VERSION,'selected_by':'validation_macro_auroc_then_macro_auprc',
              'epochs_requested':int(args.epochs),
              'early_stopping_patience':int(args.early_stopping_patience),
              'early_stopping_min_delta':float(args.min_delta),
              'early_stopping_min_epochs':int(args.min_epochs)}
    for k,v in expected.items():
        if details.get(k)!=v:
            raise ValueError(f'Existing run incompatible with resume: {out}, {k}={details.get(k)!r}, expected {v!r}')
    previous_args=details.get('args',{})
    for k in ('encoder_lr','head_lr','batch_size','weight_decay'):
        if previous_args.get(k)!=getattr(args,k):
            raise ValueError(f'Existing run incompatible with resume: {out}, {k} differs')
    if not (out/'best.pt').exists() or not (out/'predictions.npz').exists():
        raise ValueError(f'Existing run incomplete: {out}; inspect and move it before resuming')
    print('RESUME SKIP',out,flush=True)
    return True


def save_run(out, args, meta, history, validation_y, validation_p, test_y, test_p, labels, model):
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(history).to_csv(out/'epoch_history.tsv',sep='\t',index=False)
    np.savez_compressed(out/'predictions.npz',validation_y=validation_y,validation_probability=validation_p,
                        test_y=test_y,test_probability=test_p,label_columns=np.asarray(labels))
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2,default=str)+'\n')
    torch.save(model.state_dict(),out/'best.pt')


def summarize(args, model):
    rows=[]
    for mode in ['random','scaffold']:
        for seed in args.split_seeds:
            file=output_dir(args,model,mode,seed)/'run_metadata.json'
            if file.exists():
                meta=json.loads(file.read_text())
                rows.append({k:meta[k] for k in ['feature','adaptation_mode','split_mode','split_seed','selected_epoch',
                           'n_train','n_validation','n_test','validation_macro_auroc','validation_macro_auprc',
                           'test_macro_auroc','test_macro_auprc','training_seconds','trainable_parameters']})
    if rows:
        frame=pd.DataFrame(rows).sort_values(['split_mode','split_seed'])
        base=Path(args.output_root)/model
        base.mkdir(parents=True,exist_ok=True)
        frame.to_csv(base/'e7_kegg_full_repeated_runs.tsv',sep='\t',index=False)
        s=frame.groupby(['feature','adaptation_mode','split_mode'])['test_macro_auroc'].agg(['mean','std','count']).reset_index()
        s.to_csv(base/'e7_kegg_full_summary.tsv',sep='\t',index=False)
        print('RESULTS',base/'e7_kegg_full_repeated_runs.tsv',flush=True)
        print(s.to_string(index=False),flush=True)

#!/usr/bin/env python3
"""E7 NP-likeness shared data/QC, metrics, stopping, and result handling.

Uses the ORIGINAL 411,448-compound COCONUT-versus-ZINC task and EXACT saved
frozen-benchmark split files. The property-matched E1 task is deliberately
excluded from this primary frozen-versus-fine-tuned comparison.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

VERSION = 'e7_np_likeness_20ep_auc_v1'
SEEDS = [2026, 2027, 2028, 2029, 2030]
ID = 'compound_id'
LABEL = 'is_np'
PARTS = ('train', 'validation', 'test')


def add_data_args(p):
    p.add_argument('--task-file', default='data/tasks/np_likeness/np_likeness_task.tsv')
    p.add_argument('--split-dir', default='data/splits/np_likeness')
    p.add_argument('--frozen-qc', default='results/np_likeness/full/np_likeness_full_ensemble_split_qc.tsv')
    p.add_argument('--expected-compounds', type=int, default=411448)
    p.add_argument('--split-mode', choices=('random', 'scaffold', 'both'), default='both')
    p.add_argument('--split-seeds', nargs='+', type=int, default=SEEDS)
    p.add_argument('--output-root', default='results/e7/np_likeness_20ep')
    p.add_argument('--smoke-limit', type=int, default=0,
                   help='Limit EACH partition to this many class-stratified examples; only for isolated smoke output')
    p.add_argument('--data-check-only', action='store_true')
    p.add_argument('--resume', action='store_true')


def add_stopping_args(p):
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--early-stopping-patience', type=int, default=3)
    p.add_argument('--min-delta', type=float, default=0.001)
    p.add_argument('--min-epochs', type=int, default=5)


def check_args(args):
    if args.epochs < 1 or args.min_epochs < 1 or args.min_epochs > args.epochs:
        raise ValueError('Invalid epochs or min_epochs')
    if args.early_stopping_patience < 1 or not 0 <= args.min_delta < 1:
        raise ValueError('Use patience >= 1 and 0 <= min_delta < 1')
    if args.smoke_limit and args.smoke_limit < 4:
        raise ValueError('Smoke-limit must be >= 4 for binary stratification')
    if len(set(args.split_seeds)) != len(args.split_seeds):
        raise ValueError('Duplicate seed')


def seed_all(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_task(args):
    f = Path(args.task_file)
    if not f.is_file():
        raise FileNotFoundError(f)
    task = pd.read_csv(f, sep='\t', low_memory=False)
    required = {ID, LABEL, 'canonical_smiles', 'inchikey', 'scaffold_split_key', 'source'}
    if not required.issubset(task.columns):
        raise ValueError(f'Original task lacks {sorted(required - set(task.columns))}')
    if args.expected_compounds and len(task) != args.expected_compounds:
        raise ValueError(f'Expected original NP-likeness task with {args.expected_compounds:,} rows; found {len(task):,}. Do not substitute E1 matched data.')
    if task[[ID, LABEL, 'canonical_smiles', 'inchikey', 'scaffold_split_key']].isna().any().any():
        raise ValueError('Original NP-likeness task contains missing key columns')
    task[ID] = task[ID].astype(str)
    task['inchikey'] = task['inchikey'].astype(str)
    task['canonical_smiles'] = task['canonical_smiles'].astype(str)
    task[LABEL] = task[LABEL].astype(int)
    if task[ID].duplicated().any() or task['inchikey'].duplicated().any():
        raise ValueError('Duplicate compound_id or InChIKey')
    if task['canonical_smiles'].str.strip().eq('').any():
        raise ValueError('Empty SMILES')
    if set(task[LABEL].unique()) != {0, 1}:
        raise ValueError('Expected two binary NP-likeness classes')
    if args.expected_compounds == 411448:
        counts = task[LABEL].value_counts()
        if int(counts[0]) != 205724 or int(counts[1]) != 205724:
            raise ValueError(f'Original COCONUT/ZINC balance unexpected: {counts.to_dict()}')
    print('NP-likeness ORIGINAL task', len(task), 'class_counts=', task[LABEL].value_counts().to_dict(), 'version=', VERSION, flush=True)
    return task


def split_path(args, mode, seed):
    return Path(args.split_dir) / f'np_likeness_{mode}_split_seed{seed}.tsv'


def load_split(task, args, mode, seed):
    path = split_path(args, mode, seed)
    if not path.is_file():
        raise FileNotFoundError(f'Missing original frozen split: {path}')
    saved = pd.read_csv(path, sep='\t', low_memory=False)
    if not {ID, 'split'}.issubset(saved.columns):
        raise ValueError(f'{path} lacks compound_id and split')
    saved[ID] = saved[ID].astype(str)
    if len(saved) != len(task) or saved[ID].duplicated().any():
        raise ValueError(f'{path} contains duplicated or missing compound IDs')
    merged = task[[ID, LABEL, 'inchikey', 'source']].merge(
        saved, on=ID, how='left', sort=False, validate='one_to_one', suffixes=('', '_saved'))
    if len(merged) != len(task) or merged['split'].isna().any():
        raise ValueError(f'{path} missing task compound IDs')
    for c in ('inchikey', LABEL, 'source'):
        if f'{c}_saved' in merged.columns:
            if c == LABEL:
                equal = merged[c].astype(int) == merged[f'{c}_saved'].astype(int)
            else:
                equal = merged[c].astype(str) == merged[f'{c}_saved'].astype(str)
            if not equal.all():
                raise ValueError(f'{path}: {c} disagrees with original task')
    if set(merged['split'].unique()) != set(PARTS):
        raise ValueError(f'{path}: unexpected split values {merged["split"].unique()}')
    ix = {name: np.flatnonzero(merged['split'].eq(name).to_numpy()) for name in PARTS}
    y = task[LABEL].to_numpy(dtype=np.int8)
    for name, ids in ix.items():
        if np.unique(y[ids]).size != 2:
            raise ValueError(f'{mode}/{seed} {name} has one class')
    if mode == 'scaffold':
        keys = task['scaffold_split_key'].astype(str).to_numpy()
        groups = {name: set(keys[ids]) for name, ids in ix.items()}
        for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
            if groups[a] & groups[b]:
                raise ValueError(f'Scaffold leakage: {mode} {seed} {a}/{b}')
    qc_path = Path(args.frozen_qc)
    if qc_path.is_file():
        qc = pd.read_csv(qc_path, sep='\t')
        for part, ids in ix.items():
            q = qc.loc[(qc.split_mode == mode) & (qc.split_seed == int(seed)) & (qc.split == part)]
            if len(q) != 1:
                raise ValueError(f'Original frozen QC row missing: {mode} {seed} {part}')
            q = q.iloc[0]
            if int(q.n_compounds) != len(ids) or int(q.n_positive) != int(y[ids].sum()) or int(q.n_negative) != int(len(ids) - y[ids].sum()):
                raise ValueError(f'Frozen QC mismatch: {mode} {seed} {part}')
    print('SPLIT-QC PASS', mode, seed,
          {name: (len(ids), int(y[ids].sum())) for name, ids in ix.items()}, flush=True)
    return ix, path


def stratified_smoke_indices(ix, y, limit, seed):
    if not limit:
        return ix
    out = {}
    for part, ids in ix.items():
        rng = np.random.default_rng(int(seed) + {'train': 0, 'validation': 100, 'test': 200}[part])
        positive = ids[y[ids] == 1]
        negative = ids[y[ids] == 0]
        take_pos = min(len(positive), limit // 2)
        take_neg = min(len(negative), limit - take_pos)
        if take_pos < 1 or take_neg < 1:
            raise ValueError('Smoke subset has only one class')
        out[part] = np.sort(np.concatenate([
            rng.choice(positive, size=take_pos, replace=False),
            rng.choice(negative, size=take_neg, replace=False)]))
    return out


def scores(y, p):
    y = np.asarray(y, dtype=int).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    if y.shape != p.shape or not np.isfinite(p).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError('Invalid binary labels/probabilities for AUROC')
    return {'auroc': float(roc_auc_score(y, p)), 'auprc': float(average_precision_score(y, p))}


def balanced_binary_loss(logits, y, wpos, wneg):
    y = y.float()
    raw = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y, reduction='none')
    return (raw * (y * wpos + (1 - y) * wneg)).mean()


def balanced_binary_weights(y):
    y = np.asarray(y, dtype=int)
    pos = int(y.sum()); neg = int(len(y) - pos)
    if pos < 1 or neg < 1:
        raise ValueError('Training has only one class')
    return len(y) / (2.0 * pos), len(y) / (2.0 * neg)


def threshold_from_validation(y, prob):
    # Matches the original frozen benchmark threshold grid and tie-breaking.
    best_thr = None
    best_f1 = -np.inf
    best_gap = np.inf
    y = np.asarray(y)
    prob = np.asarray(prob)
    for thr in (0.2, 0.4, 0.6, 0.8):
        pred = prob >= thr
        f1 = float(f1_score(y, pred, zero_division=0))
        prevalence_gap = abs(float(np.mean(pred)) - float(np.mean(y)))
        if f1 > best_f1 or (np.isclose(f1, best_f1) and prevalence_gap < best_gap):
            best_thr, best_f1, best_gap = thr, f1, prevalence_gap
    return float(best_thr), float(best_f1)


def is_better(auc, pr, best_auc, best_pr):
    return auc > best_auc + 1e-12 or (abs(auc - best_auc) <= 1e-12 and pr > best_pr + 1e-12)


class EarlyStop:
    def __init__(self, patience=3, min_delta=0.001, min_epochs=5):
        self.patience, self.min_delta, self.min_epochs = patience, min_delta, min_epochs
        self.reference_auc = -np.inf
        self.stale = 0
        self.stopped = False

    def step(self, epoch, auc):
        if not np.isfinite(auc):
            raise ValueError(f'Nonfinite validation AUROC at epoch {epoch}')
        meaningful = not np.isfinite(self.reference_auc) or auc >= self.reference_auc + self.min_delta - 1e-12
        if meaningful:
            self.reference_auc = float(auc)
            self.stale = 0
        else:
            self.stale += 1
        self.stopped = epoch >= self.min_epochs and self.stale >= self.patience
        return self.stopped, meaningful


def run_path(args, feature, adaptation, split, seed):
    return Path(args.output_root) / feature / adaptation / split / f'seed{seed}'


def run_config(args, **extra):
    # Store all relevant hyperparameters and artifact provenance for safe resume.
    skip = {'resume', 'data_check_only', 'split_seeds', 'split_mode', 'output_root'}
    config = {k: v for k, v in vars(args).items() if k not in skip}
    return {**config, **extra, 'script_version': VERSION}


def check_run(out, args, config):
    meta = out / 'run_metadata.json'
    if not out.exists():
        return False
    if not meta.is_file():
        raise FileExistsError(f'Incomplete existing run {out}. Move that folder aside before resuming.')
    if not args.resume:
        raise FileExistsError(f'Existing run {out}. Use --resume.')
    record = json.loads(meta.read_text())
    if record.get('run_config') != config or record.get('script_version') != VERSION:
        raise ValueError(f'Resume configuration mismatch: {out}')
    for name in ('epoch_history.tsv', 'predictions.npz', 'best.pt'):
        if not (out / name).is_file():
            raise FileNotFoundError(f'Incomplete run {out}: {name}')
    print('RESUME SKIP', out, flush=True)
    return True


def save_run(out, config, meta, history, val_y, val_p, test_y, test_p, checkpoint):
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out / 'epoch_history.tsv', sep='\t', index=False)
    np.savez_compressed(out / 'predictions.npz',
                        validation_y=val_y, validation_probability=val_p,
                        test_y=test_y, test_probability=test_p)
    torch.save(checkpoint, out / 'best.pt')
    (out / 'run_metadata.json').write_text(json.dumps({**meta, 'run_config': config}, indent=2, default=str) + '\n')


def write_summary(root, feature):
    folder = Path(root) / feature
    records = []
    for meta in folder.glob('*/*/seed*/run_metadata.json'):
        j = json.loads(meta.read_text())
        records.append({k: j.get(k) for k in (
            'feature', 'adaptation_mode', 'split_mode', 'split_seed',
            'epochs_requested', 'epochs_trained', 'stopped_early', 'selected_epoch',
            'n_train', 'n_validation', 'n_test', 'validation_auroc', 'validation_auprc',
            'test_auroc', 'test_auprc', 'test_f1', 'selected_threshold',
            'training_seconds', 'trainable_parameters', 'peak_gpu_allocated_bytes')})
    if not records:
        return
    frame = pd.DataFrame(records).sort_values(['adaptation_mode', 'split_mode', 'split_seed'])
    frame.to_csv(folder / 'e7_np_likeness_repeated_runs.tsv', sep='\t', index=False)
    s = frame.groupby(['feature', 'adaptation_mode', 'split_mode'])['test_auroc'].agg(['mean','std','count']).reset_index()
    s.to_csv(folder / 'e7_np_likeness_summary.tsv', sep='\t', index=False)
    print('RESULTS', folder / 'e7_np_likeness_summary.tsv', flush=True)
    print(s.to_string(index=False), flush=True)


def final_metadata(args, feature, adaptation, mode, seed, split_file, ix, best_epoch, best_auc, best_pr,
                   history, stopper, val, test, threshold, test_f1, trainable, total, elapsed, device, **extra):
    return {'script_version': VERSION, 'task': 'original_np_likeness_coconut_vs_zinc',
            'feature': feature, 'adaptation_mode': adaptation,
            'split_mode': mode, 'split_seed': int(seed), 'split_file': str(split_file),
            'selected_epoch': int(best_epoch), 'selected_by': 'validation_auroc_then_auprc',
            'checkpoint_selection_metric': 'validation_auroc_then_auprc',
            'epochs_requested': int(args.epochs), 'epochs_trained': len(history),
            'stopped_early': bool(stopper.stopped),
            'early_stopping_patience': int(args.early_stopping_patience),
            'early_stopping_min_delta': float(args.min_delta), 'early_stopping_min_epochs': int(args.min_epochs),
            'n_train': len(ix['train']), 'n_validation': len(ix['validation']), 'n_test': len(ix['test']),
            'validation_auroc': val['auroc'], 'validation_auprc': val['auprc'],
            'test_auroc': test['auroc'], 'test_auprc': test['auprc'],
            'selected_threshold': threshold, 'test_f1': test_f1,
            'best_validation_auroc_at_epoch': best_auc, 'best_validation_auprc_at_epoch': best_pr,
            'total_parameters': total, 'trainable_parameters': trainable,
            'training_seconds': elapsed,
            'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'smoke_limit': args.smoke_limit, **extra}

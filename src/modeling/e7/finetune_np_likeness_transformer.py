#!/usr/bin/env python3
"""E7 NP-likeness: ORIGINAL task, ChemBERTa/MolFormer full fine-tuning and LoRA.

Reuses the pretrained encoder/LoRA implementation of
src/modeling/np_likeness/finetune_np_likeness_lora_v2.py.
Checkpoint by raw validation AUROC (AUPRC tie breaker), early stop by
meaningful +0.001 AUROC. Test is evaluated only on selected checkpoint.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from np_likeness_ft_common import (
    VERSION, add_data_args, add_stopping_args, check_args, load_task, load_split,
    stratified_smoke_indices, seed_all, run_path, run_config, check_run,
    save_run, write_summary, final_metadata, scores, is_better, EarlyStop,
    balanced_binary_weights, balanced_binary_loss, threshold_from_validation, sha256,
)

ROOT = Path(__file__).resolve().parents[3]
ORIGINAL = ROOT / 'src/modeling/np_likeness/finetune_np_likeness_lora_v2.py'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    add_data_args(p); add_stopping_args(p)
    p.add_argument('--model-name-or-path', required=True)
    p.add_argument('--feature-name', required=True)
    p.add_argument('--adaptation-modes', nargs='+', choices=['full', 'lora'], default=['full','lora'])
    p.add_argument('--train-batch-size', type=int, default=16)
    p.add_argument('--eval-batch-size', type=int, default=64)
    p.add_argument('--gradient-accumulation-steps', type=int, default=4)
    p.add_argument('--full-learning-rate', type=float, default=2e-5)
    p.add_argument('--lora-learning-rate', type=float, default=1e-4)
    p.add_argument('--head-learning-rate', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--warmup-ratio', type=float, default=0.06)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--head-dropout', type=float, default=0.0)
    p.add_argument('--lora-r', type=int, default=8)
    p.add_argument('--lora-alpha', type=int, default=16)
    p.add_argument('--lora-dropout', type=float, default=0.1)
    p.add_argument('--lora-target-modules', nargs='+', default=['query','value'])
    p.add_argument('--precision', choices=['bf16','fp16','fp32'], default='bf16')
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--gradient-checkpointing', action='store_true')
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--tokenize-chunk-size', type=int, default=4096)
    return p.parse_args()


def prior_module():
    if not ORIGINAL.is_file():
        raise FileNotFoundError(f'Existing benchmark LoRA implementation required: {ORIGINAL}')
    spec = importlib.util.spec_from_file_location('e7_np_likeness_prior_lora', ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TokenizedDataset(Dataset):
    def __init__(self, encoded, labels, indices):
        self.encoded = encoded
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        ix = int(self.indices[i])
        return ix, self.encoded[ix], float(self.labels[ix])


class Collator:
    def __init__(self, tokenizer, precision):
        self.pad = DataCollatorWithPadding(tokenizer, padding=True,
                  pad_to_multiple_of=8 if precision != 'fp32' else None,
                  return_tensors='pt')
    def __call__(self, examples):
        ids, encodings, labels = zip(*examples)
        out = self.pad(list(encodings))
        out['labels'] = torch.tensor(labels, dtype=torch.float32)
        out['row_indices'] = np.asarray(ids, dtype=np.int64)
        return out


def tokenize_task(task, tokenizer, args):
    # Pre-tokenize ONCE and reuse through all 10 splits/seeds; avoid repeated
    # costly tokenization of 411k molecules each training epoch.
    encoded = []
    smiles = task['canonical_smiles'].tolist()
    for begin in range(0, len(smiles), args.tokenize_chunk_size):
        chunk = tokenizer(smiles[begin:begin+args.tokenize_chunk_size],
                truncation=True, max_length=args.max_length,
                padding=False, return_attention_mask=True)
        for j in range(len(chunk['input_ids'])):
            encoded.append({name: values[j] for name, values in chunk.items()})
        if (begin // args.tokenize_chunk_size) % 20 == 0:
            print(f'TOKENIZE {min(begin+args.tokenize_chunk_size,len(smiles)):,}/{len(smiles):,}', flush=True)
    assert len(encoded) == len(smiles)
    return encoded


def loader(dataset, collator, batch, shuffle, seed, workers):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle,
            collate_fn=collator, num_workers=workers, pin_memory=True,
            persistent_workers=workers > 0,
            generator=torch.Generator().manual_seed(seed) if shuffle else None)


def batch_inputs(batch, device):
    return {k:v.to(device, non_blocking=True) for k,v in batch.items()
            if k not in ('labels','row_indices')}


@torch.no_grad()
def evaluate(model, dl, device, old, precision):
    model.eval()
    ys, ps, idxs = [], [], []
    total = 0.0
    n = 0
    for batch in dl:
        yy = batch['labels'].to(device, non_blocking=True)
        with old.autocast_context(precision):
            logits = model(**batch_inputs(batch, device)).float()
        total += float(nn.functional.binary_cross_entropy_with_logits(logits, yy, reduction='sum').item())
        n += yy.numel()
        ys.append(yy.cpu().numpy())
        ps.append(logits.sigmoid().cpu().numpy())
        idxs.append(batch['row_indices'])
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    idx = np.concatenate(idxs)
    metric = scores(y,p)
    metric['unweighted_loss'] = total / n
    return metric, y, p, idx


def train_one(task, y, encoded, tokenizer, old, args, adaptation, mode, seed, ix, path, task_hash, split_hash):
    seed_all(seed)
    device = torch.device('cuda')
    out = run_path(args, args.feature_name, adaptation, mode, seed)
    config = run_config(args, task_sha256=task_hash, split_sha256=split_hash,
                        checkpoint=args.model_name_or_path)
    if check_run(out,args,config):
        return
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 not supported on this GPU')

    if adaptation == 'lora':
        encoder, hidden, matched = old.configure_encoder(args.model_name_or_path,'lora',args)
    else:
        encoder = AutoModel.from_pretrained(args.model_name_or_path,
                  trust_remote_code=args.trust_remote_code, local_files_only=args.local_files_only)
        hidden = old.infer_hidden_size(encoder)
        matched = []
        if hasattr(encoder.config,'use_cache'):
            encoder.config.use_cache = False
        if args.gradient_checkpointing and hasattr(encoder,'gradient_checkpointing_enable'):
            encoder.gradient_checkpointing_enable()
    model = old.MeanPooledBinaryClassifier(encoder, hidden, args.head_dropout).to(device)
    total_params, trainable = old.count_parameters(model)
    if not any(p.requires_grad for p in encoder.parameters()):
        raise RuntimeError('Encoder not trainable in fine-tuning mode')

    collate = Collator(tokenizer,args.precision)
    dls = {part: loader(TokenizedDataset(encoded,y,ids),collate,
           args.train_batch_size if part=='train' else args.eval_batch_size,
           part=='train',seed,args.num_workers) for part,ids in ix.items()}
    wp, wn = balanced_binary_weights(y[ix['train']])
    wp, wn = torch.tensor(wp,device=device),torch.tensor(wn,device=device)
    groups = [
        {'params':[p for p in encoder.parameters() if p.requires_grad],
         'lr':args.lora_learning_rate if adaptation=='lora' else args.full_learning_rate,
         'weight_decay':0.0 if adaptation=='lora' else args.weight_decay},
        {'params':list(model.classifier.parameters()),'lr':args.head_learning_rate,'weight_decay':args.weight_decay},
    ]
    optimizer = torch.optim.AdamW(groups)
    nupdates = math.ceil(len(dls['train']) / args.gradient_accumulation_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
              num_warmup_steps=round(nupdates*args.warmup_ratio),num_training_steps=nupdates)
    scaler = torch.amp.GradScaler('cuda',enabled=args.precision=='fp16')
    stopper = EarlyStop(args.early_stopping_patience,args.min_delta,args.min_epochs)
    best_auc=-np.inf; best_pr=-np.inf; best_epoch=0; best_state=None; history=[]
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    print('START', args.feature_name,adaptation,mode,seed,'n_train=',len(ix['train']),
          'class_weights=',float(wp),float(wn),'trainable=',trainable,flush=True)
    for epoch in range(1,args.epochs+1):
        t0=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True)
        n=0; running=0.0
        for step,batch in enumerate(dls['train'],1):
            yy=batch['labels'].to(device,non_blocking=True)
            with old.autocast_context(args.precision):
                logits=model(**batch_inputs(batch,device)).float()
                loss=balanced_binary_loss(logits, yy, wp,wn)
            # Correct final incomplete accumulation group's scale.
            start_idx=((step-1)//args.gradient_accumulation_steps)*args.gradient_accumulation_steps
            group_size=min(args.gradient_accumulation_steps, len(dls['train'])-start_idx)
            if scaler.is_enabled():scaler.scale(loss/group_size).backward()
            else:(loss/group_size).backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(dls['train']):
                if scaler.is_enabled():scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],args.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer);scaler.update()
                else:optimizer.step()
                scheduler.step();optimizer.zero_grad(set_to_none=True)
            running+=float(loss.detach().item())*len(yy);n+=len(yy)
        val, _, _, _ = evaluate(model,dls['validation'],device,old,args.precision)
        if is_better(val['auroc'],val['auprc'],best_auc,best_pr):
            best_auc=val['auroc'];best_pr=val['auprc'];best_epoch=epoch
            best_state=old.snapshot_trainable_parameters(model)
        stop, meaningful=stopper.step(epoch,val['auroc'])
        history.append({'epoch':epoch,'train_balanced_loss':running/n,
                        'validation_auroc':val['auroc'],'validation_auprc':val['auprc'],
                        'validation_unweighted_loss':val['unweighted_loss'],
                        'early_stop_stale_epochs':stopper.stale,
                        'meaningful_auroc_improvement':meaningful,
                        'seconds':time.monotonic()-t0})
        print(args.feature_name,adaptation,mode,seed,'epoch',epoch,
              'val_auc',round(val['auroc'],6),'val_auprc',round(val['auprc'],6),
              'stale',stopper.stale,flush=True)
        if stop:
            print('EARLY STOP',args.feature_name,adaptation,mode,seed,
                  'epoch',epoch,'best_epoch',best_epoch,flush=True)
            break
    assert best_state is not None
    old.restore_trainable_parameters(model,best_state)
    val,vy,vp,_=evaluate(model,dls['validation'],device,old,args.precision)
    threshold,val_f1=threshold_from_validation(vy,vp)
    test,ty,tp,_=evaluate(model,dls['test'],device,old,args.precision)
    from sklearn.metrics import f1_score
    test_f1=float(f1_score(ty,tp>=threshold,zero_division=0))
    elapsed=time.monotonic()-started
    metadata=final_metadata(args,args.feature_name,adaptation,mode,seed,path,ix,
            best_epoch,best_auc,best_pr,history,stopper,val,test,threshold,test_f1,
            trainable,total_params,elapsed,device,
            training_loss='balanced_binary_cross_entropy',
            positive_weight=float(wp),negative_weight=float(wn),
            matched_lora_modules=matched,
            validation_f1_at_selected_threshold=val_f1)
    # Store trainable weights, including all encoder weights in full mode.
    save_run(out,config,metadata,history,vy,vp,ty,tp,best_state)
    print('TEST',json.dumps({k:metadata[k] for k in ('feature','adaptation_mode','split_mode','split_seed',
                      'selected_epoch','validation_auroc','test_auroc','epochs_trained','training_seconds')}),flush=True)
    del model,encoder,optimizer,scheduler,dls,best_state
    torch.cuda.empty_cache()


def main():
    args=parse_args();check_args(args)
    if args.train_batch_size<1 or args.eval_batch_size<1 or args.gradient_accumulation_steps<1 or args.tokenize_chunk_size<1:
        raise ValueError('Invalid batch/accumulation/tokenization size')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    task=load_task(args)
    y=task['is_np'].to_numpy(dtype=np.int8)
    modes=('random','scaffold') if args.split_mode=='both' else (args.split_mode,)
    if args.data_check_only:
        for mode in modes:
            for seed in args.split_seeds:
                load_split(task,args,mode,seed)
        print('PASS: original NP-likeness splits; no model loaded',flush=True)
        return
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    if not ORIGINAL.is_file():raise FileNotFoundError(ORIGINAL)
    old=prior_module()
    tok=AutoTokenizer.from_pretrained(args.model_name_or_path,
                 trust_remote_code=args.trust_remote_code,local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token=tok.eos_token or tok.sep_token or tok.cls_token
    if tok.pad_token is None:raise ValueError('No usable pad token')
    encoded=tokenize_task(task,tok,args)
    task_hash=sha256(args.task_file)
    for mode in modes:
        for seed in args.split_seeds:
            ix,path=load_split(task,args,mode,seed)
            ix=stratified_smoke_indices(ix,y,args.smoke_limit,seed)
            split_hash=sha256(path)
            for adaptation in args.adaptation_modes:
                train_one(task,y,encoded,tok,old,args,adaptation,mode,seed,ix,path,task_hash,split_hash)
                write_summary(args.output_root,args.feature_name)
    print('PASS transformer',args.feature_name,flush=True)

if __name__=='__main__':main()

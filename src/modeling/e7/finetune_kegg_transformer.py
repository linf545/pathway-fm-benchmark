#!/usr/bin/env python3
"""E7 KEGG multilabel fine-tuning: ChemBERTa / MolFormer, full or LoRA.

Reuses the exact Bao C/KEGG 11-label task and frozen benchmark's 10 saved
random/scaffold split files. LoRA encoder construction and mean pooling
reuse the prior NP-classification LoRA v2 implementation in this repository.

Checkpoint: maximum validation macro AUROC, validation macro AUPRC tie-breaker.
Train loss: sklearn balanced class weights for EACH binary pathway outcome.
Validation/test cross entropy is unweighted QC only.
"""
from __future__ import annotations
import argparse
import contextlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup
from kegg_ft_common import MeaningfulEarlyStopper

ROOT = Path(__file__).resolve().parents[3]
OLD_LORA = ROOT / 'src/modeling/np_classification/finetune_npclassification_lora_v2.py'
VERSION = 'e7_kegg_transformer_balanced_auc_earlystop_v2'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compound-file', default='data/processed_bao/C_kegg/compounds_master.tsv')
    p.add_argument('--label-file', default='data/processed_bao/C_kegg/bao_compound_labels_multilabel.tsv')
    p.add_argument('--split-dir', default='data/splits/kegg_pathway')
    p.add_argument('--frozen-split-qc', default='results/kegg_pathway/full/kegg_pathway_full_ensemble_split_qc.tsv')
    p.add_argument('--model-name-or-path', required=True)
    p.add_argument('--feature-name', required=True)
    p.add_argument('--adaptation-modes', nargs='+', choices=['lora','full'], default=['full','lora'])
    p.add_argument('--split-mode', choices=['both','random','scaffold'], default='both')
    p.add_argument('--split-seeds', nargs='+', type=int, default=[2026,2027,2028,2029,2030])
    p.add_argument('--output-root', default='results/e7/kegg_pathway')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--early-stopping-patience', type=int, default=3)
    p.add_argument('--min-delta', type=float, default=0.001)
    p.add_argument('--min-epochs', type=int, default=5)
    p.add_argument('--train-batch-size', type=int, default=16)
    p.add_argument('--eval-batch-size', type=int, default=64)
    p.add_argument('--gradient-accumulation-steps', type=int, default=4)
    p.add_argument('--full-learning-rate', type=float, default=2e-5)
    p.add_argument('--lora-learning-rate', type=float, default=1e-4)
    p.add_argument('--head-learning-rate', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--warmup-ratio', type=float, default=.06)
    p.add_argument('--max-grad-norm', type=float, default=1.)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--head-dropout', type=float, default=0.)
    p.add_argument('--lora-r', type=int, default=8)
    p.add_argument('--lora-alpha', type=int, default=16)
    p.add_argument('--lora-dropout', type=float, default=.1)
    p.add_argument('--lora-target-modules', nargs='+', default=['query','value'])
    p.add_argument('--precision', choices=['bf16','fp16','fp32'], default='bf16')
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--gradient-checkpointing', action='store_true')
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--no-save-full-encoder', action='store_true', help='Do not save selected full encoder (not recommended for reproducibility)')
    p.add_argument('--data-check-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


def load_prior_lora_module():
    if not OLD_LORA.is_file():
        raise FileNotFoundError(f'Prior LoRA source required: {OLD_LORA}')
    spec = importlib.util.spec_from_file_location('npclass_lora_v2_for_e7_kegg', OLD_LORA)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_task(args):
    comp = pd.read_csv(args.compound_file, sep='\t', low_memory=False)
    labels = pd.read_csv(args.label_file, sep='\t', low_memory=False)
    col = 'internal_compound_id'
    required = {col,'canonical_smiles','inchikey','murcko_scaffold'}
    if not required.issubset(comp):
        raise ValueError(f'Missing compound columns {required-set(comp)}')
    columns = sorted((c for c in labels if c.startswith('label_')), key=lambda x: int(x.split('_')[1]))
    if len(columns)!=11:
        raise ValueError(f'Expected 11 KEGG labels; found {columns}')
    expected_prefix = [f'label_{i}_' for i in range(11)]
    if not all(c.startswith(prefix) for c,prefix in zip(columns,expected_prefix)):
        raise ValueError('KEGG labels are not in label_0..label_10 numeric order')
    comp[col] = comp[col].astype(str)
    labels[col] = labels[col].astype(str)
    if comp[col].duplicated().any() or labels[col].duplicated().any():
        raise ValueError('Duplicate KEGG compound identifiers')
    task = comp.merge(labels[[col]+columns], on=col, how='inner', validate='one_to_one', sort=False)
    if len(task)!=len(comp) or len(task)!=len(labels):
        raise ValueError('Compound/label join does not retain all rows')
    if task[col].duplicated().any() or task['inchikey'].duplicated().any():
        raise ValueError('Duplicate compound or InChIKey')
    if task['canonical_smiles'].isna().any():
        raise ValueError('Missing canonical SMILES')
    y = task[columns].to_numpy(dtype=np.float32)
    if not np.all((y==0)|(y==1)) or (y.sum(axis=1)==0).any():
        raise ValueError('Expected multilabel binary matrix with >=1 positive per compound')
    return task,y,columns


def load_saved_split(task,y,columns,mode,seed,args):
    path = Path(args.split_dir)/f'kegg_pathway_{mode}_split_seed{seed}.tsv'
    if not path.is_file():
        raise FileNotFoundError(f'REQUIRED original split missing: {path}')
    split = pd.read_csv(path,sep='\t',low_memory=False)
    col='internal_compound_id'
    if not {col,'split'}.issubset(split):
        raise ValueError(f'{path} missing columns')
    split[col] = split[col].astype(str)
    if split[col].duplicated().any() or len(split)!=len(task):
        raise ValueError(f'{path}: size or duplicate mismatch')
    values = task[col].map(split.set_index(col)['split'])
    if values.isna().any() or set(values.unique())!={'train','validation','test'}:
        raise ValueError(f'{path}: split IDs or partitions mismatch')
    ix = {name: np.flatnonzero(values.eq(name).to_numpy()) for name in ('train','validation','test')}
    assert sum(map(len,ix.values()))==len(task)
    for name,idx in ix.items():
        positives = y[idx].sum(axis=0)
        if ((positives<=0)|(positives>=len(idx))).any():
            raise ValueError(f'{path}: class positive/negative support insufficient for {name}')
    if mode=='scaffold':
        groups={name:set(task.iloc[idx]['murcko_scaffold'].fillna('').astype(str))-set(['','nan']) for name,idx in ix.items()}
        if any(groups[a]&groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
            raise ValueError(f'{path}: scaffold leakage detected')
    qc_path=Path(args.frozen_split_qc)
    if not qc_path.is_file():
        raise FileNotFoundError(f'REQUIRED frozen split QC missing: {qc_path}')
    qc = pd.read_csv(qc_path,sep='\t')
    for name,idx in ix.items():
        row=qc.loc[(qc['split_mode']==mode)&(qc['split_seed']==seed)&(qc['split']==name)]
        if len(row)!=1 or int(row.iloc[0]['n_compounds'])!=len(idx):
            raise ValueError(f'Frozen split QC mismatch: {mode} {seed} {name}')
        counts=y[idx].sum(axis=0).astype(int)
        for lab,n in zip(columns,counts):
            key=f'positive_count__{lab}'
            if key not in qc or int(row.iloc[0][key])!=int(n):
                raise ValueError(f'Frozen split label QC mismatch: {mode} {seed} {name} {lab}')
    return ix,path


class KeggDataset(Dataset):
    def __init__(self, df, y):
        self.smiles=df['canonical_smiles'].astype(str).tolist()
        self.ids=df['internal_compound_id'].astype(str).tolist()
        self.labels=np.asarray(y, dtype=np.float32)
    def __len__(self): return len(self.labels)
    def __getitem__(self,i): return {'smiles':self.smiles[i],'id':self.ids[i],'labels':self.labels[i]}


class Collate:
    def __init__(self,tok,max_length,precision):
        self.tok=tok;self.max_length=max_length
        self.pad=DataCollatorWithPadding(tok,padding=True,pad_to_multiple_of=8 if precision!='fp32' else None,return_tensors='pt')
    def __call__(self,examples):
        enc=[self.tok(x['smiles'],truncation=True,max_length=self.max_length,padding=False,return_attention_mask=True) for x in examples]
        out=self.pad(enc)
        out['labels']=torch.tensor(np.stack([x['labels'] for x in examples]),dtype=torch.float32)
        out['ids']=[x['id'] for x in examples]
        return out


def evaluate(model, loader, device, precision, old):
    model.eval(); ys=[]; probs=[]; ids=[]; total=0.; n=0
    with torch.no_grad():
        for batch in loader:
            y=batch['labels'].to(device)
            inputs=old.move_batch_to_device({k:v for k,v in batch.items() if k!='ids'},device)
            with old.autocast_context(precision):
                logits=model(**inputs).float()
                loss=nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='sum')
            ys.append(y.cpu().numpy());probs.append(torch.sigmoid(logits).cpu().numpy()); ids.extend(batch['ids'])
            total+=loss.item();n+=y.numel()
    y=np.concatenate(ys);p=np.concatenate(probs)
    auc=[roc_auc_score(y[:,j],p[:,j]) for j in range(y.shape[1])]
    auprc=[average_precision_score(y[:,j],p[:,j]) for j in range(y.shape[1])]
    return dict(auroc=float(np.mean(auc)),auprc=float(np.mean(auprc)),unweighted_loss=total/n),y,p,np.asarray(ids)


def train_run(task,y,labels,ix,split_path,mode,seed,adaptation,args,old,tokenizer):
    old.set_seed(seed)
    device=torch.device('cuda')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if args.precision=='bf16' and not torch.cuda.is_bf16_supported():raise RuntimeError('GPU lacks BF16')
    if adaptation=='lora':
        encoder,hidden,matched=old.configure_encoder(args.model_name_or_path,'lora',args)
    else:
        encoder=AutoModel.from_pretrained(args.model_name_or_path,trust_remote_code=args.trust_remote_code,local_files_only=args.local_files_only)
        hidden=old.infer_hidden_size(encoder);matched=[]
        if hasattr(encoder.config,'use_cache'):encoder.config.use_cache=False
    model=old.MeanPooledMulticlassClassifier(encoder,hidden,len(labels),args.head_dropout).to(device)
    ntotal,ntrain=old.count_parameters(model)
    if ntrain==0:raise RuntimeError('No trainable parameters')
    ds={name:KeggDataset(task.iloc[idx],y[idx]) for name,idx in ix.items()}
    collate=Collate(tokenizer,args.max_length,args.precision)
    def dl(which):
        g=torch.Generator().manual_seed(seed)
        return DataLoader(ds[which],batch_size=args.train_batch_size if which=='train' else args.eval_batch_size,shuffle=which=='train',generator=g if which=='train' else None,collate_fn=collate,num_workers=args.num_workers,pin_memory=True,persistent_workers=args.num_workers>0)
    loaders={name:dl(name) for name in ds}
    pos=y[ix['train']].sum(axis=0).astype(float)
    neg=len(ix['train'])-pos
    wpos=torch.tensor(len(ix['train'])/(2*pos),dtype=torch.float32,device=device)
    wneg=torch.tensor(len(ix['train'])/(2*neg),dtype=torch.float32,device=device)
    # Equivalent to sklearn class_weight='balanced' for each of 11 labels:
    # w_positive = N/(2*n_positive), w_negative = N/(2*n_negative).
    def train_loss(logits,truth):
        raw=nn.functional.binary_cross_entropy_with_logits(logits.float(),truth,reduction='none')
        wt=truth*wpos+(1-truth)*wneg
        return (raw*wt).mean()
    groups=[{'params':list(model.classifier.parameters()),'lr':args.head_learning_rate,'weight_decay':args.weight_decay}]
    encoder_params=[v for v in model.encoder.parameters() if v.requires_grad]
    if not encoder_params:raise RuntimeError('No trainable encoder params')
    groups.insert(0,{'params':encoder_params,'lr':args.lora_learning_rate if adaptation=='lora' else args.full_learning_rate,'weight_decay':0. if adaptation=='lora' else args.weight_decay})
    optimizer=torch.optim.AdamW(groups)
    steps_per_epoch=math.ceil(len(loaders['train'])/args.gradient_accumulation_steps)
    nsteps=steps_per_epoch*args.epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=round(args.warmup_ratio*nsteps),num_training_steps=nsteps)
    scaler=torch.amp.GradScaler('cuda',enabled=args.precision=='fp16')
    best_auc=-np.inf;best_pr=-np.inf;best_epoch=None;best_state=None;history=[]
    early_stop=MeaningfulEarlyStopper(args.early_stopping_patience,args.min_delta,args.min_epochs)
    start=time.monotonic();peak=0
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,args.epochs+1):
        t0=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True);running=0.;ns=0
        for step,batch in enumerate(loaders['train'],1):
            yy=batch['labels'].to(device)
            inputs=old.move_batch_to_device({k:v for k,v in batch.items() if k!='ids'},device)
            with old.autocast_context(args.precision):
                logits=model(**inputs);loss=train_loss(logits,yy)
            # Normalize accumulated gradients correctly for a short final group.
            group_start=((step-1)//args.gradient_accumulation_steps)*args.gradient_accumulation_steps
            group_size=min(args.gradient_accumulation_steps,len(loaders['train'])-group_start)
            if scaler.is_enabled():scaler.scale(loss/group_size).backward()
            else:(loss/group_size).backward()
            if step%args.gradient_accumulation_steps==0 or step==len(loaders['train']):
                if scaler.is_enabled():scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],args.max_grad_norm)
                if scaler.is_enabled():scaler.step(optimizer);scaler.update()
                else:optimizer.step()
                scheduler.step();optimizer.zero_grad(set_to_none=True)
            running+=float(loss.detach().item())*len(yy);ns+=len(yy)
        val,_,_,_=evaluate(model,loaders['validation'],device,args.precision,old)
        record={'epoch':epoch,'train_balanced_loss':running/ns,'validation_unweighted_loss':val['unweighted_loss'],'validation_macro_auroc':val['auroc'],'validation_macro_auprc':val['auprc'],'seconds':time.monotonic()-t0}
        if val['auroc']>best_auc+1e-12 or (abs(val['auroc']-best_auc)<=1e-12 and val['auprc']>best_pr+1e-12):
            best_auc=val['auroc'];best_pr=val['auprc'];best_epoch=epoch
            best_state=old.snapshot_trainable_parameters(model)
        stop,meaningful=early_stop.step(epoch,val['auroc'])
        record.update(early_stop_stale_epochs=early_stop.stale_epochs,
                      meaningful_auroc_improvement=meaningful)
        history.append(record)
        print(args.feature_name,adaptation,mode,seed,'epoch',epoch,'val_auc',round(val['auroc'],6),
              'val_auprc',round(val['auprc'],6),'stale',early_stop.stale_epochs,flush=True)
        if stop:
            print('EARLY STOP',args.feature_name,adaptation,mode,seed,'at',epoch,
                  'best_epoch',best_epoch,flush=True)
            break
    assert best_state is not None
    old.restore_trainable_parameters(model,best_state)
    val,vy,vp,vid=evaluate(model,loaders['validation'],device,args.precision,old)
    test,ty,tp,tid=evaluate(model,loaders['test'],device,args.precision,old)
    elapsed=time.monotonic()-start
    peak=torch.cuda.max_memory_allocated(device)
    run=Path(args.output_root)/args.feature_name/adaptation/mode/f'seed{seed}'
    run.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(history).to_csv(run/'epoch_history.tsv',sep='\t',index=False)
    meta={'script_version':VERSION,'feature':args.feature_name,'adaptation_mode':adaptation,'split_mode':mode,'split_seed':seed,'split_file':str(split_path),'task':'kegg_pathway_11label','label_columns':labels,'selected_epoch':best_epoch,'epochs_requested':args.epochs,'epochs_trained':len(history),'stopped_early':early_stop.stopped,'early_stopping_patience':args.early_stopping_patience,'early_stopping_min_delta':args.min_delta,'early_stopping_min_epochs':args.min_epochs,'checkpoint_selection_metric':'validation_macro_auroc_then_macro_auprc','best_validation_macro_auroc':best_auc,'best_validation_macro_auprc':best_pr,'class_weighting':'per-label balanced positive and negative BCE','training_label_positive_counts':pos.astype(int).tolist(),'class_positive_weights':wpos.cpu().numpy().tolist(),'class_negative_weights':wneg.cpu().numpy().tolist(),'n_train':len(ix['train']),'n_validation':len(ix['validation']),'n_test':len(ix['test']),'total_parameters':ntotal,'trainable_parameters':ntrain,'matched_lora_modules':matched,'training_seconds':elapsed,'training_gpu_hours_single_gpu':elapsed/3600,'peak_allocated_gpu_memory_bytes':peak,'args':vars(args)}
    (run/'run_metadata.json').write_text(json.dumps(meta,indent=2,default=str))
    np.savez_compressed(run/'predictions.npz',validation_ids=vid,validation_y=vy,validation_probability=vp,test_ids=tid,test_y=ty,test_probability=tp,label_columns=np.asarray(labels))
    bm=run/'best_model';bm.mkdir(exist_ok=True)
    if adaptation=='lora':model.encoder.save_pretrained(bm/'adapter')
    elif not args.no_save_full_encoder:torch.save(model.encoder.state_dict(),bm/'encoder_state.pt')
    torch.save(model.classifier.state_dict(),bm/'classifier_head.pt')
    tokenizer.save_pretrained(bm/'tokenizer')
    row={'feature':args.feature_name,'adaptation_mode':adaptation,'split_mode':mode,'split_seed':seed,'selected_epoch':best_epoch,'n_train':len(ix['train']),'n_test':len(ix['test']),'test_macro_auroc':test['auroc'],'test_macro_auprc':test['auprc'],'test_unweighted_loss':test['unweighted_loss'],'validation_macro_auroc':val['auroc'],'validation_macro_auprc':val['auprc'],'training_seconds':elapsed,'trainable_parameters':ntrain,'run_dir':str(run)}
    print('TEST',json.dumps(row),flush=True)
    del model,encoder,optimizer,scheduler,best_state
    torch.cuda.empty_cache()
    return row


def main():
    args=parse_args()
    if args.epochs<=0 or args.train_batch_size<=0 or args.gradient_accumulation_steps<=0:raise ValueError('Invalid training budget')
    if args.min_epochs>args.epochs:raise ValueError('--min-epochs cannot exceed --epochs')
    MeaningfulEarlyStopper(args.early_stopping_patience,args.min_delta,args.min_epochs)
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    task,y,labels=load_task(args)
    print('KEGG compounds=',len(task),'labels=',len(labels),'script=',VERSION,flush=True)
    experiments=[]
    modes=['random','scaffold'] if args.split_mode=='both' else [args.split_mode]
    for mode in modes:
        for seed in args.split_seeds:
            indices,path=load_saved_split(task,y,labels,mode,seed,args)
            print('SPLIT-QC PASS',mode,seed,{k:len(v) for k,v in indices.items()},flush=True)
            experiments.append((mode,seed,indices,path))
    if args.data_check_only:
        print('PASS: all original splits match frozen benchmark; no model loaded')
        return
    old=load_prior_lora_module()
    tokenizer=AutoTokenizer.from_pretrained(args.model_name_or_path,trust_remote_code=args.trust_remote_code,local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        token=tokenizer.eos_token or tokenizer.sep_token or tokenizer.cls_token
        if token is None:raise ValueError('No tokenizer padding fallback')
        tokenizer.pad_token=token
    feature=Path(args.output_root)/args.feature_name
    feature.mkdir(parents=True,exist_ok=True)
    output=feature/'e7_kegg_transformer_repeated_runs.tsv'
    all_rows=[]
    if output.exists():
        if not args.resume:raise FileExistsError(f'{output} exists; use --resume')
        all_rows=pd.read_csv(output,sep='\t').to_dict('records')
    for mode,seed,ix,path in experiments:
        for adapt in args.adaptation_modes:
            exists=any(r['adaptation_mode']==adapt and r['split_mode']==mode and int(r['split_seed'])==seed for r in all_rows)
            if exists:
                if not args.resume:raise FileExistsError('Existing result, refusing overwrite')
                run_dir=feature/adapt/mode/f'seed{seed}'
                meta_file=run_dir/'run_metadata.json'
                if not meta_file.is_file():
                    raise FileNotFoundError(f'Existing TSV row without run metadata: {meta_file}')
                metadata=json.loads(meta_file.read_text())
                required={'script_version':VERSION,'epochs_requested':args.epochs,
                          'early_stopping_patience':args.early_stopping_patience,
                          'early_stopping_min_delta':args.min_delta,
                          'early_stopping_min_epochs':args.min_epochs}
                for k,v in required.items():
                    if metadata.get(k)!=v:
                        raise ValueError(f'Cannot resume incompatible run {run_dir}: {k} differs')
                previous_args=metadata.get('args',{})
                for k in ('head_learning_rate','full_learning_rate','lora_learning_rate',
                          'train_batch_size','gradient_accumulation_steps','warmup_ratio'):
                    if previous_args.get(k)!=getattr(args,k):
                        raise ValueError(f'Cannot resume incompatible run {run_dir}: {k} differs')
                print('RESUME: skip',args.feature_name,adapt,mode,seed,flush=True);continue
            row=train_run(task,y,labels,ix,path,mode,seed,adapt,args,old,tokenizer)
            all_rows.append(row)
            pd.DataFrame(all_rows).sort_values(['adaptation_mode','split_mode','split_seed']).to_csv(output,sep='\t',index=False)
    print('RESULTS',output,flush=True)
    results=pd.DataFrame(all_rows)
    if len(results):
        summary=results.groupby(['feature','adaptation_mode','split_mode'])['test_macro_auroc'].agg(['mean','std','count']).reset_index()
        summary.to_csv(feature/'e7_kegg_transformer_summary.tsv',sep='\t',index=False)
        print(summary.to_string(index=False))


if __name__=='__main__':main()

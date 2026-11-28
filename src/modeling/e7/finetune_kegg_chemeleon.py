#!/usr/bin/env python3
"""E7 KEGG CheMeleon full fine-tuning with exact per-label macro-AUROC selection.

Reuses Chemprop v2.3.1's *native* CheMeleon message-passing foundation
initialization (BondMessagePassing from ~/.chemprop/chemeleon_mp.pt), the
Chemprop molecule graph featurizer and mean aggregation. Uses a new 11-label
linear predictor trained with per-label balanced binary cross-entropy.

The Chemprop CLI built-in `roc` metric pools binary labels, and its
`--weight-column` is per molecule, not per label. This explicit PyTorch loop
is required to match the 11-task frozen KEGG macro AUROC and loss weighting.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from kegg_ft_common import (
    VERSION, MeaningfulEarlyStopper, add_data_arguments, balanced_bce, balanced_weights, experiments,
    is_better, macro_scores, output_dir, run_is_complete, save_run,
    seed_all, summarize,
)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    add_data_arguments(p)
    p.add_argument('--checkpoint',default=str(Path.home()/'.chemprop/chemeleon_mp.pt'),
                   help='Chemprop from-foundation chemeleon cached message-passing weights')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--early-stopping-patience',type=int,default=3)
    p.add_argument('--min-delta',type=float,default=0.001)
    p.add_argument('--min-epochs',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--encoder-lr',type=float,default=2e-5)
    p.add_argument('--head-lr',type=float,default=1e-3,
                   help='Validated head LR from KEGG seed-2026 20-epoch pilot')
    p.add_argument('--weight-decay',type=float,default=0.0)
    p.add_argument('--max-grad-norm',type=float,default=1.)
    p.add_argument('--num-workers',type=int,default=0)
    return p.parse_args()


class CheMeleonKeggClassifier(nn.Module):
    def __init__(self,checkpoint,n_labels):
        super().__init__()
        from chemprop import nn as chemnn
        path=Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'CheMeleon message-passing checkpoint not found: {path}. '
                'Find the cached checkpoint with: find /root/.chemprop /private/vina/model_repos -name chemeleon_mp.pt')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if not isinstance(payload,dict) or not {'hyper_parameters','state_dict'}.issubset(payload):
            raise ValueError(f'{path} is not a Chemprop CheMeleon message passing payload')
        # This is the same pretrained encoder construction used by
        # Chemprop 2.3.1 when `--from-foundation chemeleon` is supplied.
        self.encoder=chemnn.BondMessagePassing(**payload['hyper_parameters'])
        self.encoder.load_state_dict(payload['state_dict'],strict=True)
        if self.encoder.output_dim != 2048:
            raise ValueError(f'Unexpected CheMeleon encoder dimension: {self.encoder.output_dim}, expected 2048')
        self.aggregation=chemnn.MeanAggregation()
        self.head=nn.Linear(self.encoder.output_dim,n_labels)

    def forward(self,bmg):
        node=self.encoder(bmg)
        pooled=self.aggregation(node,bmg.batch)
        if pooled.ndim!=2 or pooled.shape[1]!=self.head.in_features:
            raise ValueError(f'CheMeleon pooled feature shape {tuple(pooled.shape)}')
        return self.head(pooled)


def build_loader(task,y,ids,batch_size,shuffle,seed,num_workers):
    from chemprop import data
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    datapoints=[]
    for smi,target in zip(task.iloc[ids]['canonical_smiles'],y[ids]):
        datapoint=data.MoleculeDatapoint.from_smi(str(smi),y=target.astype(np.float32))
        if datapoint.mol is None:
            raise ValueError(f'Chemprop invalid KEGG SMILES: {smi!r}')
        datapoints.append(datapoint)
    dataset=data.MoleculeDataset(datapoints,featurizer=SimpleMoleculeMolGraphFeaturizer())
    dataset.cache=True
    return data.build_dataloader(dataset,batch_size=batch_size,num_workers=num_workers,
                                 shuffle=shuffle,seed=seed if shuffle else None)


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();ys=[];ps=[];total=0.;n=0
    for bmg,v_d,x_d,target,*_ in loader:
        bmg.to(device)
        yy=target.to(device).float()
        logits=model(bmg).float()
        if logits.shape != yy.shape:
            raise ValueError(f'CheMeleon logits {tuple(logits.shape)} != labels {tuple(yy.shape)}')
        total+=nn.functional.binary_cross_entropy_with_logits(logits,yy,reduction='sum').item()
        n+=yy.numel();ys.append(yy.cpu().numpy());ps.append(logits.sigmoid().cpu().numpy())
    y=np.concatenate(ys);p=np.concatenate(ps)
    metrics=macro_scores(y,p)
    metrics['unweighted_loss']=float(total/n)
    return metrics,y,p


def train_one(task,y,labels,mode,seed,ix,split_path,args):
    seed_all(seed)
    device=torch.device('cuda')
    out=output_dir(args,'chemeleon',mode,seed)
    if run_is_complete(out,args):return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Existing incomplete run at {out}; inspect or move it before retry')
    tr_loader=build_loader(task,y,ix['train'],args.batch_size,True,seed,args.num_workers)
    val_loader=build_loader(task,y,ix['validation'],args.batch_size,False,seed,args.num_workers)
    test_loader=build_loader(task,y,ix['test'],args.batch_size,False,seed,args.num_workers)
    model=CheMeleonKeggClassifier(args.checkpoint,len(labels)).to(device)
    encoder_params=list(model.encoder.parameters())
    if not encoder_params or not any(p.requires_grad for p in encoder_params):
        raise RuntimeError('CheMeleon pretrained encoder unexpectedly frozen')
    optimizer=torch.optim.AdamW([
        {'params':encoder_params,'lr':args.encoder_lr},
        {'params':list(model.head.parameters()),'lr':args.head_lr},
    ],weight_decay=args.weight_decay)
    wpos,wneg,pos=balanced_weights(y[ix['train']],device)
    total_params=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('CheMeleon',mode,seed,'train=',len(ix['train']),
          'weights_positive=',wpos.cpu().tolist(),'trainable_parameters=',trainable,flush=True)
    best_auc=-np.inf;best_pr=-np.inf;best_epoch=None;best_state=None;history=[]
    early_stop=MeaningfulEarlyStopper(args.early_stopping_patience,args.min_delta,args.min_epochs)
    start=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,args.epochs+1):
        t0=time.monotonic();model.train();running=0.;n=0
        for step,(bmg,v_d,x_d,target,*_) in enumerate(tr_loader):
            bmg.to(device);yy=target.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            logits=model(bmg)
            if logits.shape!=yy.shape:raise ValueError('CheMeleon training output shape differs from labels')
            loss=balanced_bce(logits,yy,wpos,wneg)
            loss.backward()
            if epoch==1 and step==0:
                grads=[p.grad for p in encoder_params if p.grad is not None]
                if not grads or sum(float(g.detach().abs().sum()) for g in grads)<=0:
                    raise RuntimeError('CheMeleon encoder gradient check failed')
                print('ENCODER-GRADIENT PASS',flush=True)
            nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)
            optimizer.step()
            running+=float(loss.item())*len(yy);n+=len(yy)
        val,_,_=evaluate(model,val_loader,device)
        if is_better(val['macro_auroc'],val['macro_auprc'],best_auc,best_pr):
            best_auc=val['macro_auroc'];best_pr=val['macro_auprc'];best_epoch=epoch
            best_state={name:p.detach().cpu().clone() for name,p in model.state_dict().items()}
        stop,meaningful=early_stop.step(epoch,val['macro_auroc'])
        history.append({'epoch':epoch,'train_balanced_loss':running/n,
              'validation_unweighted_loss':val['unweighted_loss'],
              'validation_macro_auroc':val['macro_auroc'],
              'validation_macro_auprc':val['macro_auprc'],
              'early_stop_stale_epochs':early_stop.stale_epochs,
              'meaningful_auroc_improvement':meaningful,
              'seconds':time.monotonic()-t0})
        print('chemeleon full',mode,seed,'epoch',epoch,'val_auc',round(val['macro_auroc'],6),
              'val_auprc',round(val['macro_auprc'],6),
              'stale',early_stop.stale_epochs,flush=True)
        if stop:
            print('EARLY STOP chemeleon',mode,seed,'at',epoch,'best_epoch',best_epoch,flush=True)
            break
    assert best_state is not None
    model.load_state_dict(best_state,strict=True)
    validation,vy,vp=evaluate(model,val_loader,device)
    test,ty,tp=evaluate(model,test_loader,device)
    elapsed=time.monotonic()-start
    meta={'script_version':VERSION,'feature':'chemeleon','adaptation_mode':'full','task':'kegg_pathway_11label',
          'split_mode':mode,'split_seed':seed,'split_file':str(split_path),'label_columns':labels,
          'selected_epoch':best_epoch,'selected_by':'validation_macro_auroc_then_macro_auprc',
           'epochs_requested':args.epochs,'epochs_trained':len(history),
          'stopped_early':early_stop.stopped,
          'early_stopping_patience':args.early_stopping_patience,
          'early_stopping_min_delta':args.min_delta,
          'early_stopping_min_epochs':args.min_epochs,
          'n_train':len(ix['train']),'n_validation':len(ix['validation']),
          'n_test':len(ix['test']),'training_positive_counts':pos.tolist(),
          'class_positive_weights':wpos.cpu().tolist(),'class_negative_weights':wneg.cpu().tolist(),
          'training_loss':'per-label balanced binary cross-entropy',
          'validation_macro_auroc':validation['macro_auroc'],'validation_macro_auprc':validation['macro_auprc'],
          'test_macro_auroc':test['macro_auroc'],'test_macro_auprc':test['macro_auprc'],
          'validation_unweighted_loss':validation['unweighted_loss'],'test_unweighted_loss':test['unweighted_loss'],
          'training_seconds':elapsed,'total_parameters':total_params,'trainable_parameters':trainable,
          'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(device),'args':vars(args)}
    save_run(out,args,meta,history,vy,vp,ty,tp,labels,model)
    print('TEST',json.dumps({k:meta[k] for k in ['feature','split_mode','split_seed','selected_epoch',
          'test_macro_auroc','test_macro_auprc','training_seconds','trainable_parameters']}),flush=True)


def main():
    args=parse_args()
    if args.epochs<1 or args.batch_size<1 or args.num_workers<0:raise ValueError('Invalid training configuration')
    if args.min_epochs>args.epochs:raise ValueError('--min-epochs cannot exceed --epochs')
    MeaningfulEarlyStopper(args.early_stopping_patience,args.min_delta,args.min_epochs)
    task,y,labels,conditions=experiments(args)
    if args.data_check_only:
        print('PASS: KEGG original split and frozen QC; no model loaded',flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    for mode,seed,ix,path in conditions:
        train_one(task,y,labels,mode,seed,ix,path,args)
        summarize(args,'chemeleon')

if __name__=='__main__':main()

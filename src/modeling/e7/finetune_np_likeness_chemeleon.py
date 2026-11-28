#!/usr/bin/env python3
"""E7 original NP-likeness: CheMeleon full FT with 20-epoch / patience-3.

Matches the KEGG E7 CheMeleon implementation: strict pretrained Chemprop
2.3.1 BondMessagePassing loading, mean aggregation, 1-logit linear head,
per-class balanced BCE, AUROC checkpoint selection with AUPRC tie-breaker.
A lazy graph dataset avoids building >400k RDKit graphs at once.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader

from np_likeness_ft_common import (
    add_data_args, add_stopping_args, check_args, load_task, load_split,
    stratified_smoke_indices, seed_all, run_path, run_config, check_run,
    save_run, write_summary, final_metadata, scores, is_better, EarlyStop,
    balanced_binary_weights, balanced_binary_loss, threshold_from_validation, sha256,
)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    add_data_args(p);add_stopping_args(p)
    p.add_argument('--checkpoint',default=str(Path.home()/'.chemprop/chemeleon_mp.pt'))
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--eval-batch-size',type=int,default=64)
    p.add_argument('--encoder-lr',type=float,default=2e-5)
    p.add_argument('--head-lr',type=float,default=1e-3,
                   help='Head LR selected in the controlled E7 KEGG pilot')
    p.add_argument('--weight-decay',type=float,default=0.0)
    p.add_argument('--max-grad-norm',type=float,default=1.0)
    p.add_argument('--num-workers',type=int,default=2)
    return p.parse_args()


class OnTheFlyChempropDataset(Dataset):
    """Create Chemprop 2.3.1 Datum/graph in workers rather than caching all graphs."""
    def __init__(self,smiles,y,indices):
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
        self.smiles=smiles
        self.y=y
        self.indices=np.asarray(indices,dtype=np.int64)
        self.featurizer=SimpleMoleculeMolGraphFeaturizer()
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        from rdkit import Chem
        from chemprop.data.datasets import Datum
        i=int(self.indices[index])
        molecule=Chem.MolFromSmiles(str(self.smiles[i]))
        if molecule is None:
            raise ValueError(f'Invalid Chemprop SMILES at task row {i}: {self.smiles[i]!r}')
        graph=self.featurizer(molecule)
        return Datum(graph,None,None,np.array([float(self.y[i])],dtype=np.float32),1.0,None,None)


class CheMeleonBinary(nn.Module):
    def __init__(self,checkpoint):
        super().__init__()
        from chemprop import nn as chemnn
        path=Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'CheMeleon cached weights missing: {path}')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if not isinstance(payload,dict) or not {'hyper_parameters','state_dict'}.issubset(payload):
            raise ValueError('Expected CheMeleon Chemprop message-passing checkpoint payload')
        self.encoder=chemnn.BondMessagePassing(**payload['hyper_parameters'])
        self.encoder.load_state_dict(payload['state_dict'],strict=True)
        if int(self.encoder.output_dim)!=2048:
            raise RuntimeError(f'Unexpected CheMeleon dimensionality {self.encoder.output_dim}')
        self.aggregation=chemnn.MeanAggregation()
        self.head=nn.Linear(2048,1)
    def forward(self,bmg):
        node=self.encoder(bmg)
        pooled=self.aggregation(node,bmg.batch)
        return self.head(pooled).squeeze(-1)


@torch.no_grad()
def evaluate(model,dl,device):
    model.eval();ys=[];ps=[];total=0.;n=0
    for bmg,v_d,x_d,target,*_ in dl:
        bmg.to(device)
        yy=target.to(device).float().view(-1)
        logits=model(bmg).float()
        if logits.shape!=yy.shape:
            raise ValueError(f'CheMeleon output {tuple(logits.shape)} vs labels {tuple(yy.shape)}')
        total+=float(nn.functional.binary_cross_entropy_with_logits(logits,yy,reduction='sum').item())
        n+=len(yy);ys.append(yy.cpu().numpy());ps.append(logits.sigmoid().cpu().numpy())
    y=np.concatenate(ys);p=np.concatenate(ps)
    metrics=scores(y,p);metrics['unweighted_loss']=total/n
    return metrics,y,p


def train_one(args,smiles,y,mode,seed,ix,path,task_hash,split_hash):
    from chemprop.data.collate import collate_batch
    seed_all(seed)
    device=torch.device('cuda')
    out=run_path(args,'chemeleon','full',mode,seed)
    config=run_config(args,task_sha256=task_hash,split_sha256=split_hash)
    if check_run(out,args,config):return
    def make_loader(part):
        return DataLoader(OnTheFlyChempropDataset(smiles,y,ix[part]),
            batch_size=args.batch_size if part=='train' else args.eval_batch_size,
            shuffle=part=='train',num_workers=args.num_workers,collate_fn=collate_batch,
            persistent_workers=args.num_workers>0,pin_memory=False,
            generator=torch.Generator().manual_seed(seed) if part=='train' else None)
    dls={part:make_loader(part) for part in ('train','validation','test')}
    model=CheMeleonBinary(args.checkpoint).to(device)
    encoder_params=list(model.encoder.parameters())
    if not encoder_params or not any(p.requires_grad for p in encoder_params):
        raise RuntimeError('CheMeleon pretrained encoder unexpectedly frozen')
    optimizer=torch.optim.AdamW([
       {'params':encoder_params,'lr':args.encoder_lr},
       {'params':list(model.head.parameters()),'lr':args.head_lr},
    ],weight_decay=args.weight_decay)
    total=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    wpos,wneg=balanced_binary_weights(y[ix['train']]);wp=torch.tensor(wpos,device=device);wn=torch.tensor(wneg,device=device)
    best_auc=-np.inf;best_pr=-np.inf;best_epoch=0;best_state=None;history=[]
    stopper=EarlyStop(args.early_stopping_patience,args.min_delta,args.min_epochs)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
    print('START CheMeleon',mode,seed,'n_train=',len(ix['train']),
          'weight_pos=',wpos,'trainable=',trainable,flush=True)
    for epoch in range(1,args.epochs+1):
        t0=time.monotonic();model.train();running=0.;count=0
        for step,(bmg,v_d,x_d,target,*_) in enumerate(dls['train']):
            bmg.to(device)
            yy=target.to(device).float().view(-1)
            optimizer.zero_grad(set_to_none=True)
            logits=model(bmg)
            if logits.shape!=yy.shape:raise RuntimeError('CheMeleon logit/label shape mismatch')
            loss=balanced_binary_loss(logits,yy,wp,wn)
            loss.backward()
            if epoch==1 and step==0:
                grads=[p.grad for p in encoder_params if p.grad is not None]
                if not grads or sum(float(g.detach().abs().sum()) for g in grads)<=0:
                    raise RuntimeError('CheMeleon pretrained encoder gradient check failed')
                print('ENCODER-GRADIENT PASS',flush=True)
            nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)
            optimizer.step()
            running+=float(loss.item())*len(yy);count+=len(yy)
        val,_,_=evaluate(model,dls['validation'],device)
        if is_better(val['auroc'],val['auprc'],best_auc,best_pr):
            best_auc=val['auroc'];best_pr=val['auprc'];best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        stop,meaningful=stopper.step(epoch,val['auroc'])
        history.append({'epoch':epoch,'train_balanced_loss':running/count,
           'validation_auroc':val['auroc'],'validation_auprc':val['auprc'],
           'validation_unweighted_loss':val['unweighted_loss'],
           'early_stop_stale_epochs':stopper.stale,
           'meaningful_auroc_improvement':meaningful,
           'seconds':time.monotonic()-t0})
        print('chemeleon full',mode,seed,'epoch',epoch,'val_auc',round(val['auroc'],6),
              'val_auprc',round(val['auprc'],6),'stale',stopper.stale,flush=True)
        if stop:
            print('EARLY STOP CheMeleon',mode,seed,'epoch',epoch,'best',best_epoch,flush=True)
            break
    assert best_state is not None
    model.load_state_dict(best_state,strict=True)
    val,vy,vp=evaluate(model,dls['validation'],device)
    threshold,val_f1=threshold_from_validation(vy,vp)
    test,ty,tp=evaluate(model,dls['test'],device)
    from sklearn.metrics import f1_score
    test_f1=float(f1_score(ty,tp>=threshold,zero_division=0))
    meta=final_metadata(args,'chemeleon','full',mode,seed,path,ix,
         best_epoch,best_auc,best_pr,history,stopper,val,test,threshold,test_f1,
         trainable,total,time.monotonic()-started,device,
         pretrained_checkpoint=str(args.checkpoint),training_loss='balanced_binary_cross_entropy',
         positive_weight=wpos,negative_weight=wneg,
         validation_f1_at_selected_threshold=val_f1)
    save_run(out,config,meta,history,vy,vp,ty,tp,best_state)
    print('TEST',json.dumps({k:meta[k] for k in ('feature','split_mode','split_seed',
        'selected_epoch','validation_auroc','test_auroc','epochs_trained','training_seconds')}),flush=True)
    del model,optimizer,dls,best_state
    torch.cuda.empty_cache()


def main():
    args=parse_args();check_args(args)
    if args.batch_size<1 or args.eval_batch_size<1 or args.num_workers<0:raise ValueError('Invalid loader options')
    task=load_task(args)
    y=task['is_np'].to_numpy(dtype=np.int8)
    modes=('random','scaffold') if args.split_mode=='both' else (args.split_mode,)
    if args.data_check_only:
        for mode in modes:
            for seed in args.split_seeds:load_split(task,args,mode,seed)
        print('PASS original NP-likeness splits; no model loaded',flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if not Path(args.checkpoint).expanduser().is_file():raise FileNotFoundError(args.checkpoint)
    smiles=task['canonical_smiles'].tolist();task_hash=sha256(args.task_file)
    for mode in modes:
        for seed in args.split_seeds:
            ix,path=load_split(task,args,mode,seed)
            ix=stratified_smoke_indices(ix,y,args.smoke_limit,seed)
            train_one(args,smiles,y,mode,seed,ix,path,task_hash,sha256(path))
            write_summary(args.output_root,'chemeleon')
    print('PASS CheMeleon',flush=True)

if __name__=='__main__':main()

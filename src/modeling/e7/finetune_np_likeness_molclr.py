#!/usr/bin/env python3
"""E7 original NP-likeness: MolCLR-GIN full fine-tuning, balanced binary BCE.

Same pretrained GIN architecture and projection-head exclusion as E7 KEGG.
Graphs are constructed on demand to avoid holding ~411k PyG objects in RAM.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from models.ginet_molclr import GINet
from src.modeling.extract_molclr_embeddings import smiles_to_data

from np_likeness_ft_common import (
    add_data_args, add_stopping_args, check_args, load_task, load_split,
    stratified_smoke_indices, seed_all, run_path, run_config, check_run, save_run,
    write_summary, final_metadata, scores, is_better, EarlyStop,
    balanced_binary_weights, balanced_binary_loss, threshold_from_validation, sha256,
)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    add_data_args(p);add_stopping_args(p)
    p.add_argument('--checkpoint',default='/private/vina/model_repos/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--eval-batch-size',type=int,default=128)
    p.add_argument('--encoder-lr',type=float,default=2e-5)
    p.add_argument('--head-lr',type=float,default=1e-3)
    p.add_argument('--weight-decay',type=float,default=0.01)
    p.add_argument('--max-grad-norm',type=float,default=1.0)
    p.add_argument('--num-workers',type=int,default=2)
    return p.parse_args()


class GraphDataset(Dataset):
    def __init__(self,smiles,y,ids):
        self.smiles=smiles
        self.y=y
        self.ids=np.asarray(ids,dtype=np.int64)
    def __len__(self):return len(self.ids)
    def __getitem__(self,index):
        i=int(self.ids[index])
        graph=smiles_to_data(str(self.smiles[i]))
        if graph is None:
            raise ValueError(f'Invalid MolCLR SMILES at original task row {i}: {self.smiles[i]!r}')
        graph.y=torch.tensor([float(self.y[i])],dtype=torch.float32)
        return graph


class MolCLRBinary(nn.Module):
    def __init__(self,checkpoint):
        super().__init__()
        self.encoder=GINet(num_layer=5,emb_dim=300,feat_dim=512,drop_ratio=0.0,pool='mean')
        self.encoder.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        for p in self.encoder.out_lin.parameters():
            p.requires_grad_(False)
        self.head=nn.Linear(512,1)
    def forward(self,graph):
        embeddings,_=self.encoder(graph)
        if embeddings.ndim!=2 or embeddings.shape[1]!=512:
            raise RuntimeError(f'MolCLR embedding shape {tuple(embeddings.shape)}')
        return self.head(embeddings).squeeze(1)


@torch.no_grad()
def evaluate(model,dl,device):
    model.eval();ys=[];ps=[];total=0.;n=0
    for b in dl:
        b=b.to(device)
        yy=b.y.float()
        logits=model(b).float()
        if logits.shape!=yy.shape:
            raise ValueError(f'Logits {tuple(logits.shape)} != labels {tuple(yy.shape)}')
        total+=float(nn.functional.binary_cross_entropy_with_logits(logits,yy,reduction='sum').item())
        n+=len(yy);ys.append(yy.cpu().numpy());ps.append(logits.sigmoid().cpu().numpy())
    y=np.concatenate(ys);p=np.concatenate(ps)
    metrics=scores(y,p);metrics['unweighted_loss']=total/n
    return metrics,y,p


def train_one(args,smiles,y,mode,seed,ix,path,task_hash,split_hash):
    seed_all(seed)
    device=torch.device('cuda')
    out=run_path(args,'molclr_gin','full',mode,seed)
    config=run_config(args,task_sha256=task_hash,split_sha256=split_hash)
    if check_run(out,args,config):return
    def dataloader(part):
        return DataLoader(GraphDataset(smiles,y,ix[part]),
            batch_size=args.batch_size if part=='train' else args.eval_batch_size,
            shuffle=(part=='train'),num_workers=args.num_workers,
            pin_memory=True,persistent_workers=args.num_workers>0,
            generator=torch.Generator().manual_seed(seed) if part=='train' else None)
    dls={part:dataloader(part) for part in ('train','validation','test')}
    model=MolCLRBinary(args.checkpoint).to(device)
    encoder_params=[p for p in model.encoder.parameters() if p.requires_grad]
    total=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer=torch.optim.AdamW([
        {'params':encoder_params,'lr':args.encoder_lr},
        {'params':list(model.head.parameters()),'lr':args.head_lr}
    ],weight_decay=args.weight_decay)
    wpos,wneg=balanced_binary_weights(y[ix['train']]);wp=torch.tensor(wpos,device=device);wn=torch.tensor(wneg,device=device)
    best_auc=-np.inf;best_pr=-np.inf;best_epoch=0;best_state=None;history=[]
    stopper=EarlyStop(args.early_stopping_patience,args.min_delta,args.min_epochs)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
    print('START MolCLR-GIN',mode,seed,'n=',len(ix['train']),'weight_pos=',wpos,'trainable=',trainable,flush=True)
    for epoch in range(1,args.epochs+1):
        t0=time.monotonic();model.train();running=0.;count=0
        for step,batch in enumerate(dls['train']):
            batch=batch.to(device);optimizer.zero_grad(set_to_none=True)
            yy=batch.y.float();logits=model(batch)
            loss=balanced_binary_loss(logits,yy,wp,wn)
            loss.backward()
            if epoch==1 and step==0:
                grad=model.encoder.feat_lin.weight.grad
                if grad is None or grad.detach().abs().sum().item()<=0:
                    raise RuntimeError('MolCLR pretrained encoder gradient check failed')
                print('ENCODER-GRADIENT PASS',flush=True)
            nn.utils.clip_grad_norm_(encoder_params+list(model.head.parameters()),args.max_grad_norm)
            optimizer.step()
            running+=float(loss.item())*batch.num_graphs;count+=batch.num_graphs
        val,_,_=evaluate(model,dls['validation'],device)
        if is_better(val['auroc'],val['auprc'],best_auc,best_pr):
            best_auc=val['auroc'];best_pr=val['auprc'];best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        stop,meaningful=stopper.step(epoch,val['auroc'])
        history.append({'epoch':epoch,'train_balanced_loss':running/count,
            'validation_auroc':val['auroc'],'validation_auprc':val['auprc'],
            'validation_unweighted_loss':val['unweighted_loss'],
            'early_stop_stale_epochs':stopper.stale,'meaningful_auroc_improvement':meaningful,
            'seconds':time.monotonic()-t0})
        print('molclr_gin full',mode,seed,'epoch',epoch,'val_auc',round(val['auroc'],6),
              'val_auprc',round(val['auprc'],6),'stale',stopper.stale,flush=True)
        if stop:
            print('EARLY STOP MolCLR-GIN',mode,seed,'at',epoch,'best',best_epoch,flush=True)
            break
    model.load_state_dict(best_state,strict=True)
    val,vy,vp=evaluate(model,dls['validation'],device)
    threshold,val_f1=threshold_from_validation(vy,vp)
    test,ty,tp=evaluate(model,dls['test'],device)
    from sklearn.metrics import f1_score
    test_f1=float(f1_score(ty,tp>=threshold,zero_division=0))
    meta=final_metadata(args,'molclr_gin','full',mode,seed,path,ix,
        best_epoch,best_auc,best_pr,history,stopper,val,test,threshold,test_f1,
        trainable,total,time.monotonic()-started,device,
        pretrained_checkpoint=str(args.checkpoint),training_loss='balanced_binary_cross_entropy',
        positive_weight=wpos,negative_weight=wneg,
        validation_f1_at_selected_threshold=val_f1)
    save_run(out,config,meta,history,vy,vp,ty,tp,best_state)
    print('TEST',json.dumps({k:meta[k] for k in ('feature','split_mode','split_seed','selected_epoch',
         'validation_auroc','test_auroc','epochs_trained','training_seconds')}),flush=True)
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
    if not Path(args.checkpoint).is_file():raise FileNotFoundError(args.checkpoint)
    smiles=task['canonical_smiles'].tolist();task_hash=sha256(args.task_file)
    for mode in modes:
        for seed in args.split_seeds:
            ix,path=load_split(task,args,mode,seed)
            ix=stratified_smoke_indices(ix,y,args.smoke_limit,seed)
            train_one(args,smiles,y,mode,seed,ix,path,task_hash,sha256(path))
            write_summary(args.output_root,'molclr_gin')
    print('PASS MolCLR-GIN',flush=True)

if __name__=='__main__':main()

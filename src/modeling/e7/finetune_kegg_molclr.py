#!/usr/bin/env python3
"""E7 KEGG full fine-tuning of pretrained MolCLR-GIN.

Based on src/modeling/e7/finetune_molclr_npclass.py (the supplied E7 script).
Uses the authoritative KEGG 11-label data, saved splits and frozen QC.
Training uses per-label balanced positive/negative BCE; checkpoint selection
uses validation macro AUROC with macro AUPRC tie-breaker, never test metrics.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from kegg_ft_common import (
    VERSION, MeaningfulEarlyStopper, add_data_arguments, balanced_bce, balanced_weights, experiments,
    is_better, macro_scores, output_dir, run_is_complete, save_run, seed_all,
    summarize,
)
from models.ginet_molclr import GINet
from src.modeling.extract_molclr_embeddings import smiles_to_data


class MolCLKeggClassifier(nn.Module):
    def __init__(self, checkpoint: str, n_labels: int):
        super().__init__()
        self.encoder=GINet(num_layer=5,emb_dim=300,feat_dim=512,drop_ratio=0.0,pool='mean')
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.encoder.load_state_dict(state,strict=True)
        # Preserve the original E7 MolCLR protocol: exclude projection head.
        for p in self.encoder.out_lin.parameters():
            p.requires_grad_(False)
        self.head=nn.Linear(512,n_labels)

    def forward(self,graphs):
        emb,_=self.encoder(graphs)
        if emb.ndim!=2 or emb.shape[1]!=512:
            raise RuntimeError(f'Unexpected MolCLR encoder output shape {tuple(emb.shape)}')
        return self.head(emb)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    add_data_arguments(p)
    p.add_argument('--checkpoint',default='/private/vina/model_repos/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--early-stopping-patience',type=int,default=3)
    p.add_argument('--min-delta',type=float,default=0.001)
    p.add_argument('--min-epochs',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--encoder-lr',type=float,default=2e-5)
    p.add_argument('--head-lr',type=float,default=1e-3)
    p.add_argument('--weight-decay',type=float,default=.01)
    p.add_argument('--max-grad-norm',type=float,default=1.)
    p.add_argument('--num-workers',type=int,default=0)
    return p.parse_args()


def create_loader(task,y,ids,batch_size,shuffle,num_workers):
    graphs=[]
    for smi,target in zip(task.iloc[ids]['canonical_smiles'],y[ids]):
        graph=smiles_to_data(str(smi))
        if graph is None:
            raise ValueError(f'MolCLR featurizer failed for KEGG SMILES {smi!r}')
        graph.y=torch.tensor(target[None,:],dtype=torch.float32)
        graphs.append(graph)
    return DataLoader(graphs,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();ys=[];ps=[];loss=0.;n=0
    for batch in loader:
        batch=batch.to(device)
        logits=model(batch).float()
        yy=batch.y.float()
        if logits.shape!=yy.shape:
            raise ValueError(f'MolCLR prediction shape {tuple(logits.shape)} != target {tuple(yy.shape)}')
        loss+=nn.functional.binary_cross_entropy_with_logits(logits,yy,reduction='sum').item()
        n+=yy.numel()
        ys.append(yy.cpu().numpy());ps.append(logits.sigmoid().cpu().numpy())
    y=np.concatenate(ys);p=np.concatenate(ps)
    metrics=macro_scores(y,p)
    metrics['unweighted_loss']=float(loss/n)
    return metrics,y,p


def train_one(task,y,labels,mode,seed,ix,split_path,args):
    seed_all(seed)
    device=torch.device('cuda')
    out=output_dir(args,'molclr_gin',mode,seed)
    if run_is_complete(out,args):return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Existing incomplete run at {out}: inspect or move before retry')
    tr_loader=create_loader(task,y,ix['train'],args.batch_size,True,args.num_workers)
    val_loader=create_loader(task,y,ix['validation'],args.batch_size,False,args.num_workers)
    test_loader=create_loader(task,y,ix['test'],args.batch_size,False,args.num_workers)
    model=MolCLKeggClassifier(args.checkpoint,len(labels)).to(device)
    encoder_params=[p for p in model.encoder.parameters() if p.requires_grad]
    if not encoder_params:raise RuntimeError('MolCLR pretrained encoder unexpectedly frozen')
    optim=torch.optim.AdamW([
        {'params':encoder_params,'lr':args.encoder_lr},
        {'params':list(model.head.parameters()),'lr':args.head_lr},
    ],weight_decay=args.weight_decay)
    wpos,wneg,pos=balanced_weights(y[ix['train']],device)
    total_params=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('MolCLR-GIN',mode,seed,'train=',len(ix['train']),
          'weights_positive=',wpos.cpu().tolist(), 'trainable_parameters=',trainable,flush=True)
    best_auc=-np.inf;best_pr=-np.inf;best_epoch=None;best_state=None;history=[]
    early_stop=MeaningfulEarlyStopper(args.early_stopping_patience,args.min_delta,args.min_epochs)
    start=time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,args.epochs+1):
        t0=time.monotonic();model.train();running=0.;n=0
        for step,batch in enumerate(tr_loader):
            batch=batch.to(device);optim.zero_grad(set_to_none=True)
            logits=model(batch);yy=batch.y.float()
            if logits.shape!=yy.shape:raise ValueError('MolCLR training output shape differs from labels')
            loss=balanced_bce(logits,yy,wpos,wneg)
            loss.backward()
            if epoch==1 and step==0:
                grads=[p.grad for p in encoder_params if p.grad is not None]
                if not grads or sum(float(g.detach().abs().sum()) for g in grads)<=0:
                    raise RuntimeError('MolCLR encoder gradient check failed')
                print('ENCODER-GRADIENT PASS',flush=True)
            nn.utils.clip_grad_norm_(encoder_params+list(model.head.parameters()),args.max_grad_norm)
            optim.step()
            running+=float(loss.item())*batch.num_graphs;n+=batch.num_graphs
        val,_,_=evaluate(model,val_loader,device)
        row={'epoch':epoch,'train_balanced_loss':running/n,
             'validation_unweighted_loss':val['unweighted_loss'],
             'validation_macro_auroc':val['macro_auroc'],
             'validation_macro_auprc':val['macro_auprc'],
             'seconds':time.monotonic()-t0}
        if is_better(val['macro_auroc'],val['macro_auprc'],best_auc,best_pr):
            best_auc=val['macro_auroc'];best_pr=val['macro_auprc'];best_epoch=epoch
            best_state={name:p.detach().cpu().clone() for name,p in model.state_dict().items()}
        stop,meaningful=early_stop.step(epoch,val['macro_auroc'])
        row.update(early_stop_stale_epochs=early_stop.stale_epochs,
                   meaningful_auroc_improvement=meaningful)
        history.append(row)
        print('molclr_gin full',mode,seed,'epoch',epoch,'val_auc',round(val['macro_auroc'],6),
              'val_auprc',round(val['macro_auprc'],6),
              'stale',early_stop.stale_epochs,flush=True)
        if stop:
            print('EARLY STOP molclr_gin',mode,seed,'at',epoch,'best_epoch',best_epoch,flush=True)
            break
    assert best_state is not None
    model.load_state_dict(best_state,strict=True)
    validation,vy,vp=evaluate(model,val_loader,device)
    test,ty,tp=evaluate(model,test_loader,device)
    elapsed=time.monotonic()-start
    meta={'script_version':VERSION,'feature':'molclr_gin','adaptation_mode':'full','task':'kegg_pathway_11label',
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
    if not Path(args.checkpoint).is_file():raise FileNotFoundError(args.checkpoint)
    for mode,seed,ix,split_path in conditions:
        train_one(task,y,labels,mode,seed,ix,split_path,args)
        summarize(args,'molclr_gin')

if __name__=='__main__':main()

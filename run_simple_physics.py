# -*- coding: utf-8 -*-
"""Physically motivated SIMPLE loss ablation for unchanged W01-04.
Stable 5->20->2 KAN, 7 spline nodes, 3 fixed seeds.
Only train-part data used for model selection. Test predictions never fed into optimization.
Run: python code/run_simple_physics.py
"""
import os
for e in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[e]='1'
from pathlib import Path
import json, time, random, copy, sys
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent))
import base_model as base

ROOT=Path(__file__).resolve().parent.parent; RES=ROOT/'results'; FIG=ROOT/'figures'
SEEDS=[2026,2027,2028]
for d in (RES,FIG):d.mkdir(exist_ok=True,parents=True)
torch.set_num_threads(1)

# Physically motivated soft priors; all compare on identical architecture, data and timing.
CONFIGS={
  'NoPhysics':dict(offset=0,stability=0),
  'Offset_W02_0.3':dict(offset=.3,stability=0),
  'Offset_W02_1.0':dict(offset=1.,stability=0),
  'Offset_W02_3.0':dict(offset=3.,stability=0),
  'ThermalRobust_0.1':dict(offset=0,stability=.1),
  'ThermalRobust_0.5':dict(offset=0,stability=.5),
  'ThermalRobust_2.0':dict(offset=0,stability=2.),
  'OffsetPlusRobust':dict(offset=.3,stability=.5),
}

# all physics counterfactuals constructed exclusively from training inputs
# no future input or target at validation/selection time used for any constraint
# case 1: on W02 alone, source offset drives local depth prior (supported by training correlation)
# case 2: small +/-1% I,U arc-power perturbation: output stable inside small physical tolerance
class PhysPairs:
    def __init__(self,prep):
        self.p=prep
        self.off=None; self.rob=None
        offset=[]; stable=[]
        for wi,d in enumerate(base.RAW):
            cut=prep.cuts[wi];xx=d['x'][:cut]
            # validate W02 offset only; do not transfer potentially invalid prior to other welds
            if wi==1:
                ind=np.flatnonzero((np.abs(xx[:,2])>=.10)&(np.abs(xx[:,2])<=.55))
                if len(ind)>40:ind=ind[np.linspace(0,len(ind)-1,40,dtype=int)]
                for k in ind:
                    a=xx[k].copy();b=xx[k].copy();sgn=1. if a[2]>=0 else -1.
                    a[2]=sgn*max(0,abs(a[2])-.08);b[2]=sgn*(abs(b[2])+.08)
                    offset.append((wi,int(k),a,b))
            # physical local stability under small synchronized arc-source measurement shifts.
            # Same previous predicted grey state, only I,U perturbed and E,S unchanged.
            ind=np.arange(cut)
            if len(ind)>35:ind=ind[np.linspace(0,len(ind)-1,35,dtype=int)]
            for k in ind:
                a=xx[k].copy();b=xx[k].copy()
                a[0:2]*=.99;b[0:2]*=1.01
                stable.append((wi,int(k),a,b))
        self.off=self.convert(offset)
        self.rob=self.convert(stable)
    def convert(self,items):
        if not items:return None
        inds=np.array([[p[0],p[1]] for p in items],dtype=int)
        zn=torch.from_numpy(self.p.transform(np.stack([p[2] for p in items])))
        zf=torch.from_numpy(self.p.transform(np.stack([p[3] for p in items])))
        return (inds,zn,zf)

def prior_loss(model, pred, prior, mode):
    if prior is None:return pred.new_tensor(0.)
    ids,zn,zf=prior
    state=torch.zeros((len(ids),2),dtype=pred.dtype)
    use=ids[:,1]>0
    if np.any(use):state[use]=pred[ids[use,0],ids[use,1]-1].detach()
    a=model.step(state,zn);b=model.step(state,zf)
    if mode=='offset':
        # at W02 same I U S, increased |E| should not make penetration markedly rise
        # allow a 0.005 mm tolerance (normalized by train-only D scale)
        eps=.005/float(model.phys_y_scale[0])
        return F.relu(b[:,0]-a[:,0]-eps).square().mean()
    if mode=='stability':
        # Physical continuity under 1% I,U instrumental perturbations, no output label needed
        # tolerate depth 0.008 mm, width 0.012 mm difference.
        toler=torch.tensor([.008,.012],dtype=pred.dtype)/model.phys_y_scale
        excess=F.relu((b-a).abs()-toler)
        return (excess.square()*torch.tensor([.35,.65])).sum(dim=1).mean()
    raise ValueError(mode)

def fit(seed,stage,config,epochs=None):
    base.random_seed(seed)
    pp=base.Prep(stage);z,y,tr,va=pp.tensors()
    ph=PhysPairs(pp)
    model=base.GreyKAN((5,20,2),7)
    model.phys_y_scale=torch.tensor(pp.ystd,dtype=torch.float32)
    opt=torch.optim.AdamW(model.parameters(),lr=.012,weight_decay=.001)
    best=1e8;bestep=5;beststate=None
    lastphysics=None
    maxep=125 if stage=='valid' else int(epochs)
    for ep in range(1,maxep+1):
        lr=.012 if ep<=65 else .006 if ep<=105 else .003
        for pg in opt.param_groups:pg['lr']=lr
        model.train();opt.zero_grad();prediction=model(z)
        err=F.smooth_l1_loss(prediction,y,beta=.6,reduction='none')
        loss=(err*tr*torch.tensor([.6,.4])).sum()/tr.sum()
        priorval=0.
        if config['offset']:
            v=prior_loss(model,prediction,ph.off,'offset');loss=loss+config['offset']*v;priorval+=float(v.detach())
        if config['stability']:
            v=prior_loss(model,prediction,ph.rob,'stability');loss=loss+config['stability']*v;priorval+=float(v.detach())
        lastphysics=priorval
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),4.);opt.step()
        if stage=='valid' and (ep==1 or ep%5==0):
            model.eval()
            with torch.no_grad():p=model(z)
            val=float(((p-y).abs()*va*torch.tensor([.6,.4])).sum()/va.sum())
            if val<best-1e-5:best=val;bestep=ep;beststate=copy.deepcopy(model.state_dict())
            if ep-bestep>=35:break
    if beststate is not None:model.load_state_dict(beststate)
    model.eval()
    with torch.no_grad():outs=model(z).numpy()
    pred=[outs[i]*pp.ystd+pp.ymu for i in range(4)]
    return bestep,best,pred,lastphysics

def errors(y,p):
    e=y-p
    return (float(np.mean(np.abs(e))),float(np.sqrt(np.mean(e**2))),float(100*np.mean(np.abs(e)/np.maximum(np.abs(y),1e-8))))

def score_table(df):
    rows=[]
    for (variant,tar),data in df.groupby(['model','target']):
        met=[errors(a.actual.to_numpy(),a.prediction.to_numpy()) for _,a in data.groupby('seed')]
        av=np.mean(np.array(met),axis=0)
        sd=np.std(np.array(met),axis=0,ddof=1) if len(met)>1 else np.zeros(3)
        rows.append(dict(model=variant,target=tar,MAE=av[0],RMSE=av[1],MAPE=av[2],MAE_sd=sd[0],RMSE_sd=sd[1],MAPE_sd=sd[2]))
    return pd.DataFrame(rows)

def plot(df,chosen):
    view=df.groupby(['model','weld','target','test_row'],as_index=False).agg(actual=('actual','mean'),prediction=('prediction','mean'))
    view.to_csv(RES/'mean_predictions_for_plot.csv',index=False,encoding='utf-8-sig')
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.linewidth':1})
    fig,axes=plt.subplots(4,2,figsize=(13,14))
    for i,well in enumerate(base.RAW):
        for j,tar in enumerate(('D','W')):
            ax=axes[i,j]
            src=view[(view.weld==well['id'])&(view.target==tar)]
            zero=src[src.model=='NoPhysics'].sort_values('test_row')
            cur=src[src.model==chosen].sort_values('test_row')
            tt=cur.test_row.to_numpy()
            ax.plot(tt,cur.actual,color='black',lw=2.0,marker='o',ms=3.5,label='Actual')
            ax.plot(tt,zero.prediction,color='#999999',ls='--',lw=1.65,label='No physics')
            ax.plot(tt,cur.prediction,color='#1166A5',lw=1.8,marker='s',ms=2.8,label=chosen)
            ax.set_title(well['id']+' - '+('Penetration depth' if tar=='D' else 'Weld width'),fontweight='bold')
            ax.set_xlabel('Test sample index');ax.set_ylabel('mm');ax.grid(True,ls=':',alpha=.3)
            for sp in ax.spines.values():sp.set_visible(True)
            if i==0 and j==0:ax.legend(fontsize=8,frameon=True)
    fig.tight_layout(h_pad=1.8,w_pad=1.6)
    fig.savefig(FIG/'Four_welds_actual_nophys_best_physics_300dpi.png',dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(13,14))
    for i,well in enumerate(base.RAW):
        for j,tar in enumerate(('D','W')):
            ax=axes[i,j]
            cur=view[(view.weld==well['id'])&(view.target==tar)&(view.model==chosen)].sort_values('test_row')
            ax.plot(cur.test_row,cur.actual,color='black',lw=2,marker='o',ms=3,label='Actual')
            ax.plot(cur.test_row,cur.prediction,color='#1166A5',lw=1.8,marker='s',ms=2.5,label='Physics-guided PI-NGreyKAN')
            ax.set_title(well['id']+' - '+('Penetration depth' if tar=='D' else 'Weld width'),fontweight='bold')
            ax.set_xlabel('Test sample index');ax.set_ylabel('mm');ax.grid(True,ls=':',alpha=.3)
            if i==0 and j==0:ax.legend(fontsize=8,frameon=True)
    fig.tight_layout(h_pad=1.8)
    fig.savefig(FIG/'Four_welds_actual_best_physics_only_300dpi.png',dpi=300,bbox_inches='tight')
    plt.close(fig)

def main():
    trial_path=RES/'validation_trials.csv'
    records=pd.read_csv(trial_path).to_dict('records') if trial_path.exists() else []
    for name,conf in CONFIGS.items():
        for seed in SEEDS:
            if any(r['model']==name and int(r['seed'])==seed for r in records):
                continue
            tic=time.time(); ep,v,_,phys=fit(seed,'valid',conf)
            records.append(dict(model=name,seed=seed,epochs=ep,validation_score=v,prior_value=phys,offset_weight=conf['offset'],thermal_stability_weight=conf['stability']))
            print('VAL',name,seed,'epoch',ep,'score',round(v,6),'prior',round(phys,6),'s',round(time.time()-tic,2),flush=True)
            pd.DataFrame(records).to_csv(RES/'validation_trials.csv',index=False,encoding='utf-8-sig')
    ranking=pd.DataFrame(records).groupby('model',as_index=False).agg(validation_mean=('validation_score','mean'),validation_sd=('validation_score','std'))
    ranking=ranking.sort_values('validation_mean')
    ranking.to_csv(RES/'validation_ranking.csv',index=False,encoding='utf-8-sig')
    # Best NON-ZERO physical loss selected by internal validation; separately report no-physics anchor.
    viable=ranking[ranking.model!='NoPhysics'];winner=str(viable.iloc[0].model)
    (RES/'selected_physics.json').write_text(json.dumps({'best_nonzero':winner,'best_nonzero_validation':float(viable.iloc[0].validation_mean),'no_physics_validation':float(ranking[ranking.model=='NoPhysics'].iloc[0].validation_mean),'candidates':CONFIGS},ensure_ascii=False,indent=2),encoding='utf-8')
    preds=[]
    for modelname in ['NoPhysics',winner]:
        for seed in SEEDS:
            ep=int(pd.DataFrame(records).query('model==@modelname and seed==@seed').iloc[0].epochs)
            _,_,pp,_=fit(seed,'final',CONFIGS[modelname],epochs=ep)
            for wi,d in enumerate(base.RAW):
                for t in range(d['k80'],d['n']):
                    for j,tar in enumerate(['D','W']):
                        preds.append(dict(model=modelname,seed=seed,weld=d['id'],target=tar,test_row=t-d['k80']+1,original_index=int(d['ix'][t]),actual=float(d['y'][t,j]),prediction=float(pp[wi][t,j])))
            print('TEST',modelname,seed,flush=True)
    df=pd.DataFrame(preds)
    df.to_csv(RES/'test_predictions.csv',index=False,encoding='utf-8-sig')
    metric=score_table(df);metric.to_csv(RES/'summary_metrics.csv',index=False,encoding='utf-8-sig')
    bygroup=[]
    for (model,weld,target),g in df.groupby(['model','weld','target']):
        allm=np.array([errors(s.actual.to_numpy(),s.prediction.to_numpy()) for _,s in g.groupby('seed')])
        avg=allm.mean(0)
        bygroup.append(dict(model=model,weld=weld,target=target,MAE=avg[0],RMSE=avg[1],MAPE=avg[2]))
    pd.DataFrame(bygroup).to_csv(RES/'by_weld_metrics.csv',index=False,encoding='utf-8-sig')
    print('\nVALIDATION RANKING\n',ranking.to_string(index=False))
    print('\nTEST METRICS\n',metric.to_string(index=False))
    plot(df,winner)
    print('BEST_PHYSICS',winner,flush=True)

if __name__=='__main__':main()

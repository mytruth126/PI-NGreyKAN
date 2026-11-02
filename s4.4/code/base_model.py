# -*- coding: utf-8 -*-
"""KAN width/depth + physics weight tuning, fixed original W01-W04 filtered 2/3 of six data.
Time-ordered validation and test; model selection ONLY by internal validation. No test-label updating.
Dependencies: numpy, pandas, torch, matplotlib. Python 3.9 compatible.
"""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
import copy, time, random, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as TF

try: torch.set_num_threads(1)
except: pass
ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'data'; OUT=ROOT/'results'; FIG=ROOT/'figures'
OUT.mkdir(exist_ok=True,parents=True);FIG.mkdir(exist_ok=True,parents=True)
SEEDS=[2026,2027,2028]

RAW=[]
for i in range(1,5):
    df=pd.read_csv(DATA/('W%02d_随机删去2of3的6mm_顺序保留.csv'%i), encoding='utf-8-sig')
    x=df[['I_A','U_V','E_mm','S_dB']].to_numpy(np.float32)
    y=df[['D_mm','W_mm']].to_numpy(np.float32)
    ix=df['original_index'].to_numpy(int)
    n=len(df);k80=int(n*.8);k64=int(k80*.8)
    assert (df['stage'].to_numpy()[:k80]=='train').all() and (df['stage'].to_numpy()[k80:]=='test').all()
    RAW.append(dict(id='W%02d'%i,x=x,y=y,ix=ix,n=n,k80=k80,k64=k64))


def random_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def qfeat(x):
    return (x[:,0]*x[:,1]/1000)*np.exp(-.5*(x[:,2]/.5)**2)

class Prep:
    def __init__(self,stage):
        self.stage=stage
        self.cuts=[d['k64'] if stage=='valid' else d['k80'] for d in RAW]
        xtr=np.concatenate([d['x'][:c] for d,c in zip(RAW,self.cuts)])
        ytr=np.concatenate([d['y'][:c] for d,c in zip(RAW,self.cuts)])
        self.xmu=xtr.mean(0);self.xstd=np.maximum(xtr.std(0),1e-4)
        self.ymu=np.median(ytr,axis=0);self.ystd=np.maximum(ytr.std(0),.02)
        qtr=qfeat(xtr);self.qmu=qtr.mean();self.qstd=max(qtr.std(),1e-4)
        self.z=[];self.y=[]
        for d in RAW:
            self.z.append(self.transform(d['x']))
            self.y.append(((d['y']-self.ymu)/self.ystd).astype('float32'))
    def transform(self,x):
        z=(x-self.xmu)/self.xstd
        z[:,2]=x[:,2]/self.xstd[2]
        return np.column_stack([z,(qfeat(x)-self.qmu)/self.qstd]).astype('float32')
    def tensors(self):
        n=len(RAW);maxlen=max(d['n'] for d in RAW)
        zs=np.zeros((n,maxlen,5),np.float32);ys=np.zeros((n,maxlen,2),np.float32)
        tr=np.zeros((n,maxlen,1),np.float32);val=np.zeros_like(tr)
        for i,d in enumerate(RAW):
            N=d['n'];cut=self.cuts[i]
            zs[i,:N]=self.z[i];ys[i,:N]=self.y[i]
            tr[i,:cut,0]=1.
            if self.stage=='valid':val[i,cut:d['k80'],0]=1.
            else:val[i,cut:N,0]=1.
        return tuple(torch.tensor(a) for a in (zs,ys,tr,val))
    def counterfactual(self):
        rec=[]
        for i,d in enumerate(RAW):
            x=d['x'][:self.cuts[i]]
            inds=np.flatnonzero((np.abs(x[:,2])>=.1)&(np.abs(x[:,2])<=.55))
            if len(inds)>40:inds=inds[np.linspace(0,len(inds)-1,40).astype(int)]
            for t in inds:
                near=x[t].copy();far=x[t].copy();sgn=1 if x[t,2]>=0 else -1
                near[2]=sgn*max(0.,abs(x[t,2])-.08);far[2]=sgn*(abs(x[t,2])+.08)
                rec.append((i,t,near,far))
        if not rec:return None
        return (torch.tensor(self.transform(np.stack([r[2] for r in rec]))),
                torch.tensor(self.transform(np.stack([r[3] for r in rec]))),
                np.array([[r[0],r[1]] for r in rec],dtype=int))

class CubicKANLayer(nn.Module):
    def __init__(self,ni,no,nknots=7):
        super().__init__();self.base=nn.Linear(ni,no)
        self.register_buffer('centers',torch.linspace(-3,3,nknots))
        self.coef=nn.Parameter(torch.randn(no,ni,nknots)*.008)
        nn.init.xavier_uniform_(self.base.weight,gain=.28);nn.init.zeros_(self.base.bias)
    def forward(self,x):
        z=x.clamp(-4.5,4.5)
        dist=(z.unsqueeze(-1)-self.centers).abs()
        B=((2-dist).clamp_min(0)**3-4*(1-dist).clamp_min(0)**3)/6
        return self.base(z)+torch.einsum('...im,oim->...o',B,self.coef)

class GreyKAN(nn.Module):
    def __init__(self,layers=(5,6,2),knots=7):
        super().__init__()
        self.kan=nn.ModuleList([CubicKANLayer(layers[i],layers[i+1],knots) for i in range(len(layers)-1)])
        self.gamma=nn.Linear(5,2)
        nn.init.zeros_(self.gamma.weight);nn.init.constant_(self.gamma.bias,-.37)
        self.braw=nn.Parameter(torch.full((2,),-3.))
    def drive(self,z):
        for i,layer in enumerate(self.kan):
            z=layer(z)
            if i<len(self.kan)-1:z=torch.tanh(z)
        return z
    def compute_a(self,z):return .12+2*torch.sigmoid(self.gamma(z))
    def compute_b(self):return TF.softplus(self.braw).clamp_max(2)
    def step(self,v,z):
        F=self.drive(z);a=self.compute_a(z);b=self.compute_b()
        R=v+F;A=1+a
        return 2*R/(A+torch.sqrt(A*A+4*b*R.abs()+1e-10))
    def forward(self,z):
        F=self.drive(z);a=self.compute_a(z);b=self.compute_b()
        B,T,_=F.shape;v=torch.zeros((B,2),device=z.device,dtype=z.dtype)
        out=[]
        for t in range(T):
            R=v+F[:,t];A=1+a[:,t]
            v=2*R/(A+torch.sqrt(A*A+4*b*R.abs()+1e-10))
            out.append(v)
        return torch.stack(out,dim=1)

CANDIDATES={
 'original': dict(layers=(5,6,2),knots=7,phys=.2),
 'wide12': dict(layers=(5,12,2),knots=7,phys=.2),
 'deep12_8': dict(layers=(5,12,8,2),knots=7,phys=.2),
 'wide12_lowphys': dict(layers=(5,12,2),knots=7,phys=.03),
 'deep12_8_lowphys': dict(layers=(5,12,8,2),knots=7,phys=.03),
 'wide12_nophys': dict(layers=(5,12,2),knots=7,phys=0.),
}

def fit(seed,stage,conf,epochs=None,fast=False):
    random_seed(seed);prep=Prep(stage)
    z,y,train,val=prep.tensors();counter=prep.counterfactual() if conf['phys'] else None
    model=GreyKAN(conf['layers'], conf['knots'])
    opt=torch.optim.AdamW(model.parameters(),lr=.012,weight_decay=.001)
    best=float('inf');bestep=5;beststate=None
    maxep=(90 if fast else 125) if stage=='valid' else int(epochs)
    for ep in range(1,maxep+1):
        lr=.012 if ep<=65 else (.006 if ep<=105 else .003)
        for par in opt.param_groups:par['lr']=lr
        model.train();opt.zero_grad();pr=model(z)
        er=TF.smooth_l1_loss(pr,y,beta=.6,reduction='none')
        loss=(er*train*torch.tensor([.6,.4])).sum()/train.sum()
        if counter is not None:
            zn,zf,ids=counter;prev=torch.zeros((len(ids),2))
            ok=ids[:,1]>0
            if ok.any():prev[ok]=pr[ids[ok,0],ids[ok,1]-1].detach()
            phys=TF.relu(model.step(prev,zf)[:,0]-model.step(prev,zn)[:,0]).square().mean()
            loss=loss+conf['phys']*phys
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),4.0);opt.step()
        if stage=='valid' and (ep%5==0 or ep==1):
            model.eval()
            with torch.no_grad():pred=model(z)
            v=float(((pred-y).abs()*val*torch.tensor([.6,.4])).sum()/val.sum())
            if v<best-1e-5:
                best=v;bestep=ep;beststate=copy.deepcopy(model.state_dict())
            if ep-bestep>=35:break
    if beststate is not None:model.load_state_dict(beststate)
    model.eval()
    with torch.no_grad():p=model(z).numpy()
    preds=[(p[i]*prep.ystd+prep.ymu)[:d['n']] for i,d in enumerate(RAW)]
    return bestep,best,preds,len(list(model.parameters()))

def m(y,p):
    z=y-p
    return dict(MAE=float(np.mean(np.abs(z))),RMSE=float(np.sqrt(np.mean(z*z))),MAPE=float(100*np.mean(np.abs(z)/np.maximum(np.abs(y),1e-8))))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['select','final','both'],default='both')
    parser.add_argument('--fast',action='store_true')
    args=parser.parse_args()
    record=[]
    selection_csv=OUT/'validation_selection.csv'
    if args.stage in ('select','both'):
        for name,conf in CANDIDATES.items():
            for seed in SEEDS:
                tic=time.time();ep,val,preds,_=fit(seed,'valid',conf,fast=args.fast)
                record.append(dict(model=name,seed=seed,selected_epochs=ep,validation_score=val,layers='-'.join(map(str,conf['layers'])),lambda_phys=conf['phys']))
                print('VAL',name,seed,ep,round(val,6),'sec',round(time.time()-tic,1),flush=True)
                pd.DataFrame(record).to_csv(selection_csv,index=False,encoding='utf-8-sig')
    if args.stage in ('final','both'):
        selections=pd.read_csv(selection_csv)
        ranks=selections.groupby('model')['validation_score'].mean().sort_values()
        winner=ranks.index[0]
        print('VALIDATION CHOICE:',winner, 'rank:',ranks.to_dict(),flush=True)
        (OUT/'selected_model.txt').write_text('Selected by validation: '+winner+'\n'+str(ranks.to_dict())+'\n',encoding='utf-8')
        records=[]
        for seed in SEEDS:
            ep=int(selections[(selections.model==winner)&(selections.seed==seed)].selected_epochs.iloc[0])
            tic=time.time();_,_,ps,_=fit(seed,'final',CANDIDATES[winner],epochs=ep)
            for i,d in enumerate(RAW):
                for t in range(d['k80'],d['n']):
                    for j,tar in enumerate(['D','W']):
                        records.append(dict(model='PI-NGreyKAN-'+winner,seed=seed,weld=d['id'],target=tar,
                                            test_row=t-d['k80']+1,original_index=int(d['ix'][t]),
                                            actual=float(d['y'][t,j]),prediction=float(ps[i][t,j])))
            print('TEST',winner,seed,'sec',round(time.time()-tic,1),flush=True)
        new=pd.DataFrame(records)
        new.to_csv(OUT/'improved_test_predictions.csv',index=False,encoding='utf-8-sig')
        original_path=ROOT/'original_test_predictions.csv'
        old=pd.read_csv(original_path)
        old['test_row']=old.groupby(['model','seed','weld','target']).cumcount()+1
        old=old[['model','seed','weld','target','test_row','original_index','actual','prediction']]
        together=pd.concat([old,new],ignore_index=True)
        together.to_csv(OUT/'original_and_improved_predictions.csv',index=False,encoding='utf-8-sig')
        sums=[];groups=[]
        for (name,tar),df in together.groupby(['model','target']):
            seeds=[]
            for _,g in df.groupby('seed'):seeds.append(m(g.actual.to_numpy(),g.prediction.to_numpy()))
            sums.append(dict(model=name,target=tar,n=int(len(df)/len(seeds)),seeds=len(seeds),**{k:float(np.mean([v[k] for v in seeds])) for k in ['MAE','RMSE','MAPE']}))
        for (name,weld,tar),df in together.groupby(['model','weld','target']):
            seeds=[]
            for _,g in df.groupby('seed'):seeds.append(m(g.actual.to_numpy(),g.prediction.to_numpy()))
            groups.append(dict(model=name,weld=weld,target=tar,n=int(len(df)/len(seeds)),**{k:float(np.mean([v[k] for v in seeds])) for k in ['MAE','RMSE','MAPE']}))
        pd.DataFrame(sums).to_csv(OUT/'summary_metrics.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(groups).to_csv(OUT/'by_weld_metrics.csv',index=False,encoding='utf-8-sig')
        print(pd.DataFrame(sums).to_string(index=False),flush=True)
        plot(together)

def plot(df):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.linewidth':1.0,
                        'xtick.direction':'in','ytick.direction':'in'})
    # average across seeds for both models
    mean=df.groupby(['model','weld','target','test_row'],as_index=False).agg(actual=('actual','mean'),prediction=('prediction','mean'))
    models=list(mean.model.unique()); improved=next(m for m in models if m!='PI-NGreyKAN')
    fig,axes=plt.subplots(4,2,figsize=(13,14))
    for i,d in enumerate(RAW):
        for j,target in enumerate(['D','W']):
            ax=axes[i,j];tmp=mean[(mean.weld==d['id'])&(mean.target==target)]
            aa=tmp[tmp.model=='PI-NGreyKAN'].sort_values('test_row')
            bb=tmp[tmp.model==improved].sort_values('test_row')
            x=aa.test_row.to_numpy()
            ax.plot(x,aa.actual,'-o',color='black',linewidth=2.1,markersize=3.2,label='Actual')
            ax.plot(x,aa.prediction,'--',color='#B7A59A',linewidth=1.65,label='Original PI-NGreyKAN')
            ax.plot(x,bb.prediction,'-s',color='#1564A0',linewidth=1.8,markersize=2.7,label='Validation-selected PI-NGreyKAN')
            ax.set_title(d['id']+' – '+('Penetration depth' if target=='D' else 'Weld width'))
            ax.set_xlabel('Test sample index');ax.set_ylabel('mm');ax.grid(True,linestyle=':',alpha=.27)
            for s in ax.spines.values():s.set_visible(True)
            if i==0 and j==0:ax.legend(frameon=True,fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG/'W01-W04_original_vs_KAN_upgrade_300dpi.png',dpi=300,bbox_inches='tight')
    plt.close(fig)
    for d in RAW:
        fig,axes=plt.subplots(1,2,figsize=(11,4.2))
        for j,target in enumerate(['D','W']):
            ax=axes[j];tmp=mean[(mean.weld==d['id'])&(mean.target==target)]
            a=tmp[tmp.model=='PI-NGreyKAN'].sort_values('test_row')
            b=tmp[tmp.model==improved].sort_values('test_row')
            x=a.test_row.to_numpy()
            ax.plot(x,a.actual,'-o',color='black',linewidth=2.1,markersize=3.5,label='Actual')
            ax.plot(x,a.prediction,'--',color='#B7A59A',linewidth=1.7,label='Original')
            ax.plot(x,b.prediction,'-s',color='#1564A0',linewidth=1.8,markersize=2.7,label='Upgraded')
            ax.set_title('Penetration depth' if target=='D' else 'Weld width');ax.set_xlabel('Test sample index');ax.set_ylabel('mm')
            ax.grid(True,linestyle=':',alpha=.28)
            if j==0:ax.legend(frameon=True,fontsize=8)
        fig.suptitle(d['id']);fig.tight_layout()
        fig.savefig(FIG/(d['id']+'_original_vs_KAN_upgrade_300dpi.png'),dpi=300,bbox_inches='tight')
        plt.close(fig)

if __name__=='__main__':main()

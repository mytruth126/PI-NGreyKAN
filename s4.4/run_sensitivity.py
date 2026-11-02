# -*- coding: utf-8 -*-
"""Standalone PI-NGreyKAN (ell, alpha) sensitivity experiment and plotting.
Usage:
 python run_sensitivity.py --mode quick  # 49x1 seed, short training, diagnostics only
 python run_sensitivity.py --mode full   # 49x3 seeds, internal validation and retraining
 python run_sensitivity.py --mode plot   # replot cached CSV only

IMPORTANT: train/test split is fixed; testing labels never enter the optimizer.
All MAEs are reported SEPARATELY for penetration depth and weld width.
Pooled MAE over four cases: sum absolute errors / total sample count.
"""
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v]='1'
import argparse,sys,copy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import base_model as base
import run_simple_physics as physics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle

ELL_VALUES=[.20,.30,.40,.50,.65,.80,1.00]
ALPHA_VALUES=[.20,.30,.40,.50,.60,.70,.80]
DEFAULT_ELL=.50
DEFAULT_ALPHA=.60
PHYS_WEIGHT=2.0
SEEDS=[2026,2027,2028]
OUT=ROOT/'results';FIG=ROOT/'figures'
OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
torch.set_num_threads(1)

# Reuse ORIGINAL KAN, grey equation, normalization, physics perturbation,
# plus EXACT SAME fixed data. Only substitute the queried ell in qfeat.
# The original Prep reads base.qfeat at runtime, so it automatically uses ell.
def set_ell(ell):
    def qfeat(x):
        return (x[:,0]*x[:,1]/1000.)*np.exp(-.5*(x[:,2]/ell)**2)
    base.qfeat=qfeat


def train_one(seed, ell, alpha, stage='final', epochs=10):
    """One model fit. Validation stage selects epochs, final stage trains on 80%."""
    set_ell(ell)
    base.random_seed(seed)
    prep=base.Prep(stage)
    z,y,tr,va=prep.tensors()
    pairs=physics.PhysPairs(prep)
    model=base.GreyKAN((5,20,2),7)
    model.phys_y_scale=torch.tensor(prep.ystd,dtype=torch.float32)
    opt=torch.optim.AdamW(model.parameters(),lr=.012,weight_decay=.001)
    target_weight=torch.tensor([alpha,1.-alpha])
    best=float('inf');bestep=5;best_state=None
    max_epochs=125 if stage=='valid' else int(epochs)
    for epoch in range(1,max_epochs+1):
        lr=.012 if epoch<=65 else .006 if epoch<=105 else .003
        for group in opt.param_groups:group['lr']=lr
        model.train();opt.zero_grad();pred=model(z)
        error=F.smooth_l1_loss(pred,y,beta=.6,reduction='none')
        loss=(error*tr*target_weight).sum()/tr.sum()
        loss=loss+PHYS_WEIGHT*physics.prior_loss(model,pred,pairs.rob,'stability')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),4.)
        opt.step()
        if stage=='valid' and (epoch==1 or epoch%5==0):
            model.eval()
            with torch.no_grad():vpred=model(z)
            score=float(((vpred-y).abs()*va*target_weight).sum()/va.sum())
            if score<best-1e-5:
                best=score;bestep=epoch;best_state=copy.deepcopy(model.state_dict())
            if epoch-bestep>=35:break
    if best_state is not None:model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(): predicted=model(z).numpy()
    preds=[(predicted[i]*prep.ystd+prep.ymu)[:d['n']] for i,d in enumerate(base.RAW)]
    return preds,bestep,best


def pooled_mae(preds,split):
    errors=[]
    for i,d in enumerate(base.RAW):
        cut=d['k80']
        sl=slice(0,cut) if split=='train' else slice(cut,d['n'])
        errors.append(np.abs(preds[i][sl]-d['y'][sl]))
    return np.concatenate(errors,axis=0).mean(axis=0)


def create_metrics(mode):
    quick=(mode=='quick')
    seeds=[2026] if quick else SEEDS
    rows=[]
    filename=OUT/('sensitivity_metrics_quick.csv' if quick else 'sensitivity_metrics_by_target.csv')
    if filename.exists():
        rows=pd.read_csv(filename).to_dict('records')
    known={(round(float(r['alpha']),5),round(float(r['ell']),5),int(r['seed'])) for r in rows}
    total=len(ELL_VALUES)*len(ALPHA_VALUES)*len(seeds)
    count=len(known)
    for alpha in ALPHA_VALUES:
        for ell in ELL_VALUES:
            for seed in seeds:
                key=(round(alpha,5),round(ell,5),seed)
                if key in known:continue
                st=time.time()
                if quick:
                    p,ep,_=train_one(seed,ell,alpha,'final',epochs=8)
                    validation=float('nan')
                else:
                    _,ep,validation=train_one(seed,ell,alpha,'valid')
                    p,_,_=train_one(seed,ell,alpha,'final',epochs=ep)
                tr=pooled_mae(p,'train');te=pooled_mae(p,'test')
                rows.append(dict(alpha=alpha,ell=ell,seed=seed,epochs=ep,validation_score=validation,
                                 train_mae_D=tr[0],train_mae_W=tr[1],test_mae_D=te[0],test_mae_W=te[1]))
                pd.DataFrame(rows).sort_values(['alpha','ell','seed']).to_csv(filename,index=False,encoding='utf-8-sig')
                count+=1
                print(f'{count}/{total} alpha={alpha:.2f}, ell={ell:.2f}, seed={seed}: '
                      f'D={te[0]:.5f}, W={te[1]:.5f}, seconds={time.time()-st:.1f}',flush=True)
    return filename


def make_plots(csv_path, preview=False):
    df=pd.read_csv(csv_path)
    cols=['train_mae_D','test_mae_D','train_mae_W','test_mae_W']
    # Expected one record per seed per combination; first mean seed-level MAEs.
    agg=df.groupby(['alpha','ell'],as_index=False)[cols].mean()
    if len(agg)!=49:
        raise ValueError(f'Only {len(agg)}/49 parameter combinations completed; finish experiment before plotting')
    agg.to_csv(OUT/('sensitivity_mean_quick.csv' if preview else 'sensitivity_metrics_mean.csv'),index=False,encoding='utf-8-sig')
    installed=set(plt.rcParams['font.family'])
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11,'axes.unicode_minus':False,'pdf.fonttype':42,'ps.fonttype':42})
    for target,label,filename in [('D','Penetration depth','Sensitivity_Penetration_Depth'),('W','Weld width','Sensitivity_Weld_Width')]:
        tr=agg.pivot(index='alpha',columns='ell',values=f'train_mae_{target}').reindex(index=ALPHA_VALUES,columns=ELL_VALUES).to_numpy()
        te=agg.pivot(index='alpha',columns='ell',values=f'test_mae_{target}').reindex(index=ALPHA_VALUES,columns=ELL_VALUES).to_numpy()
        bi=ALPHA_VALUES.index(DEFAULT_ALPHA);bj=ELL_VALUES.index(DEFAULT_ELL)
        baseline=te[bi,bj]
        relative=100*(te-baseline)/baseline
        X,Y=np.meshgrid(ELL_VALUES,ALPHA_VALUES)
        fig=plt.figure(figsize=(13.8,5.75))
        gs=fig.add_gridspec(1,2,width_ratios=[1.05,1.0],left=.035,right=.97,bottom=.12,top=.86,wspace=.23)
        ax=fig.add_subplot(gs[0,0],projection='3d')
        surf=ax.plot_surface(X,Y,tr,cmap='viridis',edgecolor='white',linewidth=.8,antialiased=True)
        zrange=max(float(np.ptp(tr)),1e-4)
        floor=float(tr.min())-.19*zrange
        ax.contourf(X,Y,tr,zdir='z',offset=floor,levels=10,cmap='viridis',alpha=.85)
        ax.contour(X,Y,tr,zdir='z',offset=floor,levels=7,colors='.35',linewidths=.45)
        ax.set_zlim(floor,float(tr.max())+.15*zrange)
        ax.set_xlabel(r'Thermal decay scale $\ell$',labelpad=8)
        ax.set_ylabel(r'Depth-loss weight $\alpha$',labelpad=8)
        ax.set_zlabel('Training MAE (mm)',labelpad=7)
        ax.set_title('(a) Training-fit MAE sensitivity surface',fontweight='bold',pad=14,fontsize=12)
        ax.view_init(elev=26,azim=-59)
        ax.tick_params(labelsize=8)
        c1=fig.colorbar(surf,ax=ax,shrink=.65,pad=.08);c1.set_label('Training MAE (mm)',fontsize=10)
        ax2=fig.add_subplot(gs[0,1])
        lim=max(np.abs(relative).max(),.01)
        norm=colors.TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim)
        im=ax2.imshow(relative,cmap='RdYlGn_r',norm=norm,aspect='auto')
        ax2.set_xticks(range(7),[f'{x:g}' for x in ELL_VALUES]);ax2.set_yticks(range(7),[f'{x:g}' for x in ALPHA_VALUES]);ax2.tick_params(labelsize=10)
        ax2.set_xlabel(r'Thermal decay scale $\ell$');ax2.set_ylabel(r'Depth-loss weight $\alpha$')
        ax2.set_title('(b) Relative test MAE change (%)',fontweight='bold',pad=14,fontsize=12)
        ax2.set_xticks(np.arange(-.5,7,1),minor=True);ax2.set_yticks(np.arange(-.5,7,1),minor=True)
        ax2.grid(which='minor',color='white',lw=1.5);ax2.tick_params(which='minor',bottom=False,left=False)
        for i in range(7):
            for j in range(7):
                rgba=im.cmap(im.norm(relative[i,j]));luma=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax2.text(j,i,f'{relative[i,j]:+.1f}',ha='center',va='center',fontsize=9,fontweight='bold',color='white' if luma<.44 else 'black')
        ax2.add_patch(Rectangle((bj-.5,bi-.5),1,1,fill=False,ec='black',lw=2.4))
        c2=fig.colorbar(im,ax=ax2,shrink=.82,pad=.03);c2.set_label('Relative test MAE change (%)',fontsize=10)
        name=filename+('_QUICK_PREVIEW' if preview else '_300dpi')
        fig.savefig(FIG/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white')
        fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',facecolor='white')
        plt.close(fig)
        print('FIG',FIG/(name+'.png'),'baseline test MAE',baseline)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['quick','full','plot'],default='quick')
    parser.add_argument('--quick-plot',action='store_true',help='plot the quick CSV instead of full CSV')
    args=parser.parse_args()
    if args.mode in ('quick','full'):
        csv_path=create_metrics(args.mode)
        make_plots(csv_path,preview=args.mode=='quick')
    else:
        csv_path=OUT/('sensitivity_metrics_quick.csv' if args.quick_plot else 'sensitivity_metrics_by_target.csv')
        make_plots(csv_path,preview=args.quick_plot)

if __name__=='__main__':main()

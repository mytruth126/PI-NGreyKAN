# -*- coding: utf-8 -*-
"""Pairwise ablation figures (real predictions; NO model fitting).

Input: results/ablation/ablation_test_predictions.csv
Output: figures/ablation/Pairwise_Abla...D/W_300dpi.png and PDF

Top: per-case, per-seed MAE of M1 versus each variant (4 cases x 3 seeds
     = 12 paired points per panel). Case colors distinguish W01-04.
Bottom: MAE across ALL 108 held-out points, computed within EACH seed;
        bars = mean over 3 seeds, whiskers = sample SD over 3 seeds.
The % annotation is a relative error change, NOT a significance claim.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

BASE = Path(__file__).resolve().parents[1]
FILE = BASE / 'results' / 'ablation' / 'ablation_test_predictions.csv'
OUT = BASE / 'figures' / 'ablation'
ORDER = [
    'M1: PI-NGreyKAN',
    'M2: without KAN',
    'M3: without Grey Development',
    'M4: without Adaptive Development',
    'M5: without Thermal Feature',
]
MODEL_COLORS = {
    'M1: PI-NGreyKAN': '#335fd1',
    'M2: without KAN': '#53a633',
    'M3: without Grey Development': '#10a8c1',
    'M4: without Adaptive Development': '#ee9727',
    'M5: without Thermal Feature': '#c76181',
}
CASE_COLORS = {'W01':'#3366cc', 'W02':'#1ca3b3', 'W03':'#65a845', 'W04':'#cc6683'}


def load_predictions(path=FILE):
    if not path.exists():
        raise FileNotFoundError(f'Missing predictions: {path}\nRun python code/run_ablation.py first.')
    df = pd.read_csv(path, encoding='utf-8-sig')
    need={'variant','seed','weld','target','test_row','actual','prediction'}
    if need-set(df.columns):
        raise ValueError(f'Missing columns: {need-set(df.columns)}')
    df = df[df.variant.isin(ORDER)].copy()
    if df.empty:
        raise ValueError('No M1-M5 model results found in predictions CSV.')
    df['seed']=pd.to_numeric(df['seed'],errors='raise').astype(int)
    for c in ('actual','prediction','test_row'):
        df[c]=pd.to_numeric(df[c],errors='raise')
    if not np.isfinite(df[['actual','prediction','test_row']].to_numpy()).all():
        raise ValueError('Nonfinite predictions found')
    key=['variant','seed','weld','target','test_row']
    if df.duplicated(key).any():
        raise ValueError('Duplicate ablation test points. Check input.')
    # Require same case/seed/target/sample index for every variant.
    base = df[df.variant == ORDER[0]][['seed','weld','target','test_row','actual']]
    if base.empty:
        raise ValueError('No full-model results found')
    for name in ORDER[1:]:
        other = df[df.variant == name][['seed','weld','target','test_row','actual']]
        aligned = base.merge(other, on=['seed','weld','target','test_row'], how='outer',
                             suffixes=('_m1','_mi'), indicator=True)
        if not (aligned['_merge']=='both').all() or len(aligned)!=len(base):
            raise ValueError(f'Unmatched test samples: {name}')
        if not np.allclose(aligned.actual_m1,aligned.actual_mi,atol=1e-6,rtol=0):
            raise ValueError(f'Actual values differ between models: {name}')
    return df


def metrics(df,target):
    selected=df[df.target==target].copy()
    selected['abs_error']=np.abs(selected.actual-selected.prediction)
    # Scatter: MAE grouped by case and random seed, each matched point = one case/seed
    per_case=(selected.groupby(['variant','seed','weld'],as_index=False)
                    .agg(mae=('abs_error','mean'), n=('abs_error','size')))
    # Bars: pooled test MAE for all 108 points, then mean/std over seeds
    per_seed=(selected.groupby(['variant','seed'],as_index=False)
                    .agg(mae=('abs_error','mean'), n=('abs_error','size')))
    n_expected=selected[selected.variant==ORDER[0]].groupby('seed').size().unique()
    if len(n_expected)!=1:
        raise ValueError('Full model seeds have differing test lengths')
    for _,r in per_seed.iterrows():
        if r['n']!=n_expected[0]:
            raise ValueError(f'Missing test points in {r.variant}, seed={r.seed}')
    return per_case,per_seed


def plot_target(df,target):
    per_case,per_seed=metrics(df,target)
    label='Penetration Depth' if target=='D' else 'Weld Width'
    fig, axes=plt.subplots(2,4,figsize=(15.4,7.5),dpi=130)
    fig.suptitle(f'Pairwise Comparison of Structural Ablations: {label}',
                 y=.976,fontsize=18,fontweight='bold')
    fig.subplots_adjust(left=.062,right=.991,bottom=.14,top=.87,hspace=.54,wspace=.34)
    base=ORDER[0]
    for i, variant in enumerate(ORDER[1:]):
        axt, axb=axes[0,i],axes[1,i]
        m1=(per_case[per_case.variant==base][['weld','seed','mae']]
                .rename(columns={'mae':'mae_m1'}))
        mi=(per_case[per_case.variant==variant][['weld','seed','mae']]
                .rename(columns={'mae':'mae_mi'}))
        paired=m1.merge(mi,on=['weld','seed'],how='inner',validate='one_to_one')
        if len(paired)!=len(m1) or len(paired)!=len(mi):
            raise ValueError(f'Case/seed mismatch for {variant}')
        xs=paired.mae_m1.to_numpy()
        ys=paired.mae_mi.to_numpy()
        highest=max(float(xs.max()),float(ys.max()),.005)*1.12
        axt.plot([0,highest],[0,highest],'-',lw=1.1,c='#515b64',zorder=1)
        for weld,grp in paired.groupby('weld'):
            axt.scatter(grp.mae_m1,grp.mae_mi,s=36,
                c=CASE_COLORS.get(weld,'#222222'),marker='o',
                edgecolors='white',linewidths=.5,alpha=.95,zorder=3)
        axt.set(xlim=(0,highest),ylim=(0,highest))
        axt.set_aspect('equal',adjustable='box')
        axt.tick_params(labelsize=10,direction='in',top=True,right=True)
        axt.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axt.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axt.set_xlabel('M1 MAE (mm)',fontsize=11)
        if i==0: axt.set_ylabel('Variant MAE (mm)',fontsize=11)
        axt.text(.06,.92,'M1',transform=axt.transAxes,ha='left',va='top',
                 fontsize=12,fontweight='bold')
        axt.text(.95,.06,f'M{i+2}',transform=axt.transAxes,ha='right',va='bottom',
                 fontsize=12,fontweight='bold')
        # Bar heights are the correct sample-pooled metrics averaged over seeds
        full=per_seed.loc[per_seed.variant==base,'mae'].to_numpy()
        other=per_seed.loc[per_seed.variant==variant,'mae'].to_numpy()
        m=np.array([np.mean(full),np.mean(other)])
        s=np.array([np.std(full,ddof=1),np.std(other,ddof=1)])
        axb.bar([0,1],m,color=[MODEL_COLORS[base],MODEL_COLORS[variant]],
                width=.65,zorder=2)
        axb.errorbar([0,1],m,yerr=s,fmt='none',ecolor='#606060',
                     elinewidth=1.15,capsize=4,zorder=3)
        axb.set_xticks([0,1],['M1',f'M{i+2}'])
        axb.tick_params(labelsize=11,direction='in')
        axb.grid(axis='y',linestyle='--',alpha=.25,zorder=0)
        if i==0: axb.set_ylabel('Pooled test MAE (mm)',fontsize=11)
        ymax=max(float((m+s).max()),.001)
        by=ymax*1.11
        tick=ymax*.03
        axb.plot([0,0,1,1],[by-tick,by,by,by-tick],c='#656c75',lw=1.0)
        pct=(m[1]/m[0]-1)*100
        axb.text(.5,by+tick*.4,f'{pct:+.2f}%',ha='center',va='bottom',
                 fontsize=12,fontweight='bold')
        axb.set_ylim(0,ymax*1.35)
        axb.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axb.text(.5,-.24,f'({chr(97+i)}) M1 vs. M{i+2}',transform=axb.transAxes,
                 ha='center',fontsize=12)
        for ax in (axt,axb):
            for spine in ax.spines.values():spine.set_linewidth(1.0)
    handles=[Line2D([0],[0],marker='o',color='w',label=weld,
             markerfacecolor=CASE_COLORS[weld],markersize=7)
             for weld in CASE_COLORS]
    fig.legend(handles=handles,loc='upper right',ncol=4,bbox_to_anchor=(.990,.930),
               frameon=False,fontsize=10,columnspacing=1.25)
    OUT.mkdir(parents=True,exist_ok=True)
    out=OUT/f'Pairwise_Ablation_{target}_300dpi'
    fig.savefig(out.with_suffix('.png'),dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(out.with_suffix('.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print('Saved',out.with_suffix('.png'))
    return per_case,per_seed


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--target',choices=['D','W','both'],default='both')
    args=p.parse_args()
    available={f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        'font.family':next((x for x in ['Times New Roman','Liberation Serif','DejaVu Serif'] if x in available),'DejaVu Serif'),
        'axes.unicode_minus':False,'pdf.fonttype':42,'ps.fonttype':42,
    })
    df=load_predictions()
    targets=['D','W'] if args.target=='both' else [args.target]
    for target in targets:
        _,seed=plot_target(df,target)
        print(f'Pooled MAE results for {target}:')
        print(seed.groupby('variant',sort=False).mae.mean().round(6).to_string())

if __name__=='__main__':
    main()

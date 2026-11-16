# -*- coding: utf-8 -*-
"""Draw tested PI-NGreyKAN vs true values and optionally ALL available baselines.
No training; read only saved CSVs. Average independent seed forecasts per point.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
RESULT=ROOT/'results'
FIG=ROOT/'figures'
FIG.mkdir(parents=True,exist_ok=True)
STYLE={
    'PI-NGreyKAN':('#C0392B','-',2.2,'s'),
    'LSTM-Lite':('#2874A6','--',1.6,'^'),
    'Holt':('#117A65','-.',1.5,None),
    'GRU':('#884EA0',':',1.6,'d'),
    'GM(1,N)':('#BA7D14','--',1.4,'v'),
    'Lag-GM(1,N)':('#626567',':',1.4,None),
    'SVR':('#1F618D','--',1.5,None),
    'XGBoost':('#229954','-.',1.5,None),
    'CatBoost':('#7D3C98',':',1.4,None),
    'LSTM':('#1B4F72','--',1.5,None),
    'RandomForest':('#7B7D7D',':',1.5,None),
    'ExtraTrees':('#B9770E','-.',1.5,None),
}
DEFAULT=['PI-NGreyKAN','LSTM-Lite','Holt','GRU','GM(1,N)','Lag-GM(1,N)']


def load_predictions():
    path=RESULT/'selected_test_predictions.csv'
    if not path.exists():
        old=RESULT/'test_predictions.csv'
        if not old.exists():raise FileNotFoundError('Run selected model first: '+str(path))
        df=pd.read_csv(old,encoding='utf-8-sig');df=df[df.model=='ThermalRobust_2.0'].copy()
        df['model']='PI-NGreyKAN'
    else:df=pd.read_csv(path,encoding='utf-8-sig')
    other=RESULT/'baseline_test_predictions.csv'
    if other.exists():df=pd.concat([df,pd.read_csv(other,encoding='utf-8-sig')],ignore_index=True)
    required=['model','seed','weld','target','test_row','original_index','actual','prediction']
    missing=[c for c in required if c not in df.columns]
    if missing:raise ValueError('Missing columns: '+str(missing))
    if df.duplicated(['model','seed','weld','target','test_row']).any():
        raise ValueError('Duplicate predictions for a single model / seed / test point')
    if not np.isfinite(df[['actual','prediction']].to_numpy()).all():
        raise ValueError('Non-finite ground truth/predictions')
    truth=df.groupby(['weld','target','test_row'])['actual'].agg(['min','max'])
    if ((truth['max']-truth['min']).abs()>1e-4).any():
        raise ValueError('Mismatched ground truth between selected and baseline prediction CSVs')
    return df


def mean_dataframe(df):
    avg=df.groupby(['model','weld','target','test_row','original_index'],as_index=False).agg(actual=('actual','mean'),prediction=('prediction','mean'))
    avg.to_csv(RESULT/'mean_predictions_for_plot_all_models.csv',index=False,encoding='utf-8-sig')
    return avg


def metrics(df):
    rows=[]
    for (name,target),g in df.groupby(['model','target']):
        vals=[]
        for seed,gg in g.groupby('seed'):
            y=gg.actual.to_numpy();p=gg.prediction.to_numpy();diff=y-p
            vals.append((np.abs(diff).mean(),np.sqrt(np.mean(diff**2)),100*np.mean(np.abs(diff)/np.maximum(np.abs(y),1e-8))))
        av=np.mean(vals,axis=0);sd=np.std(vals,axis=0,ddof=1) if len(vals)>1 else [0]*3
        rows.append(dict(model=name,target=target,n=int(len(g)/len(vals)),seeds=len(vals),MAE=av[0],RMSE=av[1],MAPE=av[2],MAE_sd=sd[0],RMSE_sd=sd[1],MAPE_sd=sd[2]))
    out=pd.DataFrame(rows)
    out.to_csv(RESULT/'all_model_metrics.csv',index=False,encoding='utf-8-sig')
    return out


def curve(ax,df,weld,target,selected):
    sub=df[(df.weld==weld)&(df.target==target)]
    actual=sub.groupby('test_row')['actual'].mean().sort_index()
    x=actual.index.to_numpy()
    ax.plot(x,actual.to_numpy(),color='#111111',linewidth=2.4,marker='o',markersize=3.5,label='Actual',zorder=10)
    for name in selected:
        s=sub[sub.model==name].sort_values('test_row')
        if len(s)==0:continue
        if len(s)!=len(actual) or not np.array_equal(s.test_row.to_numpy(),x):
            raise ValueError('Missing or unsorted test samples for '+weld+' '+target+' '+name)
        col,ls,lw,mark=STYLE.get(name,(None,'--',1.4,None))
        ax.plot(x,s.prediction.to_numpy(),color=col,ls=ls,lw=lw,marker=mark,markersize=2.4,label=name)
    ax.set_title(weld+' - '+('Penetration depth' if target=='D' else 'Weld width'),fontweight='bold',fontsize=11)
    ax.set_xlabel('Test sample index');ax.set_ylabel('mm');ax.grid(True,linestyle=':',alpha=.30)
    ax.tick_params(direction='in')
    for sp in ax.spines.values():sp.set_visible(True)


def plot_grid(df,selected,basename):
    fig,axes=plt.subplots(4,2,figsize=(14,14))
    for i in range(1,5):
        for j,target in enumerate(['D','W']):
            curve(axes[i-1,j],df,'W%02d'%i,target,selected)
    hand,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(hand,labels,ncol=min(7,len(labels)),loc='lower center',frameon=True,fontsize=9,bbox_to_anchor=(.5,.012))
    fig.tight_layout(rect=[0,.05,1,1],h_pad=1.8,w_pad=1.4)
    dst=FIG/(basename+'_300dpi.png')
    fig.savefig(dst,dpi=300,bbox_inches='tight');plt.close(fig)
    print('figure',dst)


def plot_individual(df,selected):
    for i in range(1,5):
        weld='W%02d'%i
        fig,axes=plt.subplots(1,2,figsize=(12,4.2))
        for j,target in enumerate(['D','W']):curve(axes[j],df,weld,target,selected)
        h,l=axes[0].get_legend_handles_labels()
        fig.legend(h,l,ncol=min(7,len(l)),loc='lower center',fontsize=8,frameon=True,bbox_to_anchor=(.5,-.02))
        fig.tight_layout(rect=[0,.04,1,1])
        dst=FIG/(weld+'_actual_vs_models_300dpi.png')
        fig.savefig(dst,dpi=300,bbox_inches='tight');plt.close(fig)
        print('figure',dst)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+',default=DEFAULT)
    args=parser.parse_args()
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.linewidth':1.0,'figure.dpi':120})
    data=load_predictions();agg=mean_dataframe(data)
    metric=metrics(data)
    available=set(agg.model.unique())
    selected=[m for m in args.models if m in available]
    if 'PI-NGreyKAN' not in available:raise ValueError('Selected main model predictions missing')
    if not selected:raise ValueError('None of the chosen models is available in the prediction files')
    missing=set(args.models)-available
    if missing:print('WARNING: models not yet run, omitted from graphs:',sorted(missing))
    plot_grid(agg,['PI-NGreyKAN'],'Four_welds_actual_vs_PI_selected')
    plot_grid(agg,selected,'Four_welds_actual_vs_comparison')
    plot_individual(agg,selected)
    print(metric[metric.model.isin(selected)].to_string(index=False))

if __name__=='__main__':main()

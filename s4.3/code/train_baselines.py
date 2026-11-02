# -*- coding: utf-8 -*-
"""Fair, compact baselines for fixed four chronological welding datasets.
Train on each weld's initial 64% for validation; refit on initial 80%; test last 20%.
No test TARGET used to fit/scale/select models, nor hidden teacher forcing.
Input-powered baselines use current and past sensors; univariate TS baselines
forecast without test target access. Classic models assume compressed time index.

Usage:
 python code/train_baselines.py --models LSTM-Lite GRU 'GM(1,N)' 'Lag-GM(1,N)' Holt
 python code/train_baselines.py --models ALL
"""
import os
for env in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(env, '1')
import argparse, warnings, random, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as TF
import base_model as base

try:torch.set_num_threads(1)
except:pass
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results'
OUT.mkdir(exist_ok=True,parents=True)
SEEDS=[2026,2027,2028]
MODELS=['LSTM-Lite','LSTM','GRU','Holt','Holt-Winters','SES','ARIMA',
        'GM(1,N)','Lag-GM(1,N)','DGM(1,N)',
        'SVR','XGBoost','CatBoost','RandomForest','ExtraTrees','MLP','Ridge','PLS','KRR']


def base_data(stage):
    p=base.Prep(stage)
    cuts=p.cuts
    x_train=np.concatenate([x[:cut] for x,cut in zip(p.z,cuts)])
    y_train=np.concatenate([y[:cut] for y,cut in zip(p.y,cuts)])
    lo=[d['k64'] if stage=='valid' else d['k80'] for d in base.RAW]
    hi=[d['k80'] if stage=='valid' else d['n'] for d in base.RAW]
    return p,x_train,y_train,lo,hi


def validation_loss(prep,outputs):
    # train-only normalization; balanced tasks
    errs=[]
    for wi,d in enumerate(base.RAW):
        a=d['k64']; b=d['k80']
        e=(np.abs(outputs[wi][a:b]-d['y'][a:b])/prep.ystd)*np.array([.6,.4])
        errs.append(e.sum(axis=1))
    return float(np.mean(np.concatenate(errs)))


def predict_for_test(outputs):
    return [out[d['k80']:d['n']] for d,out in zip(base.RAW,outputs)]


def add_records(name, seed, outputs):
    rows=[]
    for wi,d in enumerate(base.RAW):
        test=outputs[wi][d['k80']:d['n']]
        if not np.isfinite(test).all() or test.shape!=(d['n']-d['k80'],2):
            raise ValueError('%s %s invalid prediction shape/values'%(name,d['id']))
        for j,tgt in enumerate(['D','W']):
            for t,value in enumerate(test[:,j]):
                rows.append(dict(model=name,seed=seed,weld=d['id'],target=tgt,test_row=t+1,
                                 original_index=int(d['ix'][d['k80']+t]),
                                 actual=float(d['y'][d['k80']+t,j]), prediction=float(value)))
    return rows


def safe_xgb():
    from xgboost import XGBRegressor
    return XGBRegressor

def safe_catboost():
    from catboost import CatBoostRegressor
    return CatBoostRegressor


def ml_grid(name):
    grids={
        'SVR':[{'C':a,'epsilon':e} for a in [1,10] for e in [.05,.15]],
        'KRR':[{'alpha':a,'gamma':g} for a in [.1,1] for g in [.05,.2]],
        'Ridge':[{'alpha':a} for a in [.1,10,100]],
        'PLS':[{'n':k} for k in [1,2,3,4]],
        'RandomForest':[{'leaf':l,'depth':d} for l in [2,5] for d in [5,None]],
        'ExtraTrees':[{'leaf':l,'depth':d} for l in [2,5] for d in [5,None]],
        'XGBoost':[{'n':n,'depth':d} for n in [100,220] for d in [2,4]],
        'CatBoost':[{'n':n,'depth':d} for n in [120,240] for d in [3,5]],
        'MLP':[{'hidden':h,'alpha':a} for h in [16,32] for a in [.01,.1]],
    }
    return grids[name]


def ml_model(name,hp,seed):
    if name=='SVR':return MultiOutputRegressor(SVR(C=hp['C'],epsilon=hp['epsilon'],gamma='scale'))
    if name=='KRR':return KernelRidge(alpha=hp['alpha'],kernel='rbf',gamma=hp['gamma'])
    if name=='Ridge':return Ridge(alpha=hp['alpha'])
    if name=='PLS':return PLSRegression(n_components=hp['n'])
    if name=='RandomForest':return RandomForestRegressor(n_estimators=120,min_samples_leaf=hp['leaf'],max_depth=hp['depth'],random_state=seed,n_jobs=1)
    if name=='ExtraTrees':return ExtraTreesRegressor(n_estimators=120,min_samples_leaf=hp['leaf'],max_depth=hp['depth'],random_state=seed,n_jobs=1)
    if name=='XGBoost':
        XGBRegressor=safe_xgb()
        return MultiOutputRegressor(XGBRegressor(n_estimators=hp['n'],max_depth=hp['depth'],learning_rate=.05,reg_lambda=3,subsample=.9,n_jobs=1,random_state=seed,objective='reg:squarederror'))
    if name=='CatBoost':
        CatBoostRegressor=safe_catboost()
        return CatBoostRegressor(iterations=hp['n'],depth=hp['depth'],learning_rate=.05,l2_leaf_reg=5,verbose=False,thread_count=1,random_seed=seed,loss_function='MultiRMSE',allow_writing_files=False)
    if name=='MLP':return MLPRegressor(hidden_layer_sizes=(hp['hidden'],),activation='tanh',alpha=hp['alpha'],max_iter=300,learning_rate_init=.003,random_state=seed)
    raise ValueError(name)


def run_ml(name):
    seeds=SEEDS if name in ('XGBoost','CatBoost','RandomForest','ExtraTrees','MLP') else [SEEDS[0]]
    p, xx, yy, _, _=base_data('valid')
    xvalid=np.concatenate([p.z[i][d['k64']:d['k80']] for i,d in enumerate(base.RAW)])
    yvalid=np.concatenate([p.y[i][d['k64']:d['k80']] for i,d in enumerate(base.RAW)])
    choices=[]
    for hp in ml_grid(name):
        vals=[]
        for seed in seeds:
            model=ml_model(name,hp,seed)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(xx,yy)
            pr=np.asarray(model.predict(xvalid)).reshape(-1,2)
            vals.append(float(np.mean((np.abs(pr-yvalid)/np.array([1.,1.])*np.array([.6,.4])).sum(1))))
        choices.append(np.mean(vals))
    idx=int(np.argmin(choices)); hp=ml_grid(name)[idx]
    prep, xx, yy, _, _=base_data('final')
    results=[]
    for seed in seeds:
        model=ml_model(name,hp,seed)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');model.fit(xx,yy)
        outputs=[np.asarray(model.predict(z)).reshape(-1,2)*prep.ystd+prep.ymu for z in prep.z]
        results.extend(add_records(name,seed,outputs))
    print('ML',name,'best',hp,'validation',round(choices[idx],6),flush=True)
    return results,dict(model=name,config=json.dumps(hp),validation=float(choices[idx]),seeds=len(seeds))


def rnn_tensor(z,window):
    allseq=[]
    for t in range(len(z)):
        a=z[max(0,t-window+1):t+1]
        if len(a)<window:a=np.concatenate([np.repeat(a[:1],window-len(a),axis=0),a],axis=0)
        allseq.append(a)
    return torch.tensor(np.stack(allseq),dtype=torch.float32)

class CausalRNN(nn.Module):
    def __init__(self,kind,hidden):
        super().__init__()
        rnn=nn.GRU if kind=='GRU' else nn.LSTM
        self.rnn=rnn(5,hidden,batch_first=True)
        self.head=nn.Linear(hidden,2)
    def forward(self,x):
        return self.head(self.rnn(x)[0][:,-1,:])


def rnn_once(name,stage,seed,hp):
    base.random_seed(seed)
    p=base.Prep(stage)
    zz=[rnn_tensor(z,hp['window']) for z in p.z]
    xs=torch.cat([zz[i][:c] for i,c in enumerate(p.cuts)],dim=0)
    ys=torch.tensor(np.concatenate([d[:c] for d,c in zip(p.y,p.cuts)]),dtype=torch.float32)
    net=CausalRNN(name,hp['hidden'])
    opt=torch.optim.AdamW(net.parameters(),lr=.004 if name in ('GRU','LSTM-Lite') else .003,weight_decay=.012)
    for ep in range(hp['epochs']):
        perm=torch.randperm(len(xs))
        net.train()
        for chunk in perm.split(32):
            yhat=net(xs[chunk]); er=TF.smooth_l1_loss(yhat,ys[chunk],beta=.5,reduction='none')
            loss=(er*torch.tensor([.6,.4])).sum(dim=1).mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),3.);opt.step()
    net.eval()
    with torch.no_grad():outputs=[net(z).numpy()*p.ystd+p.ymu for z in zz]
    return p,outputs


def run_rnn(name):
    if name=='LSTM-Lite':
        grid=[dict(window=6,hidden=4,epochs=16),dict(window=8,hidden=8,epochs=24)]
    elif name=='LSTM':
        grid=[dict(window=8,hidden=16,epochs=75),dict(window=12,hidden=24,epochs=50)]
    else:
        grid=[dict(window=8,hidden=16,epochs=75),dict(window=12,hidden=24,epochs=50)]
    vals=[]
    for hp in grid:
        vals.append(float(np.mean([validation_loss(*rnn_once(name,'valid',seed,hp)) for seed in SEEDS])))
    k=int(np.argmin(vals));hp=grid[k]
    records=[]
    for seed in SEEDS:
        prep,output=rnn_once(name,'final',seed,hp)
        records.extend(add_records(name,seed,output))
    print('RNN',name,'best',hp,'validation',round(vals[k],6),flush=True)
    return records,dict(model=name,config=json.dumps(hp),validation=float(vals[k]),seeds=len(SEEDS))


def run_univariate_model(name):
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import Holt,ExponentialSmoothing,SimpleExpSmoothing
    def forecast(train,horizon,hp):
        # Constant or near-constant target segments are not identifiable for ARIMA.
        # Exact-constant last-value forecast uses training labels ONLY.
        if float(np.std(train))<1e-6:
            return np.full(horizon,float(train[-1]),dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if name=='ARIMA':
                order=tuple(hp['order']);trend='c' if order[1]==0 else 'n'
                f=ARIMA(train,order=order,trend=trend).fit(method='innovations_mle')
            elif name=='Holt':
                f=Holt(train,damped_trend=True,initialization_method='estimated').fit(optimized=True)
            elif name=='SES':
                f=SimpleExpSmoothing(train,initialization_method='estimated').fit(optimized=True)
            else:
                f=ExponentialSmoothing(train,trend='add',damped_trend=True,
                      seasonal='add',seasonal_periods=hp['period'],initialization_method='estimated').fit(optimized=True,use_brute=False)
            out=np.asarray(f.forecast(horizon),dtype=float)
            if len(out)!=horizon or not np.all(np.isfinite(out)):
                raise ValueError(name+' forecast invalid')
            return out
    hp_grid=( [{'order':[0,0,0]},{'order':[1,0,0]},{'order':[1,1,0]}] if name=='ARIMA' else
              [{'period':6},{'period':12}] if name=='Holt-Winters' else [{}])
    prep=base.Prep('valid');scores=[]
    for hp in hp_grid:
        score=[]
        for wi,d in enumerate(base.RAW):
            a,b=d['k64'],d['k80']
            try:
                p=np.column_stack([forecast(d['y'][:a,j],b-a,hp) for j in (0,1)])
                score.extend(((np.abs(p-d['y'][a:b])/prep.ystd)*np.array([.6,.4])).sum(1).tolist())
            except Exception:
                score.extend([float('inf')]*(b-a))
        scores.append(float(np.mean(score)))
    k=int(np.argmin(scores));hp=hp_grid[k]
    if not np.isfinite(scores[k]):
        raise RuntimeError('Cannot fit '+name+' on validation')
    outputs=[]
    for d in base.RAW:
        cut=d['k80']
        p=np.column_stack([forecast(d['y'][:cut,j],d['n']-cut,hp) for j in (0,1)])
        output=np.full((d['n'],2),np.nan)
        output[cut:]=p
        outputs.append(output)
    records=add_records(name,2026,outputs)
    print('TS',name,'best',hp,'validation',round(scores[k],6),flush=True)
    return records,dict(model=name,config=json.dumps(hp),validation=float(scores[k]),seeds=1)


def grey_predict(stage,name,lam,lag):
    # Engineering GM(1,N) approximation (not identical to every published formulation)
    prep=base.Prep(stage)
    outputs=[]
    for wi,d in enumerate(base.RAW):
        cut=prep.cuts[wi];z=np.asarray(prep.z[wi],float)
        y=np.asarray(d['y'],float)
        X1=np.cumsum(z,axis=0)
        XL=np.vstack([np.repeat(X1[:1],lag,axis=0),X1[:-lag]]) if lag else X1
        out=np.full((len(y),2),np.nan)
        for j in range(2):
            med=np.median(y[:cut,j]);scale=max(y[:cut,j].std(),.05)
            target=(y[:cut,j]-med)/scale
            y1=np.cumsum(target)
            if name=='DGM(1,N)':
                # Discrete accumulated-response regression with exogenous driving
                xx=np.column_stack([y1[:-1],XL[1:cut],np.ones(cut-1)])
                pp=np.column_stack([XL,np.ones(len(y))])
                sc=np.maximum(xx.std(0),.1);sc[-1]=1
                coef=Ridge(alpha=lam,fit_intercept=False).fit(xx/sc,y1[1:]).coef_/sc
                alpha=float(np.clip(coef[0],.1,.99));b=coef[1:]
                prev=float(y1[-1]);result=[]
                for k in range(cut,len(y)):
                    ynew=alpha*prev+pp[k]@b
                    result.append(med+scale*(ynew-prev));prev=ynew
            else:
                xx=np.column_stack([-.5*(y1[1:]+y1[:-1]),XL[1:cut],np.ones(cut-1)])
                sc=np.maximum(xx.std(0),.1);sc[-1]=1
                coef=Ridge(alpha=lam,fit_intercept=False).fit(xx/sc,target[1:]).coef_/sc
                a=float(np.clip(coef[0],.001,.5));b=coef[1:]
                prev=float(y1[-1]);result=[]
                for k in range(cut,len(y)):
                    drive=XL[k]@b[:-1]+b[-1]
                    pred=(drive-a*prev)/(1+.5*a)
                    prev+=pred
                    result.append(med+scale*pred)
            out[cut:,j]=np.clip(result,0,20)
        outputs.append(out)
    return prep,outputs


def run_grey(name):
    grid=[dict(lam=x,lag=lag) for x in [1.,10.,100.] for lag in ([1,2] if name=='Lag-GM(1,N)' else [0])]
    vals=[]
    for hp in grid:
        prep,outs=grey_predict('valid',name,**hp)
        vals.append(validation_loss(prep,outs))
    k=int(np.argmin(vals));hp=grid[k]
    _,outs=grey_predict('final',name,**hp)
    print('GREY',name,'best',hp,'validation',round(vals[k],6),flush=True)
    return add_records(name,2026,outs),dict(model=name,config=json.dumps(hp),validation=float(vals[k]),seeds=1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+',default=['LSTM-Lite','Holt','GRU','GM(1,N)','Lag-GM(1,N)','SVR','XGBoost'])
    parser.add_argument('--output',default='baseline_test_predictions.csv')
    args=parser.parse_args()
    wanted=MODELS if args.models==['ALL'] else args.models
    bad=set(wanted)-set(MODELS)
    if bad:raise ValueError('Unknown model(s): '+str(sorted(bad)))
    rows=[];settings=[]
    # Always create new output on start; do not mix with an old set of predictions
    for name in wanted:
        tic=time.time()
        if name in ('GRU','LSTM','LSTM-Lite'):
            predicted,hp=run_rnn(name)
        elif name in ('ARIMA','Holt','Holt-Winters','SES'):
            predicted,hp=run_univariate_model(name)
        elif name in ('GM(1,N)','Lag-GM(1,N)','DGM(1,N)'):
            predicted,hp=run_grey(name)
        else:
            predicted,hp=run_ml(name)
        rows.extend(predicted);settings.append(hp)
        pd.DataFrame(rows).to_csv(OUT/args.output,index=False,encoding='utf-8-sig')
        pd.DataFrame(settings).to_csv(OUT/'baseline_hyperparameters.csv',index=False,encoding='utf-8-sig')
        print('DONE',name,'elapsed_sec',round(time.time()-tic,1),flush=True)
    print('Completed',len(wanted),'models: ',wanted,flush=True)

if __name__=='__main__':main()

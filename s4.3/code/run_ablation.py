# -*- coding: utf-8 -*-
"""PI-NGreyKAN reproducible single-component ablation study.

Put this script in <project>/code/run_ablation.py, then run:
    python code/run_ablation.py

The experiment uses exactly the data and model from the 'PI_NGreyKAN_完整复现项目'
project. Same 4 cases, 80/20 ordered split, 3 seeds. Selection uses the first
80% of each case only. No target-label access in test optimization.

Five configurations (including complete):
  M1: complete model with physics lambda=2
  M2: replace KAN drive by a linear 5->2 drive
  M3: disable linear grey development a_{j,t}v_j
  M4: learn a constant a_j for each target instead of adaptive a_{j,t}
  M5: set standardized thermal feature Q to zero in train/test and physics pairs

Default training uses per-variant best epochs selected on chronological 64/16
internal train/validation and refits with all 80% training samples. Each seed
is evaluated on pooled 108 test points for each target (D/W).
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import sys
import copy
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))
import base_model as base
import run_simple_physics as physics

SEEDS = [2026, 2027, 2028]
OUTPUT = ROOT / 'results' / 'ablation'
OUTPUT.mkdir(parents=True, exist_ok=True)

# DO NOT change if comparing with the originally selected PI-NGreyKAN
PHYS_LAMBDA = 2.0
VARIANTS = [
    'M1: PI-NGreyKAN',
    'M2: without KAN',
    'M3: without Grey Development',
    'M4: without Adaptive Development',
    'M5: without Thermal Feature',
]

class AblationGreyKAN(base.GreyKAN):
    """Same discrete dynamics and output heads, with one specified module disabled."""
    def __init__(self, variant):
        super().__init__((5, 20, 2), 7)
        self.variant = variant
        if variant == 'M2: without KAN':
            self.kan = nn.ModuleList([nn.Linear(5, 2)])
            nn.init.xavier_uniform_(self.kan[0].weight, gain=.28)
            nn.init.zeros_(self.kan[0].bias)
        if variant == 'M4: without Adaptive Development':
            # The complete model starts with a=0.12+2*sigmoid(-0.37).
            # Learn a constant separately for D and W; don't remove the linear term.
            self.constant_a_raw = nn.Parameter(torch.full((2,), -.37))

    def compute_a(self, z):
        if self.variant == 'M3: without Grey Development':
            return torch.zeros_like(z[..., :2])
        if self.variant == 'M4: without Adaptive Development':
            const = .12 + 2 * torch.sigmoid(self.constant_a_raw)
            return const.expand_as(z[..., :2])
        return super().compute_a(z)


def masked_no_q(z, variant):
    if variant == 'M5: without Thermal Feature':
        z = z.clone()
        z[..., 4] = 0.0  # zero standardized Q = training-mean reference
    return z


def select_and_train(seed, stage, variant, epochs=None, max_validation_epochs=125):
    base.random_seed(seed)
    prep = base.Prep(stage)
    z, y, tr, va = prep.tensors()
    z = masked_no_q(z, variant)

    model = AblationGreyKAN(variant)
    model.phys_y_scale = torch.tensor(prep.ystd, dtype=torch.float32)
    phys = physics.PhysPairs(prep)
    if variant == 'M5: without Thermal Feature':
        ids, zn, zf = phys.rob
        phys.rob = (ids, masked_no_q(zn, variant), masked_no_q(zf, variant))

    opt = torch.optim.AdamW(model.parameters(), lr=.012, weight_decay=.001)
    best = float('inf')
    best_epoch = 5
    best_state = None
    n_epochs = max_validation_epochs if stage == 'valid' else int(epochs)
    weights = torch.tensor([.6, .4])

    for ep in range(1, n_epochs + 1):
        lr = .012 if ep <= 65 else .006 if ep <= 105 else .003
        for group in opt.param_groups:
            group['lr'] = lr

        model.train()
        opt.zero_grad()
        pr = model(z)
        loss_rec = F.smooth_l1_loss(pr, y, beta=.6, reduction='none')
        loss = (loss_rec * tr * weights).sum() / tr.sum()

        prior = physics.prior_loss(model, pr, phys.rob, 'stability')
        loss = loss + PHYS_LAMBDA * prior

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 4.)
        opt.step()

        if stage == 'valid' and (ep == 1 or ep % 5 == 0):
            model.eval()
            with torch.no_grad():
                prediction = model(z)
                score = float(((prediction - y).abs() * va * weights).sum() / va.sum())
            if score < best - 1e-5:
                best = score
                best_epoch = ep
                best_state = copy.deepcopy(model.state_dict())
            if ep - best_epoch >= 35:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(z).numpy()
    predictions = [(out[i] * prep.ystd + prep.ymu)[:d['n']] for i, d in enumerate(base.RAW)]
    return best_epoch, best, predictions


def error_metrics(y, prediction):
    err = y - prediction
    return dict(
        MAE=float(np.mean(np.abs(err))),
        RMSE=float(np.sqrt(np.mean(err * err))),
        MAPE=float(100 * np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-8))),
    )


def pooled_summary(pred):
    rows=[]
    seedwise=[]
    casewise=[]
    for (variant, target), df in pred.groupby(['variant', 'target'], sort=False):
        seed_errors=[]
        for seed, g in df.groupby('seed', sort=True):
            score=error_metrics(g.actual.to_numpy(), g.prediction.to_numpy())
            seedwise.append(dict(variant=variant, seed=seed, target=target, n=len(g), **score))
            seed_errors.append(score)
        rows.append(dict(variant=variant, target=target,
            MAE=np.mean([m['MAE'] for m in seed_errors]),
            RMSE=np.mean([m['RMSE'] for m in seed_errors]),
            MAPE=np.mean([m['MAPE'] for m in seed_errors]),
            MAE_sd=np.std([m['MAE'] for m in seed_errors], ddof=1),
            RMSE_sd=np.std([m['RMSE'] for m in seed_errors], ddof=1),
            MAPE_sd=np.std([m['MAPE'] for m in seed_errors], ddof=1)))
    for (variant, weld, target), df in pred.groupby(['variant', 'weld', 'target'], sort=False):
        vals=[]
        for seed, g in df.groupby('seed'):
            vals.append(error_metrics(g.actual.to_numpy(),g.prediction.to_numpy()))
        casewise.append(dict(variant=variant, weld=weld, target=target, n=len(g),
                             **{m:np.mean([v[m] for v in vals]) for m in ('MAE','RMSE','MAPE')}))
    overall=pd.DataFrame(rows)
    order={x:i for i,x in enumerate(VARIANTS)}
    overall['order']=overall.variant.map(order)
    overall=overall.sort_values(['order','target']).drop(columns='order')
    wide=overall.pivot(index='variant',columns='target',values=['MAE','RMSE','MAPE'])
    wide=wide.reindex(VARIANTS)
    wide.columns=[f'{tar}_{metric}' for metric,tar in wide.columns]
    wide=wide.reset_index()
    return overall,wide,pd.DataFrame(seedwise),pd.DataFrame(casewise)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    p.add_argument('--max-val-epochs',type=int,default=125)
    p.add_argument('--only',nargs='+',default=VARIANTS)
    p.add_argument('--resume',action='store_true',help='skip completed validation/test seed variants')
    args=p.parse_args()
    variants=args.only
    if any(v not in VARIANTS for v in variants):
        raise ValueError('Unknown variant in --only')
    logs_path=OUTPUT/'validation_log.csv'
    pred_path=OUTPUT/'ablation_test_predictions.csv'
    logs=pd.read_csv(logs_path).to_dict('records') if args.resume and logs_path.is_file() else []
    preds=pd.read_csv(pred_path).to_dict('records') if args.resume and pred_path.is_file() else []
    for variant in variants:
        for seed in args.seeds:
            if any(row['variant']==variant and int(row['seed'])==seed for row in logs) and any(
                row['variant']==variant and int(row['seed'])==seed for row in preds):
                continue
            tm=time.time()
            epoch,valid,_=select_and_train(seed,'valid',variant,max_validation_epochs=args.max_val_epochs)
            _,_,pred=select_and_train(seed,'final',variant,epochs=epoch)
            logs.append(dict(variant=variant,seed=seed,selected_epoch=epoch,validation_score=valid))
            for wi,weld in enumerate(base.RAW):
                for t in range(weld['k80'],weld['n']):
                    for j,target in enumerate(('D','W')):
                        preds.append(dict(variant=variant,seed=seed,weld=weld['id'],target=target,
                                          test_row=t-weld['k80']+1,original_index=int(weld['ix'][t]),
                                          actual=float(weld['y'][t,j]),prediction=float(pred[wi][t,j])))
            pd.DataFrame(logs).to_csv(logs_path,index=False,encoding='utf-8-sig')
            pd.DataFrame(preds).to_csv(pred_path,index=False,encoding='utf-8-sig')
            print(f'{variant:30s} seed={seed} ep={epoch} val={valid:.6f} seconds={time.time()-tm:.1f}',flush=True)
    pred_df=pd.DataFrame(preds)
    # Only summarize complete combos (all requested seeds per variant)
    pred_df=pred_df[pred_df.variant.isin(variants)&pred_df.seed.isin(args.seeds)]
    overall,wide,seedwise,casewise=pooled_summary(pred_df)
    overall.to_csv(OUTPUT/'ablation_metrics_long.csv',index=False,encoding='utf-8-sig')
    wide.to_csv(OUTPUT/'ablation_table.csv',index=False,encoding='utf-8-sig')
    seedwise.to_csv(OUTPUT/'ablation_by_seed.csv',index=False,encoding='utf-8-sig')
    casewise.to_csv(OUTPUT/'ablation_by_case.csv',index=False,encoding='utf-8-sig')
    print('\nPooled over four cases / mean across seeds:\n',wide.round(5).to_string(index=False),flush=True)

if __name__=='__main__':
    main()

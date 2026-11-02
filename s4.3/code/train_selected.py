# -*- coding: utf-8 -*-
"""Reproduce SELECTED 5->20->2 PI-NGreyKAN with local thermal-input consistency.
Datasets and masks are fixed. No test-quality labels enter optimization.
Uses exactly base_model.py / run_simple_physics.py training implementation.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import run_simple_physics as runphys
import base_model as base

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
MODEL = 'PI-NGreyKAN'
VARIANT = 'ThermalRobust_2.0'
SEEDS = [2026, 2027, 2028]


def run(seeds=None):
    if seeds is None:
        seeds = SEEDS
    selected = runphys.CONFIGS[VARIANT]
    records = []
    epochs_log = []
    for seed in seeds:
        ep, valid, _, _ = runphys.fit(seed, 'valid', selected)
        _, _, full_predict, _ = runphys.fit(seed, 'final', selected, epochs=ep)
        epochs_log.append(dict(model=MODEL, seed=seed, selected_epoch=ep, validation_score=valid,
                               layers='5-20-2', phys_lambda=2.0, physics='LocalThermalInputConsistency'))
        for wi, weld in enumerate(base.RAW):
            for t in range(weld['k80'], weld['n']):
                for j, tar in enumerate(('D', 'W')):
                    records.append(dict(model=MODEL, seed=seed, weld=weld['id'], target=tar,
                                        test_row=t-weld['k80']+1, original_index=int(weld['ix'][t]),
                                        actual=float(weld['y'][t, j]), prediction=float(full_predict[wi][t,j])))
        print('SELECTED', seed, 'epoch', ep, 'validation', round(valid,6), flush=True)
    result=pd.DataFrame(records)
    result.to_csv(RESULTS/'selected_test_predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(epochs_log).to_csv(RESULTS/'selected_training_log.csv',index=False,encoding='utf-8-sig')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    args=parser.parse_args()
    run(args.seeds)

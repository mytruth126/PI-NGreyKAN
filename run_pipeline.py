# -*- coding: utf-8 -*-
"""Portable launcher for Spyder or cmd.exe; uses current Python interpreter.

Examples:
  python run_pipeline.py --step plot
  python run_pipeline.py --step selected
  python run_pipeline.py --step baselines --models LSTM-Lite GRU Holt 'GM(1,N)'
  python run_pipeline.py --step all
"""
import sys,subprocess,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CODE=ROOT/'code'

def call(file,args=None):
    cmd=[sys.executable,'-u',str(CODE/file)]+list(args or [])
    print('EXEC:', ' '.join(cmd),flush=True)
    # Direct console output (including full inner Python traceback) in Spyder/terminal
    subprocess.run(cmd,cwd=str(ROOT),check=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--step',choices=['prepare','selected','baselines','plot','all'],default='all')
    parser.add_argument('--models',nargs='+',default=['LSTM-Lite','Holt','GRU','GM(1,N)','Lag-GM(1,N)','SVR','XGBoost'])
    a=parser.parse_args()
    if a.step in ['prepare']:
        call('recreate_data.py');return
    if a.step in ('selected','all'):call('train_selected.py')
    if a.step in ('baselines','all'):call('train_baselines.py',['--models']+a.models)
    if a.step in ('plot','all'):call('plot_predictions.py')
    print('Pipeline step completed:',a.step,flush=True)

if __name__=='__main__':main()

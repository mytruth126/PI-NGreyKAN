# -*- coding: utf-8 -*-
"""Unified entry: python run_ablation_and_plot.py --step plot|train|all."""
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def run(*args):
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--step',choices=['plot','train','all'],default='plot',help='plot uses included finished 3-seed results and needs no training')
    a=p.parse_args()
    if a.step in ('train','all'):
        run(ROOT/'code'/'run_ablation.py')
    if a.step in ('plot','all'):
        run(ROOT/'code'/'plot_ablation_pairwise.py')
if __name__=='__main__':main()

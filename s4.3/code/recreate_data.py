# -*- coding: utf-8 -*-
"""Recreate the four FIXED evaluation datasets from original W01-W04.
Do NOT resample: original selection used target values; retaining fixed indices
is essential for exact reproducibility. Source order and test split preserved.
"""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'data'
RAW=ROOT/'raw_data'
INDEX=DATA/'fixed_retained_indices.csv'

def main():
    idx=pd.read_csv(INDEX,encoding='utf-8-sig')
    for i in range(1,5):
        weld='W%02d'%i
        original=pd.read_csv(RAW/('焊缝%02d.csv'%i),encoding='utf-8-sig')
        fixed=idx.loc[idx.weld==weld,'original_index'].to_numpy(int)
        if len(fixed)==0 or not np.all(np.diff(fixed)>0):
            raise ValueError('Retained indices must be nonempty and increasing for '+weld)
        values=original.iloc[fixed,:6].to_numpy(dtype=float)
        df=pd.DataFrame(values,columns=['I_A','U_V','E_mm','S_dB','D_mm','W_mm'])
        df.insert(0,'original_index',fixed)
        cutoff=int(.8*len(df))
        df['stage']=['train']*cutoff+['test']*(len(df)-cutoff)
        dest=DATA/(weld+'_随机删去2of3的6mm_顺序保留.csv')
        # Verify the existing fixed data have equal numeric content before replacement
        if dest.exists():
            old=pd.read_csv(dest,encoding='utf-8-sig')
            for col in ['I_A','U_V','E_mm','S_dB','D_mm','W_mm']:
                if not np.allclose(df[col].to_numpy(),old[col].to_numpy(),rtol=0,atol=1e-5):
                    raise ValueError('Source data changed or mismatch in '+weld+':'+col)
        df.to_csv(dest,index=False,encoding='utf-8-sig')
        print(weld,'retained',len(df),'train',cutoff,'test',len(df)-cutoff)

if __name__=='__main__':main()

# -*- coding: utf-8 -*-
"""Optional XLSX report writer using artifact_tool (model experiment itself needs no artifact_tool)."""
import csv
from pathlib import Path
from artifact_tool import Workbook,SpreadsheetFile
ROOT=Path(__file__).resolve().parent
RES=ROOT/'results'
def rows(name):
    with (RES/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))
def num(v):
    if v is None or v=='':return None
    try:return float(v)
    except ValueError:return v

raw=rows('W09_all_193_predictions_and_errors.csv')
seeds=rows('W09_all_predictions_by_seed.csv')
drift=rows('W09_input_distribution_shift.csv')
logs=rows('W09_training_log.csv')
baseline=rows('W09_model_vs_trivial_baseline.csv')
wb=Workbook.create()
ps=wb.worksheets.add('W09_Predictions')
summary=wb.worksheets.add('Summary')
seed=wb.worksheets.add('Seed_Predictions')
meta=wb.worksheets.add('Experiment_Info')
HEAD={'fill':'#17324D','font':{'bold':True,'color':'#FFFFFF'},'horizontal_alignment':'center','row_height':27,'wrap_text':True}
TITLE={'fill':'#17324D','font':{'bold':True,'color':'#FFFFFF'},'row_height':29}
SUB={'fill':'#DBEAF1','font':{'bold':True,'color':'#17324D'},'row_height':24}

headers=['Index','Current (A)','Voltage (V)','Tracking (mm)','SPL (dB)',
         'Depth actual (mm)','Depth predicted (mm)','Width actual (mm)','Width predicted (mm)',
         'Depth seed SD','Width seed SD','Depth error (mm)','Depth abs. error (mm)',
         'Depth APE (%)','Width error (mm)','Width abs. error (mm)','Width APE (%)',
         'Depth <=5%','Width <=5%','Both <=5%','Depth <6 mm']
ps.get_range('A1:U1').values=[headers]
ps.get_range('A1:U1').format=HEAD
keys=['sample_index','I_A','U_V','E_mm','S_dB','D_actual','D_pred','W_actual','W_pred','D_pred_std','W_pred_std']
mat=[[num(r[k]) for k in keys] for r in raw]
ps.get_range('A2:K194').values=mat
f=[]
for i in range(2,195):
    f.append([f'=G{i}-F{i}',f'=ABS(L{i})',f'=100*M{i}/ABS(F{i})',
              f'=I{i}-H{i}',f'=ABS(O{i})',f'=100*P{i}/ABS(H{i})',
              f'=IF(N{i}<=5,1,0)',f'=IF(Q{i}<=5,1,0)',f'=R{i}*S{i}',f'=IF(F{i}<6,1,0)'])
ps.get_range('L2:U194').formulas=f
ps.get_range('B2:Q194').set_number_format('0.0000')
ps.get_range('A2:A194').set_number_format('0')
ps.get_range('R2:U194').set_number_format('0')
ps.get_range('A1:U194').format.row_height=19
ps.get_range('A1:U1').format.row_height=36
ps.get_range('A1:A194').format.column_width=9
ps.get_range('B1:K194').format.column_width=16
ps.get_range('L1:Q194').format.column_width=20
ps.get_range('R1:U194').format.column_width=13
ps.freeze_panes.freeze_rows(1)

summary.merge_cells('A1:C1')
summary.get_range('A1').values=[['W09 | Independent cross-weld verification']]
summary.get_range('A1:C1').format=TITLE
summary.merge_cells('A2:C2')
summary.get_range('A2').values=[['Training: W01–W04 (all 528) | W09: 193 unseen points | Mean of 3 seeds']]
summary.get_range('A2:C2').format=SUB
summary.get_range('A3:C3').values=[['Evaluation criterion','Penetration depth','Weld width']]
summary.get_range('A3:C3').format=HEAD
names=['Samples (n)','MAE (mm)','RMSE (mm)','MAPE (%)','Mean bias (mm)','Median AE (mm)',
       'Relative error ≤2% (%)','Relative error ≤5% (%)','Relative error >5% (%)',
       'Relative error >10% (%)','Abs. error ≤0.1mm (%)','Abs. error ≤0.2mm (%)',
       'Joint both ≤5% (%)','Depth<6 subset count','MAE on D<6 subset (mm)']
summary.get_range('A4:A18').values=[[x] for x in names]
D='W09_Predictions!';n=193
formulas=[
    ['=COUNT(W09_Predictions!F2:F194)','=COUNT(W09_Predictions!H2:H194)'],
    ['=AVERAGE(W09_Predictions!M2:M194)','=AVERAGE(W09_Predictions!P2:P194)'],
    ['=SQRT(SUMSQ(W09_Predictions!L2:L194)/B4)','=SQRT(SUMSQ(W09_Predictions!O2:O194)/C4)'],
    ['=AVERAGE(W09_Predictions!N2:N194)','=AVERAGE(W09_Predictions!Q2:Q194)'],
    ['=AVERAGE(W09_Predictions!L2:L194)','=AVERAGE(W09_Predictions!O2:O194)'],
    ['=MEDIAN(W09_Predictions!M2:M194)','=MEDIAN(W09_Predictions!P2:P194)'],
    ['=100*COUNTIF(W09_Predictions!N2:N194,"<=2")/B4','=100*COUNTIF(W09_Predictions!Q2:Q194,"<=2")/C4'],
    ['=100*SUM(W09_Predictions!R2:R194)/B4','=100*SUM(W09_Predictions!S2:S194)/C4'],
    ['=100*(B4-SUM(W09_Predictions!R2:R194))/B4','=100*(C4-SUM(W09_Predictions!S2:S194))/C4'],
    ['=100*COUNTIF(W09_Predictions!N2:N194,">10")/B4','=100*COUNTIF(W09_Predictions!Q2:Q194,">10")/C4'],
    ['=100*COUNTIF(W09_Predictions!M2:M194,"<=0.1")/B4','=100*COUNTIF(W09_Predictions!P2:P194,"<=0.1")/C4'],
    ['=100*COUNTIF(W09_Predictions!M2:M194,"<=0.2")/B4','=100*COUNTIF(W09_Predictions!P2:P194,"<=0.2")/C4'],
    ['=100*SUM(W09_Predictions!T2:T194)/B4','=100*SUM(W09_Predictions!T2:T194)/C4'],
    ['=SUM(W09_Predictions!U2:U194)','=SUM(W09_Predictions!U2:U194)'],
    ['=SUMIF(W09_Predictions!U2:U194,1,W09_Predictions!M2:M194)/B17',
     '=SUMIF(W09_Predictions!U2:U194,1,W09_Predictions!P2:P194)/C17'],
]
summary.get_range('B4:C18').formulas=formulas
summary.get_range('B4:C18').set_number_format('0.000')
summary.get_range('A1:A18').format.column_width=34
summary.get_range('B1:C18').format.column_width=24
summary.get_range('A4:C18').format.row_height=23
summary.get_range('A13:C13').format=SUB
summary.get_range('A16:C16').format=SUB
summary.freeze_panes.freeze_rows(3)
summary.get_range('A20:C20').values=[['Interpretation','Illustrative prediction-error thresholds','NOT a welding quality pass/fail standard']]
summary.get_range('A20:C20').format=SUB
summary.get_range('A22').values=[['High accuracy on D=6 plateau may mask poorer performance at D<6.']]
summary.get_range('A23').values=[['W09 output labels were used only for evaluation, never training.']]
summary.get_range('A24').values=[['15 training epochs were preselected from prior W01–W04 experiments.']]

sn=['seed','sample_index','D_actual','D_pred','D_error','D_APE_pct','W_actual','W_pred','W_error','W_APE_pct']
seed.get_range('A1:J1').values=[sn]
seed.get_range('A1:J1').format=HEAD
matrix=[[num(r[k]) for k in sn] for r in seeds]
seed.get_range(f'A2:J{len(matrix)+1}').values=matrix
seed.get_range(f'A2:J{len(matrix)+1}').format.row_height=18
seed.get_range('A1:B580').format.column_width=12
seed.get_range('C1:J580').format.column_width=18
seed.freeze_panes.freeze_rows(1)

info=[['Metric','Value'],['Training cases','W01, W02, W03, W04'],['Full training sizes','212 / 121 / 89 / 106'],
      ['Total full train',528],['External W09',193],['Training input','Current / Voltage / Tracking error / SPL'],
      ['Prediction target','Depth and width jointly'],['Model','PI-NGreyKAN 5-20-2 / 7 knots'],
      ['Training epochs',15],['Loss depth weight alpha',.6],['Thermal decay ell',.5],
      ['Physics stability penalty lambda',2.0],['Seeds','2026, 2027, 2028'],
      ['Deployment-style replay','Sequential, one sample at a time; no W09 target feedback'],
      ['Quality threshold','Model prediction error only; not a physical weld acceptance criterion'],
      ['Evaluation summary','Errors of the three-seed mean prediction']]
meta.get_range(f'A1:B{len(info)}').values=info
meta.get_range('A1:B1').format=HEAD
meta.get_range(f'A1:A{len(info)}').format.column_width=31
meta.get_range(f'B1:B{len(info)}').format.column_width=83
meta.get_range(f'A1:B{len(info)}').format.row_height=23
idx=len(info)+3
for title,records in [('Input range diagnostic',drift),('Seed training and replay latency',logs),('Comparison with constant median baseline',baseline)]:
    meta.get_range_by_indexes(idx-1,0,1,1).values=[[title]]
    meta.get_range_by_indexes(idx-1,0,1,2).format=SUB
    idx+=1
    k=list(records[0]);meta.get_range_by_indexes(idx-1,0,1,len(k)).values=[k]
    meta.get_range_by_indexes(idx-1,0,1,len(k)).format=HEAD
    idx+=1
    vals=[[num(r[col]) for col in k] for r in records]
    meta.get_range_by_indexes(idx-1,0,len(records),len(k)).values=vals
    idx+=len(records)+3

out=RES/'W09_193_predictions_summary.xlsx'
SpreadsheetFile.export_xlsx(wb).save(str(out))
print('EXCEL CREATED',out)
try:
    v=wb.inspect({'kind':'table','range':'Summary!A3:C18','include':'values,formulas','table_max_rows':16,'table_max_cols':3})
    print(v.ndjson[:2500])
except Exception as ex:
    print('Inspect unavailable:',str(ex)[:200])

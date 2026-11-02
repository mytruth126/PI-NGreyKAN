# -*- coding: utf-8 -*-
"""PI-NGreyKAN cross-weld external W09 verification (offline training + online-like replay).

HOW TO RUN:
  python run_w09_online_validation.py
  python run_w09_online_validation.py --seeds 2026 2027 2028
  python run_w09_online_validation.py --plots-only

Data:
  Train: ALL 528 cases W01-W04, with NO holdout for final training.
  No W01--W04 resplitting in this external validation run: all 528 observations train.
  Fixed 15 epochs are taken from earlier W01--W04 model-selection experiments, not W09.
  W09: untouched hold-out, 193 consecutive observations; labels used ONLY AFTER prediction.

Method identical to earlier PI-NGreyKAN core: (5,20,2) spline KAN, nonlinear grey recursion,
  Q=I*U/1000*exp(-0.5*(E/0.5)^2), alpha=.6, physics stability lambda=2,
  AdamW, three preset seeds.

W09 observations fed as input signal once per sample, no true target is used in forward pass.
Predictions reported as mean across trained seeds (ensemble; also each seed preserved).
Threshold compliance is illustrative error-tolerance, NOT a weld acceptance criterion.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import sys, random, argparse, copy, json, time, re
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
from matplotlib.ticker import MaxNLocator

RES=ROOT/'results'; FIG=ROOT/'figures';DATA=ROOT/'data'
RES.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
SEEDS=(2026,2027,2028)
ALPHA=.60;ELL=.50;PHYS_LAMBDA=2.0
THRESHOLDS=(2.,5.,10.)
FIXED_EPOCHS=15 # preselected in the prior W01-W04 M1 validation experiment (each seed=15)
torch.set_num_threads(1)

# Enforce reference Q feature even if others monkeypatch base.qfeat
base.qfeat=lambda x: (x[:,0]*x[:,1]/1000.)*np.exp(-.5*(x[:,2]/ELL)**2)

class FullTrainPrep:
    def __init__(self, stage):
        self.stage=stage
        self.cuts=[d['k80'] if stage=='valid' else d['n'] for d in base.RAW]
        X=np.concatenate([d['x'][:n] for d,n in zip(base.RAW,self.cuts)])
        Y=np.concatenate([d['y'][:n] for d,n in zip(base.RAW,self.cuts)])
        self.xmu=X.mean(0);self.xstd=np.maximum(X.std(0),1e-4)
        self.ymu=np.median(Y,axis=0);self.ystd=np.maximum(Y.std(0),.02)
        q=base.qfeat(X);self.qmu=q.mean();self.qstd=max(q.std(),1e-4)
    def transform(self,x):
        x=np.asarray(x,dtype=np.float32)
        z=(x-self.xmu)/self.xstd
        z[:,2]=x[:,2]/self.xstd[2] # exact earlier representation
        return np.column_stack([z,(base.qfeat(x)-self.qmu)/self.qstd]).astype('float32')
    def tensors(self):
        m=len(base.RAW);T=max(d['n'] for d in base.RAW)
        Z=np.zeros((m,T,5),np.float32);Y=np.zeros((m,T,2),np.float32)
        TR=np.zeros((m,T,1),np.float32); VA=np.zeros((m,T,1),np.float32)
        for i,d in enumerate(base.RAW):
            n=d['n'];cut=self.cuts[i]
            Z[i,:n]=self.transform(d['x'])
            Y[i,:n]=(d['y']-self.ymu)/self.ystd
            TR[i,:cut]=1.
            if self.stage=='valid':VA[i,cut:n]=1.
        return tuple(torch.from_numpy(t) for t in (Z,Y,TR,VA))


def fit_one(seed,stage='valid',epochs=None):
    base.random_seed(seed)
    p=FullTrainPrep(stage)
    z,y,tr,va=p.tensors()
    model=base.GreyKAN((5,20,2),7)
    model.phys_y_scale=torch.tensor(p.ystd,dtype=torch.float32)
    phys=physics.PhysPairs(p)
    opt=torch.optim.AdamW(model.parameters(),lr=.012,weight_decay=.001)
    best=np.inf;best_ep=5;best_state=None
    limit=125 if stage=='valid' else int(epochs)
    for ep in range(1,limit+1):
        lr=.012 if ep<=65 else .006 if ep<=105 else .003
        for group in opt.param_groups:group['lr']=lr
        model.train();opt.zero_grad();pred=model(z)
        err=F.smooth_l1_loss(pred,y,beta=.6,reduction='none')
        loss=(err*tr*torch.tensor([ALPHA,1-ALPHA])).sum()/tr.sum()
        loss=loss+PHYS_LAMBDA*physics.prior_loss(model,pred,phys.rob,'stability')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),4.)
        opt.step()
        if stage=='valid' and (ep==1 or ep%5==0):
            model.eval()
            with torch.no_grad():vpred=model(z)
            val=float(((vpred-y).abs()*va*torch.tensor([ALPHA,1-ALPHA])).sum()/va.sum())
            if val<best-1e-5:
                best=val;best_ep=ep;best_state=copy.deepcopy(model.state_dict())
            if ep-best_ep>=35:break
    if stage=='valid' and best_state is not None:model.load_state_dict(best_state)
    return model,p,best_ep,float(best) if stage=='valid' else None


def predict_w09(model,p,w09):
    inputs=w09[['I_A','U_V','E_mm','S_dB']].to_numpy(np.float32)
    z=torch.from_numpy(p.transform(inputs))[None,:,:]
    model.eval()
    with torch.no_grad():normalized=model(z)[0].cpu().numpy()
    return normalized*p.ystd+p.ymu


def benchmark_sequential_inference(model,p,w09,repetitions=3):
    """Replay one sample at a time, retaining prior grey state; no labels or future inputs."""
    X=w09[['I_A','U_V','E_mm','S_dB']].to_numpy(np.float32)
    Z=p.transform(X)
    times=[];out=[]
    model.eval()
    with torch.no_grad():
        for repeat in range(repetitions):
            state=torch.zeros((1,2),dtype=torch.float32)
            ret=[]
            t0=time.perf_counter()
            for row in Z:
                state=model.step(state,torch.from_numpy(row[None,:]))
                ret.append(state[0].numpy().copy())
            elapsed=time.perf_counter()-t0
            times.append(elapsed*1000/len(Z))
            out=np.stack(ret)*p.ystd+p.ymu
    return out,float(np.median(times))


def calc_metrics(actual,pred):
    error=np.asarray(pred)-np.asarray(actual)
    relative=100*np.abs(error)/np.maximum(np.abs(actual),1e-12)
    return dict(n=len(actual),MAE=float(np.abs(error).mean()),
                RMSE=float(np.sqrt((error**2).mean())),MAPE=float(relative.mean()),
                Bias=float(error.mean()),MedAE=float(np.median(np.abs(error))),
                P95_AE=float(np.percentile(np.abs(error),95)),MaxAE=float(np.abs(error).max()),
                Within_2pct=int((relative<=2).sum()),Rate_2pct=float(100*(relative<=2).mean()),
                Within_5pct=int((relative<=5).sum()),Rate_5pct=float(100*(relative<=5).mean()),
                Over_5pct=int((relative>5).sum()),Rate_over_5pct=float(100*(relative>5).mean()),
                Over_10pct=int((relative>10).sum()),Rate_over_10pct=float(100*(relative>10).mean()),
                Within_0p1mm=int((np.abs(error)<=.1).sum()),Rate_0p1mm=float(100*(np.abs(error)<=.1).mean()),
                Within_0p2mm=int((np.abs(error)<=.2).sum()),Rate_0p2mm=float(100*(np.abs(error)<=.2).mean()))


def summarize(df):
    rec=[]
    for label,mask in [('W09 all 193',np.ones(len(df),bool)),
                       ('D=6 mm subset',df['D_actual'].eq(6).to_numpy()),
                       ('D<6 mm subset',df['D_actual'].lt(6).to_numpy())]:
        if not np.any(mask):continue
        for j,t in enumerate(['D','W']):
            d=calc_metrics(df.loc[mask,f'{t}_actual'],df.loc[mask,f'{t}_pred'])
            rec.append(dict(Group=label,Target='Penetration depth' if t=='D' else 'Weld width',**d))
    ret=pd.DataFrame(rec)
    ret.to_csv(RES/'W09_error_accuracy_summary.csv',index=False,encoding='utf-8-sig')
    # combined table, left D / right W, single row per metric (for manuscript)
    all_rows=[]
    for r in ['n','MAE','RMSE','MAPE','Bias','MedAE','P95_AE','MaxAE',
              'Within_2pct','Rate_2pct','Within_5pct','Rate_5pct','Over_5pct','Rate_over_5pct',
              'Over_10pct','Rate_over_10pct','Within_0p1mm','Rate_0p1mm','Within_0p2mm','Rate_0p2mm']:
        r1=ret[(ret.Group=='W09 all 193')&(ret.Target=='Penetration depth')].iloc[0]
        r2=ret[(ret.Group=='W09 all 193')&(ret.Target=='Weld width')].iloc[0]
        all_rows.append(dict(Metric=r,Depth=r1[r],Width=r2[r]))
    pd.DataFrame(all_rows).to_csv(RES/'W09_combined_depth_width_table.csv',index=False,encoding='utf-8-sig')
    return ret


def summaries_with_baselines(df, training):
    baseline={'D':6.0,'W':float(np.median(np.concatenate([d['y'][:,1] for d in base.RAW])))}
    rows=[]
    for t in ['D','W']:
        actual=df[f'{t}_actual'].to_numpy()
        for method,pred in [('PI-NGreyKAN (mean of 3 seeds)',df[f'{t}_pred'].to_numpy()),
                            ('Constant train-median baseline',np.full(len(actual),baseline[t]))]:
            rows.append(dict(Target=t,Method=method,**calc_metrics(actual,pred)))
    pd.DataFrame(rows).to_csv(RES/'W09_model_vs_trivial_baseline.csv',index=False,encoding='utf-8-sig')


def make_figures(df):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, ConnectionPatch
    from pathlib import Path

    # =====================================================
    # 1. Output directory
    # =====================================================

    FIG = Path(__file__).resolve().parent / "figures"
    FIG.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # 2. Parameters
    # =====================================================

    ZOOM_START = 181
    ZOOM_END = 190

    COLOR_ACTUAL = "#181818"
    COLOR_PRED = "#ba642b"
    COLOR_CONNECTOR = "#53616d"

    # Title and legend positioning
    TITLE_X = 0.02
    LEGEND_X = 0.74
    LEGEND_Y = 1.075

    # =====================================================
    # 3. Plot style
    # =====================================================

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "DejaVu Serif"
        ],
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False
    })

    # =====================================================
    # 4. Read prediction results
    # =====================================================

    x = df["sample_index"].to_numpy()

    d_actual = df["D_actual"].to_numpy()
    d_pred = df["D_pred"].to_numpy()

    w_actual = df["W_actual"].to_numpy()
    w_pred = df["W_pred"].to_numpy()

    # =====================================================
    # 5. Extract local zoom region
    # =====================================================

    mask = (
        (x >= ZOOM_START)
        & (x <= ZOOM_END)
    )

    if not np.any(mask):
        raise ValueError(
            "The selected zoom window has no samples."
        )

    x_zoom = x[mask]

    w_actual_zoom = w_actual[mask]
    w_pred_zoom = w_pred[mask]

    y_min = float(
        min(
            w_actual_zoom.min(),
            w_pred_zoom.min()
        )
    )

    y_max = float(
        max(
            w_actual_zoom.max(),
            w_pred_zoom.max()
        )
    )

    y_pad = max(
        0.014,
        0.09 * (y_max - y_min)
    )

    zoom_ymin = y_min - y_pad
    zoom_ymax = y_max + y_pad

    # =====================================================
    # 6. Figure layout
    # =====================================================

    fig = plt.figure(figsize=(13.2, 7.4))

    # [left, bottom, width, height]
    #
    # Left: two equal-sized primary plots
    # Right: compact, vertically centered zoom plot

    ax_depth = fig.add_axes([
        0.100, 0.570, 0.558, 0.365
    ])

    ax_width = fig.add_axes([
        0.100, 0.125, 0.558, 0.365
    ])

    ax_zoom = fig.add_axes([
        0.748, 0.365, 0.218, 0.275
    ])

    # =====================================================
    # 7. Common axis formatting
    # =====================================================

    def decorate(ax):

        ax.grid(
            axis="y",
            color="#dddddd",
            linestyle=":",
            linewidth=0.8,
            zorder=0
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#252525")
            spine.set_linewidth(1.0)

        ax.tick_params(
            direction="in",
            length=3.5
        )

    # =====================================================
    # 8. Common legend formatting
    # =====================================================

    def add_top_legend(ax):

        ax.legend(
            loc="center",
            bbox_to_anchor=(
                LEGEND_X,
                LEGEND_Y
            ),
            bbox_transform=ax.transAxes,
            ncol=2,
            frameon=True,
            framealpha=1,
            edgecolor="#888888",
            fontsize=8.8,
            columnspacing=1.3,
            handlelength=2.5,
            handletextpad=0.5
        )

    # =====================================================
    # 9. (a) Penetration depth
    # =====================================================

    ax_depth.plot(
        x,
        d_actual,
        label="Observed",
        color=COLOR_ACTUAL,
        linewidth=1.9,
        zorder=3
    )

    ax_depth.plot(
        x,
        d_pred,
        label="PI-NGreyKAN",
        color=COLOR_PRED,
        linewidth=1.75,
        linestyle=(0, (5, 2.5)),
        zorder=4
    )

    # No 6 mm reference line

    ax_depth.set_xlim(
        x.min(),
        x.max()
    )

    ax_depth.set_ylabel(
        "Penetration depth (mm)"
    )

    # Left-aligned title
    ax_depth.set_title(
        "(a) Penetration depth",
        fontweight="bold",
        loc="left",
        x=TITLE_X,
        pad=9
    )

    decorate(ax_depth)

    # Legend positioned to the right of the title
    add_top_legend(ax_depth)

    # =====================================================
    # 10. (b) Weld width
    # =====================================================

    ax_width.plot(
        x,
        w_actual,
        label="Observed",
        color=COLOR_ACTUAL,
        linewidth=1.9,
        zorder=3
    )

    ax_width.plot(
        x,
        w_pred,
        label="PI-NGreyKAN",
        color=COLOR_PRED,
        linewidth=1.75,
        linestyle=(0, (5, 2.5)),
        zorder=4
    )

    ax_width.set_xlim(
        x.min(),
        x.max()
    )

    ax_width.set_ylabel(
        "Weld width (mm)"
    )

    ax_width.set_xlabel(
        "Sequential sample index"
    )

    # Same horizontal title position as (a)
    ax_width.set_title(
        "(b) Weld width",
        fontweight="bold",
        loc="left",
        x=TITLE_X,
        pad=9
    )

    decorate(ax_width)

    # Same legend height and horizontal position as (a)
    add_top_legend(ax_width)

    # =====================================================
    # 11. Zoom rectangle on original weld-width plot
    # =====================================================

    rectangle = Rectangle(
        xy=(
            ZOOM_START,
            zoom_ymin
        ),
        width=ZOOM_END - ZOOM_START,
        height=zoom_ymax - zoom_ymin,
        facecolor="none",
        edgecolor=COLOR_CONNECTOR,
        linewidth=1.35,
        linestyle="-",
        zorder=8
    )

    ax_width.add_patch(rectangle)

    # =====================================================
    # 12. (c) Local enlargement
    # =====================================================

    ax_zoom.plot(
        x_zoom,
        w_actual_zoom,
        color=COLOR_ACTUAL,
        linewidth=1.9,
        zorder=3
    )

    ax_zoom.plot(
        x_zoom,
        w_pred_zoom,
        color=COLOR_PRED,
        linewidth=1.75,
        linestyle=(0, (5, 2.5)),
        zorder=4
    )

    ax_zoom.set_xlim(
        ZOOM_START,
        ZOOM_END
    )

    ax_zoom.set_ylim(
        zoom_ymin,
        zoom_ymax
    )

    ax_zoom.set_xticks([
        181, 184, 187, 190
    ])

    ax_zoom.set_xlabel(
        "Sample index"
    )

    ax_zoom.set_title(
        "(c) Local enlargement",
        fontweight="bold",
        pad=9
    )

    decorate(ax_zoom)

    # =====================================================
    # 13. Two connecting lines
    # =====================================================

    # Upper connection
    upper_line = ConnectionPatch(
        xyA=(
            ZOOM_END,
            zoom_ymax
        ),
        coordsA=ax_width.transData,

        xyB=(
            ZOOM_START,
            zoom_ymax
        ),
        coordsB=ax_zoom.transData,

        color=COLOR_CONNECTOR,
        linewidth=1.05,
        zorder=5,
        clip_on=False
    )

    # Lower connection
    lower_line = ConnectionPatch(
        xyA=(
            ZOOM_END,
            zoom_ymin
        ),
        coordsA=ax_width.transData,

        xyB=(
            ZOOM_START,
            zoom_ymin
        ),
        coordsB=ax_zoom.transData,

        color=COLOR_CONNECTOR,
        linewidth=1.05,
        zorder=5,
        clip_on=False
    )

    fig.add_artist(upper_line)
    fig.add_artist(lower_line)

    # =====================================================
    # 14. Save publication-quality figures
    # =====================================================

    png_path = (
        FIG /
        "W09_Online_Replay_Depth_Width_300dpi.png"
    )

    pdf_path = (
        FIG /
        "W09_Online_Replay_Depth_Width.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print("Figure saved successfully.")
    print("PNG:", png_path)
    print("PDF:", pdf_path)




def make_tex_table(ret):
    d=ret[(ret.Group=='W09 all 193')&(ret.Target=='Penetration depth')].iloc[0]
    w=ret[(ret.Group=='W09 all 193')&(ret.Target=='Weld width')].iloc[0]
    dsub=ret[(ret.Group=='D<6 mm subset')&(ret.Target=='Penetration depth')].iloc[0]
    wsub=ret[(ret.Group=='D<6 mm subset')&(ret.Target=='Weld width')].iloc[0]
    d6=ret[(ret.Group=='D=6 mm subset')&(ret.Target=='Penetration depth')].iloc[0]
    w6=ret[(ret.Group=='D=6 mm subset')&(ret.Target=='Weld width')].iloc[0]
    template=r'''\begin{table}[htbp]
    \centering
    \caption{External W09 prediction accuracy and error-tolerance coverage for weld penetration depth and width.}
    \label{tab:w09_online_metrics}
    \setlength{\tabcolsep}{5pt}
    \renewcommand{\arraystretch}{1.08}
    \begin{tabular}{lrr}
        \toprule
        Metric & Penetration depth & Weld width \\
        \midrule
        Number of observations & {n} & {n} \\
        MAE (mm) & {DMAE:.4f} & {WMAE:.4f} \\
        RMSE (mm) & {DRMSE:.4f} & {WRMSE:.4f} \\
        MAPE (\%) & {DMAPE:.3f} & {WMAPE:.3f} \\
        Mean bias (mm) & {DBias:+.4f} & {WBias:+.4f} \\
        MAE for $D=6$ mm (153 points) & {DMAE6:.4f} & {WMAE6:.4f} \\
        MAE for $D<6$ mm (40 points) & {DMAEunder:.4f} & {WMAEunder:.4f} \\
        Relative error $\leq 2\%$ (\%) & {DR2:.1f} & {WR2:.1f} \\
        Relative error $\leq 5\%$ (\%) & {DR5:.1f} & {WR5:.1f} \\
        Relative error $>5\%$ (\%) & {DO5:.1f} & {WO5:.1f} \\
        Relative error $>10\%$ (\%) & {DO10:.1f} & {WO10:.1f} \\
        Absolute error $\leq 0.1$ mm (\%) & {DA1:.1f} & {WA1:.1f} \\
        Absolute error $\leq 0.2$ mm (\%) & {DA2:.1f} & {WA2:.1f} \\
        \midrule
        Both targets within 5\% & \multicolumn{2}{c}{{JOINT5}} \\
        \bottomrule
    \end{tabular}
    \par\vspace{3pt}
    \begin{minipage}{0.90\linewidth}\footnotesize
    \emph{Note:} This is an external, label-blind sequential replay of W09
    (193 observations) after fitting on all 528 observations of Cases 1--4.
    Results use the mean prediction from three independently fitted seeds.
    Percentages denote illustrative prediction-error tolerances, not industrial
    weld acceptance specifications. W09 labels are used only for post-hoc scoring.
    \end{minipage}
\end{table}
'''
    replacements=dict(n=int(d.n),DMAE=d.MAE,WMAE=w.MAE,DRMSE=d.RMSE,WRMSE=w.RMSE,DMAPE=d.MAPE,WMAPE=w.MAPE,
         DBias=d.Bias,WBias=w.Bias,DR2=d.Rate_2pct,WR2=w.Rate_2pct,DR5=d.Rate_5pct,WR5=w.Rate_5pct,
         DO5=d.Rate_over_5pct,WO5=w.Rate_over_5pct,DO10=d.Rate_over_10pct,WO10=w.Rate_over_10pct,
         DA1=d.Rate_0p1mm,WA1=w.Rate_0p1mm,DA2=d.Rate_0p2mm,WA2=w.Rate_0p2mm,
         DMAE6=d6.MAE,WMAE6=w6.MAE,DMAEunder=dsub.MAE,WMAEunder=wsub.MAE)
    # Replace only named data placeholders; keep LaTeX grouping braces intact.
    s=re.sub(r'\{([A-Za-z][A-Za-z0-9_]*)(?::([^}]+))?\}',
        lambda m: (format(replacements[m.group(1)],m.group(2)) if m.group(2) else str(replacements[m.group(1)]))
        if m.group(1) in replacements else m.group(0),template)
    n_joint=int(pd.read_csv(RES/'W09_all_193_predictions_and_errors.csv')['Both_within_5pct'].sum())
    s=s.replace('{JOINT5}',f'{100*n_joint/int(d.n):.1f}\\% ({n_joint}/{int(d.n)})')
    (RES/'W09_external_validation_table.tex').write_text(s,encoding='utf-8')
    print('D<6 group',dict(n=dsub.n,MAE=dsub.MAE,MAPE=dsub.MAPE,Rate_5pct=dsub.Rate_5pct))


def analyze_drift(w09):
    train=np.concatenate([d['x'] for d in base.RAW]);x=w09[['I_A','U_V','E_mm','S_dB']].to_numpy()
    names=('I_A','U_V','E_mm','S_dB');rows=[]
    for j,n in enumerate(names):
        lo,hi=float(train[:,j].min()),float(train[:,j].max());ex=(x[:,j]<lo)|(x[:,j]>hi)
        rows.append(dict(Input=n,Train_min=lo,Train_max=hi,W09_min=float(x[:,j].min()),W09_max=float(x[:,j].max()),Outside_train_range_count=int(ex.sum()),Outside_train_range_pct=float(ex.mean()*100)))
    df=pd.DataFrame(rows);df.to_csv(RES/'W09_input_distribution_shift.csv',index=False,encoding='utf-8-sig')
    return df


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    parser.add_argument('--plots-only',action='store_true')
    parser.add_argument('--epochs',type=int,default=FIXED_EPOCHS,help='Predeclared epochs; default from prior W01-W04 experiments, NEVER optimized on W09')
    args=parser.parse_args()
    w09=pd.read_csv(DATA/'W09_standardized.csv',encoding='utf-8-sig')
    assert len(w09)==193 and not w09.isna().any().any()
    print('W01-W04 full training sizes:',[d['n'] for d in base.RAW],'total',sum(d['n'] for d in base.RAW),flush=True)
    print('W09 independent validation:',len(w09),'rows',flush=True)
    if not args.plots_only:
        log=[];all_predictions=[]
        for seed in args.seeds:
            tic=time.time()
            bestep=int(args.epochs)
            print('Fixed full-training epochs',bestep,'seed',seed,flush=True)
            model,p,_,_=fit_one(seed,'final',epochs=bestep)
            # External replay. W09 targets never supplied to model or normalizer.
            pred=predict_w09(model,p,w09)
            pred_stream,latency=benchmark_sequential_inference(model,p,w09)
            mismatch=float(np.max(np.abs(pred_stream-pred)))
            assert mismatch<1e-5, f'Streaming/full-sequence mismatch={mismatch}'
            for idx,(v1,v2) in enumerate(pred,1):
                all_predictions.append(dict(seed=seed,sample_index=idx,D_pred=float(v1),W_pred=float(v2)))
            log.append(dict(seed=seed,fixed_epochs=bestep,epoch_selection='previous W01-W04 experiments',
                            final_training_samples=sum(d['n'] for d in base.RAW),
                            per_sample_streaming_latency_ms=latency,
                            streaming_vs_batch_max_diff_mm=mismatch,elapsed_s=time.time()-tic))
            torch.save({'model_state_dict':model.state_dict(),'xmu':p.xmu,'xstd':p.xstd,
                        'ymu':p.ymu,'ystd':p.ystd,'qmu':p.qmu,'qstd':p.qstd,
                        'alpha':ALPHA,'ell':ELL,'seed':seed,'architecture':'GreyKAN-5-20-2-7knots'},
                       RES/f'pi_ngreykan_w09_seed{seed}.pt')
            print('Completed seed',seed,'time',time.time()-tic,flush=True)
        pd.DataFrame(log).to_csv(RES/'W09_training_log.csv',index=False,encoding='utf-8-sig')
        perseed=pd.DataFrame(all_predictions)
        perseed=perseed.merge(w09[['sample_index','D_mm','W_mm']].rename(columns={'D_mm':'D_actual','W_mm':'W_actual'}),on='sample_index',validate='many_to_one')
        for t in ('D','W'):
            perseed[f'{t}_error']=perseed[f'{t}_pred']-perseed[f'{t}_actual']
            perseed[f'{t}_APE_pct']=100*np.abs(perseed[f'{t}_error'])/np.maximum(np.abs(perseed[f'{t}_actual']),1e-12)
        perseed.to_csv(RES/'W09_all_predictions_by_seed.csv',index=False,encoding='utf-8-sig')
        agg=perseed.groupby('sample_index')[['D_pred','W_pred']].agg(['mean','std'])
        agg.columns=['_'.join(col) for col in agg.columns]
        agg=agg.reset_index().rename(columns={'D_pred_mean':'D_pred','W_pred_mean':'W_pred'})
        df=w09.copy().rename(columns={'D_mm':'D_actual','W_mm':'W_actual'})
        df=df.merge(agg,on='sample_index',validate='one_to_one')
        for t in ('D','W'):
            df[f'{t}_error']=df[f'{t}_pred']-df[f'{t}_actual']
            df[f'{t}_AE_mm']=df[f'{t}_error'].abs()
            df[f'{t}_APE_pct']=100*df[f'{t}_AE_mm']/np.maximum(np.abs(df[f'{t}_actual']),1e-12)
            df[f'{t}_within_2pct']=df[f'{t}_APE_pct']<=2
            df[f'{t}_within_5pct']=df[f'{t}_APE_pct']<=5
            df[f'{t}_over_10pct']=df[f'{t}_APE_pct']>10
        df['Both_within_2pct']=df['D_within_2pct']&df['W_within_2pct']
        df['Both_within_5pct']=df['D_within_5pct']&df['W_within_5pct']
        df.to_csv(RES/'W09_all_193_predictions_and_errors.csv',index=False,encoding='utf-8-sig')
    else:
        df=pd.read_csv(RES/'W09_all_193_predictions_and_errors.csv',encoding='utf-8-sig')
    summary=summarize(df)
    summaries_with_baselines(df,base.RAW)
    make_figures(df)
    make_tex_table(summary)
    print('INPUT DRIFT\n',analyze_drift(w09).to_string(index=False),flush=True)
    print('SUMMARY\n',summary[summary.Group=='W09 all 193'].to_string(index=False),flush=True)
    print('BOTH: within2%',int(df.Both_within_2pct.sum()),'within5%',int(df.Both_within_5pct.sum()),flush=True)
    print('Finished. Results:',RES,flush=True)

if __name__=='__main__':main()

# PI-NGreyKAN — four-case full training and independent Weld 09 online-style replay

## Scope

- Train **only PI-NGreyKAN**, with all **528 observations** from W01–W04, with no final train/test split.
- Use **W09's all 193 chronological observations as a completely unseen validation weld**. W09 measurements used as inputs; its penetration-depth, width and tensile-strength labels **are never supplied to the model** (the first two labels are used **only after** inference to score predictions). Tensile strength is not predicted.
- This is a **deployment-like sequential replay on an existing dataset**, not an actual industrial online integration or prospective field trial.
- Three fixed seeds: 2026, 2027, 2028. Prediction curve and main table show the **mean of the three models' predictions at each observation** (metrics computed afterward); per-seed predictions and weight files are saved separately.
- Fixed training epochs **15**, preselected from previous W01-W04 PI-NGreyKAN M1 experiments (all seeds selected 15). This particular experiment **does not split W01-W04 for tuning**, nor does it use W09 to select hyperparameters or epochs. Modify `--epochs` only on independent evidence; otherwise you would make the validation less convincing.
- Architecture and optimizer: same PI-NGreyKAN (5-20-2 shared KAN, 7 knots, nonlinear grey state dynamics, physics stability regularization weight 2.0), ell=0.5, depth-loss alpha=0.6, AdamW.

## Directory

- `run_w09_online_validation.py`: **main one-click script**, reads data, trains, exports every prediction, accuracy tables, and charts.
- `data/`: the four existing real welding case CSV files (their `stage` column is deliberately **ignored for the full-training run**); `W09_original_193.csv` preserves the original Chinese-column GB18030 upload; `W09_standardized.csv` supplies the exact same 193 rows with standard headings.
- `code/base_model.py`, `code/run_simple_physics.py`: model architecture, state recursion, and stability-based physics loss reused from earlier PI-NGreyKAN projects. This experiment uses its own `FullTrainPrep`, not the `base_model.Prep` 80/20 split.
- `results/W09_all_193_predictions_and_errors.csv`: one row per W09 observation, all four input features, target observations, 3-seed mean predictions and their standard deviations, signed and absolute errors, APE%, 2%/5%/10% flags, dual-target joint coverage.
- `results/W09_all_predictions_by_seed.csv`: all **579** seed-by-observation predictions.
- `results/W09_error_accuracy_summary.csv`: per-target and by penetration-depth subgroup.
- `results/W09_model_vs_trivial_baseline.csv`: comparison against fixed training-median predictions (D=6.0mm, W=3.92mm).
- `results/W09_external_validation_table.tex`: one joint D/W error and tolerance table for SCI.
- `results/W09_193_predictions_summary.xlsx`: prebuilt Excel with formula-driven summary, full sequential predictions, seed predictions, and experiment metadata.
- `results/W09_training_log.csv`: exact training epochs, measured per-sample streaming CPU latency and streaming-to-batch equivalence check.
- `figures/`: scientific figures at 300 dpi, PDF vector versions for the main line and absolute-error charts.
- `Section_5_Discussion_and_Online_Validation.tex`: paper-ready English section, integrated table, figures and limitations.
- `create_report_xlsx.py`: **optional** Excel generator (requires `artifact_tool` package, not needed to run the experiment).

## Run in a clean Anaconda Prompt (recommended on Windows)

```bat
cd /d "D:\your_folder\PI_NGreyKAN_W09_独立在线模拟验证"
python run_w09_online_validation.py
```

Use Python 3.9+ with `numpy`, `pandas`, `torch`, and `matplotlib` (example: `python -m pip install numpy pandas torch matplotlib`). The bundled `create_report_xlsx.py` is optional.

For redraw without retraining:

```bat
python run_w09_online_validation.py --plots-only
```

If Spyder dies with `OMP Error #15`, prefer a new Anaconda Prompt or fresh isolated environment. Do not use `KMP_DUPLICATE_LIB_OK=TRUE` for production/research reproduction.

## Important interpretation

W09 has **153/193 observations at exactly D=6.00 mm**. This makes overall depth MAPE look excellent even though a rare **32–39 sample under-penetration episode is substantially underestimated**. Inspect the D<6 subset and constant-median baseline in addition to the overall statistics.

We use **relative error <=5%** as a descriptive *prediction agreement coverage* threshold. It is **not** an established welding-quality pass/fail standard and does not verify weld mechanical strength or safety. Offline replay time is not certified factory real-time performance.

All input data, raw predictions and output tables here are experimental records; no synthetic observations or invented accuracy results are used.

---
name: tabular-automl
description: Complete AutoML pipeline for tabular binary classification. Profiles the data, handles categorical and missing values, fits a cross-validated model zoo (regularized LightGBM, HistGradientBoosting, CatBoost, regularized logistic regression) under a time budget, and writes a valid submission from the CV-best model or blend. Always use this instead of writing modelling code.
---

# Tabular AutoML Skill

One tested script does the entire modelling job. It is time-bounded, degrades gracefully, and always writes a valid submission file — even if every model fails.

## Script: `scripts/run_automl.py`

Reads `train.csv` / `test.csv` / `sample_submission.csv` from the working directory. Writes a submission CSV with the exact id column and row order of `sample_submission.csv`.

```bash
# safety submission, ~5-30s
python skills/tabular-automl/scripts/run_automl.py --mode fast --out sub_fast.csv

# full cross-validated zoo, bounded to 25 minutes
python skills/tabular-automl/scripts/run_automl.py --mode full --time-budget 1500 --out sub_full.csv

# a second opinion from a different seed
python skills/tabular-automl/scripts/run_automl.py --mode full --seed 1 --time-budget 900 --out sub_alt.csv
```

## Reading the output

```
n_train=10803 n_test=10000 n_feat=9 n_cat=9 pos_rate=0.496 missing=0.131
  [    11s] lgbm_reg   0.8038
  [    12s] logreg     0.8074
  [    18s] hgb        0.7995
  [    41s] catboost   0.8051
  [    41s] blend      0.8086  (lgbm_reg+hgb+catboost)
CV(5-fold roc_auc), 5 candidates:
  blend        0.8086
  ...
CHOSEN: blend (CV 0.8086) -> sub_full.csv  [41s total]
```

The number after `CHOSEN:` is the honest cross-validated score. **Report that number back when selecting final submissions** — it is more trustworthy than the public score, which is computed on a small subset.

`(partial, N/M rows)` means that model ran out of time and only completed some folds. Its score is optimistic; the script already discounts it when choosing.

## Behaviour worth knowing

- `--time-budget` is a soft deadline in seconds. Models run cheapest-first and are **skipped rather than killed**, so a tight budget still produces a usable submission. Always pass it — leave at least 10 minutes of session time as margin.
- Fold count adapts: 10 folds below 2000 rows, 5 up to 20000, 4 above.
- Below ~2000 rows the linear model joins the blend; above it, it is excluded. Benchmarking across 16 datasets from this family showed that is worth about +0.013 on small data and −0.007 on large.
- `row_id`-style id columns and the target are dropped automatically. Missing values and `cat_*`/`ord_*` string levels are handled internally — **do not preprocess anything yourself.**
- **Never impute missing values before calling this script.** The tree models handle NaN natively by learning a split direction for it, which is strictly better: median-imputing first costs −0.0024 AUC on average across the 16 benchmark datasets, and up to −0.024 on some. The linear model does its own median imputation *with* missingness indicator columns.

## If it fails

Read the traceback, then try once with explicit column names:

```bash
python skills/tabular-automl/scripts/run_automl.py --target target --id-col row_id --mode fast --out sub.csv
```

If that also fails, submit any CSV that already exists, select it, and stop. Do not rewrite this script.

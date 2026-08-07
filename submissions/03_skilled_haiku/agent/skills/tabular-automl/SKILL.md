---
name: tabular-automl
description: Complete AutoML pipeline for tabular binary classification. Profiles the data, handles categorical and missing values, fits a cross-validated model zoo (regularized LightGBM, a CV-tuned LightGBM variant on datasets under 20000 rows, HistGradientBoosting, CatBoost, regularized logistic regression) under a time budget, and writes a valid submission file. Always use this instead of writing modelling code by hand.
---

# Tabular AutoML Skill

One tested script does the entire modelling job. It is time-bounded, degrades gracefully, and always writes a valid submission file — even if every model fails.

## Invocation

Run it with the `run_skill_script` tool. Pass `args` as a flat list of strings:

```
run_skill_script(skill_name="tabular-automl", file_path="scripts/run_automl.py",
                 args=["--mode", "fast", "--out", "sub_fast.csv"])
```

| Purpose | `args` |
|---|---|
| Safety submission (~5–30s) | `["--mode", "fast", "--out", "sub_fast.csv"]` |
| Full zoo, capped at 15 min | `["--mode", "full", "--time-budget", "900", "--out", "sub_full.csv"]` |
| Second opinion, different seed | `["--mode", "full", "--seed", "1", "--time-budget", "900", "--out", "sub_alt.csv"]` |

The script runs in the same working directory as `run_command`, so it reads `train.csv` / `test.csv` / `sample_submission.csv` directly and the file it writes is immediately submittable.

## Reading the output

```
n_train=10803 n_test=10000 n_feat=9 n_cat=9 pos_rate=0.496 missing=0.131
  [    11s] lgbm_reg   0.8038
  [    12s] logreg     0.8074
  [    18s] hgb        0.7995
  [    41s] catboost   0.8051
  [    41s] blend      0.8086  (lgbm_reg+hgb+catboost)
CHOSEN: blend (CV 0.8086) -> sub_full.csv  [41s total]
```

The number after `CHOSEN:` is the honest cross-validated score. **Report it when selecting final submissions** — it is more trustworthy than the public score, which is computed on a small subset.

`(partial, N/M rows)` means that model ran out of time and completed only some folds. Its score is optimistic; the script already discounts it when choosing.

## Behaviour worth knowing

- `--time-budget` is a soft deadline in seconds. Models run cheapest-first and are **skipped rather than killed**, so a tight budget still yields a usable submission. Always pass it.
- Fold count adapts: 10 folds below 2000 rows, 5 up to 20000, 4 above.
- Below ~2000 rows the linear model joins the blend; above it, it is excluded. Across 16 datasets from this family that is worth about +0.013 on small data and −0.007 on large.
- The id column and target are dropped automatically. `cat_*` / `ord_*` string levels are handled internally.
- **Never impute missing values before calling this script.** The tree models handle NaN natively by learning a split direction for it, which is strictly better: median-imputing first costs −0.0024 AUC on average across the 16 benchmark datasets, and up to −0.024 on some. The linear model does its own median imputation *with* missingness indicator columns.

## If it fails

Read the traceback, then retry once with explicit column names:

```
args=["--target", "target", "--id-col", "row_id", "--mode", "fast", "--out", "sub.csv"]
```

If that also fails, submit any CSV that already exists, select it, and stop. Do not rewrite this script.

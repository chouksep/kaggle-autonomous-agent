---
name: tabular-automl
description: Pre-tested AutoML pipeline for tabular binary classification. Loads train.csv/test.csv, handles categorical and missing values, fits a cross-validated model zoo (regularized LightGBM, HistGradientBoosting, CatBoost, regularized logistic regression), and writes a submission from the CV-best model or blend. Use this instead of writing modelling code by hand.
---

# Tabular AutoML Skill

This skill contains a single tested script that does the entire modelling job. It is faster, cheaper and far more reliable than authoring code turn-by-turn.

## Script: `scripts/run_automl.py`

Reads `train.csv` and `test.csv` from the working directory, writes a submission CSV matching `sample_submission.csv`.

### Modes

| Mode | What it does | Typical runtime |
|---|---|---|
| `--mode fast` | One regularized LightGBM, 3-fold CV. Use this for the safety submission. | under 60s |
| `--mode full` | Fits the full zoo, 5-fold CV, writes the CV-best single model or auto-blend. | 2–8 min |
| `--mode blend --members a,b` | Rank-averages the named zoo members. | 2–8 min |

### Usage

This script does not run directly out of `skills/` via `run_command` -- fetch it once, write it into the working directory, then run it like any other file:

```
load_skill_resource("tabular-automl", "scripts/run_automl.py")   # returns the script's text
write_file("run_automl.py", <that text>)
```

Then, for the rest of the session:

```bash
python run_automl.py --mode fast --out sub_fast.csv
python run_automl.py --mode full --out sub_full.csv
python run_automl.py --mode blend --members lgbm_reg,logreg --out sub_blend.csv
```

### What it prints

A compact CV table, one line per zoo member, plus the chosen strategy. Example:

```
n_train=10803 n_feat=9 n_cat=9 pos_rate=0.496 missing=0.131
CV(5-fold roc_auc):
  lgbm_reg   0.8038
  hgb        0.7995
  catboost   0.8051
  logreg     0.8074
  blend_auto 0.8086
CHOSEN: blend_auto (0.8086) -> sub_full.csv
```

Read the numbers. If the top two are within 0.002, a blend of them is usually worth trying.

### Built-in behaviour worth knowing

- **Small datasets (n < 2000) automatically get a linear model in the blend.** Benchmarking across 16 datasets from this family showed blending gains about +0.013 AUC below 2000 rows and *loses* about 0.007 above it.
- Categorical columns (`cat_*`, `ord_*` string levels) are handled natively by the tree models and one-hot encoded for the linear model.
- Missing values are left to the tree models and median-imputed for the linear model. Do not impute manually first.
- `row_id` is dropped automatically. The output always uses the exact id column and row order of `sample_submission.csv`.

### If it fails

Read the traceback. The two realistic failure causes are a missing `target` column name or a non-standard id column. Both are handled by flags:

```bash
python run_automl.py --target target --id-col row_id --mode fast --out sub.csv
```

Do not rewrite this script. Fall back to `--mode fast`, submit whatever it produces, and select it.

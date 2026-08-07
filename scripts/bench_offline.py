#!/usr/bin/env python3
"""Offline benchmark of the AutoML pipeline across all 16 training datasets.

This is the cheap iteration loop: it costs zero LLM tokens and shows whether
a modelling change is actually worth anything before a Kaggle submission
(or a local API key) is spent on a full agent session.

Usage
-----
    python scripts/bench_offline.py                  # all 16 datasets
    python scripts/bench_offline.py --datasets 1 5 13 15
    python scripts/bench_offline.py --folds 5 --out bench_results.csv

Reference numbers from the initial run (5-fold, ROC AUC, mean over 16 datasets):

    oracle best-per-dataset   0.7975
    regularized LightGBM      0.7886
    rank blend (3 models)     0.7880
    HistGradientBoosting      0.7856
    default LightGBM          0.7798
    logistic regression       0.7295

Anything that does not beat 0.7886 mean is not an improvement.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 0


def prep(df: pd.DataFrame):
    y = df["target"].values.astype(int)
    X = df.drop(columns=[c for c in ("row_id", "target") if c in df.columns])
    cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    Xc = X.copy()
    for c in cats:
        Xc[c] = pd.Categorical(X[c].astype(str))
    Xd = pd.get_dummies(X, columns=cats, dummy_na=True)
    Xd = Xd.loc[:, Xd.nunique() > 1]
    return y, Xc, Xd, cats


def cv(model_fn, X, y, folds, cats=None, use_cat_idx=False):
    oof = np.zeros(len(y))
    skf = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    for trn, val in skf.split(X, y):
        m = model_fn()
        if use_cat_idx and cats:
            m.set_params(categorical_features=[X.columns.get_loc(c) for c in cats])
        m.fit(X.iloc[trn], y[trn])
        oof[val] = m.predict_proba(X.iloc[val])[:, 1]
    return oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=int, nargs="*", default=list(range(1, 17)))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="bench_results.csv")
    args = ap.parse_args()

    import lightgbm as lgb

    rows = []
    for i in args.datasets:
        d = DATA / f"train_{i:02d}"
        if not (d / "train.csv").exists():
            print(f"[skip] {d} not found")
            continue

        y, Xc, Xd, cats = prep(pd.read_csv(d / "train.csv"))
        oofs = {}

        oofs["lgbm_default"] = cv(
            lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=SEED, verbose=-1),
            Xc, y, args.folds,
        )
        oofs["lgbm_reg"] = cv(
            lambda: lgb.LGBMClassifier(
                n_estimators=1500, learning_rate=0.02, num_leaves=15, min_child_samples=40,
                colsample_bytree=0.7, subsample=0.8, subsample_freq=1, reg_lambda=5.0,
                random_state=SEED, verbose=-1,
            ),
            Xc, y, args.folds,
        )
        oofs["hgb"] = cv(
            lambda: HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=SEED,
            ),
            Xc, y, args.folds, cats=cats, use_cat_idx=True,
        )
        oofs["logreg"] = cv(
            lambda: make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                LogisticRegression(max_iter=3000, C=0.3 if len(y) >= 2000 else 0.1),
            ),
            Xd, y, args.folds,
        )

        rk = lambda v: rankdata(v) / len(v)  # noqa: E731
        oofs["blend_trees"] = np.mean([rk(oofs["lgbm_reg"]), rk(oofs["hgb"])], axis=0)
        oofs["blend_all"] = np.mean([rk(v) for k, v in oofs.items() if not k.startswith("blend")], axis=0)

        row = {"ds": f"train_{i:02d}", "n": len(y), "n_feat": Xc.shape[1], "n_cat": len(cats)}
        row.update({k: round(roc_auc_score(y, v), 4) for k, v in oofs.items()})
        rows.append(row)
        print(row, flush=True)

    df = pd.DataFrame(rows).set_index("ds")
    model_cols = [c for c in df.columns if c not in ("n", "n_feat", "n_cat")]

    print("\n=== mean AUC per pipeline ===")
    print(df[model_cols].mean().round(4).sort_values(ascending=False).to_string())
    print(f"\noracle (best per dataset): {df[model_cols].max(axis=1).mean():.4f}")

    small, large = df[df.n < 2000], df[df.n >= 2000]
    if len(small):
        print(f"\n--- n < 2000 ({len(small)} datasets) ---")
        print(small[model_cols].mean().round(4).sort_values(ascending=False).to_string())
    if len(large):
        print(f"\n--- n >= 2000 ({len(large)} datasets) ---")
        print(large[model_cols].mean().round(4).sort_values(ascending=False).to_string())

    df.to_csv(args.out)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()

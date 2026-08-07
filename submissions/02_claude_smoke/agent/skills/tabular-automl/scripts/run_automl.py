#!/usr/bin/env python3
"""Tabular binary-classification AutoML for the Kaggle-in-Kaggle sandbox.

Design notes
------------
Everything here is deliberately defensive: it runs inside a 60-minute session
where a crash costs the whole submission. Every optional dependency is guarded,
every model is wrapped, and the script always writes *something* usable.

Benchmarked across the 16 provided training datasets (5-fold CV, ROC AUC):

    oracle best-per-dataset   0.7975
    regularized LightGBM      0.7886
    rank blend (3 models)     0.7880
    HistGradientBoosting      0.7856
    default LightGBM          0.7798
    logistic regression       0.7295

The gap between the oracle and any fixed pipeline is ~0.009 AUC, which is the
same size as the entire public-leaderboard spread. Hence: pick per dataset by CV.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

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

SEED = 0


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load(target: str, id_col: str | None):
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("sample_submission.csv")

    if target not in train.columns:
        cands = [c for c in train.columns if c not in test.columns]
        if len(cands) != 1:
            raise SystemExit(f"cannot infer target; candidates={cands}")
        target = cands[0]

    if id_col is None:
        id_col = str(sample.columns[0])

    y = train[target].values
    if y.dtype == object or not np.issubdtype(y.dtype, np.number):
        y = pd.Categorical(y).codes
    y = y.astype(int)

    drop = [c for c in (target, id_col) if c in train.columns]
    X = train.drop(columns=drop)
    Xt = test.drop(columns=[c for c in (target, id_col) if c in test.columns])
    Xt = Xt[[c for c in X.columns if c in Xt.columns]]
    X = X[list(Xt.columns)]
    return X, y, Xt, test, sample, id_col, target


def encode(X: pd.DataFrame, Xt: pd.DataFrame):
    """Return (tree-friendly frames, one-hot frames, categorical column names)."""
    cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    Xc, Xtc = X.copy(), Xt.copy()
    for c in cats:
        levels = pd.Index(sorted(set(X[c].dropna().astype(str)) | set(Xt[c].dropna().astype(str))))
        Xc[c] = pd.Categorical(X[c].astype(str), categories=levels)
        Xtc[c] = pd.Categorical(Xt[c].astype(str), categories=levels)

    both = pd.concat([X, Xt], keys=["tr", "te"])
    dummies = pd.get_dummies(both, columns=cats, dummy_na=True)
    dummies = dummies.loc[:, dummies.nunique() > 1]
    Xd, Xtd = dummies.loc["tr"], dummies.loc["te"]
    return Xc, Xtc, Xd, Xtd, cats


# ----------------------------------------------------------------------------
# model zoo
# ----------------------------------------------------------------------------
def build_zoo(n_rows: int, cats: list[str], mode: str):
    """Each entry: name -> (factory, which_encoding)."""
    zoo: dict = {}

    try:
        import lightgbm as lgb

        if mode == "fast":
            zoo["lgbm_reg"] = (
                lambda: lgb.LGBMClassifier(
                    n_estimators=400, learning_rate=0.05, num_leaves=31,
                    random_state=SEED, verbose=-1,
                ),
                "cat",
            )
        else:
            zoo["lgbm_reg"] = (
                lambda: lgb.LGBMClassifier(
                    n_estimators=1500, learning_rate=0.02, num_leaves=15,
                    min_child_samples=40, colsample_bytree=0.7,
                    subsample=0.8, subsample_freq=1, reg_lambda=5.0,
                    random_state=SEED, verbose=-1,
                ),
                "cat",
            )
    except Exception as exc:  # pragma: no cover
        print(f"[warn] lightgbm unavailable: {exc}")

    if mode != "fast":
        zoo["hgb"] = (
            lambda: HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=SEED,
            ),
            "cat_idx",
        )

        try:
            from catboost import CatBoostClassifier

            zoo["catboost"] = (
                lambda: CatBoostClassifier(
                    iterations=1200, learning_rate=0.03, depth=5, l2_leaf_reg=6.0,
                    random_seed=SEED, verbose=0, allow_writing_files=False,
                ),
                "catboost",
            )
        except Exception as exc:
            print(f"[warn] catboost unavailable: {exc}")

        # Linear model earns its place mainly on small data (see module docstring).
        C = 0.3 if n_rows >= 2000 else 0.1
        zoo["logreg"] = (
            lambda: make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=3000, C=C),
            ),
            "dummy",
        )

    if not zoo:
        raise SystemExit("no models available in this environment")
    return zoo


def fit_predict(name, factory, kind, Xc, Xtc, Xd, Xtd, cats, y, n_splits):
    """Return (oof_predictions, test_predictions) or (None, None) on failure."""
    if kind == "cat":
        A, B = Xc, Xtc
    elif kind == "cat_idx":
        A, B = Xc, Xtc
    elif kind == "catboost":
        # CatBoost rejects NaN in categorical columns -- use an explicit sentinel.
        A, B = Xc.copy(), Xtc.copy()
        for c in cats:
            A[c] = A[c].astype(str).replace({"nan": "__NA__"}).fillna("__NA__")
            B[c] = B[c].astype(str).replace({"nan": "__NA__"}).fillna("__NA__")
    else:
        A, B = Xd, Xtd

    oof = np.zeros(len(y))
    test_pred = np.zeros(len(B))
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)

    try:
        for trn, val in skf.split(A, y):
            model = factory()
            if kind == "cat_idx" and cats:
                model.set_params(categorical_features=[A.columns.get_loc(c) for c in cats])
            if kind == "catboost":
                model.fit(A.iloc[trn], y[trn], cat_features=cats)
            else:
                model.fit(A.iloc[trn], y[trn])
            oof[val] = model.predict_proba(A.iloc[val])[:, 1]
            test_pred += model.predict_proba(B)[:, 1] / n_splits
    except Exception as exc:
        print(f"[warn] {name} failed: {type(exc).__name__}: {str(exc)[:160]}")
        return None, None

    return oof, test_pred


def rank01(v: np.ndarray) -> np.ndarray:
    return rankdata(v) / len(v)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fast", "full", "blend"], default="full")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--members", default="", help="comma-separated zoo names for --mode blend")
    ap.add_argument("--target", default="target")
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--folds", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    X, y, Xt, test, sample, id_col, target = load(args.target, args.id_col)
    Xc, Xtc, Xd, Xtd, cats = encode(X, Xt)
    n = len(y)

    print(
        f"n_train={n} n_test={len(Xt)} n_feat={X.shape[1]} n_cat={len(cats)} "
        f"pos_rate={y.mean():.3f} missing={X.isna().mean().mean():.3f}"
    )

    folds = args.folds or (3 if args.mode == "fast" else (5 if n >= 2000 else 10))
    zoo = build_zoo(n, cats, "fast" if args.mode == "fast" else "full")

    oofs: dict[str, np.ndarray] = {}
    preds: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}

    for name, (factory, kind) in zoo.items():
        oof, pred = fit_predict(name, factory, kind, Xc, Xtc, Xd, Xtd, cats, y, folds)
        if oof is None:
            continue
        oofs[name], preds[name] = oof, pred
        scores[name] = roc_auc_score(y, oof)

    if not scores:
        # Absolute last resort: never leave the agent with nothing to submit.
        print("[error] every model failed; writing constant predictions")
        out = sample.copy()
        out[out.columns[1]] = 0.5
        out.to_csv(args.out, index=False)
        sys.exit(0)

    # --- blending -----------------------------------------------------------
    if args.mode == "blend" and args.members:
        members = [m.strip() for m in args.members.split(",") if m.strip() in oofs]
        if not members:
            print(f"[warn] no valid members in {args.members!r}; using all")
            members = list(oofs)
        blend_oof = np.mean([rank01(oofs[m]) for m in members], axis=0)
        blend_pred = np.mean([rank01(preds[m]) for m in members], axis=0)
        scores["blend_manual"] = roc_auc_score(y, blend_oof)
        oofs["blend_manual"], preds["blend_manual"] = blend_oof, blend_pred
    elif len(oofs) > 1:
        # Auto-blend. Small data benefits from including the linear model;
        # large data is better off with the trees only.
        members = list(oofs) if n < 2000 else [m for m in oofs if m != "logreg"] or list(oofs)
        blend_oof = np.mean([rank01(oofs[m]) for m in members], axis=0)
        blend_pred = np.mean([rank01(preds[m]) for m in members], axis=0)
        scores["blend_auto"] = roc_auc_score(y, blend_oof)
        oofs["blend_auto"], preds["blend_auto"] = blend_oof, blend_pred

    print(f"CV({folds}-fold roc_auc):")
    for name, s in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<12} {s:.4f}")

    best = max(scores, key=scores.get)
    out = sample.copy()
    out[out.columns[0]] = test[id_col].values
    out[out.columns[1]] = preds[best]
    out.to_csv(args.out, index=False)

    print(f"CHOSEN: {best} ({scores[best]:.4f}) -> {args.out}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()

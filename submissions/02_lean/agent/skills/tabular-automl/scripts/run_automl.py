#!/usr/bin/env python3
"""Time-bounded tabular binary-classification AutoML for the Kaggle-in-Kaggle sandbox.

v2 changes (after 01_baseline scored 0.812 vs a 0.822 field):
  * --time-budget: the zoo degrades gracefully instead of being killed mid-fit.
    Models run cheapest-first and each one is skipped if there isn't time.
  * Adaptive fold count: more folds on small data (where CV noise dominates),
    fewer on large data (where fits are expensive).
  * A submission file is written no matter what -- even if only one model
    survived, even if all of them failed.
  * --seed for cheap ensembling across runs.

Benchmarked across the 16 provided training datasets (5-fold CV, ROC AUC):

    oracle best-per-dataset   0.7975
    regularized LightGBM      0.7886
    rank blend (3 models)     0.7880
    HistGradientBoosting      0.7856
    default LightGBM          0.7798
    logistic regression       0.7295

The oracle-vs-fixed gap (~0.009) is the whole competition. Hence: pick by CV.
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

T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


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

    X = train.drop(columns=[c for c in (target, id_col) if c in train.columns])
    Xt = test.drop(columns=[c for c in (target, id_col) if c in test.columns])
    common = [c for c in X.columns if c in Xt.columns]
    return X[common], y, Xt[common], test, sample, id_col


def encode(X: pd.DataFrame, Xt: pd.DataFrame):
    cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    Xc, Xtc = X.copy(), Xt.copy()
    for c in cats:
        levels = pd.Index(sorted(set(X[c].dropna().astype(str)) | set(Xt[c].dropna().astype(str))))
        Xc[c] = pd.Categorical(X[c].astype(str), categories=levels)
        Xtc[c] = pd.Categorical(Xt[c].astype(str), categories=levels)

    both = pd.concat([X, Xt], keys=["tr", "te"])
    dummies = pd.get_dummies(both, columns=cats, dummy_na=True)
    dummies = dummies.loc[:, dummies.nunique() > 1]
    return Xc, Xtc, dummies.loc["tr"], dummies.loc["te"], cats


# ----------------------------------------------------------------------------
# zoo -- ordered cheapest-first so a tight time budget still gets the good ones
# ----------------------------------------------------------------------------
def build_zoo(n_rows: int, mode: str, seed: int):
    zoo: list[tuple[str, object, str, float]] = []  # (name, factory, encoding, cost_weight)

    try:
        import lightgbm as lgb

        if mode == "fast":
            zoo.append((
                "lgbm_fast",
                lambda: lgb.LGBMClassifier(
                    n_estimators=300, learning_rate=0.05, num_leaves=31,
                    random_state=seed, verbose=-1,
                ),
                "cat", 1.0,
            ))
            return zoo

        zoo.append((
            "lgbm_reg",
            lambda: lgb.LGBMClassifier(
                n_estimators=1500, learning_rate=0.02, num_leaves=15,
                min_child_samples=40, colsample_bytree=0.7,
                subsample=0.8, subsample_freq=1, reg_lambda=5.0,
                random_state=seed, verbose=-1,
            ),
            "cat", 3.0,
        ))
    except Exception as exc:
        print(f"[warn] lightgbm unavailable: {exc}")

    if mode == "fast":
        return zoo

    # Cheap and it wins outright on ~a third of the datasets in this family.
    #
    # add_indicator=True appends a binary "was missing" column per imputed
    # feature. Median imputation otherwise destroys that information, and
    # missingness here is mildly predictive (a model on the missingness mask
    # alone averages 0.527 AUC, 0.587 on train_02). Measured over the 16
    # datasets: mean +0.0010, best +0.0078 (train_05), worst -0.0017. Small,
    # but the upside tail is 4x the downside tail, so it is worth taking.
    C = 0.3 if n_rows >= 2000 else 0.1
    zoo.append((
        "logreg",
        lambda: make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
            LogisticRegression(max_iter=3000, C=C, random_state=seed),
        ),
        "dummy", 1.0,
    ))

    zoo.append((
        "hgb",
        lambda: HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=seed,
        ),
        "cat_idx", 2.5,
    ))

    try:
        from catboost import CatBoostClassifier

        zoo.append((
            "catboost",
            lambda: CatBoostClassifier(
                iterations=1200, learning_rate=0.03, depth=5, l2_leaf_reg=6.0,
                random_seed=seed, verbose=0, allow_writing_files=False,
            ),
            "catboost", 5.0,
        ))
    except Exception as exc:
        print(f"[warn] catboost unavailable: {exc}")

    return zoo


def frames_for(kind, Xc, Xtc, Xd, Xtd, cats):
    if kind == "dummy":
        return Xd, Xtd
    if kind == "catboost":
        # CatBoost rejects NaN in categorical columns -- use an explicit sentinel.
        # Note: .astype(str) does NOT reliably stringify NaN for Categorical
        # dtype, so fill first, then cast.
        A, B = Xc.copy(), Xtc.copy()
        for c in cats:
            A[c] = A[c].astype(object).where(A[c].notna(), "__NA__").astype(str)
            B[c] = B[c].astype(object).where(B[c].notna(), "__NA__").astype(str)
        return A, B
    return Xc, Xtc


def fit_predict(name, factory, kind, A, B, cats, y, folds, deadline):
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(B))
    skf = StratifiedKFold(folds, shuffle=True, random_state=0)
    done = 0

    try:
        for trn, val in skf.split(A, y):
            if deadline is not None and elapsed() > deadline:
                print(f"[time] {name}: stopping after {done}/{folds} folds")
                break
            model = factory()
            if kind == "cat_idx" and cats:
                model.set_params(categorical_features=[A.columns.get_loc(c) for c in cats])
            if kind == "catboost":
                model.fit(A.iloc[trn], y[trn], cat_features=cats)
            else:
                model.fit(A.iloc[trn], y[trn])
            oof[val] = model.predict_proba(A.iloc[val])[:, 1]
            test_pred += model.predict_proba(B)[:, 1]
            done += 1
    except Exception as exc:
        print(f"[warn] {name} failed: {type(exc).__name__}: {str(exc)[:160]}")
        return None, None

    if done == 0:
        return None, None
    if done < folds:
        # Partial CV: score only the folds we actually filled.
        mask = oof != 0
        if mask.sum() < 50 or len(np.unique(y[mask])) < 2:
            return None, None
        return (oof, mask), test_pred / done
    return (oof, np.ones(len(y), bool)), test_pred / done


def rank01(v: np.ndarray) -> np.ndarray:
    return rankdata(v) / len(v)


def write_submission(sample, test, id_col, values, out):
    sub = sample.copy()
    sub[sub.columns[0]] = test[id_col].values
    sub[sub.columns[1]] = values
    sub.to_csv(out, index=False)


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fast", "full"], default="full")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="seconds; models are skipped rather than killed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default="target")
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--folds", type=int, default=None)
    args = ap.parse_args()

    X, y, Xt, test, sample, id_col = load(args.target, args.id_col)
    Xc, Xtc, Xd, Xtd, cats = encode(X, Xt)
    n = len(y)

    print(
        f"n_train={n} n_test={len(Xt)} n_feat={X.shape[1]} n_cat={len(cats)} "
        f"pos_rate={y.mean():.3f} missing={X.isna().mean().mean():.3f}"
    )

    # Small data => CV noise dominates => more folds. Large data => fits cost more.
    if args.folds:
        folds = args.folds
    elif args.mode == "fast":
        folds = 3
    elif n < 2000:
        folds = 10
    elif n < 20000:
        folds = 5
    else:
        folds = 4

    budget = args.time_budget
    zoo = build_zoo(n, args.mode, args.seed)

    oofs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    preds: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    weight_total = sum(w for *_, w in zoo) or 1.0

    for name, factory, kind, weight in zoo:
        if budget is not None and elapsed() > budget * 0.9:
            print(f"[time] skipping {name}: {elapsed():.0f}s of {budget:.0f}s used")
            continue

        # Give each model a slice of the remaining time proportional to its cost.
        deadline = None
        if budget is not None:
            deadline = min(budget, elapsed() + max(30.0, (budget - elapsed()) * (weight / weight_total) * 2.5))

        A, B = frames_for(kind, Xc, Xtc, Xd, Xtd, cats)
        res, pred = fit_predict(name, factory, kind, A, B, cats, y, folds, deadline)
        if res is None:
            continue
        oof, mask = res
        oofs[name], preds[name] = (oof, mask), pred
        scores[name] = roc_auc_score(y[mask], oof[mask])
        print(f"  [{elapsed():6.0f}s] {name:<10} {scores[name]:.4f}"
              + ("" if mask.all() else f"  (partial, {mask.sum()}/{len(y)} rows)"))

    if not scores:
        print("[error] every model failed; writing constant predictions")
        write_submission(sample, test, id_col, 0.5, args.out)
        sys.exit(0)

    # --- blend ---------------------------------------------------------------
    # Only blend over models with complete OOF, so the comparison is honest.
    full = [k for k in oofs if oofs[k][1].all()]
    if len(full) > 1:
        # Below ~2000 rows the linear model earns its place in the blend
        # (+0.013 in benchmark); above it, it costs about 0.007.
        members = full if n < 2000 else ([m for m in full if m != "logreg"] or full)
        if len(members) > 1:
            b_oof = np.mean([rank01(oofs[m][0]) for m in members], axis=0)
            b_pred = np.mean([rank01(preds[m]) for m in members], axis=0)
            scores["blend"] = roc_auc_score(y, b_oof)
            oofs["blend"] = (b_oof, np.ones(len(y), bool))
            preds["blend"] = b_pred
            print(f"  [{elapsed():6.0f}s] {'blend':<10} {scores['blend']:.4f}  ({'+'.join(members)})")

    print(f"CV({folds}-fold roc_auc), {len(scores)} candidates:")
    for name, s in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<12} {s:.4f}")

    # Prefer a complete-CV model when a partial one is only marginally ahead --
    # partial CV scores are optimistic and unstable.
    def keyfn(name):
        return scores[name] - (0.003 if not oofs[name][1].all() else 0.0)

    best = max(scores, key=keyfn)
    write_submission(sample, test, id_col, preds[best], args.out)
    print(f"CHOSEN: {best} (CV {scores[best]:.4f}) -> {args.out}  [{elapsed():.0f}s total]")


if __name__ == "__main__":
    main()

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

v3 additions (research-driven, each verified with scripts/bench_v2.py --
imports this file directly, zero drift from what actually runs):
  * A critical fix: an id column can crash pd.read_csv entirely if a hex
    row_id coincidentally parses as scientific notation with a huge exponent
    (numpy 2.5.1 segfaults on the overflow instead of raising). id_col is
    now always forced to string dtype at read time. This is the single
    highest-value fix in this file -- a crash forfeits the whole submission.
  * lgbm_tuned: a per-dataset CV-based iteration count (lgb.cv + early
    stopping) for lgbm_reg's own hyperparameters, gated to n<20000 where the
    cost/benefit holds (measured +0.006 mean AUC where it runs; negligible
    and expensive above that size). Hard-capped at 240s wall-clock,
    independent of --time-budget, since it runs before the outer per-model
    deadline loop even starts.
  * Tried and DROPPED: smoothed target/frequency encoding as extra features
    for LightGBM. Measured ~0 mean AUC delta on the datasets it applies to
    (n_cat>0) -- not worth the added columns/compute. See git history if
    revisiting.
  * Missingness indicators for the tree models (see encode()). Error analysis
    found the worst-performing segment on two datasets was defined by one
    feature simply being missing -- logreg already got this via
    SimpleImputer(add_indicator=True), the trees didn't. Measured: +0.004
    mean AUC on lgbm_tuned specifically (where it runs), oracle ceiling
    essentially flat (+0.0001) -- a real but modest effect, kept since it's
    free and nothing regressed.
  * A second critical fix, found while benchmarking the above: LightGBM and
    HistGB both default to using every available core, and on this dev
    machine that measured a ~900x slowdown (0.23s vs 204s for 200 trees on
    3.5k rows) -- an OpenMP thread-pool pathology, not a per-dataset cost.
    Both are now pinned to a fixed thread count. A single .fit() call can't
    be interrupted mid-flight by --time-budget's between-model checks, so an
    unlucky environment hitting this could lose the whole session to one
    model; this is cheap insurance, not just a local speed fix.
  * Two new ensemble candidates alongside the existing plain blend, both in
    combine_predictions(): blend_weighted (an NNLS-fit weighted average) and
    stack (a logistic-regression meta-model). Both operate on rank01-
    transformed out-of-fold predictions, fit via an honest repeated 2-fold
    cross-fit (never the raw dataset features) so the meta-level
    weights/model are never trained on the same rows they predict. Measured
    across all 16 datasets: blend_weighted +0.0029 mean AUC vs plain blend,
    stack +0.0023 (raw per-candidate column means); oracle best-per-dataset
    rose from 0.8032 to 0.8036; no regressions beyond per-dataset noise
    (worst case -0.0033 on one dataset). Those column means overstate what
    production actually captures, though: selection is via
    max(scores, key=keyfn), so what matters is how much the SELECTED
    candidate's CV improves, not the new columns' own means. Simulating the
    actual selection rule on the same benchmark data, the mean CV of the
    picked candidate moves from 0.8032 to 0.8036 -- only +0.0004, because the
    new candidates only change which model wins on 7 of 16 datasets (most by
    small margins); two of the biggest per-column gains (train_16, train_11)
    contribute exactly zero pick-improvement because a different candidate
    (catboost, lgbm_reg respectively) already tied or nearly tied for the
    win there. Both are just two more entries in the scores/oofs/preds
    dicts, so the existing CV-based selection picks whichever candidate wins
    per dataset automatically -- no new selection logic needed.

v4 additions (04_reflective variant only -- forked from 03_skilled, not yet
promoted to the real submission):
  * A live error-analysis pass (diagnose_weak_segment()) runs after every
    --mode full, reusing the winning model's own OOF predictions (no extra
    model fit) to honest-split-diagnose a statistically significant weak
    segment, same p<0.01 bar as scripts/error_analysis.py. If found, the
    orchestrator can run --mode apply-recipe --recipe discover_weak_segments,
    which converts THAT diagnosed segment into a binary feature and re-runs
    the zoo -- deliberately NOT the archived discover_weak_segments()'s own
    from-scratch rediscovery (a separate preliminary model + its own
    honest-split), since the real zoo's OOF is already honest and a cheap
    proxy model adds nothing.
  * Measured via scripts/bench_v2.py --reflective across all 16 datasets:
    a segment was diagnosed on 12/16 (vs the archived version's 2/12 on the
    same 12 -- using the real zoo's OOF instead of a cheap preliminary model
    finds real segments far more often). Of those 12: 8 wins / 3 losses /
    1 tie, mean CV delta +0.0003 on the datasets where it was attempted.
    Comparable in size to the ensemble combiners' own accepted
    selection-rule delta (+0.0004, kept) and a materially better outcome
    than the archived discover_weak_segments' blanket-always-on approach
    (+0.00004, not wired in) -- targeting the actual diagnosed segment
    instead of a blanket flag looks like a real, if modest, improvement --
    though on much weaker evidence than the combiners had (single-seed,
    un-repeated, n=12; a sign test on 8W/3L is not distinguishable from a
    coin flip at p~0.23, and half the wins are within one rounding unit of
    zero).
  * Before promoting this to 03_skilled, two things need fixing/measuring
    first, both found by review rather than by the sweep -- the sweep
    couldn't have caught either:
      1. The with-recipe CV comparison is not fully honest: the diagnosed
         segment's boundaries are fit on labels from half the training
         rows (auc_attribution + the diagnostic tree both use y), and that
         SAME segment definition is then used as a feature for the
         with-recipe zoo's own cross-validation -- a row that helped
         define the segment can later score itself in a validation fold.
         The bias favors the recipe, and the measured effect is small
         enough (+0.0003 mean, sign-test p~0.23 on 8W/3L/1T, half the wins
         within one rounding unit, and the whole mean driven mostly by one
         dataset) that an unmeasured optimistic bias of any size matters.
         The real test: re-derive the segment INSIDE each CV fold rather
         than once on half the data, and see if the effect survives.
      2. diagnose_weak_segment() is only ever fed the winning model's OOF
         when that model completed ALL folds (see the args.mode=="full"
         guard above) -- this closes a real corruption risk on
         hidden datasets where a model could time out mid-CV, but it also
         means the offline sweep (which already only benchmarks complete
         models) and the diagnosis path now agree, which was NOT true
         before this fix.
    Single-split honest-split noise (the same noise source
    combine_predictions() needed 5-repeat averaging to control) affects
    WHICH segment gets diagnosed or whether one does at all, not the
    with-recipe CV that actually decides submission -- that CV is a full
    4/5/10-fold zoo comparison, a much more stable arbiter. Lower priority
    than the two issues above, but still worth measuring before promotion.

Benchmarked across the 16 provided training datasets (per-dataset adaptive
folds, ROC AUC, catboost + lgbm_tuned + missingness indicators included):

    # oracle moved 0.8021 -> 0.8036 in this refresh; only the 0.8032 -> 0.8036
    # slice is from the ensemble combiners above -- 0.8021 -> 0.8032 was this
    # table catching up to already-shipped fixes it never reflected.
    oracle best-per-dataset   0.8036
    blend_weighted            0.8025
    stack                     0.8019
    blend (per-dataset)       0.7996
    lgbm_tuned (where run)    0.7982
    catboost                  0.7956
    regularized LightGBM      0.7895
    HistGradientBoosting      0.7883
    logistic regression       0.7331

The oracle-vs-fixed gap is the whole competition. Hence: pick by CV.
"""

from __future__ import annotations

import os

# Must be set before sklearn/lightgbm import -- both use OpenMP internally,
# and an unbounded thread count (their default) measured a ~900x slowdown in
# testing on a many-core machine (0.23s vs 204s for 200 trees on 3.5k rows) --
# almost certainly an OpenMP thread-pool pathology, not a per-dataset cost.
# LightGBM's own n_jobs is also pinned per-instance below; HistGB has no
# constructor param for this, so it can only be capped via the environment.
# A single .fit() call can't be interrupted mid-flight by --time-budget's
# between-model checks, so leaving this unbounded risks losing the whole
# session to one model on an unlucky environment.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_ind
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings("ignore")

T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


ERROR_ANALYSIS_PATH = "error_analysis.json"


# ----------------------------------------------------------------------------
# working directory
# ----------------------------------------------------------------------------
def find_workdir() -> "Path":
    """Locate the directory holding the competition CSVs, and chdir into it.

    Skill scripts launched via `run_skill_script` do NOT run with cwd set to the
    sandbox working directory. The harness materialises the skill into its own
    directory and runs `runpy.run_path('scripts/run_automl.py')` from there, so
    a bare `pd.read_csv("train.csv")` raises FileNotFoundError even though
    `run_command("ls")` clearly shows train.csv. This cost a whole local
    evaluation to diagnose -- do not "simplify" this away.

    Resolving here also means the output CSV lands next to the data, which is
    where `submit_predictions` resolves relative paths from.
    """
    from pathlib import Path

    def has_data(d: Path) -> bool:
        try:
            return (d / "train.csv").is_file() and (d / "test.csv").is_file()
        except OSError:
            return False

    cwd = Path.cwd()
    candidates = [cwd, *list(cwd.parents)[:4],
                  Path("/work"), Path("/kaggle/working"), Path("/kaggle/input"),
                  Path.home()]

    seen = set()
    for d in candidates:
        if d in seen:
            continue
        seen.add(d)
        if has_data(d):
            return d

    # Last resort: shallow scan of likely roots.
    for root in (Path("/work"), Path("/kaggle"), Path("/tmp")):
        if not root.is_dir():
            continue
        try:
            for sub in list(root.iterdir())[:200]:
                if sub.is_dir() and has_data(sub):
                    return sub
        except OSError:
            continue

    raise SystemExit(
        f"could not locate train.csv/test.csv (cwd={cwd}, "
        f"contents={sorted(p.name for p in list(cwd.iterdir())[:20])})"
    )


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load(target: str, id_col: str | None):
    # The id column must never be numeric-inferred: a purely coincidental hex
    # row_id like "5e1541736541025" is valid scientific-notation syntax (5 *
    # 10^1541736541025), and numpy 2.5.1's float parser SEGFAULTS on the
    # absurd exponent instead of raising OverflowError -- confirmed
    # reproducible on one of the 16 training datasets, and it can happen on
    # any dataset's row_id column by chance. A crash here forfeits the whole
    # session, so the id column is forced to string dtype before pandas gets
    # a chance to infer anything.
    if id_col is None:
        id_col = str(pd.read_csv("sample_submission.csv", nrows=0).columns[0])
    str_dtype = {id_col: str}

    train = pd.read_csv("train.csv", dtype=str_dtype)
    test = pd.read_csv("test.csv", dtype=str_dtype)
    sample = pd.read_csv("sample_submission.csv", dtype=str_dtype)

    if target not in train.columns:
        cands = [c for c in train.columns if c not in test.columns]
        if len(cands) != 1:
            raise SystemExit(f"cannot infer target; candidates={cands}")
        target = cands[0]

    y = train[target].values
    if y.dtype == object or not np.issubdtype(y.dtype, np.number):
        y = pd.Categorical(y).codes
    y = y.astype(int)

    X = train.drop(columns=[c for c in (target, id_col) if c in train.columns])
    Xt = test.drop(columns=[c for c in (target, id_col) if c in test.columns])
    common = [c for c in X.columns if c in Xt.columns]
    X, Xt = X[common], Xt[common]

    # Near-empty columns: NOT dropped, on measured evidence, not guesswork.
    # train_07's feature_16 is 80.5% missing (the worst of any column across
    # all 16 known datasets) -- tested dropping it at that exact threshold
    # and every candidate got WORSE (lgbm_reg -0.0065, hgb -0.0060, blend
    # -0.0058, catboost -0.0044, on bench_v2.py --dataset 7). The ~20% of
    # rows where it IS present apparently carry real signal; the
    # missingness indicator (see encode()) already flags absence separately,
    # so the tree models get both "was it there" and "what was the value"
    # rather than losing the latter. Threshold set well above the worst
    # column actually observed -- this is dormant insurance against a
    # genuinely degenerate future column (99%+ missing), not an active
    # behavior change on any of the 16 known datasets.
    miss_rate = X.isna().mean()
    near_empty = miss_rate[miss_rate > 0.98].index.tolist()
    if near_empty:
        print(f"[warn] dropping near-empty columns (>98% missing): {near_empty}")
        X = X.drop(columns=near_empty)
        Xt = Xt.drop(columns=near_empty)

    return X, y, Xt, test, sample, id_col


def log_transform_skewed(X: pd.DataFrame, Xt: pd.DataFrame):
    """log1p numeric columns that are heavily right-skewed and non-negative.

    Applied globally (not scoped to logreg) because it's monotonic: tree
    models only ever act on rank order within a column, so this is a
    complete no-op for lgbm/hgb/catboost's splits, and only actually changes
    anything for logreg's linear decision boundary, which is sensitive to
    heavy tails in a way trees aren't. Test can have a small number of
    negative values even where train has none (sampling noise) -- clip at 0
    first rather than letting log1p produce NaN, which SimpleImputer would
    otherwise silently treat as "missing" and median-impute away.
    """
    X2, Xt2 = X.copy(), Xt.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            continue
        vals = X[c].dropna()
        if len(vals) == 0 or (vals < 0).any():
            continue
        if vals.skew() > 1.0:
            X2[c] = np.log1p(X[c])
            Xt2[c] = np.log1p(Xt[c].clip(lower=0))
    return X2, Xt2


def duplicate_groups(X: pd.DataFrame) -> np.ndarray | None:
    """Integer group id per row; rows with identical feature values share a
    group. Returns None if there are no duplicates (the common case) so
    callers can skip group-aware CV entirely when it isn't needed.

    Found on real data, not hypothetical: train_06 has 768/10803 (~7%)
    duplicate feature-rows, and roughly half of those duplicate GROUPS have
    CONFLICTING labels (same features, different target) -- irreducible
    noise in the data-generating process, not a cleaning bug, and the reason
    this dataset caps out around 0.80 AUC regardless of model. Dropping
    duplicates would be wrong here (which copy's label would you keep?).
    The actual risk is a duplicate group split across CV folds: the model
    then gets to see an identical row's label in its training fold and
    "predicts" the fold-mate almost for free, inflating CV. Group-aware CV
    (StratifiedGroupKFold, keeping each duplicate group in one fold) fixes
    that for conflicting and non-conflicting groups alike, with no need to
    decide which row to keep.
    """
    key = X.astype(str).agg("|".join, axis=1)
    codes, _ = pd.factorize(key)
    if len(np.unique(codes)) == len(codes):
        return None
    return codes


def encode(X: pd.DataFrame, Xt: pd.DataFrame):
    cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    Xc, Xtc = X.copy(), Xt.copy()
    for c in cats:
        levels = pd.Index(sorted(set(X[c].dropna().astype(str)) | set(Xt[c].dropna().astype(str))))
        Xc[c] = pd.Categorical(X[c].astype(str), categories=levels)
        Xtc[c] = pd.Categorical(Xt[c].astype(str), categories=levels)

    # Missingness indicators for the TREE models. Error analysis on the 16
    # known datasets found the single worst-performing segment on two of
    # them is defined by one feature simply being missing (not an extreme
    # value) -- e.g. train_14's worst leaf is exactly "feature_22 is NaN",
    # AUC contribution 0.77 vs 0.94-0.99 elsewhere. LightGBM/CatBoost/HistGB
    # only see the raw NaN and route it via their own native split-finding,
    # with no explicit "this row is one of the ones missing this feature"
    # signal. logreg already gets this via SimpleImputer(add_indicator=True)
    # -- this extends the same idea to the rest of the zoo.
    for c in X.columns:
        if X[c].isna().any() or Xt[c].isna().any():
            Xc[f"{c}_was_missing"] = X[c].isna().astype(np.int8)
            Xtc[f"{c}_was_missing"] = Xt[c].isna().astype(np.int8)

    both = pd.concat([X, Xt], keys=["tr", "te"])
    dummies = pd.get_dummies(both, columns=cats, dummy_na=True)
    dummies = dummies.loc[:, dummies.nunique() > 1]
    return Xc, Xtc, dummies.loc["tr"], dummies.loc["te"], cats


# ----------------------------------------------------------------------------
# zoo -- ordered cheapest-first so a tight time budget still gets the good ones
# ----------------------------------------------------------------------------
def tune_lgbm(A, y, cats, seed, n_rows, imbalance_kwargs=None):
    """Cheap, dependency-free companion to lgbm_reg: a single lgb.cv() call
    with early stopping picks a per-dataset iteration count for the SAME
    hyperparameters lgbm_reg already uses, instead of a fixed n_estimators=
    1500 regardless of dataset size. This is the CV-based early-stopping
    fix -- carving out one held-out split for stopping is unreliable on the
    small end of this family (down to 500 rows); lgb.cv's internal k-fold
    average is the standard, cheaper alternative. No optuna needed.

    An earlier version also swept a small (num_leaves, reg_lambda) grid --
    dropped after it cost 25 minutes on the largest dataset (four lgb.cv
    calls) for a gain of +0.002, worse than CatBoost anyway. One cv() call
    only.

    This runs during zoo construction, BEFORE the outer per-model deadline
    loop (see fit_predict's projected-cost check) even starts -- so it is
    NOT covered by --time-budget. Measured: on a dataset with 15 categorical
    columns, LightGBM's categorical split-finding made a single cv() call
    slow enough to eat a real fraction of a 900s budget on its own. A hard
    wall-clock cap here, independent of the outer budget, closes that gap --
    worst case, tuning aborts and lgbm_tuned is simply skipped (see the
    `if tuned_params:` guard below), which is always safe.

    num_threads is pinned (never left at LightGBM's n_jobs=-1 default): on
    this machine, unbounded thread count measured a ~900x slowdown (0.23s
    vs 204s for 200 trees on 3.5k rows) -- almost certainly an OpenMP
    thread-pool pathology, not a per-dataset cost. A single .fit() call
    can't be interrupted mid-flight by --time-budget's between-model checks,
    so an environment that hits this could blow the whole session on one
    model. A bounded thread count is cheap insurance against an unknown
    failure mode, not just a speed tweak.
    """
    import lightgbm as lgb

    max_rounds = 1500 if n_rows < 20000 else 600
    params = dict(
        objective="binary", metric="auc", num_leaves=15, learning_rate=0.02,
        reg_lambda=5.0, min_child_samples=40, colsample_bytree=0.7,
        subsample=0.8, subsample_freq=1, verbosity=-1, seed=seed, num_threads=4,
        **(imbalance_kwargs or {}),
    )

    tune_deadline = elapsed() + 240.0  # hard cap, independent of --time-budget

    def _timeout_guard(_env):
        if elapsed() > tune_deadline:
            raise TimeoutError(f"tune_lgbm exceeded its {240}s cap")

    try:
        ds = lgb.Dataset(A, label=y, categorical_feature=(cats or "auto"), free_raw_data=False)
        cv_res = lgb.cv(
            params, ds, num_boost_round=max_rounds, nfold=4, stratified=True, seed=seed,
            callbacks=[lgb.early_stopping(50, verbose=False), _timeout_guard],
        )
    except Exception as exc:
        print(f"[warn] tune_lgbm failed: {type(exc).__name__}: {str(exc)[:120]}")
        return None
    key = next((k for k in cv_res if k.endswith("-mean")), None)
    if key is None:
        return None
    n_rounds = max(len(cv_res[key]), 50)
    return dict(
        n_estimators=n_rounds, num_leaves=15, reg_lambda=5.0, learning_rate=0.02,
        min_child_samples=40, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
        **(imbalance_kwargs or {}),
    )


def build_zoo(n_rows: int, mode: str, seed: int, cats: list[str] | None = None,
              Xc=None, y=None):
    zoo: list[tuple[str, object, str, float]] = []  # (name, factory, encoding, cost_weight)

    # Defensive only -- every one of the 16 known datasets in this family is
    # balanced (pos_rate 0.494-0.511), so this can't be validated as a
    # measured win, only as "does nothing on data we've actually seen and
    # doesn't crash on data we haven't." Threshold is well outside anything
    # observed. HGB has no constructor-level class_weight in sklearn, so it
    # isn't covered here; not worth threading sample_weight through
    # fit_predict's generic .fit() call for a scenario with zero observed
    # instances.
    pos_rate = float(y.mean()) if y is not None else 0.5
    imbalanced = pos_rate < 0.35 or pos_rate > 0.65
    lgb_imbalance_kwargs = {"is_unbalance": True} if imbalanced else {}

    try:
        import lightgbm as lgb

        if mode == "fast":
            zoo.append((
                "lgbm_fast",
                lambda: lgb.LGBMClassifier(
                    n_estimators=300, learning_rate=0.05, num_leaves=31,
                    random_state=seed, verbose=-1, n_jobs=4,
                    **lgb_imbalance_kwargs,
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
                random_state=seed, verbose=-1, n_jobs=4,
                **lgb_imbalance_kwargs,
            ),
            "cat", 3.0,
        ))

        # CV-based iteration count for lgbm_reg's own hyperparameters.
        # Measured: +0.018 AUC on the smallest dataset (n=500, ~1.5 min added)
        # vs +0.002 on the largest (n=49432, ~9 min added, still lost to
        # CatBoost) -- matches the research finding that naive early-stopping
        # splits hurt most on small data, and the payoff just isn't there on
        # large data. Gate it off above this dataset's own size class.
        if Xc is not None and y is not None and n_rows < 20000:
            tuned_params = tune_lgbm(Xc, y, cats, seed, n_rows, lgb_imbalance_kwargs)
            if tuned_params:
                zoo.append((
                    "lgbm_tuned",
                    lambda p=tuned_params: lgb.LGBMClassifier(random_state=seed, verbose=-1, n_jobs=4, **p),
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
            # Clip to +/-3 SD post-scaling. Scoped to logreg only, never
            # applied to the tree models: clipping isn't rank-preserving the
            # way log_transform_skewed's log1p is, so it would cost the
            # trees real information at the tails. logreg's linear boundary
            # is the one that's actually sensitive to a handful of extreme
            # values dominating a fold's coefficient fit.
            FunctionTransformer(lambda A: np.clip(A, -3, 3)),
            LogisticRegression(
                max_iter=3000, C=C, random_state=seed,
                class_weight="balanced" if imbalanced else None,
            ),
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

        # CatBoost is by far the most expensive member and the sandbox CPU is
        # much slower than a dev box -- 1200 iterations on 15k rows overran a
        # 25-minute budget there. Scale iterations down as n grows.
        cb_iters = 1200 if n_rows < 5000 else (700 if n_rows < 20000 else 400)
        cb_kwargs = {"auto_class_weights": "Balanced"} if imbalanced else {}
        zoo.append((
            "catboost",
            lambda: CatBoostClassifier(
                iterations=cb_iters, learning_rate=0.05, depth=5, l2_leaf_reg=6.0,
                random_seed=seed, verbose=0, allow_writing_files=False,
                thread_count=-1, **cb_kwargs,
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


def fit_predict(name, factory, kind, A, B, cats, y, folds, deadline, groups=None):
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(B))
    if groups is not None:
        # Keep duplicate-feature rows together in one fold -- see
        # duplicate_groups()'s docstring for why this matters.
        skf = StratifiedGroupKFold(folds, shuffle=True, random_state=0)
        split_args = (A, y, groups)
    else:
        skf = StratifiedKFold(folds, shuffle=True, random_state=0)
        split_args = (A, y)
    done = 0

    try:
        for trn, val in skf.split(*split_args):
            if deadline is not None and elapsed() > deadline:
                print(f"[time] {name}: stopping after {done}/{folds} folds")
                break
            # Projected-cost check. Checking only *between* folds is not enough:
            # one slow fold can blow the whole budget (observed in the sandbox,
            # where CatBoost on 15k rows overran a 1500s budget by >10 min).
            # After the first fold we know the per-fold cost, so we stop early
            # rather than starting a fold we cannot afford to finish.
            if deadline is not None and done > 0:
                per_fold = (elapsed() - t_start) / done
                if elapsed() + per_fold > deadline:
                    print(f"[time] {name}: stopping after {done}/{folds} folds "
                          f"({per_fold:.0f}s/fold would overrun)")
                    break
            if done == 0:
                t_start = elapsed()

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


def combine_predictions(
    oof_dict: dict[str, np.ndarray], y: np.ndarray, members: list[str], seed: int = 0,
    preds: dict[str, np.ndarray] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """NNLS-weighted blend and logistic stacking as extra ensemble
    candidates over `members`' rank01 OOF predictions (never the raw
    dataset features -- the combiner's input has only len(members)
    columns, so overfitting risk stays low even on small datasets).

    Honest CV score via REPEATED 2-fold cross-fitting: each combiner is
    fit on one half of the rows and scored on the OTHER half (and vice
    versa, concatenated), so no row's reported OOF prediction came from a
    combiner that saw that row's own label -- directly analogous to why
    the member models' own OOF predictions are honest. A single random
    2-way split is noisy enough on its own to matter: measured swings of
    up to 0.0044 std at n=500, larger than this feature's whole mean
    improvement, big enough to flip which candidate wins the final
    CV-based pick on the smallest datasets in this family. 5 repeated
    splits (seed, seed+1, ..., seed+4) are averaged elementwise instead of
    using just one -- measured ~190ms even at the largest dataset
    (n=49432), so 5x that is still under a second, and it's purely
    internal here (no interface change; callers still get one final oof
    array back).

    Test-set predictions are produced differently from the honest score:
    when `preds` is given, each combiner is ALSO refit on the FULL n rows
    of OOF data and applied to `preds`, mirroring tune_lgbm's existing
    pattern (CV picks something honestly, the real artifact is fit on all
    available data). This full-data refit is NOT repeated -- it isn't
    subject to split-choice noise, since it always fits on all n rows
    regardless of which cross-fit split produced the honest score. Without
    `preds` (e.g. bench_v2.py, which never needs a deployable prediction),
    test_pred is None for every combiner.

    Returns {} if there's nothing to combine (len(members) < 2), or if
    input construction or the repeated split itself fails (if even one of
    the 5 repeats can't split -- e.g. class imbalance -- that's a signal
    the whole approach won't work for this dataset, so this bails out
    rather than partially averaging). A combiner that individually fails
    once splitting has succeeded is simply absent from the result, never a
    crash -- this step is pure upside for the rest of the zoo.
    """
    if len(members) < 2:
        return {}

    def _matrix(source: dict[str, np.ndarray]) -> np.ndarray:
        return np.column_stack([rank01(source[m]) for m in members])

    try:
        P = _matrix(oof_dict)
        P_preds = _matrix(preds) if preds is not None else None
    except Exception as exc:
        print(f"[warn] combine_predictions: input matrix construction failed: {type(exc).__name__}: {str(exc)[:160]}")
        return {}

    n_repeats = 5
    try:
        splits = []
        for i in range(n_repeats):
            skf = StratifiedKFold(2, shuffle=True, random_state=seed + i)
            splits.append(next(skf.split(P, y)))
    except Exception as exc:
        print(f"[warn] combine_predictions: cross-fit split failed: {type(exc).__name__}: {str(exc)[:160]}")
        return {}

    results: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}

    def _fit_weights(Pf: np.ndarray, yf: np.ndarray) -> np.ndarray:
        w, _ = nnls(Pf, yf.astype(float))
        if w.sum() <= 0:
            return np.full(Pf.shape[1], 1.0 / Pf.shape[1])
        return w / w.sum()

    try:
        oof_bw_sum = np.zeros(len(y))
        for h1, h2 in splits:
            w_h1 = _fit_weights(P[h1], y[h1])
            w_h2 = _fit_weights(P[h2], y[h2])
            oof_bw = np.empty(len(y))
            oof_bw[h2] = P[h2] @ w_h1
            oof_bw[h1] = P[h1] @ w_h2
            oof_bw_sum += oof_bw
        oof_bw = oof_bw_sum / n_repeats

        test_bw = None
        if P_preds is not None:
            w_full = _fit_weights(P, y)
            test_bw = P_preds @ w_full

        results["blend_weighted"] = (oof_bw, test_bw)
    except Exception as exc:
        print(f"[warn] blend_weighted failed: {type(exc).__name__}: {str(exc)[:160]}")

    try:
        oof_st_sum = np.zeros(len(y))
        for h1, h2 in splits:
            m_h1 = LogisticRegression(max_iter=1000, random_state=seed).fit(P[h1], y[h1])
            m_h2 = LogisticRegression(max_iter=1000, random_state=seed).fit(P[h2], y[h2])
            oof_st = np.empty(len(y))
            oof_st[h2] = m_h1.predict_proba(P[h2])[:, 1]
            oof_st[h1] = m_h2.predict_proba(P[h1])[:, 1]
            oof_st_sum += oof_st
        oof_st = oof_st_sum / n_repeats

        test_st = None
        if P_preds is not None:
            m_full = LogisticRegression(max_iter=1000, random_state=seed).fit(P, y)
            test_st = m_full.predict_proba(P_preds)[:, 1]

        results["stack"] = (oof_st, test_st)
    except Exception as exc:
        print(f"[warn] stack failed: {type(exc).__name__}: {str(exc)[:160]}")

    return results


# ----------------------------------------------------------------------------
# error analysis + recipes
# ----------------------------------------------------------------------------
def auc_attribution(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Per-example contribution to the overall AUC (Mann-Whitney identity).
    mean(attr[y==1]) == mean(attr[y==0]) == roc_auc_score(y, scores). Ported
    from scripts/error_analysis.py -- that script isn't bundled into the
    Kaggle submission, so the math is duplicated here rather than imported.
    """
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    attr = np.zeros(len(y))

    rank_all = rankdata(scores, method="average")
    pos_mask = y == 1
    rank_pos = rankdata(scores[pos_mask], method="average")
    attr[pos_mask] = (rank_all[pos_mask] - rank_pos) / n_neg

    neg_mask = y == 0
    rank_neg = rankdata(scores[neg_mask], method="average")
    n_pos_leq = rank_all[neg_mask] - rank_neg
    attr[neg_mask] = 1.0 - (n_pos_leq / n_pos)
    return attr


def _leaf_conditions(tree_, leaf_id: int) -> list[tuple[int, float, str]]:
    """Walk a fitted sklearn Tree's raw arrays from the root to `leaf_id`,
    returning the (feature_index, threshold, direction) splits that define
    it. direction is "<=" for the left branch, ">" for the right."""
    path: list[tuple[int, float, str]] = []

    def dfs(node: int) -> bool:
        if node == leaf_id:
            return True
        left, right = tree_.children_left[node], tree_.children_right[node]
        if left == -1:  # true leaf, not the one we're looking for
            return False
        feat, thr = int(tree_.feature[node]), float(tree_.threshold[node])
        path.append((feat, thr, "<="))
        if dfs(left):
            return True
        path.pop()
        path.append((feat, thr, ">"))
        if dfs(right):
            return True
        path.pop()
        return False

    dfs(0)
    return path


def _apply_segment_mask(df_filled: pd.DataFrame, conditions: list[tuple[int, float, str]]) -> np.ndarray:
    """Evaluate a leaf's split conditions against any frame sharing
    df_filled's column order (train or test alike). df_filled must already
    have NaN filled the same way the diagnostic tree was fit on."""
    mask = np.ones(len(df_filled), dtype=bool)
    for feat_idx, thr, direction in conditions:
        col = df_filled.columns[feat_idx]
        vals = df_filled[col].to_numpy()
        mask &= (vals <= thr) if direction == "<=" else (vals > thr)
    return mask


def diagnose_weak_segment(Xd: pd.DataFrame, y: np.ndarray, oof: np.ndarray, seed: int = 0) -> dict | None:
    """Honest-split weak-segment diagnostic, reusing the winning zoo
    model's own OOF predictions (already honest, from real k-fold CV) --
    unlike the archived discover_weak_segments() in scripts/error_analysis.py,
    there's no extra preliminary model fit here, just one train/eval split
    for the diagnostic tree itself, since the OOF input is already honest.

    Returns the worst significantly-underperforming segment (p<0.01,
    honest held-out half, same bar as error_analysis.py's analyze_one()) as
    {gap, pval, n, conditions}, where `conditions` is the leaf-path
    (feature_index, threshold, direction) list needed to reproduce the
    segment mask on any frame sharing Xd's columns -- or None if nothing
    significant is found, or on any failure (never raises).
    """
    try:
        attr = auc_attribution(y, oof)

        skf = StratifiedKFold(2, shuffle=True, random_state=seed)
        fit_idx, eval_idx = next(skf.split(Xd, y))

        Xd_filled = Xd.fillna(-999)
        tree = DecisionTreeRegressor(
            max_depth=3, min_samples_leaf=max(30, len(y) // 40), random_state=seed,
        )
        tree.fit(Xd_filled.iloc[fit_idx], attr[fit_idx])

        leaf_eval = tree.apply(Xd_filled.iloc[eval_idx])
        eval_attr = attr[eval_idx]
        overall_mean = eval_attr.mean()

        flagged = []
        for leaf in np.unique(leaf_eval):
            seg = eval_attr[leaf_eval == leaf]
            rest = eval_attr[leaf_eval != leaf]
            if len(seg) < 20 or len(rest) < 20:
                continue
            _tstat, pval = ttest_ind(seg, rest, equal_var=False)
            if pval < 0.01 and seg.mean() < overall_mean:
                flagged.append((leaf, overall_mean - seg.mean(), pval, len(seg)))

        if not flagged:
            return None
        flagged.sort(key=lambda f: -f[1])
        leaf, gap, pval, n_seg = flagged[0]
        conditions = _leaf_conditions(tree.tree_, int(leaf))
        return {
            "gap": round(float(gap), 4),
            "pval": float(pval),
            "n": int(n_seg),
            "conditions": conditions,
        }
    except Exception as exc:
        print(f"[warn] diagnose_weak_segment failed: {type(exc).__name__}: {str(exc)[:160]}")
        return None


def write_error_analysis(diag: dict | None, baseline_cv: float, baseline_model: str, path: str) -> None:
    payload = {
        "baseline_model": baseline_model,
        "baseline_cv": round(float(baseline_cv), 4),
        "worst_segment": diag,
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def read_error_analysis_baseline(path: str) -> float | None:
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("baseline_cv")
    except (OSError, json.JSONDecodeError):
        return None


def _recipe_discover_weak_segments(Xd: pd.DataFrame, Xtd: pd.DataFrame, error_analysis_path: str):
    """Converts error_analysis.json's diagnosed weak segment (already
    validated by diagnose_weak_segment() during the preceding --mode full
    run) into a binary column -- does NOT rediscover its own segment the
    way the archived discover_weak_segments() in scripts/error_analysis.py
    did. The segment being tested is the one diagnosed on the real zoo's
    actual winning model, not a cheap proxy -- cheaper and more principled
    than a fresh preliminary-model discovery pass.
    """
    try:
        with open(error_analysis_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] apply_recipe: could not read {error_analysis_path}: {type(exc).__name__}: {exc}")
        return None

    seg = data.get("worst_segment")
    if not seg:
        return None

    conditions = [(int(c[0]), float(c[1]), c[2]) for c in seg["conditions"]]
    Xd_filled = Xd.fillna(-999)
    Xtd_filled = Xtd.fillna(-999)
    train_mask = _apply_segment_mask(Xd_filled, conditions)
    test_mask = _apply_segment_mask(Xtd_filled, conditions)
    return "is_in_diagnosed_weak_segment", train_mask, test_mask


def apply_recipe(name: str, Xd: pd.DataFrame, Xtd: pd.DataFrame, error_analysis_path: str):
    """Recipe registry. Each recipe reads error_analysis.json (already
    written by a prior --mode full run in this same working directory) and
    returns (column_name, train_mask, test_mask) to append as a new binary
    feature, or None if there's nothing usable to apply. Adding a new
    recipe means adding a new entry to RECIPES, not new orchestrator-visible
    plumbing -- the CLI surface (--recipe <name>) doesn't change.
    """
    RECIPES = {
        "discover_weak_segments": _recipe_discover_weak_segments,
    }
    fn = RECIPES.get(name)
    if fn is None:
        print(f"[warn] apply_recipe: unknown recipe {name!r}; known recipes: {sorted(RECIPES)}")
        return None
    return fn(Xd, Xtd, error_analysis_path)


def write_submission(sample, test, id_col, values, out):
    sub = sample.copy()
    sub[sub.columns[0]] = test[id_col].values
    sub[sub.columns[1]] = values
    sub.to_csv(out, index=False)


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fast", "full", "apply-recipe"], default="full")
    ap.add_argument("--recipe", default=None, help="recipe name, required when --mode apply-recipe")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="seconds; models are skipped rather than killed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default="target")
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--folds", type=int, default=None)
    args = ap.parse_args()

    import os
    workdir = find_workdir()
    os.chdir(workdir)
    print(f"workdir={workdir}")

    X, y, Xt, test, sample, id_col = load(args.target, args.id_col)
    n = len(y)

    # Computed on the raw, pre-transform values -- duplicate/group identity
    # shouldn't depend on whether a column later gets log-transformed.
    groups = duplicate_groups(X)

    X, Xt = log_transform_skewed(X, Xt)
    Xc, Xtc, Xd, Xtd, cats = encode(X, Xt)

    if args.mode == "apply-recipe":
        if not args.recipe:
            raise SystemExit("--recipe is required when --mode apply-recipe")
        applied = apply_recipe(args.recipe, Xd, Xtd, ERROR_ANALYSIS_PATH)
        if applied is None:
            print(f"[error] apply-recipe: no usable segment for recipe {args.recipe!r}; nothing to apply")
            sys.exit(1)
        col_name, train_mask, test_mask = applied
        Xc[col_name] = train_mask.astype(np.int8)
        Xd[col_name] = train_mask.astype(np.int8)
        Xtc[col_name] = test_mask.astype(np.int8)
        Xtd[col_name] = test_mask.astype(np.int8)
        print(f"[info] applied recipe {args.recipe!r}: added column {col_name!r}")

    if groups is not None:
        group_sizes = pd.Series(groups).value_counts()
        n_dup_rows = int(group_sizes[group_sizes > 1].sum())
        print(f"[warn] {n_dup_rows} of {n} rows share a feature-duplicate with another row; "
              f"using group-aware CV so a duplicate is never split across train/val "
              f"(see duplicate_groups() docstring)")

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
    zoo = build_zoo(n, args.mode, args.seed, cats, Xc, y)

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
        res, pred = fit_predict(name, factory, kind, A, B, cats, y, folds, deadline, groups)
        if res is None:
            continue
        oof, mask = res
        oofs[name], preds[name] = (oof, mask), pred
        scores[name] = roc_auc_score(y[mask], oof[mask])
        print(f"  [{elapsed():6.0f}s] {name:<16} {scores[name]:.4f}"
              + ("" if mask.all() else f"  (partial, {mask.sum()}/{len(y)} rows)"))

    if not scores:
        print("[error] every model failed; writing constant predictions")
        write_submission(sample, test, id_col, 0.5, args.out)
        sys.exit(0)

    # --- blend + ensemble combiners -------------------------------------------
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
            print(f"  [{elapsed():6.0f}s] {'blend':<16} {scores['blend']:.4f}  ({'+'.join(members)})")

            oof_only = {m: oofs[m][0] for m in members}
            combos = combine_predictions(oof_only, y, members, args.seed, preds=preds)
            for name, (c_oof, c_pred) in combos.items():
                if c_pred is None:
                    # A combiner should only omit test_pred here when preds
                    # (a non-None dict) was passed in and its own refit still
                    # failed -- writing a None into preds would corrupt
                    # write_submission() if this candidate were ever selected.
                    # Unreachable today given how main() calls this, but cheap
                    # insurance.
                    print(f"[warn] {name}: no test prediction, skipping candidate")
                    continue
                scores[name] = roc_auc_score(y, c_oof)
                oofs[name] = (c_oof, np.ones(len(y), bool))
                preds[name] = c_pred
                print(f"  [{elapsed():6.0f}s] {name:<16} {scores[name]:.4f}  ({'+'.join(members)})")

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

    if args.mode == "full":
        if oofs[best][1].all():
            diag = diagnose_weak_segment(Xd, y, oofs[best][0], args.seed)
        else:
            print(f"[info] {best} has partial CV; skipping weak-segment diagnosis")
            diag = None
        write_error_analysis(diag, scores[best], best, ERROR_ANALYSIS_PATH)
        if diag:
            print(f"[info] weak segment diagnosed: gap={diag['gap']:.4f} pval={diag['pval']:.1e} n={diag['n']}")
        else:
            print("[info] no significant weak segment diagnosed")
    elif args.mode == "apply-recipe":
        baseline = read_error_analysis_baseline(ERROR_ANALYSIS_PATH)
        if baseline is not None:
            print(f"[info] apply-recipe result: baseline CV {baseline:.4f} vs with-recipe CV {scores[best]:.4f}")


if __name__ == "__main__":
    main()

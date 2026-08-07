#!/usr/bin/env python3
"""AUC-attribution error analysis (arXiv:2205.11781's core method, adapted).

For each of the 16 training datasets, this:
  1. Computes lgbm_reg's OOF predictions (via the real run_automl.py code,
     zero drift from what the agent actually runs).
  2. Assigns each example an AUC "attribution" -- its own contribution to
     the overall AUC score, derived from the Mann-Whitney rank-sum identity.
     Positives get credit for each negative they outrank; negatives get
     credit for each positive that outranks them. The mean attribution over
     positives (and, separately, over negatives) both equal the overall AUC.
  3. Fits a SHALLOW decision tree to predict attribution from the original
     (one-hot encoded) features -- its leaves define candidate "segments"
     where the model might be over/underperforming.
  4. Uses an HONEST estimate: the tree is fit on one half of the data, and
     each leaf's mean attribution is then measured on the OTHER half (never
     the half that defined the split), with a t-test against the dataset's
     overall mean attribution to flag segments whose underperformance looks
     real rather than a fitting artifact.

This is a diagnostic, not a zoo candidate -- it doesn't change any score, it
tells us WHERE the pipeline is weak and whether that's dataset-specific or a
uniform family-wide pattern (the open question from the research pass).

This file also hosts discover_weak_segments(), a runtime version of the same
honest-split method that turns flagged segments into binary features rather
than a report. It lives here (not in run_automl.py, the file that actually
ships to Kaggle) because nothing in the shipped submission calls it -- see
its own docstring for the measured result and why it isn't wired in.
scripts/bench_v2.py's --with-segments flag imports it from here for offline
comparison.

Usage
-----
    python scripts/error_analysis.py                  # all 16 datasets
    python scripts/error_analysis.py --datasets 1 9 13
"""
from __future__ import annotations

import os

# See run_automl.py's module docstring: unbounded OpenMP threads measured a
# ~900x slowdown on this machine. Set before sklearn/lightgbm import.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import importlib.util
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_ind
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor, export_text

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SKILL = ROOT / "submissions/03_skilled/agent/skills/tabular-automl/scripts/run_automl.py"

T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def load_automl_module():
    # Never leave __pycache__/*.pyc behind in the live submission directory
    # -- compile_submission rejects it on the next real invocation. See
    # scripts/bench_v2.py's load_automl_module for the incident this fixes.
    import sys
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("run_automl_ea", SKILL)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = prev
    return mod


def auc_attribution(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Per-example contribution to the overall AUC (Mann-Whitney identity).
    mean(attr[y==1]) == mean(attr[y==0]) == roc_auc_score(y, scores)."""
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    attr = np.zeros(len(y))

    rank_all = rankdata(scores, method="average")
    pos_mask = y == 1
    rank_pos = rankdata(scores[pos_mask], method="average")
    # (# negatives outranked by this positive) / n_neg
    attr[pos_mask] = (rank_all[pos_mask] - rank_pos) / n_neg

    neg_mask = y == 0
    rank_neg = rankdata(scores[neg_mask], method="average")
    # (# positives with score <= this negative) / n_pos, then complemented
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
        if left == -1:  # true leaf, not the target node
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


def discover_weak_segments(
    Xd: pd.DataFrame, Xtd: pd.DataFrame, y: np.ndarray, seed: int, deadline: float,
    max_segments: int = 3,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Find rows in a statistically real underperforming segment and return
    them as binary masks the caller can attach as new features.

    Lives here rather than in run_automl.py: nothing in the shipped Kaggle
    submission calls it, so it doesn't need to be self-contained inside the
    bundled skill script -- it can just import auc_attribution from this
    same module. bench_v2.py's --with-segments flag imports this function
    to benchmark it; run_automl.py's main() does NOT call it (see the "Tried
    and not wired in" entry in run_automl.py's module docstring for why).

    Split scheme (deliberately cheaper than analyze_one's full-CV-OOF
    version above, to fit a ~120s runtime budget): train/holdout split
    (50/50) -> cheap preliminary LightGBM fit on train -> AUC attribution
    computed on holdout only -> holdout itself split again (50/50) ->
    shallow diagnostic tree fit on one half -> each candidate leaf's mean
    attribution validated on the OTHER half via a p<0.01 t-test (never
    validate a leaf on the rows that defined it). Net effect: preliminary
    model sees 50% of the data, the tree candidate-fits on 25%, and
    validates on a disjoint 25%.

    Exception-safe: any exception or exceeded deadline returns [], never
    raises -- that guarantee holds regardless of what follows below.

    What it's not: a proven win. Measured with bench_v2.py --with-segments
    across the 12 (of 16) datasets with a real diagnosed weak segment (per
    this file's own honest-split diagnostic): this function actually found
    segments on only 2 of those 12 (train_06, train_14) -- the other 10
    correctly returned [] (nothing significant to flag). On those 2, mean
    blend AUC delta was effectively zero and slightly negative. Net effect
    across all 12: mean blend delta +0.00004, mean lgbm_tuned delta
    -0.00007 -- both inside noise. The tree models already carve out these
    regions from the existing features on their own, so the redundant flag
    adds no new information. See git history (commit 8a09553's revert
    message) for the full reasoning behind not wiring this in.
    """
    try:
        if elapsed() > deadline:
            return []

        import lightgbm as lgb

        skf = StratifiedKFold(2, shuffle=True, random_state=seed)
        fit_idx, hold_idx = next(skf.split(Xd, y))

        def _timeout_guard(_env):
            if elapsed() > deadline:
                raise TimeoutError("discover_weak_segments exceeded its deadline")

        prelim = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            random_state=seed, verbose=-1, n_jobs=4,
        )
        prelim.fit(Xd.iloc[fit_idx], y[fit_idx], callbacks=[_timeout_guard])
        scores_hold = prelim.predict_proba(Xd.iloc[hold_idx])[:, 1]
        attr = auc_attribution(y[hold_idx], scores_hold)

        if elapsed() > deadline:
            return []

        y_hold = y[hold_idx]
        skf2 = StratifiedKFold(2, shuffle=True, random_state=seed)
        tree_fit_pos, tree_eval_pos = next(skf2.split(Xd.iloc[hold_idx], y_hold))

        Xd_hold_filled = Xd.iloc[hold_idx].fillna(-999)
        tree = DecisionTreeRegressor(
            max_depth=3, min_samples_leaf=max(30, len(hold_idx) // 40), random_state=seed,
        )
        tree.fit(Xd_hold_filled.iloc[tree_fit_pos], attr[tree_fit_pos])

        if elapsed() > deadline:
            return []

        eval_leaf = tree.apply(Xd_hold_filled.iloc[tree_eval_pos])
        eval_attr = attr[tree_eval_pos]
        overall_mean = eval_attr.mean()

        flagged = []
        for leaf in np.unique(eval_leaf):
            seg = eval_attr[eval_leaf == leaf]
            rest = eval_attr[eval_leaf != leaf]
            if len(seg) < 20 or len(rest) < 20:
                continue
            _tstat, pval = ttest_ind(seg, rest, equal_var=False)
            if pval < 0.01 and seg.mean() < overall_mean:
                flagged.append((leaf, overall_mean - seg.mean()))

        flagged.sort(key=lambda lg: -lg[1])
        flagged = flagged[:max_segments]
        if not flagged:
            return []

        Xd_filled = Xd.fillna(-999)
        Xtd_filled = Xtd.fillna(-999)
        results = []
        for n, (leaf, _gap) in enumerate(flagged):
            conditions = _leaf_conditions(tree.tree_, int(leaf))
            train_mask = _apply_segment_mask(Xd_filled, conditions)
            test_mask = _apply_segment_mask(Xtd_filled, conditions)
            results.append((f"is_in_weak_segment_{n}", train_mask, test_mask))
        return results
    except Exception as exc:
        print(f"[warn] discover_weak_segments failed: {type(exc).__name__}: {str(exc)[:160]}")
        return []


def get_oof(automl, ds: str, seed: int = 0):
    d = DATA / ds
    orig_cwd = Path.cwd()
    os.chdir(d)
    try:
        X, y, Xt, test, sample, id_col = automl.load("target", None)
        Xc, Xtc, Xd, Xtd, cats = automl.encode(X, Xt)
        n = len(y)
        folds = 10 if n < 2000 else (5 if n < 20000 else 4)

        factory = lambda: __import__("lightgbm").LGBMClassifier(  # noqa: E731
            n_estimators=1500, learning_rate=0.02, num_leaves=15,
            min_child_samples=40, colsample_bytree=0.7,
            subsample=0.8, subsample_freq=1, reg_lambda=5.0,
            random_state=seed, verbose=-1,
        )
        res, _pred = automl.fit_predict("lgbm_reg", factory, "cat", Xc, Xtc, cats, y, folds, None)
    finally:
        os.chdir(orig_cwd)

    if res is None:
        return None
    oof, mask = res
    if not mask.all():
        return None
    return oof, y, Xd


def analyze_one(automl, ds: str, seed: int = 0) -> dict:
    got = get_oof(automl, ds, seed)
    if got is None:
        return {"ds": ds, "status": "failed"}
    oof, y, Xd = got

    overall_auc = roc_auc_score(y, oof)
    attr = auc_attribution(y, oof)
    # sanity check: mean attribution per class must equal the overall AUC
    check_pos = abs(attr[y == 1].mean() - overall_auc)
    check_neg = abs(attr[y == 0].mean() - overall_auc)
    if max(check_pos, check_neg) > 1e-6:
        print(f"  [warn] {ds}: attribution sanity check off by {max(check_pos, check_neg):.2e}")

    # Honest split: fit the tree on half, measure segments on the OTHER half.
    skf = StratifiedKFold(2, shuffle=True, random_state=seed)
    fit_idx, eval_idx = next(skf.split(Xd, y))

    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(30, len(y) // 40), random_state=seed)
    tree.fit(Xd.iloc[fit_idx].fillna(-999), attr[fit_idx])

    leaf_eval = tree.apply(Xd.iloc[eval_idx].fillna(-999))
    overall_mean = attr[eval_idx].mean()

    flagged = []
    for leaf in np.unique(leaf_eval):
        seg = attr[eval_idx][leaf_eval == leaf]
        rest = attr[eval_idx][leaf_eval != leaf]
        if len(seg) < 20 or len(rest) < 20:
            continue
        tstat, pval = ttest_ind(seg, rest, equal_var=False)
        if pval < 0.01 and seg.mean() < overall_mean:
            flagged.append({
                "leaf": int(leaf), "n": int(len(seg)),
                "seg_mean_attr": round(float(seg.mean()), 4),
                "overall_mean_attr": round(float(overall_mean), 4),
                "gap": round(float(overall_mean - seg.mean()), 4),
                "pval": float(pval),
            })

    flagged.sort(key=lambda f: -f["gap"])
    return {
        "ds": ds, "status": "ok", "n": len(y), "auc": round(overall_auc, 4),
        "n_flagged_segments": len(flagged),
        "worst_segment": flagged[0] if flagged else None,
        "tree_rules": export_text(tree, feature_names=list(Xd.columns), max_depth=3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=int, nargs="*", default=list(range(1, 17)))
    args = ap.parse_args()

    automl = load_automl_module()
    results = []
    for i in args.datasets:
        ds = f"train_{i:02d}"
        if not (DATA / ds / "train.csv").exists():
            continue
        r = analyze_one(automl, ds)
        results.append(r)
        if r["status"] == "ok":
            w = r["worst_segment"]
            if w:
                print(f"{ds}: AUC={r['auc']} -- worst segment: n={w['n']} "
                      f"attr {w['seg_mean_attr']} vs overall {w['overall_mean_attr']} "
                      f"(gap {w['gap']}, p={w['pval']:.1e})")
            else:
                print(f"{ds}: AUC={r['auc']} -- no segment significantly underperforms (uniform)")
        else:
            print(f"{ds}: FAILED")

    ok = [r for r in results if r["status"] == "ok"]
    n_with_gap = sum(1 for r in ok if r["worst_segment"])
    print(f"\n{n_with_gap}/{len(ok)} datasets have a significantly underperforming segment "
          f"(p<0.01, honest held-out half).")
    if n_with_gap:
        print("\nWorst segments (largest gap first):")
        for r in sorted(ok, key=lambda r: -(r["worst_segment"]["gap"] if r["worst_segment"] else 0))[:5]:
            w = r["worst_segment"]
            if not w:
                continue
            print(f"\n--- {r['ds']} (gap {w['gap']}, n={w['n']}, p={w['pval']:.1e}) ---")
            print(r["tree_rules"])


if __name__ == "__main__":
    main()

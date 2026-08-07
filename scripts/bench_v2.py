#!/usr/bin/env python3
"""Benchmark the ACTUAL run_automl.py zoo/CV code across the 16 training
datasets -- imports the real skill script directly, zero drift from what the
agent runs in the sandbox.

Each dataset runs in its OWN process invocation (see eval_bench_v2.sh), not
looped in one long-lived interpreter: running the full zoo (LightGBM +
HistGB + CatBoost) back-to-back across many datasets in a single process
segfaulted -- almost certainly a CatBoost/LightGBM OpenMP thread-pool
conflict from reuse. Production never hits this because `run_skill_script`
launches a fresh process per invocation; this harness now matches that.

Usage
-----
    python scripts/bench_v2.py --dataset 5 --row-out /tmp/rows/train_05.json
    python scripts/bench_v2.py --summarize /tmp/rows --out bench_v2_results.csv
"""
from __future__ import annotations

import os

# See run_automl.py's module docstring: unbounded OpenMP threads measured a
# ~900x slowdown on this machine. Set before sklearn/lightgbm import.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

# discover_weak_segments lives in error_analysis.py, not run_automl.py: it's
# not called from main() (see run_automl.py's "Tried and not wired in"
# docstring entry), so it doesn't need to be self-contained inside the
# bundled skill script -- only --with-segments below needs it, as a dev-side
# opt-in comparison.
import error_analysis

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SKILL = ROOT / "submissions/03_skilled/agent/skills/tabular-automl/scripts/run_automl.py"


def load_automl_module():
    # dont_write_bytecode: importing run_automl.py directly from the live
    # submission directory otherwise drops __pycache__/*.pyc right next to
    # it, which validate_submission.py / compile_submission correctly
    # rejects ("disallowed extension") the next time that submission is
    # actually used -- this broke a real local-eval run after a benchmark
    # pass left a stale .pyc behind. Never write it in the first place.
    import sys
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("run_automl_bench", SKILL)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = prev
    return mod


def load_automl_module_from(skill_path: Path):
    """Like load_automl_module(), but for an arbitrary skill script path --
    needed to benchmark submission variants other than 03_skilled (e.g.
    04_reflective) without duplicating this whole harness."""
    import sys
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("run_automl_bench_variant", skill_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = prev
    return mod


def run_one_reflective(i: int, seed: int, skill_path: Path) -> dict:
    """Simulates the full 04_reflective sequence offline: run the zoo,
    diagnose a weak segment on the winning model's OOF, and if one is
    found, apply the discover_weak_segments recipe and re-run, recording
    both the baseline and with-recipe CV so it's possible to see how often
    the recipe is even attempted vs how often it actually helps."""
    automl = load_automl_module_from(skill_path)
    rank01 = lambda v: rankdata(v) / len(v)  # noqa: E731

    ds = f"train_{i:02d}"
    d = DATA / ds
    orig_cwd = Path.cwd()
    os.chdir(d)
    try:
        X, y, Xt, test, sample, id_col = automl.load("target", None)
        n = len(y)
        groups = automl.duplicate_groups(X)
        X, Xt = automl.log_transform_skewed(X, Xt)
        Xc, Xtc, Xd, Xtd, cats = automl.encode(X, Xt)
        folds = 10 if n < 2000 else (5 if n < 20000 else 4)
        zoo = automl.build_zoo(n, "full", seed, cats, Xc, y)

        def run_zoo(Xc_, Xtc_, Xd_, Xtd_):
            scores, oofs = {}, {}
            for name, factory, kind, _weight in zoo:
                A, B = automl.frames_for(kind, Xc_, Xtc_, Xd_, Xtd_, cats)
                res, _pred = automl.fit_predict(name, factory, kind, A, B, cats, y, folds, None, groups)
                if res is None:
                    continue
                oof, mask = res
                if not mask.all():
                    continue
                oofs[name] = oof
                scores[name] = roc_auc_score(y, oof)
            full = list(oofs)
            if len(full) > 1:
                members = full if n < 2000 else ([m for m in full if m != "logreg"] or full)
                if len(members) > 1:
                    b_oof = np.mean([rank01(oofs[m]) for m in members], axis=0)
                    scores["blend"] = roc_auc_score(y, b_oof)
                    oofs["blend"] = b_oof
                    combos = automl.combine_predictions(oofs, y, members, seed)
                    for name, (c_oof, _c_pred) in combos.items():
                        scores[name] = roc_auc_score(y, c_oof)
                        oofs[name] = c_oof
            return scores, oofs

        scores, oofs = run_zoo(Xc, Xtc, Xd, Xtd)
        if not scores:
            return {"ds": ds, "n": n, "n_feat": X.shape[1], "n_cat": len(cats), "segment_found": False}

        best = max(scores, key=scores.get)
        diag = automl.diagnose_weak_segment(Xd, y, oofs[best], seed)

        row = {
            "ds": ds, "n": n, "n_feat": X.shape[1], "n_cat": len(cats),
            "baseline_best": best, "baseline_cv": round(scores[best], 4),
            "segment_found": diag is not None,
        }
        if diag is not None:
            row["segment_gap"] = diag["gap"]
            row["segment_pval"] = diag["pval"]

            Xd_filled = Xd.fillna(-999)
            Xtd_filled = Xtd.fillna(-999)
            mask_train = automl._apply_segment_mask(Xd_filled, diag["conditions"])
            mask_test = automl._apply_segment_mask(Xtd_filled, diag["conditions"])

            Xc2, Xtc2, Xd2, Xtd2 = Xc.copy(), Xtc.copy(), Xd.copy(), Xtd.copy()
            col = "is_in_diagnosed_weak_segment"
            Xc2[col] = mask_train.astype("int8")
            Xd2[col] = mask_train.astype("int8")
            Xtc2[col] = mask_test.astype("int8")
            Xtd2[col] = mask_test.astype("int8")

            scores2, _oofs2 = run_zoo(Xc2, Xtc2, Xd2, Xtd2)
            if scores2:
                best2 = max(scores2, key=scores2.get)
                row["with_recipe_best"] = best2
                row["with_recipe_cv"] = round(scores2[best2], 4)
    finally:
        os.chdir(orig_cwd)

    return row


def run_one(i: int, seed: int, with_segments: bool = False) -> dict:
    automl = load_automl_module()
    rank01 = lambda v: rankdata(v) / len(v)  # noqa: E731

    ds = f"train_{i:02d}"
    d = DATA / ds
    orig_cwd = Path.cwd()
    os.chdir(d)
    try:
        X, y, Xt, test, sample, id_col = automl.load("target", None)
        n = len(y)
        groups = automl.duplicate_groups(X)  # before log_transform_skewed, see run_automl.py main()
        X, Xt = automl.log_transform_skewed(X, Xt)
        Xc, Xtc, Xd, Xtd, cats = automl.encode(X, Xt)

        n_segments = 0
        if with_segments:
            segs = error_analysis.discover_weak_segments(Xd, Xtd, y, seed, error_analysis.elapsed() + 120.0)
            for name, train_mask, test_mask in segs:
                Xc[name] = train_mask.astype(np.int8)
                Xd[name] = train_mask.astype(np.int8)
                Xtc[name] = test_mask.astype(np.int8)
                Xtd[name] = test_mask.astype(np.int8)
            n_segments = len(segs)

        folds = 10 if n < 2000 else (5 if n < 20000 else 4)
        zoo = automl.build_zoo(n, "full", seed, cats, Xc, y)

        scores, oofs = {}, {}
        for name, factory, kind, _weight in zoo:
            A, B = automl.frames_for(kind, Xc, Xtc, Xd, Xtd, cats)
            res, _pred = automl.fit_predict(name, factory, kind, A, B, cats, y, folds, None, groups)
            if res is None:
                continue
            oof, mask = res
            if not mask.all():
                print(f"  [warn] {ds}/{name}: partial CV, skipping from bench")
                continue
            oofs[name] = oof
            scores[name] = roc_auc_score(y, oof)

        full = list(oofs)
        if len(full) > 1:
            members = full if n < 2000 else ([m for m in full if m != "logreg"] or full)
            if len(members) > 1:
                b_oof = np.mean([rank01(oofs[m]) for m in members], axis=0)
                scores["blend"] = roc_auc_score(y, b_oof)

                oof_only = {m: oofs[m] for m in members}
                combos = automl.combine_predictions(oof_only, y, members, seed)
                for name, (c_oof, _c_pred) in combos.items():
                    scores[name] = roc_auc_score(y, c_oof)
    finally:
        os.chdir(orig_cwd)

    row = {"ds": ds, "n": n, "n_feat": X.shape[1], "n_cat": len(cats), "n_segments": n_segments}
    row.update({k: round(v, 4) for k, v in scores.items()})
    return row


def summarize(rows_dir: Path, out: str) -> None:
    rows = []
    for f in sorted(rows_dir.glob("*.json")):
        rows.append(json.loads(f.read_text()))
    if not rows:
        raise SystemExit(f"no row files found in {rows_dir}")

    df = pd.DataFrame(rows).set_index("ds")
    model_cols = [c for c in df.columns if c not in ("n", "n_feat", "n_cat", "n_segments")]

    print(df.to_string())
    print("\n=== mean AUC per candidate (NaN-safe over datasets present) ===")
    print(df[model_cols].mean(skipna=True).round(4).sort_values(ascending=False).to_string())
    print(f"\noracle (best per dataset): {df[model_cols].max(axis=1).mean():.4f}")
    print(f"n datasets: {len(df)}")

    df.to_csv(out)
    print(f"\nwritten: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, help="single dataset index (1-16) to run")
    ap.add_argument("--row-out", help="write this dataset's result row as JSON here")
    ap.add_argument("--summarize", help="directory of per-dataset JSON rows to summarize")
    ap.add_argument("--out", default="bench_v2_results.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-segments", action="store_true",
                     help="enable discover_weak_segments (opt-in, for before/after comparison)")
    ap.add_argument("--reflective", action="store_true",
                     help="simulate the 04_reflective diagnose+recipe sequence instead of the standard zoo")
    ap.add_argument("--skill-path", default=str(SKILL),
                     help="path to run_automl.py to benchmark (defaults to 03_skilled's)")
    args = ap.parse_args()

    if args.summarize:
        summarize(Path(args.summarize), args.out)
        return

    if args.dataset is None or args.row_out is None:
        raise SystemExit("need --dataset N --row-out path.json (or --summarize dir)")

    if args.reflective:
        row = run_one_reflective(args.dataset, args.seed, Path(args.skill_path))
    else:
        row = run_one(args.dataset, args.seed, args.with_segments)
    print(row, flush=True)
    Path(args.row_out).write_text(json.dumps(row))


if __name__ == "__main__":
    main()

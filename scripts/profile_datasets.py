#!/usr/bin/env python3
"""Characterise the 16 provided training datasets.

Answers the question the agent has to answer in its first 3 minutes: what kind
of dataset am I looking at, and what does that imply for modelling?

    python scripts/profile_datasets.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main() -> None:
    rows = []
    for i in range(1, 17):
        d = DATA / f"train_{i:02d}"
        if not (d / "train.csv").exists():
            continue

        tr = pd.read_csv(d / "train.csv")
        te = pd.read_csv(d / "test.csv") if (d / "test.csv").exists() else None
        md = (d / "DATA.md").read_text() if (d / "DATA.md").exists() else ""
        declared = pd.Series(dict(re.findall(r"`(feature_\d+)`: (\w+)", md))).value_counts().to_dict()

        X = tr.drop(columns=[c for c in ("row_id", "target") if c in tr.columns])
        cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

        rows.append({
            "ds": f"train_{i:02d}",
            "n_train": len(tr),
            "n_test": len(te) if te is not None else None,
            "n_feat": X.shape[1],
            "n_cat": len(cats),
            "max_cat_levels": int(X[cats].nunique().max()) if cats else 0,
            "pos_rate": round(float(tr["target"].mean()), 4),
            "missing": round(float(X.isna().mean().mean()), 4),
            "declared_types": declared,
        })

    df = pd.DataFrame(rows).set_index("ds")
    print(df.drop(columns=["declared_types"]).to_string())
    print("\n=== spread ===")
    print(df[["n_train", "n_feat", "n_cat", "pos_rate", "missing"]].agg(["min", "median", "max"]).round(4).to_string())
    print("\n=== declared feature types ===")
    for ds, t in df["declared_types"].items():
        if t:
            print(f"  {ds}: {t}")


if __name__ == "__main__":
    main()

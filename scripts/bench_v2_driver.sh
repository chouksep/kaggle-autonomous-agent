#!/usr/bin/env bash
# Runs bench_v2.py once per dataset, each in its own process (avoids the
# CatBoost/LightGBM segfault seen when the full zoo is reused across many
# datasets in one long-lived interpreter). Usage:
#   bash scripts/bench_v2_driver.sh /tmp/rows_label bench_v2_label.csv [dataset_ids...]
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$HOME/.venvs/kik/bin/python"
export LD_LIBRARY_PATH="$HOME/.local/libgomp"

ROWS_DIR="${1:?rows dir required}"
OUT="${2:?output csv required}"
shift 2
DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=($(seq 1 16))
fi

rm -rf "$ROWS_DIR"
mkdir -p "$ROWS_DIR"

for i in "${DATASETS[@]}"; do
  ds=$(printf "train_%02d" "$i")
  echo "=== $ds ==="
  "$PY" "$REPO/scripts/bench_v2.py" --dataset "$i" --row-out "$ROWS_DIR/$ds.json"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[error] $ds exited $rc"
  fi
done

echo
echo "=== summary ==="
"$PY" "$REPO/scripts/bench_v2.py" --summarize "$ROWS_DIR" --out "$OUT"

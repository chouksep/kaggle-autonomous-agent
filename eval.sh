#!/usr/bin/env bash
# Run a local agent evaluation. Usage:
#
#   wsl -d Ubuntu -e bash ./eval.sh 02_lean train_01
#   wsl -d Ubuntu -e bash ./eval.sh 02_lean train_05
#
# Requires: Docker Desktop running, and GEMINI_API_KEY set in .env
# (the harness maps google models to gemini/<alias> in Direct Provider Mode).
#
# Budgets below deliberately mirror the real competition session:
#   60 minutes, $2.00 of LLM spend, 30 submissions, 2 selections.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/.venvs/kik/bin/python"

EXP="${1:-02_lean}"
DS="${2:-train_01}"

cd "$REPO"

if ! grep -qE '^GEMINI_API_KEY=.+' .env 2>/dev/null; then
  echo "ERROR: GEMINI_API_KEY is not set in .env" >&2
  exit 1
fi

echo "=== $EXP on $DS ==="
"$PY" run_local_eval.py \
  --submission-dir "submissions/$EXP/agent" \
  --dataset "$DS" \
  --metric roc_auc \
  --max-time-minutes 60 \
  --max-budget-usd 2.0 \
  --max-submissions 30 \
  --max-selections 2 \
  --max-stdout-chars 5000 \
  2>&1 | tee "submissions/$EXP/output/eval_${DS}.log"

echo
echo "=== trace ==="
"$PY" scripts/parse_eval_trace.py --experiment-dir "submissions/$EXP" 2>&1 | tail -40

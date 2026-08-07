#!/usr/bin/env bash
# Set up the Kaggle-in-Kaggle local evaluation harness inside WSL Ubuntu.
#
# Why WSL: litellm >= 1.83 has no Windows wheel and needs Rust + the MSVC
# linker to build from source. On Linux it installs from a wheel in seconds.
# WSL also talks to Docker Desktop directly, which the sandbox needs.
#
#   wsl -d Ubuntu -e bash ./setup_wsl.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HOME/.venvs/kik"
UV="$HOME/.local/bin/uv"

cd "$REPO"

if [ ! -x "$UV" ]; then
  echo ">> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo ">> creating venv at $VENV"
"$UV" venv --clear --python 3.12 "$VENV"

echo ">> installing requirements"
"$UV" pip install --python "$VENV/bin/python" -r requirements.txt

echo ">> verifying"
"$VENV/bin/python" -c "import adk_submission, kaggle_kaggle, litellm; print('HARNESS_OK')"

echo ">> done. run things with:  ~/.venvs/kik/bin/python <script>"

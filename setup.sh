#!/usr/bin/env bash
# One-time setup: Python venv + dependencies + model weights (via Git LFS).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
WEIGHTS="models/embeddinggemma-2/model.safetensors"

# 1. Python environment (uv if available, else stdlib venv).
if command -v uv >/dev/null 2>&1; then
  [[ -d .venv ]] || uv venv --python 3.12 .venv
  uv pip install --python .venv -r requirements.txt
else
  PYTHON="${PYTHON:-python3}"
  [[ -d .venv ]] || "$PYTHON" -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

# 2. Model weights. The repo stores them with Git LFS; a ~130-byte pointer file means they weren't pulled.
size_of() { stat -f%z "$1" 2>/dev/null || stat -c%s "$1"; }
if [[ ! -f "$WEIGHTS" || $(size_of "$WEIGHTS") -lt 1000000 ]]; then
  if command -v git-lfs >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then
    echo "[setup] pulling model weights with git lfs…"
    git lfs install --local >/dev/null
    git lfs pull --include "models/**" || true
  fi
fi
if [[ ! -f "$WEIGHTS" || $(size_of "$WEIGHTS") -lt 1000000 ]]; then
  echo "[setup] Git LFS weights unavailable – rag.sh will download google/embeddinggemma-2 from Hugging Face on first use."
  echo "        (install git-lfs and run 'git lfs pull' to use the bundled copy instead)"
fi

echo "[setup] done. Try:  ./rag.sh index sample_docs && ./rag.sh \"how do I roll back a deploy?\""

#!/usr/bin/env bash
# One-time setup: Python venv + 3 small dependencies, then rebuild the bundled model weights.
# Works with Python 3.9+. The model ships inside the repo, so nothing is downloaded for it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PYTHON="${PYTHON:-python3}"

if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
  echo "[setup] need Python 3.9+ (found: $("$PYTHON" --version 2>&1)). Set PYTHON=/path/to/python3." >&2
  exit 1
fi

# 1. Virtual environment + dependencies (numpy, onnxruntime, tokenizers).
[[ -d .venv ]] || "$PYTHON" -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt

# 2. Rebuild model.onnx.data from the committed parts (local, ~1 s, checksum-verified).
.venv/bin/python -m rag.model_files >/dev/null

echo "[setup] done ($(.venv/bin/python --version)). Try:  ./rag.sh index sample_docs && ./rag.sh \"how do I roll back a deploy?\""

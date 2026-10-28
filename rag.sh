#!/usr/bin/env bash
# Local RAG over your docs with EmbeddingGemma 2.
#
#   ./rag.sh index ./docs              # index (incremental) a folder of .md/.txt files
#   ./rag.sh "how do I roll back?"     # search; prints ranked chunks with file:line
#   ./rag.sh "..." --md                # LLM-ready Markdown (what Copilot receives)
#   ./rag.sh ui                        # web UI on http://127.0.0.1:7860
#   ./rag.sh mcp                       # MCP server for VS Code Copilot
#   ./rag.sh vscode /path/to/project   # wire it into another project
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "[rag] first run: setting up environment…" >&2
  "$HERE/setup.sh" >&2
fi

cd "$HERE"
exec "$PY" -m rag.cli "$@"

"""FastAPI backend for the search UI (`./rag.sh ui`)."""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from ..embedder import get_model
from ..index import index_paths, status
from ..search import default_searcher, to_markdown

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="EmbeddingGemma RAG")
_reindex_lock = threading.Lock()

# Warm the model in the background so the first search is fast.
threading.Thread(target=get_model, daemon=True).start()


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/api/search")
def search(q: str, k: int = Query(8, ge=1, le=50), path: str | None = None,
           all: bool = False):
    results = default_searcher().search(q, k=k, path_filter=path, margin=0.0 if all else None)
    return {"query": q, "results": results, "markdown": to_markdown(results, q)}


@app.get("/api/status")
def get_status():
    return status()


@app.get("/api/files")
def files():
    return default_searcher().files()


@app.get("/api/file")
def file(path: str):
    # Only serve files that are actually in the index.
    if path not in {f["path"] for f in default_searcher().files()}:
        raise HTTPException(404, "not indexed")
    return {"path": path, "text": Path(path).read_text(encoding="utf-8", errors="replace")}


@app.get("/api/chunk/{chunk_id}")
def chunk(chunk_id: int, neighbors: int = Query(1, ge=0, le=5)):
    c = default_searcher().get_chunk(chunk_id, neighbors)
    if c is None:
        raise HTTPException(404, "no such chunk")
    return c


@app.post("/api/reindex")
def reindex():
    if not _reindex_lock.acquire(blocking=False):
        raise HTTPException(409, "already indexing")
    try:
        return index_paths(None, quiet=True)
    finally:
        _reindex_lock.release()

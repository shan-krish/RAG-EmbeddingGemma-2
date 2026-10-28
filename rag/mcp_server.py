"""MCP server (stdio) so VS Code Copilot agent mode can query the index as a tool.

Copilot gets a few relevant chunks with `path:start-end` citations instead of whole files,
which keeps its context window small.
"""

from __future__ import annotations

import sys
import threading

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

from .embedder import get_model
from .index import index_paths, status
from .search import default_searcher, to_markdown

mcp = MCPServer(
    "docs-rag",
    instructions=(
        "Semantic search over this project's documentation (Markdown/text), powered by EmbeddingGemma 2. "
        "Call search_docs FIRST whenever a question may be answered by project docs, instead of reading "
        "whole files. Results cite `path:start-end`; use get_chunk to widen a hit, and only open a file "
        "when you need more than the returned chunks."
    ),
    log_level="WARNING",
)
_index_lock = threading.Lock()


@mcp.tool()
def search_docs(query: str, k: int = 5, path_filter: str | None = None) -> str:
    """Search the project's docs and return only the most relevant chunks.

    Args:
        query: A natural-language question or keywords (identifiers, error messages and config keys work too).
        k: Maximum number of chunks to return (1-15). Weak matches are dropped, so fewer may come back.
        path_filter: Optional substring to restrict results to matching file paths, e.g. "deploy" or "api/".
    """
    results = default_searcher().search(query, k=max(1, min(k, 15)), path_filter=path_filter)
    return to_markdown(results, query)


@mcp.tool()
def get_chunk(chunk_id: int, neighbors: int = 1) -> str:
    """Return a chunk (by the id shown in search results) plus `neighbors` adjacent chunks on each side."""
    c = default_searcher().get_chunk(chunk_id, max(0, min(neighbors, 5)))
    if c is None:
        return f"No chunk with id {chunk_id} (the index may have been rebuilt; search again)."
    return to_markdown([c])


@mcp.tool()
def list_indexed_files() -> str:
    """List the files in the index with their chunk counts."""
    files = default_searcher().files()
    if not files:
        return "The index is empty. Run `rag.sh index <folder>` or call reindex."
    return "\n".join(f"- {f['rel_path']} ({f['chunks']} chunks) — {f['title']}" for f in files)


@mcp.tool()
def reindex() -> str:
    """Re-index the configured source folders (only changed files are re-embedded)."""
    if not _index_lock.acquire(blocking=False):
        return "Indexing is already running."
    try:
        s = index_paths(None, quiet=True)
        return f"Indexed: {s['added']} added, {s['updated']} updated, {s['removed']} removed, {s['unchanged']} unchanged."
    except SystemExit as e:
        return str(e)
    finally:
        _index_lock.release()


def _startup(sources: list[str] | None) -> None:
    try:
        if sources:
            with _index_lock:
                index_paths(sources, quiet=True)
        else:
            get_model()
        st = status()
        print(f"[rag] MCP ready: {st['files']} files, {st['chunks']} chunks ({st['index']})", file=sys.stderr)
    except BaseException as e:  # never let a background failure kill the server
        print(f"[rag] startup indexing failed: {e}", file=sys.stderr)


def run(source: list[str] | None = None) -> None:
    threading.Thread(target=_startup, args=(source,), daemon=True).start()
    mcp.run()  # stdio

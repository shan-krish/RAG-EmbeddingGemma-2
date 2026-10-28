"""MCP server (stdio) so VS Code Copilot agent mode can query the index as a tool.

Copilot gets a few relevant chunks with `path:start-end` citations instead of whole files,
which keeps its context window small.

The protocol is implemented directly (newline-delimited JSON-RPC 2.0 over stdin/stdout) because
the official `mcp` SDK requires Python 3.10+; this keeps the server dependency-free on 3.9.
"""

from __future__ import annotations

import json
import sys
import threading
import traceback

from .embedder import get_model
from .index import index_paths, status
from .search import default_searcher, to_markdown

SERVER_INFO = {"name": "docs-rag", "version": "1.0.0"}
LATEST_PROTOCOL = "2025-06-18"
INSTRUCTIONS = (
    "Semantic search over this project's documentation (Markdown/text), powered by EmbeddingGemma 2. "
    "Call search_docs FIRST whenever a question may be answered by project docs, instead of reading "
    "whole files. Results cite `path:start-end`; use get_chunk to widen a hit, and only open a file "
    "when you need more than the returned chunks."
)
_index_lock = threading.Lock()


# ---------------------------------------------------------------- tools

def search_docs(query: str, k: int = 5, path_filter: str | None = None) -> str:
    results = default_searcher().search(query, k=max(1, min(int(k), 15)), path_filter=path_filter or None)
    return to_markdown(results, query)


def get_chunk(chunk_id: int, neighbors: int = 1) -> str:
    c = default_searcher().get_chunk(int(chunk_id), max(0, min(int(neighbors), 5)))
    if c is None:
        return f"No chunk with id {chunk_id} (the index may have been rebuilt; search again)."
    return to_markdown([c])


def list_indexed_files() -> str:
    files = default_searcher().files()
    if not files:
        return "The index is empty. Run `rag.sh index <folder>` or call reindex."
    return "\n".join(f"- {f['rel_path']} ({f['chunks']} chunks) — {f['title']}" for f in files)


def reindex() -> str:
    if not _index_lock.acquire(blocking=False):
        return "Indexing is already running."
    try:
        s = index_paths(None, quiet=True)
        return f"Indexed: {s['added']} added, {s['updated']} updated, {s['removed']} removed, {s['unchanged']} unchanged."
    except SystemExit as e:
        return str(e)
    finally:
        _index_lock.release()


TOOLS = {
    "search_docs": (search_docs, {
        "description": "Search the project's docs and return only the most relevant chunks, each cited as "
                       "`path:start-end`. Weak matches are dropped, so fewer than k may come back.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language question or keywords "
                          "(identifiers, error messages and config keys work too)."},
                "k": {"type": "integer", "minimum": 1, "maximum": 15, "default": 5,
                      "description": "Maximum number of chunks to return."},
                "path_filter": {"type": "string", "description": "Only search files whose path contains "
                                "this substring, e.g. \"deploy\" or \"api/\"."},
            },
            "required": ["query"],
        },
        "annotations": {"readOnlyHint": True},
    }),
    "get_chunk": (get_chunk, {
        "description": "Return a chunk (by the id shown in search results) plus `neighbors` adjacent chunks "
                       "on each side, to widen context without opening the whole file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "integer"},
                "neighbors": {"type": "integer", "minimum": 0, "maximum": 5, "default": 1},
            },
            "required": ["chunk_id"],
        },
        "annotations": {"readOnlyHint": True},
    }),
    "list_indexed_files": (list_indexed_files, {
        "description": "List the files in the index with their chunk counts.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    }),
    "reindex": (reindex, {
        "description": "Re-index the configured source folders (only changed files are re-embedded).",
        "inputSchema": {"type": "object", "properties": {}},
    }),
}


# ---------------------------------------------------------------- JSON-RPC plumbing

class Server:
    def __init__(self, out):
        self.out = out
        self.write_lock = threading.Lock()

    def send(self, msg: dict) -> None:
        line = json.dumps(msg, ensure_ascii=False)
        with self.write_lock:
            self.out.write(line + "\n")
            self.out.flush()

    def handle(self, msg: dict) -> dict | None:
        method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if method is None:  # a response to something we never send; ignore
            return None
        is_request = "id" in msg
        try:
            if method == "initialize":
                # Echo the client's protocol version: we only use tools, which are the same across versions.
                result = {
                    "protocolVersion": params.get("protocolVersion") or LATEST_PROTOCOL,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": SERVER_INFO,
                    "instructions": INSTRUCTIONS,
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": [{"name": name, **spec} for name, (_, spec) in TOOLS.items()]}
            elif method == "tools/call":
                name = params.get("name")
                if name not in TOOLS:
                    return self._error(mid, -32602, f"Unknown tool: {name}")
                fn = TOOLS[name][0]
                try:
                    text, is_error = fn(**(params.get("arguments") or {})), False
                except TypeError as e:
                    text, is_error = f"Invalid arguments for {name}: {e}", True
                except Exception as e:
                    traceback.print_exc(file=sys.stderr)
                    text, is_error = f"{name} failed: {e}", True
                result = {"content": [{"type": "text", "text": text}], "isError": is_error}
            elif method.startswith("notifications/"):
                return None
            else:
                return self._error(mid, -32601, f"Method not found: {method}") if is_request else None
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            return self._error(mid, -32603, str(e)) if is_request else None
        return {"jsonrpc": "2.0", "id": mid, "result": result} if is_request else None

    @staticmethod
    def _error(mid, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}

    def serve(self, inp) -> None:
        for raw in inp:
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                self.send(self._error(None, -32700, "Parse error"))
                continue
            for m in msg if isinstance(msg, list) else [msg]:
                reply = self.handle(m) if isinstance(m, dict) else self._error(None, -32600, "Invalid request")
                if reply is not None:
                    self.send(reply)


def _startup(sources: list[str] | None) -> None:
    try:
        if sources:
            with _index_lock:
                index_paths(sources, quiet=True)
        get_model()
        st = status()
        print(f"[rag] MCP ready: {st['files']} files, {st['chunks']} chunks ({st['index']})", file=sys.stderr)
    except BaseException as e:  # never let a background failure kill the server
        print(f"[rag] startup failed: {e}", file=sys.stderr)


def run(source: list[str] | None = None) -> None:
    protocol_out = sys.stdout
    sys.stdout = sys.stderr  # stray prints must never corrupt the JSON-RPC stream
    threading.Thread(target=_startup, args=(source,), daemon=True).start()
    Server(protocol_out).serve(sys.stdin)

"""Search UI backend (`./rag.sh ui`), built on the standard library's http.server: no extra deps."""

from __future__ import annotations

import json
import re
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..embedder import get_model
from ..index import index_paths, status
from ..search import default_searcher, to_markdown

STATIC = Path(__file__).parent / "static"
_reindex_lock = threading.Lock()


class ApiError(Exception):
    def __init__(self, code: int, detail: str):
        super().__init__(detail)
        self.code, self.detail = code, detail


def _int(qs: dict, name: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(qs.get(name, [default])[0])
    except ValueError:
        raise ApiError(400, f"{name} must be an integer")
    return max(lo, min(hi, v))


def _search(qs: dict):
    q = qs.get("q", [""])[0]
    k = _int(qs, "k", 8, 1, 50)
    path = qs.get("path", [None])[0] or None
    weaker = qs.get("all", ["false"])[0].lower() in ("1", "true", "yes")
    results = default_searcher().search(q, k=k, path_filter=path, margin=0.0 if weaker else None)
    return {"query": q, "results": results, "markdown": to_markdown(results, q)}


def _file(qs: dict):
    path = qs.get("path", [""])[0]
    if path not in {f["path"] for f in default_searcher().files()}:  # only serve indexed files
        raise ApiError(404, "not indexed")
    return {"path": path, "text": Path(path).read_text(encoding="utf-8", errors="replace")}


def _chunk(chunk_id: int, qs: dict):
    c = default_searcher().get_chunk(chunk_id, _int(qs, "neighbors", 1, 0, 5))
    if c is None:
        raise ApiError(404, "no such chunk")
    return c


def _reindex():
    if not _reindex_lock.acquire(blocking=False):
        raise ApiError(409, "already indexing")
    try:
        return index_paths(None, quiet=True)
    except SystemExit as e:
        raise ApiError(400, str(e))
    finally:
        _reindex_lock.release()


class Handler(BaseHTTPRequestHandler):
    server_version = "rag-ui"

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _dispatch(self, method: str) -> None:
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        try:
            if method == "GET" and url.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if method == "GET" and url.path == "/api/search":
                return self._json(200, _search(qs))
            if method == "GET" and url.path == "/api/status":
                return self._json(200, status())
            if method == "GET" and url.path == "/api/files":
                return self._json(200, default_searcher().files())
            if method == "GET" and url.path == "/api/file":
                return self._json(200, _file(qs))
            m = re.fullmatch(r"/api/chunk/(\d+)", url.path)
            if method == "GET" and m:
                return self._json(200, _chunk(int(m.group(1)), qs))
            if method == "POST" and url.path == "/api/reindex":
                return self._json(200, _reindex())
            raise ApiError(404, "not found")
        except ApiError as e:
            self._json(e.code, {"detail": e.detail})
        except Exception as e:  # surface errors to the UI instead of hanging
            print(f"[rag] {method} {self.path} failed: {e!r}", file=sys.stderr)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": str(e)})

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")


def serve(host: str = "127.0.0.1", port: int = 7860) -> None:
    threading.Thread(target=get_model, daemon=True).start()  # warm the model so the first search is fast
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

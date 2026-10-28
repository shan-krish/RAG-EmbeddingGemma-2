"""SQLite-backed index. One file (`index.db`) holds files, chunks and their embeddings.

Indexing is incremental: a file is only re-chunked and re-embedded when its content hash
changes, and files that disappeared from an indexed folder are dropped.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from . import config
from .chunker import chunk_text, doc_title
from .embedder import embed_documents, format_document

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta  (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, hash TEXT, title TEXT, indexed_at REAL);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
    ord INTEGER, start_line INTEGER, end_line INTEGER,
    headings TEXT, text TEXT, embedding BLOB
);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
"""


def db_path(index_dir: Path | None = None) -> Path:
    return (index_dir or config.INDEX_DIR) / "index.db"


def connect(index_dir: Path | None = None) -> sqlite3.Connection:
    p = db_path(index_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    con.executescript(SCHEMA)
    return con


def get_meta(con: sqlite3.Connection, key: str, default=None):
    row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def set_meta(con: sqlite3.Connection, key: str, value) -> None:
    con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, json.dumps(value)))


def iter_files(root: Path, exts: set[str]):
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in config.IGNORE_DIRS and not d.startswith(".")]
        for name in sorted(filenames):
            if Path(name).suffix.lower() in exts:
                yield Path(dirpath) / name


def _log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg, file=sys.stderr)


def index_paths(paths: list[str | Path] | None = None, exts: set[str] | None = None,
                index_dir: Path | None = None, quiet: bool = False) -> dict:
    """Index the given files/folders (or the previously indexed sources if none are given)."""
    con = connect(index_dir)
    dim = get_meta(con, "dim")
    if dim is not None and dim != config.EMBED_DIM:
        raise SystemExit(f"Index was built with dim={dim} but RAG_DIM={config.EMBED_DIM}. "
                         f"Delete {db_path(index_dir)} or set RAG_DIM={dim}.")
    set_meta(con, "dim", config.EMBED_DIM)
    # Chunking settings changed since the last run -> re-chunk every file even if unchanged.
    chunking = f"v1:{config.CHUNK_TARGET}:{config.CHUNK_MAX}:{config.CHUNK_MIN}"
    rechunk = get_meta(con, "chunking") not in (None, chunking)
    set_meta(con, "chunking", chunking)

    sources = [str(Path(p).expanduser().resolve()) for p in paths] if paths else get_meta(con, "sources", [])
    if not sources:
        raise SystemExit("Nothing to index: pass a folder, e.g. `rag.sh index ./docs`.")
    exts = exts or set(get_meta(con, "exts", sorted(config.DEFAULT_EXTS)))
    set_meta(con, "sources", sorted(set(get_meta(con, "sources", [])) | set(sources)))
    set_meta(con, "exts", sorted(exts))

    stats = {"added": 0, "updated": 0, "unchanged": 0, "removed": 0, "chunks": 0}
    seen: set[str] = set()
    t0 = time.time()
    for src in sources:
        root = Path(src)
        if not root.exists():
            _log(f"[rag] skip missing source {root}", quiet)
            continue
        for f in iter_files(root, exts):
            path = str(f.resolve())
            seen.add(path)
            try:
                raw = f.read_bytes()
            except OSError:
                continue
            h = hashlib.sha1(raw).hexdigest()
            row = con.execute("SELECT hash FROM files WHERE path = ?", (path,)).fetchone()
            if row and row["hash"] == h and not rechunk:
                stats["unchanged"] += 1
                continue
            text = raw.decode("utf-8", errors="replace")
            title = doc_title(text, path)
            chunks = chunk_text(text, path)
            _log(f"[rag] {'update' if row else 'add   '} {path} ({len(chunks)} chunks)", quiet)
            vecs = embed_documents([format_document(title, c.breadcrumb, c.text) for c in chunks]) if chunks else []
            with con:
                con.execute("DELETE FROM files WHERE path = ?", (path,))
                con.execute("INSERT INTO files(path, hash, title, indexed_at) VALUES (?,?,?,?)",
                            (path, h, title, time.time()))
                con.executemany(
                    "INSERT INTO chunks(path, ord, start_line, end_line, headings, text, embedding) VALUES (?,?,?,?,?,?,?)",
                    [(path, i, c.start_line, c.end_line, json.dumps(c.headings), c.text,
                      vecs[i].astype(np.float16).tobytes()) for i, c in enumerate(chunks)])
            stats["updated" if row else "added"] += 1
            stats["chunks"] += len(chunks)

    # Drop files that were deleted, or that vanished from a folder scanned in this run.
    scanned = [Path(s) for s in sources]
    for row in con.execute("SELECT path FROM files").fetchall():
        p = Path(row["path"])
        if row["path"] in seen:
            continue
        if not p.exists() or any(p == r or r in p.parents for r in scanned):
            con.execute("DELETE FROM files WHERE path = ?", (row["path"],))
            stats["removed"] += 1
    con.commit()
    stats["seconds"] = round(time.time() - t0, 2)
    _log(f"[rag] done: {stats}", quiet)
    con.close()
    return stats


def status(index_dir: Path | None = None) -> dict:
    con = connect(index_dir)
    out = {
        "index": str(db_path(index_dir)),
        "sources": get_meta(con, "sources", []),
        "dim": get_meta(con, "dim"),
        "files": con.execute("SELECT COUNT(*) FROM files").fetchone()[0],
        "chunks": con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
    }
    con.close()
    return out

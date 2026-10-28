"""Hybrid retrieval: EmbeddingGemma cosine similarity + BM25, merged with reciprocal rank fusion."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .embedder import embed_query
from .index import connect, get_meta

RRF_K = 60
CANDIDATES = 50
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[_\-.][A-Za-z0-9]+)*")
_STOP = set("""a an and are as at be but by can do does for from has have how i if in into is it its
me my no not of on or our so than that the their then there these this to us was we what when where
which who why will with you your""".split())


def tokenize(text: str) -> list[str]:
    out = []
    for w in _WORD_RE.findall(text):
        lw = w.lower()
        if lw in _STOP:
            continue
        out.append(lw)
        # Also index the parts of snake_case / kebab-case / dotted / camelCase identifiers.
        parts = re.split(r"[_\-.]|(?<=[a-z])(?=[A-Z])", w)
        if len(parts) > 1:
            out.extend(p.lower() for p in parts if p)
    return out


class Searcher:
    """Holds the embedding matrix and BM25 index in memory; reloads when the index changes."""

    def __init__(self, index_dir: Path | None = None):
        self.index_dir = index_dir
        self._sig = None
        self._lock = threading.Lock()
        self.rows: list[dict] = []
        self.matrix = np.zeros((0, config.EMBED_DIM), dtype=np.float32)
        self.bm25 = None
        self.sources: list[str] = []

    def _signature(self, con):
        return con.execute("SELECT COUNT(*), COALESCE(MAX(id), 0), "
                           "(SELECT COALESCE(MAX(indexed_at), 0) FROM files) FROM chunks").fetchone()[:]

    def _load(self):
        con = connect(self.index_dir)
        try:
            sig = self._signature(con)
            if sig == self._sig:
                return
            dim = get_meta(con, "dim", config.EMBED_DIM)
            if dim != config.EMBED_DIM:
                raise SystemExit(f"Index dim={dim} but RAG_DIM={config.EMBED_DIM}.")
            rows = con.execute(
                "SELECT c.id, c.path, c.ord, c.start_line, c.end_line, c.headings, c.text, c.embedding, f.title "
                "FROM chunks c JOIN files f ON f.path = c.path ORDER BY c.id").fetchall()
            self.sources = get_meta(con, "sources", [])
        finally:
            con.close()
        self.rows = [{
            "id": r["id"], "path": r["path"], "ord": r["ord"], "start_line": r["start_line"],
            "end_line": r["end_line"], "headings": json.loads(r["headings"]), "text": r["text"], "title": r["title"],
        } for r in rows]
        self.matrix = (np.stack([np.frombuffer(r["embedding"], dtype=np.float16) for r in rows]).astype(np.float32)
                       if rows else np.zeros((0, dim), dtype=np.float32))
        corpus = [tokenize(f"{r['title']} {' '.join(r['headings'])} {r['text']}") for r in self.rows]
        self.bm25 = BM25Okapi(corpus) if corpus else None
        self._sig = sig

    def rel_path(self, path: str) -> str:
        best = None
        for s in self.sources:
            sp = Path(s)
            base = sp.parent if sp.is_file() else sp
            try:
                rel = Path(path).relative_to(base)
            except ValueError:
                continue
            if best is None or len(rel.parts) < len(best.parts):
                best = rel
        return best.as_posix() if best else path

    def search(self, query: str, k: int = 5, path_filter: str | None = None,
               min_score: float | None = None, margin: float | None = None) -> list[dict]:
        """Top-k chunks for `query`. Pass min_score=0, margin=0 to disable relevance filtering."""
        with self._lock:
            self._load()
            if not self.rows or not query.strip():
                return []
            min_score = config.MIN_SCORE if min_score is None else min_score
            margin = config.SCORE_MARGIN if margin is None else margin

            mask = np.ones(len(self.rows), dtype=bool)
            if path_filter:
                pf = path_filter.lower()
                mask = np.array([pf in r["path"].lower() for r in self.rows])
                if not mask.any():
                    return []

            cos = self.matrix @ embed_query(query)
            cos[~mask] = -1.0
            dense_rank = np.argsort(-cos)[:CANDIDATES]

            bm = self.bm25.get_scores(tokenize(query)) if self.bm25 else np.zeros(len(self.rows))
            bm[~mask] = 0.0
            sparse_rank = [i for i in np.argsort(-bm)[:CANDIDATES] if bm[i] > 0]

            fused: dict[int, float] = {}
            for rank, i in enumerate(dense_rank):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank + 1)
            for rank, i in enumerate(sparse_rank):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank + 1)
            # Keep hits close to the best one; the top keyword hits only need to clear min_score
            # (exact identifiers like `NIMBUS_DB_URL` can embed less distinctively).
            floor = max(min_score, float(cos[dense_rank[0]]) - margin) if margin > 0 else min_score
            top_keyword = set(int(i) for i in sparse_rank[:2])

            results = []
            for i in sorted(fused, key=fused.get, reverse=True):
                if cos[i] < floor and not (i in top_keyword and cos[i] >= min_score):
                    continue
                r = self.rows[i]
                results.append({
                    "id": r["id"], "path": r["path"], "rel_path": self.rel_path(r["path"]),
                    "start_line": r["start_line"], "end_line": r["end_line"],
                    "title": r["title"], "headings": r["headings"], "text": r["text"],
                    "score": round(float(cos[i]), 4), "keyword": float(bm[i]) > 0,
                    "fused": round(fused[i], 5),
                })
                if len(results) >= k:
                    break
            return results

    def get_chunk(self, chunk_id: int, neighbors: int = 0) -> dict | None:
        """Return a chunk merged with up to `neighbors` adjacent chunks on each side."""
        with self._lock:
            self._load()
            by_id = {r["id"]: r for r in self.rows}
            r = by_id.get(chunk_id)
            if r is None:
                return None
            same = sorted((x for x in self.rows if x["path"] == r["path"]
                           and abs(x["ord"] - r["ord"]) <= neighbors), key=lambda x: x["ord"])
            return {
                "id": r["id"], "path": r["path"], "rel_path": self.rel_path(r["path"]),
                "start_line": same[0]["start_line"], "end_line": same[-1]["end_line"],
                "title": r["title"], "headings": r["headings"],
                "text": "\n\n".join(x["text"] for x in same),
            }

    def files(self) -> list[dict]:
        with self._lock:
            self._load()
            counts: dict[str, int] = {}
            for r in self.rows:
                counts[r["path"]] = counts.get(r["path"], 0) + 1
            titles = {r["path"]: r["title"] for r in self.rows}
            return [{"path": p, "rel_path": self.rel_path(p), "title": titles[p], "chunks": n}
                    for p, n in sorted(counts.items())]


def to_markdown(results: list[dict], query: str | None = None) -> str:
    """Compact, citation-friendly rendering meant to be dropped straight into an LLM context."""
    if not results:
        return f"No relevant chunks found{f' for: {query}' if query else ''}."
    out = []
    for n, r in enumerate(results, 1):
        crumb = " › ".join(r["headings"]) or r["title"]
        score = f"score {r['score']:.2f}, " if "score" in r else ""
        out.append(f"### [{n}] {r['rel_path']}:{r['start_line']}-{r['end_line']} — {crumb}\n"
                   f"<sub>{score}chunk id {r['id']} · full path {r['path']}</sub>\n\n{r['text']}")
    return "\n\n---\n\n".join(out)


_default: Searcher | None = None


def default_searcher() -> Searcher:
    global _default
    if _default is None:
        _default = Searcher()  # follows config.INDEX_DIR
    return _default

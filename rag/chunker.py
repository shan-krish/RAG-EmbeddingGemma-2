"""Heading-aware chunking for Markdown (and plain paragraph chunking for everything else).

Each chunk remembers its file, 1-based line range and heading breadcrumb, so a hit can be
cited as `path:12-40` and Copilot can open exactly that slice of the file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass
class Chunk:
    path: str
    start_line: int
    end_line: int
    headings: list[str] = field(default_factory=list)
    text: str = ""

    @property
    def breadcrumb(self) -> str:
        return " › ".join(self.headings)


@dataclass
class _Section:
    headings: list[str]
    lines: list[tuple[int, str]]  # (line_no, text)


def _strip_front_matter(lines: list[str]) -> int:
    """Return the index of the first line after a YAML front-matter block (0 if none)."""
    if lines and lines[0].strip() == "---":
        for i in range(1, min(len(lines), 200)):
            if lines[i].strip() in ("---", "..."):
                return i + 1
    return 0


def _markdown_sections(lines: list[str]) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    current = _Section([], [])
    in_fence = False
    start = _strip_front_matter(lines)
    for no, line in enumerate(lines[start:], start=start + 1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
        m = None if in_fence else HEADING_RE.match(line)
        if m:
            if current.lines:
                sections.append(current)
            level, title = len(m.group(1)), m.group(2).strip()
            stack = [(lv, t) for lv, t in stack if lv < level] + [(level, title)]
            current = _Section([t for _, t in stack], [(no, line)])
        else:
            current.lines.append((no, line))
    if current.lines:
        sections.append(current)
    return sections


def _blocks(lines: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    """Group lines into paragraphs / code fences, never splitting inside a fence."""
    blocks, cur, in_fence = [], [], False
    for no, line in lines:
        if FENCE_RE.match(line):
            in_fence = not in_fence
        if not in_fence and not line.strip():
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append((no, line))
    if cur:
        blocks.append(cur)
    return blocks


def _hard_split(block: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    """Split an oversized block by lines; a single giant line stays whole."""
    out, cur, size = [], [], 0
    for item in block:
        if cur and size + len(item[1]) > config.CHUNK_MAX:
            out.append(cur)
            cur, size = [], 0
        cur.append(item)
        size += len(item[1]) + 1
    if cur:
        out.append(cur)
    return out


def _pack(section: _Section, path: str) -> list[Chunk]:
    """Pack a section's paragraphs into chunks of roughly CHUNK_TARGET characters."""
    pieces: list[list[tuple[int, str]]] = []
    for b in _blocks(section.lines):
        pieces.extend(_hard_split(b) if sum(len(l) + 1 for _, l in b) > config.CHUNK_MAX else [b])

    chunks, cur, size = [], [], 0
    for p in pieces:
        psize = sum(len(l) + 1 for _, l in p)
        if cur and size + psize > config.CHUNK_TARGET:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(p)
        size += psize + 1
    if cur:
        chunks.append(cur)

    out = []
    for group in chunks:
        flat = [item for p in group for item in p]
        text = "\n\n".join("\n".join(l for _, l in p) for p in group).strip()
        if text:
            out.append(Chunk(path, flat[0][0], flat[-1][0], list(section.headings), text))
    return out


def _merged_headings(a: list[str], b: list[str]) -> list[str]:
    """Breadcrumb for a chunk spanning sections a then b: b if a is its parent, else the shared prefix."""
    if b[: len(a)] == a:
        return list(b)
    common = []
    for x, y in zip(a, b):
        if x != y:
            break
        common.append(x)
    return common


def _merge_small(chunks: list[Chunk]) -> list[Chunk]:
    """Fold a tiny parent section (e.g. an H1 with one intro line) into its first child section.

    Small sibling sections are kept as separate chunks: they are precise hits and merging them
    would blur the heading breadcrumb.
    """
    out: list[Chunk] = []
    carry: Chunk | None = None
    for c in chunks:
        if carry is not None:
            if c.headings[: len(carry.headings)] == carry.headings and len(carry.text) + len(c.text) <= config.CHUNK_MAX:
                c = Chunk(c.path, carry.start_line, c.end_line,
                          _merged_headings(carry.headings, c.headings), carry.text + "\n\n" + c.text)
            else:
                out.append(carry)
            carry = None
        if len(c.text) < config.CHUNK_MIN:
            carry = c
        else:
            out.append(c)
    if carry is not None:
        out.append(carry)
    return out


def chunk_text(text: str, path: str) -> list[Chunk]:
    lines = text.splitlines()
    if Path(path).suffix.lower() in {".md", ".markdown", ".mdx"}:
        sections = _markdown_sections(lines)
    else:
        sections = [_Section([], list(enumerate(lines, start=1)))]
    chunks = [c for s in sections for c in _pack(s, path)]
    return _merge_small(chunks)


def doc_title(text: str, path: str) -> str:
    """First H1 in the file, else the file name."""
    for line in text.splitlines()[:50]:
        m = HEADING_RE.match(line)
        if m and len(m.group(1)) == 1:
            return m.group(2).strip()
    return Path(path).stem.replace("-", " ").replace("_", " ")

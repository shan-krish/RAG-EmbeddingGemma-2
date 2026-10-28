"""Command-line entry point (wrapped by ./rag.sh)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from . import config

COMMANDS = {"search", "index", "status", "files", "ui", "mcp", "watch", "vscode"}


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() and not os.environ.get("NO_COLOR") else s


def _print_human(results: list[dict], query: str, snippet_lines: int) -> None:
    if not results:
        print(_c("33", f"No relevant chunks for “{query}”."))
        return
    width = min(shutil.get_terminal_size().columns, 110)
    for n, r in enumerate(results, 1):
        crumb = " › ".join(r["headings"]) or r["title"]
        loc = f"{r['rel_path']}:{r['start_line']}-{r['end_line']}"
        score = f"score {r['score']:.2f}"
        print(f"{_c('1;36', f'[{n}]')} {_c('1', loc)}  {_c('2', score)}"
              f"{_c('35', '  +keyword') if r.get('keyword') else ''}")
        print(f"    {_c('34', crumb)}")
        lines = [l for l in r["text"].splitlines() if l.strip()]
        shown = lines if snippet_lines <= 0 else lines[:snippet_lines]
        for l in shown:
            print("    " + (l if len(l) <= width - 4 else l[: width - 6] + " …"))
        if len(shown) < len(lines):
            print(_c("2", f"    … {len(lines) - len(shown)} more lines (use --full)"))
        print()


def cmd_search(a) -> None:
    from .search import default_searcher, to_markdown

    query = " ".join(a.query).strip()
    if not query:
        sys.exit("usage: rag.sh \"your question\" [-k 5] [--json|--md|--files-only]")
    results = default_searcher().search(query, k=a.k, path_filter=a.path, min_score=a.min_score,
                                        margin=0.0 if a.all else None)
    if a.json:
        print(json.dumps({"query": query, "results": results}, indent=2))
    elif a.md:
        print(to_markdown(results, query))
    elif a.files_only:
        seen = []
        for r in results:
            if r["path"] not in seen:
                seen.append(r["path"])
        print("\n".join(seen))
    else:
        _print_human(results, query, 0 if a.full else a.lines)


def cmd_index(a) -> None:
    from .index import index_paths

    exts = {e if e.startswith(".") else f".{e}" for e in a.ext.split(",")} if a.ext else None
    index_paths(a.paths or None, exts=exts)


def cmd_status(a) -> None:
    from .index import status

    print(json.dumps(status(), indent=2))


def cmd_files(a) -> None:
    from .search import default_searcher

    for f in default_searcher().files():
        print(f"{f['chunks']:4d}  {f['rel_path']}")


def cmd_ui(a) -> None:
    from .web.app import serve

    url = f"http://{a.host}:{a.port}"
    print(f"[rag] UI on {url}  (Ctrl-C to stop)", file=sys.stderr)
    if not a.no_open:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    serve(a.host, a.port)


def cmd_mcp(a) -> None:
    from .mcp_server import run

    run(source=a.source)


def cmd_watch(a) -> None:
    import time

    from .index import get_meta, connect, index_paths, iter_files, status

    index_paths(a.paths or None)
    targets = status()["sources"]
    con = connect()
    exts = set(get_meta(con, "exts", sorted(config.DEFAULT_EXTS)))
    con.close()

    def snapshot() -> dict:
        snap = {}
        for t in targets:
            for f in iter_files(Path(t), exts):
                try:
                    st = f.stat()
                    snap[str(f)] = (st.st_mtime_ns, st.st_size)
                except OSError:
                    pass
        return snap

    print(f"[rag] watching {', '.join(targets)} every {a.interval}s (Ctrl-C to stop)", file=sys.stderr)
    last = snapshot()
    try:
        while True:
            time.sleep(a.interval)
            now = snapshot()
            if now != last:
                index_paths(None)
                last = now
    except KeyboardInterrupt:
        pass


def cmd_vscode(a) -> None:
    from .vscode import install

    install(Path(a.project).expanduser().resolve(), docs=a.docs)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag.sh", description="Local RAG over your docs with EmbeddingGemma 2.")
    p.add_argument("--index-dir", help=f"index location (default {config.INDEX_DIR}, env RAG_INDEX_DIR)")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("search", help="search the index (default command)")
    s.add_argument("query", nargs="*")
    s.add_argument("-k", type=int, default=5, help="max results (default 5)")
    s.add_argument("-p", "--path", help="only paths containing this substring")
    s.add_argument("--min-score", type=float, help=f"cosine threshold (default {config.MIN_SCORE})")
    s.add_argument("--all", action="store_true", help="don't trim hits that score far below the best one")
    s.add_argument("-n", "--lines", type=int, default=8, help="snippet lines per hit in terminal output")
    s.add_argument("--full", action="store_true", help="print whole chunks")
    fmt = s.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="machine-readable output")
    fmt.add_argument("--md", action="store_true", help="LLM-ready Markdown (what Copilot gets)")
    fmt.add_argument("--files-only", action="store_true", help="just the matching file paths")
    s.set_defaults(fn=cmd_search)

    i = sub.add_parser("index", help="index files/folders (no args = refresh existing sources)")
    i.add_argument("paths", nargs="*")
    i.add_argument("--ext", help=f"comma-separated extensions (default {','.join(sorted(config.DEFAULT_EXTS))})")
    i.set_defaults(fn=cmd_index)

    sub.add_parser("status", help="show index stats").set_defaults(fn=cmd_status)
    sub.add_parser("files", help="list indexed files").set_defaults(fn=cmd_files)

    u = sub.add_parser("ui", help="start the web UI")
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--port", type=int, default=7860)
    u.add_argument("--no-open", action="store_true", help="don't open a browser")
    u.set_defaults(fn=cmd_ui)

    m = sub.add_parser("mcp", help="run the MCP server for VS Code Copilot (stdio)")
    m.add_argument("--source", action="append", help="folder to (re)index on startup; repeatable")
    m.set_defaults(fn=cmd_mcp)

    w = sub.add_parser("watch", help="re-index automatically when files change")
    w.add_argument("paths", nargs="*")
    w.add_argument("--interval", type=float, default=2.0, help="seconds between checks (default 2)")
    w.set_defaults(fn=cmd_watch)

    v = sub.add_parser("vscode", help="wire this RAG into a VS Code project for Copilot")
    v.add_argument("project", nargs="?", default=".", help="project folder (default: current)")
    v.add_argument("--docs", action="append", help="docs folder(s) inside the project to index (default: whole project)")
    v.set_defaults(fn=cmd_vscode)
    return p


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `rag.sh "some question"` == `rag.sh search "some question"`
    pos = 2 if argv[:1] == ["--index-dir"] else 0
    if len(argv) > pos and argv[pos] not in COMMANDS and argv[pos] not in ("-h", "--help"):
        argv.insert(pos, "search")
    a = build_parser().parse_args(argv)
    if a.index_dir:
        config.INDEX_DIR = Path(a.index_dir).expanduser().resolve()
        os.environ["RAG_INDEX_DIR"] = str(config.INDEX_DIR)
    if not getattr(a, "fn", None):
        build_parser().print_help()
        return
    a.fn(a)


if __name__ == "__main__":
    main()

# RAG-EmbeddingGemma-2

A small, fully local RAG retriever for your Markdown/text docs, built on
[Google EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2).

It's built to keep an AI assistant's context window **clean and small**. Instead of reading whole files,
Copilot (or any MCP client, or you in a terminal) asks a question and gets back only the few chunks that
are actually relevant, each with a `path:start-end` citation.

```
$ ./rag.sh "client keeps getting token_expired after refresh"

[1] troubleshooting.md:14-17  score 0.81  +keyword
    Troubleshooting › Error: token_expired loops
    If a client keeps receiving `token_expired` right after refreshing, its clock is probably
    skewed. JWT validation allows only 30 seconds of clock skew. …

[2] authentication.md:19-23  score 0.81  +keyword
    Authentication and Authorization › Refreshing tokens
    …
```

## Features

- **Everything is in the repo.** The model (text-only EmbeddingGemma 2, as ONNX) is committed as
  plain git files, with no Git LFS. After `git clone` it never touches the network: no Hugging Face
  downloads, no telemetry, no web fonts.
- **Python 3.9+ and three dependencies:** `numpy`, `onnxruntime` and `tokenizers`, pinned to
  long-established releases. The web UI, MCP server and file watcher use only the standard library.
- **Heading-aware chunking.** Markdown is split by section, so each chunk knows its breadcrumb
  (`Deployment Guide › Rolling back manually`) and its exact line range.
- **Hybrid search.** EmbeddingGemma cosine similarity and BM25 keyword scores are merged with
  reciprocal rank fusion. BM25 catches identifiers, config keys and error codes that embeddings blur.
- **Relevance cut-off.** Weak hits are dropped, so off-topic queries return *nothing* rather than filler.
- **Incremental indexing.** Only changed files are re-embedded, and deleted files drop out of the index.
- **Four ways to use it:** a CLI script, a web UI, an MCP server for VS Code Copilot, and `--json` output for your own scripts.

## Quick start

Requirements: macOS or Linux with Python 3.9 or newer. The macOS system `python3` (3.9.6) works.

```bash
git clone https://github.com/shan-krish/RAG-EmbeddingGemma-2.git   # ~600 MB, includes the model
cd RAG-EmbeddingGemma-2
./setup.sh                      # venv + 3 deps, then rebuilds the model weights locally (~1 s)
./rag.sh index sample_docs      # index the demo docs
./rag.sh "how do I roll back a deploy?"
```

`rag.sh` runs `setup.sh` automatically the first time if you skip it. To use a specific interpreter,
run `PYTHON=/path/to/python3 ./setup.sh`.

### Dependencies

| Package | Python 3.9 | Python 3.10+ |
|---|---|---|
| numpy | `==2.0.2` (Aug 2024) | `>=2.2,<3` |
| onnxruntime | `==1.19.2` (Sep 2024) | `>=1.22,<2` |
| tokenizers | `==0.22.2` (Jan 2026) | `>=0.22,<0.23` |

On Python 3.9 these are the newest releases that still support it. They're all months old, so
they pass "too new / quarantined" rules on curated package mirrors (Artifactory, Nexus and similar).
`pip install` is the only step that needs a package index. Point pip at your mirror as usual, for
example with `PIP_INDEX_URL`.

## CLI

```bash
./rag.sh index ~/notes ./docs          # index folders/files (incremental; remembers the sources)
./rag.sh index                         # refresh all remembered sources
./rag.sh index ./src --ext md,py,ts    # choose extensions (default: md, markdown, mdx, txt, rst, adoc)
./rag.sh "your question"               # search (same as: ./rag.sh search "…")
./rag.sh "…" -k 3                      # at most 3 chunks
./rag.sh "…" -p deploy                 # only files whose path contains "deploy"
./rag.sh "…" --md                      # LLM-ready Markdown, exactly what Copilot receives
./rag.sh "…" --json                    # machine-readable
./rag.sh "…" --files-only              # just the matching file paths
./rag.sh "…" --full                    # whole chunks instead of snippets
./rag.sh "…" --all                     # keep hits that score well below the best one
./rag.sh status | files                # index stats / indexed files
./rag.sh watch                         # re-index automatically on file changes
./rag.sh --index-dir /path/.rag_index …  # use a separate index (e.g. one per project)
```

### Tips

```bash
# Start fresh with only your own docs (drops the sample docs from the index)
rm -rf .rag_index && ./rag.sh index ~/notes

# Run it from anywhere as `rag`
echo "alias rag=\"$PWD/rag.sh\"" >> ~/.zshrc && source ~/.zshrc

# Pipe results into other tools
rag "rate limits" --md | pbcopy
rag "rate limits" --json | jq '.results[].rel_path'
```

The first command after a fresh start takes a few seconds while the model loads; after that, searches are fast.

## Web UI

```bash
./rag.sh ui            # http://127.0.0.1:7860
```

The UI has a search box, result cards with highlighted terms, a score and `file:lines` on each hit, and a
file filter. You can expand a hit to its neighbouring chunks, open the whole file in a side panel with the
matched lines highlighted, and use **Copy as context** to get Markdown you can paste into any chat.
Supports light and dark mode; press `/` to focus the search box.

> Open the UI through `./rag.sh ui`, not by double-clicking `rag/web/static/index.html`. The page needs
> the Python server behind it to search; opened as a plain file, it shows a "server isn't reachable" notice.

## Use with VS Code + GitHub Copilot

`./rag.sh mcp` runs an [MCP](https://modelcontextprotocol.io) server over stdio with these tools:

| Tool | What it does |
|---|---|
| `search_docs(query, k=5, path_filter=None)` | Top relevant chunks as compact Markdown with `path:start-end` citations |
| `get_chunk(chunk_id, neighbors=1)` | Widen a hit with adjacent chunks instead of opening the file |
| `list_indexed_files()` | What's in the index |
| `reindex()` | Pick up doc changes (only changed files are re-embedded) |

The server implements MCP's stdio transport directly (newline-delimited JSON-RPC), because the
official `mcp` SDK needs Python 3.10+. It has been tested against the official MCP client.

**Option A: one command per project**

```bash
/path/to/RAG-EmbeddingGemma-2/rag.sh vscode /path/to/your-project --docs docs
```

This writes `.vscode/mcp.json` (registering `docs-rag`, with the index stored in
`your-project/.rag_index`) and appends a short section to `.github/copilot-instructions.md`. That section
tells Copilot to call `search_docs` before reading docs. It also adds `.rag_index/` to `.gitignore` if the
project has one. Omit `--docs` to index the whole project.

**Option B: by hand.** Copy [`examples/vscode/mcp.json`](examples/vscode/mcp.json) to
`.vscode/mcp.json` in your project and set the absolute path to `rag.sh`. Then copy
[`examples/vscode/copilot-instructions.md`](examples/vscode/copilot-instructions.md) into
`.github/copilot-instructions.md`.

Then open Copilot Chat in **Agent** mode. Start `docs-rag` from the tools picker (or from `mcp.json`'s
inline "Start" action) and ask questions. On startup the server incrementally indexes the `--source`
folders, so the index stays current.

> Why this keeps context small: a typical `search_docs` answer is 3–5 chunks of ~100–400 tokens each,
> rather than several whole documents. Weak matches are filtered out entirely.

## How it works

```
 .md/.txt files ──► chunker ──► "title: Doc › Section | text: …" ──► EmbeddingGemma 2 (ONNX) ──► SQLite (.rag_index/index.db)
                   (headings,                                        768d, normalised          text + metadata
                    line ranges)                                                                + fp16 vectors

 query ──► "task: search result | query: …" ──► cosine top-50 ─┐
       └─► BM25 top-50 ───────────────────────────────────────┴─► RRF fusion ─► relevance cut-off ─► top-k chunks
```

- Prompts follow the model card: queries use `task: search result | query: …`, and documents use
  `title: {title} | text: {chunk}`, where the title is the file's H1 plus the section breadcrumb.
- Vectors are kept in memory as a NumPy matrix. Search is a single matrix-vector product, which is
  plenty fast for tens of thousands of chunks.

### The bundled model

`models/embeddinggemma-2/` holds a text-only ONNX export of EmbeddingGemma 2 (270M params). Mean
pooling and L2 normalisation are built into the graph. GitHub rejects files over 100 MB, so the
weights are stored like this:

- Google released the weights in bf16. In the fp32 ONNX file, nearly every value is an exact bf16
  upcast, so the repo stores only the upper 16 bits of each value: 542 MB split into six parts
  (≤ 90 MB each) under `weights/`.
- 382 values (out of 271 million) carry extra low bits. They're kept in `weights/fixups.npz`.
- On first run, `rag/model_files.py` widens the parts back to fp32 into `model.onnx.data`
  (gitignored), applies the fix-ups, and verifies every SHA-256 in `SHA256SUMS`. The rebuild is
  bit-exact.

The ONNX export reproduces the original PyTorch / sentence-transformers embeddings (cosine 1.000000,
including batched and padded inputs and texts over the 512-token sliding window). Search scores
match to 4 decimal places.

To regenerate the model from Google's release, a maintainer task that needs the heavy dev stack:

```bash
pip install -r requirements-dev.txt     # Python 3.11+: torch, transformers, sentence-transformers, onnx
python scripts/export_model.py          # downloads google/embeddinggemma-2 once, rewrites models/embeddinggemma-2/
```

## Configuration

Set environment variables to override the defaults (see [`rag/config.py`](rag/config.py)):

| Variable | Default | Meaning |
|---|---|---|
| `RAG_INDEX_DIR` | `./.rag_index` | Where the index lives |
| `RAG_DIM` | `768` | Matryoshka dimension: 768/512/256/128 (re-index after changing) |
| `RAG_MIN_SCORE` | `0.65` | Cosine floor; lower it if obvious answers are missed |
| `RAG_SCORE_MARGIN` | `0.12` | Drop hits scoring more than this below the best hit |
| `RAG_CHUNK_TARGET` / `RAG_CHUNK_MAX` | `1800` / `2800` | Chunk size in characters |
| `RAG_THREADS` | auto | ONNX Runtime CPU threads |
| `RAG_MAX_SEQ_LEN` | `1024` | Max tokens per chunk fed to the model |

## Repo layout

```
rag.sh / setup.sh          entry point and one-time setup
rag/chunker.py             heading-aware Markdown chunking
rag/embedder.py            ONNX Runtime inference, prompts, Matryoshka truncation
rag/model_files.py         rebuilds the model weights from the committed parts
rag/index.py               incremental SQLite index
rag/search.py              hybrid search (cosine + BM25) + Markdown formatting for LLMs
rag/mcp_server.py          MCP server for Copilot (stdlib JSON-RPC)
rag/web/                   stdlib HTTP server + single-page UI
rag/vscode.py              `rag.sh vscode` project installer
scripts/export_model.py    maintainer tool: re-export the model from Google's release
examples/vscode/           mcp.json + copilot-instructions templates
models/embeddinggemma-2/   bundled model (model.onnx, tokenizer, weights/ parts, SHA256SUMS)
sample_docs/               demo corpus
```

## Model and license

The bundled model is derived from `google/embeddinggemma-2`, which Google DeepMind released under
the Apache 2.0 license. `models/embeddinggemma-2/NOTICE` describes the changes, and
`models/embeddinggemma-2/README.md` is the original model card.

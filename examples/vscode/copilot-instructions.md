## Project docs: use the `docs-rag` tools

This project's documentation is indexed by a local RAG server (`docs-rag`, EmbeddingGemma 2).

- Before answering questions about how this project works, or before reading Markdown/docs files,
  call `search_docs` with a focused query. Use 3–5 results unless more are clearly needed.
- Work from the returned chunks. They cite `path:start-end`; quote or open only those line ranges.
- If a chunk is cut off, call `get_chunk` with its id (`neighbors: 1`) instead of opening the whole file.
- Open a full doc only when the chunks are clearly not enough.
- If nothing relevant comes back, say so and try one rephrased query before falling back to file search.

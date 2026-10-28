"""Paths and tunables. Everything can be overridden with RAG_* environment variables."""

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = Path(os.environ.get("RAG_MODEL_DIR", REPO_DIR / "models" / "embeddinggemma-2"))
MODEL_HUB_ID = "google/embeddinggemma-2"
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", REPO_DIR / ".rag_index"))

# Matryoshka dimension: 768 (full), 512, 256 or 128. Must match the index.
EMBED_DIM = int(os.environ.get("RAG_DIM", "768"))
DEVICE = os.environ.get("RAG_DEVICE")  # None = auto (mps > cuda > cpu)
BATCH_SIZE = int(os.environ.get("RAG_BATCH_SIZE", "16"))
MAX_SEQ_LEN = int(os.environ.get("RAG_MAX_SEQ_LEN", "1024"))

# Chunking, in characters (~4 chars per token).
CHUNK_TARGET = int(os.environ.get("RAG_CHUNK_TARGET", "1800"))
CHUNK_MAX = int(os.environ.get("RAG_CHUNK_MAX", "2800"))
CHUNK_MIN = int(os.environ.get("RAG_CHUNK_MIN", "160"))

# Relevance cut-offs (cosine similarity). With EmbeddingGemma 2 search prompts, off-topic text
# scores ~0.45-0.52, same-domain-but-unhelpful text ~0.58-0.64, and real answers ~0.68+.
# A hit is kept only if it clears MIN_SCORE and is within SCORE_MARGIN of the best hit,
# so Copilot isn't fed filler. Lower RAG_MIN_SCORE if your corpus misses obvious answers.
MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.65"))
SCORE_MARGIN = float(os.environ.get("RAG_SCORE_MARGIN", "0.12"))

DEFAULT_EXTS = {".md", ".markdown", ".mdx", ".txt", ".rst", ".adoc"}
IGNORE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
    ".rag_index", "models", "dist", "build", ".next", ".idea", ".vscode",
}

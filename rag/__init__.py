"""Local RAG over Markdown/text files, powered by Google's EmbeddingGemma 2 (ONNX, fully offline)."""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

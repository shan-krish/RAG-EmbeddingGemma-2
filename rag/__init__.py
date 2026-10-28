"""Local RAG over Markdown/text files, powered by Google's EmbeddingGemma 2."""

import logging
import os

# Keep stdout clean (the MCP server speaks JSON-RPC over it) and stderr quiet.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

"""EmbeddingGemma 2 via ONNX Runtime: task prefixes, batching, Matryoshka truncation.

Runtime needs only onnxruntime, tokenizers and numpy (works on Python 3.9+); mean pooling and
L2 normalisation are part of the exported graph (see scripts/export_model.py).
"""

from __future__ import annotations

import sys
import threading

import numpy as np

from . import config
from .model_files import ensure_model

QUERY_PREFIX = "task: search result | query: "
PAD_ID = 0
_session = None
_tokenizer = None
_lock = threading.Lock()


def get_model():
    """Load (once) and return (onnxruntime session, tokenizer)."""
    global _session, _tokenizer
    with _lock:
        if _session is None:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            graph = ensure_model()  # rebuilds the committed weight parts on first run; never downloads
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            if config.THREADS:
                opts.intra_op_num_threads = config.THREADS
            _session = ort.InferenceSession(str(graph), sess_options=opts, providers=["CPUExecutionProvider"])
            tok = Tokenizer.from_file(str(config.MODEL_DIR / "tokenizer.json"))
            tok.no_padding()
            tok.enable_truncation(max_length=config.MAX_SEQ_LEN)
            _tokenizer = tok
        return _session, _tokenizer


def format_document(title: str, breadcrumb: str, text: str) -> str:
    full_title = f"{title} › {breadcrumb}" if breadcrumb and breadcrumb != title else (title or "none")
    return f"title: {full_title} | text: {text}"


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _encode(texts: list[str], show_progress: bool = False) -> np.ndarray:
    session, tok = get_model()
    ids = [e.ids for e in tok.encode_batch(texts)]
    out = np.zeros((len(texts), 768), dtype=np.float32)
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))  # similar lengths -> less padding
    for start in range(0, len(order), config.BATCH_SIZE):
        batch = order[start:start + config.BATCH_SIZE]
        width = max(len(ids[i]) for i in batch)
        input_ids = np.full((len(batch), width), PAD_ID, dtype=np.int64)
        mask = np.zeros((len(batch), width), dtype=np.int64)
        for row, i in enumerate(batch):
            input_ids[row, :len(ids[i])] = ids[i]
            mask[row, :len(ids[i])] = 1
        out[batch] = session.run(None, {"input_ids": input_ids, "attention_mask": mask})[0]
        if show_progress:
            print(f"\r[rag] embedded {min(start + config.BATCH_SIZE, len(order))}/{len(order)}", end="",
                  file=sys.stderr)
    if show_progress:
        print(file=sys.stderr)
    # Matryoshka: keep the leading dims, then re-normalise.
    return _normalize(out[:, :config.EMBED_DIM]).astype(np.float32)


def embed_documents(docs: list[str], show_progress: bool = False) -> np.ndarray:
    return _encode(docs, show_progress)


def embed_query(query: str) -> np.ndarray:
    return _encode([QUERY_PREFIX + query])[0]

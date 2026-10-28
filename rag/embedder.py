"""EmbeddingGemma 2 wrapper: text-only loading, task prefixes, Matryoshka truncation."""

from __future__ import annotations

import sys
import threading

import numpy as np

from . import config

QUERY_PREFIX = "task: search result | query: "
_model = None
_lock = threading.Lock()


def _pick_device() -> str:
    if config.DEVICE:
        return config.DEVICE
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _local_model_ok() -> bool:
    weights = config.MODEL_DIR / "model.safetensors"
    # A git-lfs pointer file is ~130 bytes; the real weights are ~540 MB.
    return weights.exists() and weights.stat().st_size > 1_000_000


def get_model():
    global _model
    with _lock:
        if _model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            if _local_model_ok():
                source = str(config.MODEL_DIR)
            else:
                print(f"[rag] local model missing at {config.MODEL_DIR}; using {config.MODEL_HUB_ID} from Hugging Face",
                      file=sys.stderr)
                source = config.MODEL_HUB_ID
            _model = SentenceTransformer(
                source,
                device=_pick_device(),
                # Text-only: skip the vision/audio encoders (740M -> 270M params).
                config_kwargs={"vision_config": None, "audio_config": None},
                model_kwargs={"dtype": torch.float32},
            )
            _model.max_seq_length = config.MAX_SEQ_LEN
        return _model


def format_document(title: str, breadcrumb: str, text: str) -> str:
    full_title = f"{title} › {breadcrumb}" if breadcrumb and breadcrumb != title else (title or "none")
    return f"title: {full_title} | text: {text}"


def _encode(texts: list[str], show_progress: bool = False) -> np.ndarray:
    model = get_model()
    emb = model.encode(
        texts,
        batch_size=config.BATCH_SIZE,
        truncate_dim=config.EMBED_DIM,
        normalize_embeddings=True,  # re-normalises after Matryoshka truncation
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return emb.astype(np.float32)


def embed_documents(docs: list[str], show_progress: bool = False) -> np.ndarray:
    return _encode(docs, show_progress)


def embed_query(query: str) -> np.ndarray:
    return _encode([QUERY_PREFIX + query])[0]

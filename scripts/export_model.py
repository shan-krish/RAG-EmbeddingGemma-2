"""Maintainer tool: rebuild the bundled ONNX model from Google's EmbeddingGemma 2.

You do NOT need this to use the RAG; the result is already committed under models/.
It needs the heavy dev stack (Python 3.11+, torch, transformers, sentence-transformers,
onnx, onnxscript; see requirements-dev.txt) and downloads the original model once.

    python scripts/export_model.py                      # from google/embeddinggemma-2 on the Hub
    python scripts/export_model.py --source ./local-dir # from a local copy

What it produces in models/embeddinggemma-2/:
  model.onnx            graph: (input_ids, attention_mask) -> mean-pooled, L2-normalised 768d embedding
  weights/*.part-NN     the fp32 external weights stored as their upper 16 bits (bf16), split < 100 MB
  weights/fixups.npz    the few fp32 words whose lower 16 bits are non-zero (makes the rebuild bit-exact)
  SHA256SUMS            checksums of every part and of the rebuilt model.onnx.data
  tokenizer.json etc.   copied from the source model
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

import numpy as np

PART_BYTES = 90 * 1024 * 1024  # GitHub rejects files > 100 MB
COPY = ["tokenizer.json", "tokenizer_config.json", "config.json", "README.md"]


def export_onnx(source: str, out: Path) -> Path:
    import torch
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(
        source, device="cpu",
        config_kwargs={"vision_config": None, "audio_config": None},  # text-only: 740M -> 270M params
        model_kwargs={"dtype": torch.bfloat16},  # released weights are bf16
    )
    st = st.float().eval()  # exact bf16 -> fp32 upcast for a portable CPU graph

    class Embed(torch.nn.Module):
        def __init__(self, st):
            super().__init__()
            self.st = st

        def forward(self, input_ids, attention_mask):
            return self.st({"input_ids": input_ids, "attention_mask": attention_mask})["sentence_embedding"]

    enc = st.tokenizer(["title: none | text: hello", "task: search result | query: a longer query"],
                       padding=True, return_tensors="pt")
    path = out / "model.onnx"
    torch.onnx.export(
        Embed(st).eval(), (enc["input_ids"], enc["attention_mask"]), str(path),
        input_names=["input_ids", "attention_mask"], output_names=["sentence_embedding"],
        dynamic_shapes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}},
        dynamo=True, external_data=True, opset_version=18,
    )
    src_dir = Path(st[0].auto_model.config._name_or_path)
    if not src_dir.is_dir():
        from huggingface_hub import snapshot_download
        src_dir = Path(snapshot_download(source, allow_patterns=COPY))
    for name in COPY:
        if (src_dir / name).exists():
            shutil.copy2(src_dir / name, out / name)
    return path


def pack_weights(data_file: Path, model_dir: Path) -> None:
    words = np.fromfile(data_file, dtype="<u4")
    low = (words & 0xFFFF).astype("<u2")
    fix_idx = np.flatnonzero(low).astype("<u8")
    print(f"{words.size} fp32 words, {fix_idx.size} need their low 16 bits stored")
    hi = (words >> 16).astype("<u2").tobytes()

    wdir = model_dir / "weights"
    shutil.rmtree(wdir, ignore_errors=True)
    wdir.mkdir(parents=True)
    sums = []
    for n, start in enumerate(range(0, len(hi), PART_BYTES)):
        name = f"model.onnx.data.bf16.part-{n:02d}"
        chunk = hi[start:start + PART_BYTES]
        (wdir / name).write_bytes(chunk)
        sums.append((hashlib.sha256(chunk).hexdigest(), f"weights/{name}"))
    np.savez(wdir / "fixups.npz", index=fix_idx, low=low[fix_idx])
    sums.append((hashlib.sha256((wdir / "fixups.npz").read_bytes()).hexdigest(), "weights/fixups.npz"))
    sums.append((hashlib.sha256(data_file.read_bytes()).hexdigest(), "model.onnx.data"))
    (model_dir / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for h, n in sums))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="google/embeddinggemma-2", help="Hub id or local model folder")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "models" / "embeddinggemma-2"))
    a = ap.parse_args()
    out = Path(a.out)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        onnx_path = export_onnx(a.source, tmp)
        out.mkdir(parents=True, exist_ok=True)
        for name in ["model.onnx", *COPY]:
            if (tmp / name).exists():
                shutil.copy2(tmp / name, out / name)
        pack_weights(tmp / "model.onnx.data", out)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()

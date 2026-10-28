"""Rebuild model.onnx.data from the <100 MB parts committed to git (local, no network).

The ONNX graph's fp32 weights are exact upcasts of Google's bf16 release, so the repo stores
only their upper 16 bits (`weights/*.bf16.part-NN`, split to stay under GitHub's 100 MB file
limit) plus `weights/fixups.npz` for the few words whose lower 16 bits are non-zero. The first
run widens them back to fp32 and checks every SHA-256 in SHA256SUMS, so the result is bit-exact.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

from . import config

DATA = "model.onnx.data"


def _manifest(model_dir: Path) -> dict[str, str]:
    out = {}
    for line in (model_dir / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name] = digest
    return out


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_model(model_dir: Path | None = None) -> Path:
    """Return the path to model.onnx, rebuilding model.onnx.data first if needed."""
    model_dir = model_dir or config.MODEL_DIR
    graph, target = model_dir / "model.onnx", model_dir / DATA
    if not graph.exists():
        raise SystemExit(f"[rag] model not found: {graph}")
    manifest = _manifest(model_dir)
    parts = sorted(n for n in manifest if ".bf16.part-" in n)
    expected_size = sum((model_dir / p).stat().st_size for p in parts if (model_dir / p).exists()) * 2
    if target.exists() and target.stat().st_size == expected_size:
        return graph  # already rebuilt

    missing = [p for p in parts + ["weights/fixups.npz"] if not (model_dir / p).exists()]
    if missing:
        raise SystemExit(f"[rag] missing model files: {', '.join(missing)} (re-clone the repo)")

    print(f"[rag] rebuilding {target.name} from {len(parts)} parts (one-time, ~1 s)…", file=sys.stderr)
    tmp = target.with_name(DATA + ".tmp")
    with open(tmp, "wb") as out:
        for name in parts:
            raw = (model_dir / name).read_bytes()
            if _sha256(raw) != manifest[name]:
                tmp.unlink()
                raise SystemExit(f"[rag] checksum mismatch for {name} (re-clone the repo)")
            hi = np.frombuffer(raw, dtype="<u2")
            out.write((hi.astype("<u4") << 16).tobytes())

    fix_file = model_dir / "weights" / "fixups.npz"
    if _sha256(fix_file.read_bytes()) != manifest["weights/fixups.npz"]:
        tmp.unlink()
        raise SystemExit("[rag] checksum mismatch for weights/fixups.npz (re-clone the repo)")
    fix = np.load(fix_file, allow_pickle=False)
    words = np.memmap(tmp, dtype="<u4", mode="r+")
    words[fix["index"]] |= fix["low"].astype("<u4")
    words.flush()
    del words

    h = hashlib.sha256()
    with open(tmp, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != manifest.get(DATA, h.hexdigest()):
        tmp.unlink()
        raise SystemExit(f"[rag] rebuilt {DATA} failed its checksum (re-clone the repo)")
    tmp.replace(target)
    return graph


if __name__ == "__main__":
    print(ensure_model())

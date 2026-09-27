"""Text embedders.

`TransformerEmbedder` is the real one. `HashEmbedder` is a dependency-free lexical
embedder (feature hashing over unigrams + bigrams) used for tests and for running the whole
pipeline on a laptop without downloading a model.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...  # (n, dim), L2-normalized float32


class HashEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _features(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for feat in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class TransformerEmbedder:
    """Hugging Face encoder with CLS (BGE) or mean pooling, L2-normalized.

    Uses `transformers` directly rather than `sentence-transformers`, which pulls in scikit-learn
    (blocked by some Windows application-control policies) for features we don't need.
    """

    def __init__(self, model_name: str, batch_size: int = 64):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "torch/transformers are not installed. Run `pip install -e .[ml]` "
                "or set CLEARANCE_EMBEDDER=hash."
            ) from e
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        self.name = model_name
        self.dim = self.model.config.hidden_size
        self.batch_size = batch_size
        self.pooling = "cls" if "bge" in model_name.lower() else "mean"
        # BGE models retrieve better when queries carry an instruction prefix.
        self._query_prefix = (
            "Represent this sentence for searching relevant passages: " if "bge" in model_name.lower() else ""
        )

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        torch = self._torch
        if is_query and self._query_prefix:
            texts = [self._query_prefix + t for t in texts]
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    texts[i : i + self.batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt"
                )
                hidden = self.model(**batch).last_hidden_state
                if self.pooling == "cls":
                    vec = hidden[:, 0]
                else:
                    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out.append(torch.nn.functional.normalize(vec, dim=-1).cpu().numpy())
        return np.vstack(out).astype(np.float32) if out else np.zeros((0, self.dim), dtype=np.float32)


def embed_query(embedder: Embedder, text: str) -> np.ndarray:
    if isinstance(embedder, TransformerEmbedder):
        return embedder.embed([text], is_query=True)[0]
    return embedder.embed([text])[0]


@lru_cache(maxsize=4)
def load_embedder(kind: str, model_name: str) -> Embedder:
    if kind == "hash":
        return HashEmbedder()
    if kind == "local":
        return TransformerEmbedder(model_name)
    raise ValueError(f"Unknown embedder {kind!r} (expected 'local' or 'hash')")

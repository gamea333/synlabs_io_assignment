"""Local embeddings via fastembed (ONNX, CPU) — no API cost per vector."""
import os
from functools import lru_cache

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless on Windows

import numpy as np
from fastembed import TextEmbedding

from app.config import get_settings

# bge models are trained with this instruction on the query side only
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=".cache/fastembed")
        self.dim = len(next(iter(self._model.embed(["dim probe"]))))

    def embed_docs(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return np.array(list(self._model.embed(texts, batch_size=batch_size)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        if "bge" in self.model_name.lower():
            text = BGE_QUERY_PREFIX + text
        return np.array(next(iter(self._model.embed([text]))), dtype=np.float32)


@lru_cache
def get_embedder(model_name: str | None = None) -> Embedder:
    return Embedder(model_name or get_settings().embed_model)

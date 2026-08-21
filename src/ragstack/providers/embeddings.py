"""Embedding providers: local sentence-transformers or any OpenAI-compatible API."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import EmbeddingConfig
from ..errors import ProviderError
from ..utils import get_logger

log = get_logger("ragstack.embeddings")


class EmbeddingProvider(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (n, dim) float32 L2-normalized matrix."""


class SentenceTransformerEmbeddings(EmbeddingProvider):
    name = "sentence-transformers"

    def __init__(self, model_name: str, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self.dim = 0

    def _ensure(self):
        if self._model is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ProviderError(
                    f"sentence-transformers not available ({e}); install with: pip install torch sentence-transformers"
                ) from e
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("loading embedding model %s on %s …", self.model_name, device)
            self._model = SentenceTransformer(self.model_name, device=device)
            self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 1), dtype=np.float32)
        self._ensure()
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 128,
        )
        return np.asarray(vecs, dtype=np.float32)


class OpenAICompatEmbeddings(EmbeddingProvider):
    name = "openai-compatible"

    def __init__(self, model: str, base_url: str | None, api_key_env: str, batch_size: int = 64):
        import os

        self.model_name = model
        self.base_url = base_url
        api_key = os.environ.get(api_key_env) or "not-needed"
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ProviderError(f"openai package missing: {e}") from e
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.batch_size = batch_size
        self.dim = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 1), dtype=np.float32)
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t[:32000] for t in texts[i : i + self.batch_size]]
            resp = self.client.embeddings.create(model=self.model_name, input=batch)
            out.extend(d.embedding for d in resp.data)
        arr = np.asarray(out, dtype=np.float32)
        if self.dim == 0:
            self.dim = int(arr.shape[1])
            log.info("embedding dim=%d via %s", self.dim, self.model_name)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def make_embedding_provider(cfg: EmbeddingConfig) -> EmbeddingProvider:
    if cfg.provider == "sentence-transformers":
        return SentenceTransformerEmbeddings(cfg.model, cfg.batch_size)
    if cfg.provider == "openai-compatible":
        return OpenAICompatEmbeddings(cfg.model, cfg.base_url, cfg.api_key_env, cfg.batch_size)
    raise ProviderError(f"unknown embedding provider: {cfg.provider}")

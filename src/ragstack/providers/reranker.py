"""Reranker providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..utils import get_logger

log = get_logger("ragstack.rerank")


class Reranker(ABC):
    name = "base"

    @abstractmethod
    def rerank(self, query: str, items: list[Any], text_key: str = "text") -> list[Any]:
        """Return items sorted by relevance to query (best first)."""


class NoopReranker(Reranker):
    name = "none"

    def rerank(self, query: str, items: list[Any], text_key: str = "text") -> list[Any]:
        return items


class CrossEncoderReranker(Reranker):
    name = "cross-encoder"

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("loading reranker %s â€¦", self.model_name)
            self._model = CrossEncoder(self.model_name, max_length=512)

    def rerank(self, query: str, items: list[Any], text_key: str = "text") -> list[Any]:
        if not items:
            return items
        self._ensure()
        pairs = [[query, getattr(it, text_key)[:4000]] for it in items]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(items, scores, strict=True), key=lambda x: float(x[1]), reverse=True)
        return [it for it, _ in ranked]


class CohereReranker(Reranker):
    """Cohere Rerank API (v2). Requires COHERE_API_KEY; uses httpx (no extra dep)."""

    name = "cohere"
    _API = "https://api.cohere.com/v2/rerank"

    def __init__(self, model_name: str, api_key_env: str = "COHERE_API_KEY"):
        import os

        self.model_name = model_name
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(f"missing {api_key_env} for cohere reranker")

    def rerank(self, query: str, items: list[Any], text_key: str = "text") -> list[Any]:
        if not items:
            return items
        import httpx

        resp = httpx.post(
            self._API,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_name,
                "query": query,
                "documents": [getattr(it, text_key)[:4000] for it in items],
                "top_n": len(items),
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        scored = sorted(results, key=lambda r: r["relevance_score"], reverse=True)
        return [items[r["index"]] for r in scored]


def make_reranker(provider: str, model: str) -> Reranker:
    if provider in ("none", "noop"):
        return NoopReranker()
    if provider == "cross-encoder":
        return CrossEncoderReranker(model)
    if provider == "cohere":
        return CohereReranker(model or "rerank-v3.5")
    raise ValueError(f"unknown reranker provider: {provider}")

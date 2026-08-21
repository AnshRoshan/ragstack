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

            log.info("loading reranker %s …", self.model_name)
            self._model = CrossEncoder(self.model_name, max_length=512)

    def rerank(self, query: str, items: list[Any], text_key: str = "text") -> list[Any]:
        if not items:
            return items
        self._ensure()
        pairs = [[query, getattr(it, text_key)[:4000]] for it in items]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(items, scores), key=lambda x: float(x[1]), reverse=True)
        return [it for it, _ in ranked]


def make_reranker(provider: str, model: str) -> Reranker:
    if provider in ("none", "noop"):
        return NoopReranker()
    if provider == "cross-encoder":
        return CrossEncoderReranker(model)
    raise ValueError(f"unknown reranker provider: {provider}")

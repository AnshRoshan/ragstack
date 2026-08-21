"""Dense vector retrieval + hybrid (dense+BM25) with Reciprocal Rank Fusion."""

from __future__ import annotations

import numpy as np

from ..types import RetrievedItem
from ..utils import get_logger
from .lexical_rag import search_chunks

log = get_logger("ragstack.vector_rag")


def semantic_search(embeddings, vector_store, query: str, top_k: int = 10, reranker=None) -> list[RetrievedItem]:
    qvec = embeddings.embed([query])[0]
    rows = vector_store.search(np.asarray(qvec), top_k=max(top_k * 2, 20))
    items = [
        RetrievedItem(
            chunk_id=r["id"],
            doc_id=r["doc_id"],
            source=r["source"],
            title=r["title"],
            text=r["text"],
            score=float(r["score"]),
            origin="vector",
        )
        for r in rows
    ]
    if reranker is not None and items:
        try:
            items = reranker.rerank(query, items, text_key="text")
        except Exception as e:
            log.warning("rerank failed: %s", e)
    return items[:top_k]


def rrf(ranklists: list[list[RetrievedItem]], k: int = 60) -> list[RetrievedItem]:
    """Reciprocal Rank Fusion over lists of RetrievedItems keyed by chunk_id."""
    scores: dict[str, float] = {}
    items: dict[str, RetrievedItem] = {}
    for lst in ranklists:
        for rank, item in enumerate(lst):
            key = item.chunk_id or item.source
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in items:
                items[key] = item
                items[key].origin = "hybrid"
    fused = sorted(items.values(), key=lambda it: scores.get(it.chunk_id or it.source, 0.0), reverse=True)
    for it in fused:
        it.score = round(scores.get(it.chunk_id or it.source, 0.0), 6)
    return fused


def hybrid_search(embeddings, vector_store, lexical_store, query: str, top_k: int = 10, reranker=None) -> list[RetrievedItem]:
    dense = []
    try:
        qvec = embeddings.embed([query])[0]
        dense = [
            RetrievedItem(chunk_id=r["id"], doc_id=r["doc_id"], source=r["source"], title=r["title"], text=r["text"], score=float(r["score"]), origin="vector")
            for r in vector_store.search(np.asarray(qvec), top_k=top_k * 3)
        ]
    except Exception as e:
        log.warning("dense leg failed: %s", e)
    sparse = search_chunks(lexical_store, query, top_k=top_k * 3, reranker=None)
    fused = rrf([dense, sparse])
    if reranker is not None and fused:
        try:
            fused = reranker.rerank(query, fused, text_key="text")
        except Exception as e:
            log.warning("rerank failed: %s", e)
    return fused[:top_k]

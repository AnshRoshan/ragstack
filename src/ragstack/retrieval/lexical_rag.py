"""Vectorless page-level + chunk-level BM25 retrieval."""

from __future__ import annotations

from ..types import RetrievedItem


def search_pages(lexical_store, query: str, top_k: int = 5) -> list[RetrievedItem]:
    hits = lexical_store.search(query, kind="page", top_k=top_k)
    return [
        RetrievedItem(
            chunk_id=r["id"],
            doc_id=r["doc_id"],
            source=r["source"],
            title=r["title"],
            text=r["text"],
            score=float(r["score"]),
            origin="page",
        )
        for r in hits
    ]


def search_chunks(lexical_store, query: str, top_k: int = 10, reranker=None) -> list[RetrievedItem]:
    items = [
        RetrievedItem(
            chunk_id=r["id"],
            doc_id=r["doc_id"],
            source=r["source"],
            title=r["title"],
            text=r["text"],
            score=float(r["score"]),
            origin="lexical",
        )
        for r in lexical_store.search(query, kind="chunk", top_k=max(top_k * 3, 20))
    ]
    if reranker is not None and items:
        try:
            items = reranker.rerank(query, items, text_key="text")
        except Exception as e:
            from ..utils import get_logger

            get_logger("ragstack.lexical").warning("rerank failed: %s", e)
    return items[:top_k]

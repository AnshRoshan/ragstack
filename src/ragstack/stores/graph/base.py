"""GraphStore ABC shared by the SQLite (embedded) and Neo4j backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ...types import ChunkExtraction


class GraphStore(ABC):
    backend: str = "base"

    @abstractmethod
    def upsert_extraction(self, extraction: ChunkExtraction, chunk_id: str, doc_id: str) -> None: ...

    @abstractmethod
    def neighborhood(self, query: str, limit: int = 25, max_hops: int = 2) -> dict[str, Any]:
        """Return {'seeds': [...], 'entities': [...], 'relations': [...], 'chunk_ids': [...]}."""

    @abstractmethod
    def all_entities(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def all_relations(self) -> list[tuple[str, str, float]]: ...

    @abstractmethod
    def save_communities(self, communities: list[dict[str, Any]]) -> None:
        """Each: {'member_ids': [...], 'summary': str, 'keywords': [str]}."""

    @abstractmethod
    def community_summaries(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def cache_get(self, key: str) -> str | None: ...

    @abstractmethod
    def cache_put(self, key: str, value: str) -> None: ...

    @abstractmethod
    def stats(self) -> dict[str, int]: ...

    @abstractmethod
    def clear(self) -> None: ...

    def close(self) -> None:  # noqa: B027 — optional for backends holding open connections
        ...

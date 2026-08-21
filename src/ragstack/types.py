"""Core data types shared across RAGStack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    id: str
    source: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    id: str
    doc_id: str
    ordinal: int
    text: str
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    n_tokens: int = 0

    @property
    def embed_text(self) -> str:
        return f"{self.context}\n{self.text}" if self.context else self.text


@dataclass(slots=True)
class ExtractedEntity:
    name: str
    type: str = "concept"
    description: str = ""


@dataclass(slots=True)
class ExtractedRelation:
    source: str
    target: str
    relation: str
    description: str = ""


@dataclass(slots=True)
class ChunkExtraction:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelation] = field(default_factory=list)


@dataclass(slots=True)
class RetrievedItem:
    ref_id: str = ""
    chunk_id: str = ""
    doc_id: str = ""
    source: str = ""
    title: str = ""
    text: str = ""
    score: float = 0.0
    origin: str = "chunk"

    def snippet(self, n: int = 400) -> str:
        return self.text[:n] + (" …" if len(self.text) > n else "")


@dataclass(slots=True)
class Citation:
    ref_id: str
    source: str
    title: str
    snippet: str


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class LLMResult:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


@dataclass(slots=True)
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)

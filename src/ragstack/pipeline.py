"""Ingestion pipeline: parse → chunk → enrich → embed → index (lexical+vector) → graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import AppConfig
from .errors import ProviderError
from .graphrag.communities import build_communities
from .graphrag.extract import extract_chunks
from .ingestion.chunker import chunk_document
from .ingestion.crawler import crawl as _crawl
from .ingestion.enricher import Enricher
from .ingestion.parsers import collect_files, parse_file
from .providers.embeddings import EmbeddingProvider
from .providers.llm import LLMProvider
from .stores.graph.base import GraphStore
from .stores.lexical import LexicalStore
from .stores.vector import VectorStore
from .types import Document
from .utils import dump_jsonl, get_logger, load_jsonl, now_iso, sha1

log = get_logger("ragstack.pipeline")


@dataclass
class IngestStats:
    files_seen: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    chunks: int = 0
    entities: int = 0
    relations: int = 0
    communities: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"files={self.files_seen}",
            f"indexed={self.indexed}",
            f"skipped={self.skipped}",
            f"chunks={self.chunks}",
        ]
        if self.entities:
            parts.append(f"entities={self.entities}")
        if self.relations:
            parts.append(f"relations={self.relations}")
        if self.communities:
            parts.append(f"communities={self.communities}")
        if self.failed:
            parts.append(f"FAILED={self.failed}")
        return " ".join(parts)


class IngestionPipeline:
    def __init__(
        self,
        cfg: AppConfig,
        embeddings: EmbeddingProvider,
        llm: LLMProvider,
        lexical: LexicalStore,
        vector: VectorStore,
        graph: GraphStore | None,
    ):
        self.cfg = cfg
        self.embeddings = embeddings
        self.llm = llm
        self.lexical = lexical
        self.vector = vector
        self.graph = graph
        self.manifest_path = cfg.resolved_root() / "manifest.jsonl"

    # -- manifest ------------------------------------------------------------
    def _load_manifest(self) -> dict[str, dict]:
        return {row["path"]: row for row in load_jsonl(self.manifest_path)}

    def _save_manifest(self, manifest: dict[str, dict]) -> None:
        dump_jsonl(self.manifest_path, list(manifest.values()))

    # -- indexing ------------------------------------------------------------
    def index_documents(
        self,
        docs: list[Document],
        enrich: bool = False,
        with_graph: bool | None = None,
        force: bool = False,
    ) -> IngestStats:
        stats = IngestStats(files_seen=len(docs))
        with_graph = self.cfg.graph.enabled if with_graph is None else with_graph
        manifest = {} if force else self._load_manifest()
        manifest_dirty = False

        for doc in docs:
            try:
                content_hash = sha1(doc.text)
                prior = manifest.get(doc.source)
                if prior and prior.get("hash") == content_hash and not force:
                    stats.skipped += 1
                    continue

                chunks = chunk_document(doc, self.cfg.chunking)
                if not chunks:
                    stats.skipped += 1
                    continue

                if enrich:
                    enricher = Enricher(self.llm, self.cfg.index)
                    chunks = enricher.enrich(doc, chunks)

                vectors = self.embeddings.embed([c.embed_text for c in chunks])

                self.vector.delete_doc(doc.id)
                self.vector.add(
                    [
                        {
                            "id": c.id,
                            "doc_id": c.doc_id,
                            "ordinal": c.ordinal,
                            "title": c.metadata.get("title", ""),
                            "source": c.metadata.get("source", ""),
                            "text": c.text,
                            "context": c.context,
                            "meta": "{}",
                        }
                        for c in chunks
                    ],
                    vectors,
                )

                self.lexical.delete_doc(doc.id)
                self.lexical.add(
                    [
                        {
                            "id": doc.id,
                            "kind": "page",
                            "doc_id": doc.id,
                            "title": doc.title,
                            "body": doc.text[:100_000],
                            "source": doc.source,
                            "meta": "{}",
                        }
                    ]
                    + [
                        {
                            "id": c.id,
                            "kind": "chunk",
                            "doc_id": c.doc_id,
                            "title": c.metadata.get("title", ""),
                            "body": c.text,
                            "source": c.metadata.get("source", ""),
                            "meta": "{}",
                        }
                        for c in chunks
                    ]
                )

                if with_graph and self.graph is not None:
                    extractions = extract_chunks(
                        self.llm,
                        chunks,
                        cache=self.graph,
                        workers=self.cfg.graph.workers,
                        model_name=f"{self.llm.name}:{getattr(self.llm, 'model', '?')}",
                    )
                    for chunk, ext in zip(chunks, extractions, strict=True):
                        if ext.entities or ext.relationships:
                            self.graph.upsert_extraction(ext, chunk.id, doc.id)
                            stats.entities += len(ext.entities)
                            stats.relations += len(ext.relationships)

                manifest[doc.source] = {
                    "path": doc.source,
                    "doc_id": doc.id,
                    "hash": content_hash,
                    "chunks": len(chunks),
                    "ts": now_iso(),
                }
                manifest_dirty = True
                stats.indexed += 1
                stats.chunks += len(chunks)
            except ProviderError:
                raise
            except Exception as e:
                log.exception("failed indexing %s", doc.source)
                stats.failed += 1
                stats.errors.append(f"{doc.source}: {e}")

        if manifest_dirty:
            self._save_manifest(manifest)

        if with_graph and self.graph is not None and stats.indexed:
            try:
                communities = build_communities(self.graph, self.llm)
                stats.communities = len(communities)
            except Exception as e:
                log.warning("community build failed: %s", e)
        return stats

    def ingest_paths(
        self,
        paths: list[str | Path],
        recursive: bool = False,
        enrich: bool = False,
        with_graph: bool | None = None,
        force: bool = False,
    ) -> IngestStats:
        files = collect_files(paths, recursive=recursive)
        docs: list[Document] = []
        failed = 0
        for f in files:
            try:
                docs.append(parse_file(f))
            except Exception as e:
                log.warning("parse failed %s: %s", f, e)
                failed += 1
        stats = self.index_documents(docs, enrich=enrich, with_graph=with_graph, force=force)
        stats.files_seen += failed
        stats.failed += failed
        return stats

    def ingest_url(
        self,
        url: str,
        depth: int = 1,
        max_pages: int = 10,
        enrich: bool = False,
        with_graph: bool | None = None,
    ) -> IngestStats:
        docs = _crawl(url, depth=depth, max_pages=max_pages)
        return self.index_documents(docs, enrich=enrich, with_graph=with_graph)

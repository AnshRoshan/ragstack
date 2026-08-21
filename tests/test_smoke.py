"""Smoke tests: pure-unit, no network/LLM needed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragstack.config import AppConfig
from ragstack.ingestion.chunker import chunk_document
from ragstack.retrieval.vector_rag import rrf
from ragstack.stores.graph.sqlite_graph import SQLiteGraphStore
from ragstack.types import (
    ChunkExtraction,
    Document,
    ExtractedEntity,
    ExtractedRelation,
    RetrievedItem,
)


def _doc(text: str) -> Document:
    return Document(id="dtest", source="test.md", title="Test Doc", text=text)


class TestChunker:
    def test_splits_long_doc(self):
        cfg = AppConfig().chunking
        cfg.size = 120
        cfg.overlap = 20
        text = "# Intro\n" + ("Alpha beta gamma delta. " * 40) + "\n\n# Details\n" + ("Epsilon zeta eta theta. " * 40)
        chunks = chunk_document(_doc(text), cfg)
        assert len(chunks) >= 3
        assert all(c.n_tokens <= cfg.size + 60 for c in chunks)

    def test_heading_paths(self):
        cfg = AppConfig().chunking
        text = "# A\npara one\n\n## B\npara two"
        chunks = chunk_document(_doc(text), cfg)
        headings = [c.metadata["heading"] for c in chunks]
        assert "A" in headings and "A > B" in headings
        assert len(chunks) == 2

    def test_code_fence_kept_whole(self):
        cfg = AppConfig().chunking
        text = "before\n```python\nprint('x')\nprint('y')\n```\nafter"
        chunks = chunk_document(_doc(text), cfg)
        joined = "\n".join(c.text for c in chunks)
        assert "print('x')" in joined


class TestRRF:
    def test_fusion_prefers_consensus(self):
        a = [RetrievedItem(chunk_id="x"), RetrievedItem(chunk_id="y")]
        b = [RetrievedItem(chunk_id="z"), RetrievedItem(chunk_id="x")]
        fused = rrf([a, b])
        assert fused[0].chunk_id == "x"


class TestSQLiteGraph:
    def test_roundtrip_and_neighborhood(self, tmp_path: Path):
        store = SQLiteGraphStore(tmp_path)
        ext = ChunkExtraction(
            entities=[
                ExtractedEntity(name="Alice", type="person", description="An engineer"),
                ExtractedEntity(name="Acme", type="organization", description="Her employer"),
                ExtractedEntity(name="RAGStack", type="product", description="A RAG system"),
            ],
            relationships=[
                ExtractedRelation(source="Alice", target="Acme", relation="works_at"),
                ExtractedRelation(source="Acme", target="RAGStack", relation="develops"),
            ],
        )
        store.upsert_extraction(ext, "chunk1", "doc1")
        stats = store.stats()
        assert stats["entities"] == 3
        assert stats["relations"] == 2

        nb = store.neighborhood("alice")
        names = {e["name"] for e in nb["entities"]}
        assert "Alice" in names
        assert "Acme" in names  # 1 hop away
        assert "chunk1" in nb["chunk_ids"]

        # dedup on re-upsert
        store.upsert_extraction(ext, "chunk2", "doc1")
        assert store.stats()["entities"] == 3
        assert store.stats()["relations"] == 4  # new evidence rows appended

    def test_cache(self, tmp_path: Path):
        store = SQLiteGraphStore(tmp_path)
        assert store.cache_get("k") is None
        store.cache_put("k", "v")
        assert store.cache_get("k") == "v"


class TestLexicalStore:
    def test_index_and_search(self, tmp_path: Path):
        from ragstack.stores.lexical import LexicalStore

        store = LexicalStore(tmp_path)
        store.add(
            [
                {"id": "p1", "kind": "page", "doc_id": "d1", "title": "Kubernetes Guide", "body": "Kubernetes orchestrates containers across clusters.", "source": "k8s.md"},
                {"id": "c1", "kind": "chunk", "doc_id": "d1", "title": "Kubernetes Guide", "body": "Pods are the smallest deployable units in Kubernetes.", "source": "k8s.md"},
                {"id": "c2", "kind": "chunk", "doc_id": "d2", "title": "Cooking", "body": "Boil pasta water with salt.", "source": "cook.md"},
            ]
        )
        assert store.count() == 3
        hits = store.search("kubernetes pods", kind="chunk", top_k=2)
        assert hits and hits[0]["id"] == "c1"
        pages = store.search("containers", kind="page", top_k=5)
        assert pages and pages[0]["id"] == "p1"
        by_id = store.get_by_ids(["c1"])
        assert by_id and by_id[0]["text"].startswith("Pods")

    def test_delete_doc(self, tmp_path: Path):
        from ragstack.stores.lexical import LexicalStore

        store = LexicalStore(tmp_path)
        store.add(
            [
                {"id": "a", "kind": "chunk", "doc_id": "d1", "title": "t", "body": "unique words here", "source": "s"},
                {"id": "b", "kind": "chunk", "doc_id": "d2", "title": "t", "body": "other stuff entirely", "source": "s"},
            ]
        )
        store.delete_doc("d1")
        hits = store.search("unique words")
        assert not hits


class TestVectorStore:
    def test_add_search_delete(self, tmp_path: Path):
        import numpy as np

        from ragstack.stores.vector import VectorStore

        store = VectorStore(tmp_path)
        dim = 8
        vecs = np.eye(dim, dtype=np.float32)
        rows = [
            {"id": f"c{i}", "doc_id": "d1", "ordinal": i, "title": "t", "source": "s", "text": f"text {i}", "context": "", "meta": "{}"}
            for i in range(dim)
        ]
        store.add(rows, vecs)
        assert store.count() == dim
        res = store.search(vecs[3], top_k=2)
        assert res[0]["id"] == "c3"
        got = store.get_by_ids(["c1", "c2"])
        assert len(got) == 2
        store.delete_doc("d1")
        assert store.count() == 0


class TestConfig:
    def test_local_mode_resolution(self):
        cfg = AppConfig(mode="local").resolve_providers()
        assert cfg.embedding.provider == "sentence-transformers"
        assert cfg.llm.provider == "ollama"
        assert cfg.rerank.provider == "cross-encoder"

    def test_invalid_mode(self):
        from ragstack.errors import ConfigError

        with pytest.raises(ConfigError):
            AppConfig(mode="quantum").resolve_providers()

    def test_cloud_requires_key(self, monkeypatch):
        from ragstack.errors import ConfigError

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ConfigError):
            AppConfig(mode="cloud").resolve_providers()


class TestSQLCatalog:
    def test_readonly_guard(self, tmp_path: Path):
        from ragstack.errors import ToolError
        from ragstack.stores.sql_catalog import SQLCatalog

        db = tmp_path / "t.db"
        import sqlite3

        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE items(id INTEGER, name TEXT)")
        conn.execute("INSERT INTO items VALUES(1, 'widget')")
        conn.commit()
        conn.close()

        cat = SQLCatalog({"test": f"sqlite:///{db}"})
        out = cat.query("test", "SELECT * FROM items")
        assert "widget" in out
        with pytest.raises(ToolError):
            cat.query("test", "DELETE FROM items")
        with pytest.raises(ToolError):
            cat.query("test", "UPDATE items SET name='x'")

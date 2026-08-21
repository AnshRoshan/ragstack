"""GraphRAG extraction/communities/search tests with fakes."""

from __future__ import annotations

import json

from conftest import FakeLLM

from ragstack.graphrag.communities import build_communities
from ragstack.graphrag.extract import _normalize, _parse_json_loose, extract_chunks
from ragstack.graphrag.search import global_search, local_search
from ragstack.stores.graph.sqlite_graph import SQLiteGraphStore
from ragstack.stores.vector import VectorStore
from ragstack.types import Chunk


def _chunk(text: str, ordinal: int = 0) -> Chunk:
    return Chunk(id=f"c{ordinal}", doc_id="d1", ordinal=ordinal, text=text, metadata={"title": "T"})


class TestExtractionParsing:
    def test_parse_json_loose_handles_fences(self):
        raw = '```json\n{"entities": [], "relationships": []}\n```'
        assert _parse_json_loose(raw) == {"entities": [], "relationships": []}

    def test_parse_json_loose_with_prose(self):
        raw = 'Sure! Here it is: {"entities": [{"name": "X"}]} hope that helps'
        assert _parse_json_loose(raw)["entities"][0]["name"] == "X"

    def test_normalize_filters_bad_relations(self):
        data = {
            "entities": [
                {"name": "Alice", "type": "PERSON", "description": "d"},
                {"name": "Bob", "type": "unknown-type", "description": ""},
                {"name": "", "type": "person", "description": "skipped"},
            ],
            "relationships": [
                {"source": "Alice", "target": "Bob", "relation": "knows", "description": ""},
                {"source": "Ghost", "target": "Alice", "relation": "bad", "description": ""},
                {"source": "Alice", "target": "Alice", "relation": "self", "description": ""},
            ],
        }
        ext = _normalize(data)
        assert [e.name for e in ext.entities] == ["Alice", "Bob"]
        assert ext.entities[0].type == "person"
        assert ext.entities[1].type == "other"
        assert len(ext.relationships) == 1  # only Alice->Bob survives

    def test_extract_chunks_with_fake_llm_and_cache(self, tmp_path):
        payload = json.dumps(
            {
                "entities": [{"name": "Acme", "type": "organization", "description": "corp"}],
                "relationships": [],
            }
        )
        llm = FakeLLM([{"content": payload}, {"content": payload}])
        store = SQLiteGraphStore(tmp_path)
        chunks = [_chunk("Acme makes widgets"), _chunk("Acme is based in Delaware")]
        results = extract_chunks(llm, chunks, cache=store, workers=1, model_name="fake")
        assert all(r.entities for r in results)
        assert len(llm.calls) == 2

        # second run: everything served from cache -> no LLM calls
        llm2 = FakeLLM([])
        results2 = extract_chunks(llm2, chunks, cache=store, workers=1, model_name="fake")
        assert all(r.entities for r in results2)
        assert len(llm2.calls) == 0


class TestCommunities:
    def test_build_with_fake_llm(self, tmp_path):
        store = SQLiteGraphStore(tmp_path)
        from ragstack.types import ChunkExtraction, ExtractedEntity, ExtractedRelation

        for i in range(6):
            ext = ChunkExtraction(
                entities=[
                    ExtractedEntity(name=f"Person{i}", type="person", description="worker"),
                    ExtractedEntity(name="TheFirm", type="organization", description="employer"),
                ],
                relationships=[
                    ExtractedRelation(source=f"Person{i}", target="TheFirm", relation="works_at")
                ],
            )
            store.upsert_extraction(ext, f"ch{i}", "d1")

        llm = FakeLLM(
            [{"content": json.dumps({"summary": "People working at TheFirm.", "keywords": ["firm"]})}]
        )
        communities = build_communities(store, llm, min_size=3)
        assert len(communities) >= 1
        assert communities[0]["summary"]
        stored = store.community_summaries()
        assert stored and stored[0]["summary"].startswith("People")


class TestGraphSearch:
    def test_local_search_returns_context_and_items(self, tmp_path):
        store = SQLiteGraphStore(tmp_path)
        from ragstack.types import ChunkExtraction, ExtractedEntity, ExtractedRelation

        ext = ChunkExtraction(
            entities=[
                ExtractedEntity(name="Alice", type="person", description="engineer"),
                ExtractedEntity(name="Acme", type="organization", description="company"),
            ],
            relationships=[ExtractedRelation(source="Alice", target="Acme", relation="works_at")],
        )
        store.upsert_extraction(ext, "chunk-1", "doc-1")

        vector = VectorStore(tmp_path / "vec")
        import numpy as np

        vector.add(
            [{"id": "chunk-1", "doc_id": "doc-1", "ordinal": 0, "title": "T", "source": "s.md", "text": "Alice works at Acme.", "context": "", "meta": "{}"}],
            np.eye(64, dtype=np.float32)[:1],
        )

        context, items = local_search(store, vector, "alice", top_k=5, max_hops=2)
        assert "ENTITY" in context and "works_at" in context
        assert len(items) == 1 and items[0].chunk_id == "chunk-1"

    def test_global_search_maps_over_communities(self, tmp_path):
        store = SQLiteGraphStore(tmp_path)
        store.save_communities(
            [
                {"member_ids": ["e1"], "summary": "All about kubernetes deployments.", "keywords": ["kubernetes"]},
                {"member_ids": ["e2"], "summary": "Cooking pasta dishes.", "keywords": ["pasta"]},
            ]
        )
        llm = FakeLLM([{"content": "Kubernetes theme dominates."}])
        context, items = global_search(store, llm, "what about kubernetes?", top_k=2)
        assert "Kubernetes" in context or "kubernetes" in context
        assert len(items) == 2

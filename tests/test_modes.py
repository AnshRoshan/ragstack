"""Tests for retrieval-mode dispatch: direct single-pass pipelines vs agentic loop."""

from __future__ import annotations

from pathlib import Path

from conftest import FakeEmbeddings, FakeLLM

from ragstack.service import ALL_MODES, DIRECT_MODES, MODE_CATALOG, RAGStack


def _svc(tmp_path: Path) -> RAGStack:
    from ragstack.config import AppConfig

    cfg = AppConfig(mode="local")
    cfg.index.root = str(tmp_path / "idx")
    cfg.graph.enabled = True
    cfg.graph.backend = "sqlite"
    svc = RAGStack(cfg)
    svc._embeddings = FakeEmbeddings()
    svc._llm = FakeLLM([{"content": "Synthesized answer [S1]."}])

    doc = tmp_path / "k8s.md"
    doc.write_text(
        "# Kubernetes\n\nKubernetes orchestrates containers across a cluster of machines. " * 15,
        encoding="utf-8",
    )
    svc.ingest([doc], with_graph=False)
    return svc


class TestModeCatalog:
    def test_catalog_shape(self):
        assert len(MODE_CATALOG) == 8
        ids = {m["id"] for m in MODE_CATALOG}
        assert ids == ALL_MODES == (DIRECT_MODES | {"auto", "agentic", "sql"})
        for m in MODE_CATALOG:
            assert m["description"]
            assert m.get("tools") or m.get("pipeline")

    def test_direct_and_agentic_partition(self):
        assert DIRECT_MODES == {"vector", "lexical", "hybrid", "graph", "global"}
        assert {"auto", "agentic", "sql"} - DIRECT_MODES == {"auto", "agentic", "sql"}


class TestDirectPipelines:
    def test_vector_mode_single_llm_call(self, tmp_path):
        svc = _svc(tmp_path)
        answer = svc.query("what orchestrates containers?", mode="vector")
        assert "[S1]" in answer.text or "Synthesized" in answer.text
        assert len(svc._llm.calls) == 1  # exactly one synthesis call — no agent loop
        assert answer.citations

    def test_lexical_mode(self, tmp_path):
        svc = _svc(tmp_path)
        answer = svc.query("kubernetes containers cluster", mode="lexical")
        assert answer.text
        assert len(svc._llm.calls) == 1

    def test_hybrid_mode(self, tmp_path):
        svc = _svc(tmp_path)
        answer = svc.query("container orchestration engine", mode="hybrid")
        assert answer.text and len(svc._llm.calls) == 1

    def test_graph_mode_empty_index_still_answers(self, tmp_path):
        svc = _svc(tmp_path)  # graph disabled during ingest -> empty graph
        answer = svc.query("anything", mode="graph")
        # no entities -> no items; LLM still asked with "(nothing retrieved)"
        assert len(svc._llm.calls) == 1

    def test_stream_direct_emits_retrieval_event(self, tmp_path):
        svc = _svc(tmp_path)
        events = list(svc.stream_query("containers?", mode="vector", use_cache=False))
        types = [e["type"] for e in events]
        assert "retrieval" in types
        assert types[0] == "start" and types[-1] == "done"
        done = events[-1]
        assert done["steps"][0]["tool"].startswith("vector")

    def test_unknown_mode_errors_cleanly(self, tmp_path):
        svc = _svc(tmp_path)
        events = list(svc.stream_query("q", mode="telepathy"))
        assert events[0]["type"] == "error"


class TestAgenticDispatch:
    def test_agentic_mode_uses_tool_loop(self, tmp_path):
        svc = _svc(tmp_path)
        svc._llm = FakeLLM(
            [
                {"tool_calls": [{"name": "search_chunks", "args": {"query": "kubernetes"}}]},
                {"content": "Final [S1]."},
            ]
        )
        answer = svc.query("what runs containers?", mode="agentic")
        assert answer.text == "Final [S1]."
        assert any(step["tool"] == "search_chunks" for step in answer.steps)

    def test_auto_mode_is_agentic(self, tmp_path):
        svc = _svc(tmp_path)
        svc._llm = FakeLLM([{"content": "Direct answer."}])
        events = list(svc.stream_query("q", mode="auto"))
        start = events[0]
        assert start["type"] == "start" and len(start["tools"]) == 9


class TestModesEndpoint:
    def test_api_modes(self, tmp_path):
        from fastapi.testclient import TestClient

        from ragstack.web.app import create_app

        app = create_app(_svc(tmp_path))
        client = TestClient(app)
        r = client.get("/api/modes")
        assert r.status_code == 200
        modes = r.json()
        assert {m["id"] for m in modes} == ALL_MODES
        agentic = [m for m in modes if m["kind"] == "agentic"]
        assert all(len(m.get("tools", [])) >= 2 for m in agentic)

    def test_status_includes_modes(self, tmp_path):
        from fastapi.testclient import TestClient

        from ragstack.web.app import create_app

        client = TestClient(create_app(_svc(tmp_path)))
        body = client.get("/api/status").json()
        assert set(body["modes"]) == ALL_MODES


class TestCacheAcrossModes:
    def test_direct_mode_cache_roundtrip(self, tmp_path):
        svc = _svc(tmp_path)
        first = list(svc.stream_query("kubernetes?", mode="hybrid"))
        assert not first[0].get("cached")
        second = list(svc.stream_query("kubernetes?", mode="hybrid"))
        assert second[0].get("cached") is True
        assert second[-1]["answer"] == first[-1]["answer"]

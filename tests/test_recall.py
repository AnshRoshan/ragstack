"""Cross-session recall store, recall tool, and server auth tests."""

from __future__ import annotations

import json

from conftest import FakeEmbeddings, FakeLLM

from ragstack.memory import RecallStore
from ragstack.service import RAGStack


class TestRecallStore:
    def test_add_search_roundtrip(self, tmp_path):
        store = RecallStore(tmp_path / "idx", FakeEmbeddings())
        assert store.count() == 0
        store.add("s1", "what database does the vector index use?", "LanceDB stores the vectors.")
        store.add("s2", "how do I reset the graph?", "Run `ragstack reset -y`.")
        assert store.count() == 2
        hits = store.search("which vector database?", top_k=2)
        assert hits and hits[0]["answer"].startswith("LanceDB")
        assert hits[0]["similarity"] > hits[-1]["similarity"] or len(hits) == 1

    def test_clear(self, tmp_path):
        store = RecallStore(tmp_path / "idx", FakeEmbeddings())
        store.add("s", "q", "a")
        store.clear()
        assert store.count() == 0


class TestRecallTool:
    def test_executor_returns_hits(self, app_config, tmp_path):
        from ragstack.agent.tools import ToolContext, make_executor

        svc = RAGStack(app_config)
        svc._embeddings = FakeEmbeddings()
        doc = tmp_path / "d.md"
        doc.write_text("Content. " * 50, encoding="utf-8")
        svc.ingest([doc], with_graph=False)
        svc.recall.add("s9", "what did we decide about caching?", "We use a semantic cache at 0.95.")

        ctx = ToolContext(
            embeddings=svc.embeddings, vector_store=svc.vector,
            lexical_store=svc.lexical, recall_store=svc.recall,
        )
        out = make_executor(ctx)("recall_memory", {"query": "caching decision"})
        data = json.loads(out)
        assert isinstance(data, list) and "semantic cache" in data[0]["answer"]

    def test_disabled_returns_error(self, app_config):
        from ragstack.agent.tools import ToolContext, make_executor

        app_config.agent.recall_enabled = False
        svc = RAGStack(app_config)
        ctx = ToolContext(embeddings=svc.embeddings, vector_store=svc.vector,
                          lexical_store=svc.lexical, recall_store=None)
        out = make_executor(ctx)("recall_memory", {"query": "x"})
        assert "error" in json.loads(out)


class TestServiceRecallWiring:
    def test_answered_query_is_recallable(self, app_config, tmp_path):
        svc = RAGStack(app_config)
        svc._embeddings = FakeEmbeddings()
        svc._llm = FakeLLM([{"content": "LanceDB stores the vectors."}])
        doc = tmp_path / "d.md"
        doc.write_text("Vector content. " * 40, encoding="utf-8")
        svc.ingest([doc], with_graph=False)

        list(svc.stream_query("what stores vectors?", mode="vector", use_cache=False, session_id="s1"))
        assert svc.recall.count() == 1
        hits = svc.recall.search("vector storage engine?")
        assert hits and "LanceDB" in hits[0]["answer"]

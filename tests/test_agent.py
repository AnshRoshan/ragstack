"""Agent runner + tool executor tests with scripted LLM."""

from __future__ import annotations

import json

from conftest import FakeEmbeddings, FakeLLM

from ragstack.agent.runner import AgentRunner
from ragstack.agent.tools import ToolContext, make_executor, tools_for_mode
from ragstack.providers.llm import LLMProvider
from ragstack.stores.lexical import LexicalStore
from ragstack.stores.vector import VectorStore


def _seed_stores(tmp_path):
    lexical = LexicalStore(tmp_path / "lex")
    vector = VectorStore(tmp_path / "vec")

    emb = FakeEmbeddings()
    texts = [
        "Kubernetes orchestrates containers across a cluster of machines.",
        "RAGStack uses reciprocal rank fusion to merge dense and sparse results.",
        "Pasta water should be boiled with salt before adding noodles.",
    ]
    rows = []
    vecs = emb.embed(texts)
    for i, t in enumerate(texts):
        rows.append(
            {"id": f"c{i}", "doc_id": f"d{i}", "ordinal": 0, "title": f"T{i}", "source": f"s{i}.md", "text": t, "context": "", "meta": "{}"}
        )
        lexical.add(
            [{"id": f"c{i}", "kind": "chunk", "doc_id": f"d{i}", "title": f"T{i}", "body": t, "source": f"s{i}.md"}]
        )
    vector.add(rows, vecs)
    return emb, lexical, vector


class TestToolExecutor:
    def test_search_tools_return_refs(self, tmp_path):
        emb, lexical, vector = _seed_stores(tmp_path)
        ctx = ToolContext(embeddings=emb, vector_store=vector, lexical_store=lexical, top_k=3)
        execute = make_executor(ctx)

        out = execute("search_chunks", {"query": "reciprocal rank fusion"})
        data = json.loads(out)
        assert isinstance(data, list) and data[0]["ref"] == "S1"
        assert "fusion" in data[0]["text"].lower()
        assert any(c.ref_id == "S1" for c in ctx.citations.values())

        out = execute("semantic_search", {"query": "containers orchestration"})
        data = json.loads(out)
        assert "kubernetes" in data[0]["text"].lower()

        out = execute("hybrid_search", {"query": "rank fusion retrieval"})
        assert json.loads(out)

    def test_unknown_tool_returns_error_json(self, tmp_path):
        emb, lexical, vector = _seed_stores(tmp_path)
        ctx = ToolContext(embeddings=emb, vector_store=vector, lexical_store=lexical)
        out = make_executor(ctx)("nope", {})
        assert "error" in json.loads(out)

    def test_citation_dedup_same_chunk(self, tmp_path):
        emb, lexical, vector = _seed_stores(tmp_path)
        ctx = ToolContext(embeddings=emb, vector_store=vector, lexical_store=lexical)
        execute = make_executor(ctx)
        execute("search_chunks", {"query": "reciprocal rank fusion"})
        execute("search_chunks", {"query": "reciprocal rank fusion"})
        assert len(ctx.citations) == 1


class TestAgentRunner:
    def _ctx(self, tmp_path, llm):
        emb, lexical, vector = _seed_stores(tmp_path)
        return ToolContext(
            embeddings=emb, vector_store=vector, lexical_store=lexical,
            llm=llm, max_steps=4,
        )

    def test_single_hop_tool_then_answer(self, tmp_path):
        llm = FakeLLM(
            [
                {"tool_calls": [{"name": "search_chunks", "args": {"query": "rank fusion"}}]},
                {"content": "Fusion uses RRF [S1]."},
            ]
        )
        events = list(AgentRunner(llm, self._ctx(tmp_path, llm)).run("q", stream=False))
        done = events[-1]
        assert done["type"] == "done"
        assert done["answer"] == "Fusion uses RRF [S1]."
        assert len(done["steps"]) == 1
        assert done["steps"][0]["tool"] == "search_chunks"
        assert done["citations"][0]["ref_id"] == "S1"
        types = [e["type"] for e in events]
        assert "tool_start" in types and "tool_end" in types

    def test_streaming_events_flow(self, tmp_path):
        llm = FakeLLM(
            [
                {"tool_calls": [{"name": "search_pages", "args": {"query": "kubernetes"}}]},
                {"content": "Answer here."},
            ]
        )
        events = list(AgentRunner(llm, self._ctx(tmp_path, llm)).run("q", stream=True))
        kinds = [e["type"] for e in events]
        assert kinds[0] == "start"
        assert "thought" in kinds
        assert kinds[-1] == "done"

    def test_step_budget_forces_stop(self, tmp_path):
        script = [{"tool_calls": [{"name": "search_chunks", "args": {"query": "x"}}]}] * 20
        llm = FakeLLM(script)
        ctx = self._ctx(tmp_path, llm)
        ctx.max_steps = 2
        events = list(AgentRunner(llm, ctx).run("q", stream=False))
        done = events[-1]
        # final step runs WITHOUT tools -> model must answer (fake returns 'done')
        assert done["type"] == "done"

    def test_provider_error_surfaces_as_event(self, tmp_path):
        class Boom(LLMProvider):
            name = "boom"

            def chat(self, *a, **k):
                raise RuntimeError("kaput")

            def stream(self, *a, **k):
                raise RuntimeError("kaput")

        events = list(AgentRunner(Boom(), self._ctx(tmp_path, Boom())).run("q", stream=False))
        assert any(e["type"] == "error" and "kaput" in e["message"] for e in events)


class TestModeRouting:
    def test_modes_restrict_tools(self):
        assert {t["function"]["name"] for t in tools_for_mode("sql")} == {"sql_query", "search_chunks"}
        assert "graph_search" in [t["function"]["name"] for t in tools_for_mode("graph")]
        all_names = {t["function"]["name"] for t in tools_for_mode("auto")}
        assert len(all_names) == 11

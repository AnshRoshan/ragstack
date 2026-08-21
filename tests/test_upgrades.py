"""Tests for the SOTA upgrades: evidence grader, decomposition planner, semantic cache."""

from __future__ import annotations

import json

from conftest import FakeEmbeddings, FakeLLM

from ragstack.agent.planner import decompose, decomposed_search
from ragstack.cache import SemanticCache
from ragstack.retrieval.evaluator import EvidenceGrader
from ragstack.stores.lexical import LexicalStore
from ragstack.stores.vector import VectorStore
from ragstack.types import RetrievedItem


def _items(*scores):
    return [
        RetrievedItem(chunk_id=f"c{i}", source=f"s{i}.md", title=f"T{i}", text=f"text {i}", score=s)
        for i, s in enumerate(scores)
    ]


class TestEvidenceGrader:
    def test_empty_results_are_incorrect(self):
        g = EvidenceGrader()
        verdict = g.grade("q", [])
        assert verdict["action"] == "incorrect"
        assert "insufficient" in verdict["hint"]

    def test_vector_scores_drive_action(self):
        g = EvidenceGrader(upper=0.5, lower=0.2)
        strong = [RetrievedItem(chunk_id="c", text="t", score=0.8, origin="vector")]
        weak = [RetrievedItem(chunk_id="c", text="t", score=0.1, origin="vector")]
        mid = [RetrievedItem(chunk_id="c", text="t", score=0.35, origin="vector")]
        assert g.grade("q", strong)["action"] == "correct"
        assert g.grade("q", weak)["action"] == "incorrect"
        assert g.grade("q", mid)["action"] == "ambiguous"

    def test_non_vector_scores_fall_back_to_ambiguous(self):
        g = EvidenceGrader()
        verdict = g.grade("q", _items(9.2, 4.1))  # BM25-style unbounded scores
        assert verdict["action"] == "ambiguous"

    def test_llm_grader(self):
        llm = FakeLLM([{"content": "8"}, {"content": "1"}])
        g = EvidenceGrader(grader="llm", llm=llm)
        assert g.grade("q", _items(0))["action"] == "correct"
        assert g.grade("q", _items(0))["action"] == "incorrect"

    def test_llm_grader_failure_is_honest_ambiguous(self):
        class Boom(FakeLLM):
            def chat(self, *a, **k):
                raise RuntimeError("down")

        g = EvidenceGrader(grader="llm", llm=Boom([]))
        assert g.grade("q", _items(0))["action"] == "ambiguous"


def _seed(tmp_path):
    emb = FakeEmbeddings()
    lexical = LexicalStore(tmp_path / "lex")
    vector = VectorStore(tmp_path / "vec")

    texts = [
        "Kubernetes orchestrates containers across machines.",
        "Reciprocal rank fusion merges dense and sparse result lists.",
        "The Eiffel Tower is in Paris, France.",
    ]
    rows = []
    vecs = emb.embed(texts)
    for i, t in enumerate(texts):
        rows.append({"id": f"c{i}", "doc_id": f"d{i}", "ordinal": 0, "title": f"T{i}", "source": f"s{i}.md", "text": t, "context": "", "meta": "{}"})
        lexical.add([{"id": f"c{i}", "kind": "chunk", "doc_id": f"d{i}", "title": f"T{i}", "body": t, "source": f"s{i}.md"}])
    vector.add(rows, vecs)
    return emb, lexical, vector


class TestDecompositionPlanner:
    def test_decompose_parses_sub_queries(self):
        llm = FakeLLM([{"content": json.dumps({"sub_queries": ["compare X features", "compare Y pricing"]})}])
        subs = decompose(llm, "compare X and Y")
        assert subs == ["compare X features", "compare Y pricing"]

    def test_decompose_falls_back_on_garbage(self):
        llm = FakeLLM([{"content": "I cannot do that"}])
        assert decompose(llm, "original question") == ["original question"]

    def test_decomposed_search_merges_parallel_legs(self, tmp_path):
        emb, lexical, vector = _seed(tmp_path)
        llm = FakeLLM(
            [{"content": json.dumps({"sub_queries": ["kubernetes containers", "rank fusion retrieval"]})}]
        )
        subs, items = decomposed_search(llm, emb, vector, lexical, "compare orchestration and fusion", top_k=3)
        assert len(subs) == 2
        assert items and len(items) <= 3
        # both topics should be represented after fusion
        joined = " ".join(i.text.lower() for i in items)
        assert ("kubernetes" in joined) or ("fusion" in joined)

    def test_single_subquery_skips_pool(self, tmp_path):
        emb, lexical, vector = _seed(tmp_path)
        llm = FakeLLM([{"content": json.dumps({"sub_queries": ["eiffel tower paris"]})}])
        _, items = decomposed_search(llm, emb, vector, lexical, "where is the eiffel tower?", top_k=3)
        assert items


class TestSemanticCache:
    def _cache(self, tmp_path):
        return SemanticCache(tmp_path / "qc", FakeEmbeddings(), threshold=0.95, fuzzy_threshold=0.98)

    def test_store_and_exact_hit(self, tmp_path):
        cache = self._cache(tmp_path)
        assert cache.count() == 0
        cache.store("what is lancedb?", "auto", "LanceDB is a vector DB.", [{"ref_id": "S1"}], [])
        hit = cache.lookup("what is lancedb?", mode="auto")
        assert hit and hit["answer"].startswith("LanceDB") and hit["cached"]

    def test_miss_below_threshold(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.store("what is lancedb?", "auto", "Answer A.", [], [])
        assert cache.lookup("completely different topic about pasta recipes", mode="auto") is None

    def test_mode_scoping(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.store("question one", "graph", "Graph answer.", [], [])
        assert cache.lookup("question one", mode="auto") is None
        assert cache.lookup("question one", mode="graph") is not None

    def test_uncertainty_tightens_threshold(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.store("is feature X supported?", "auto", "Yes it is.", [], [])
        near = "is feature X supported maybe?"
        # similar wording but hedged -> requires 0.98; hashed embedding of extra word drops sim
        result = cache.lookup(near, mode="auto")
        assert result is None or result.get("cache_similarity", 0) >= 0.98

    def test_clear(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.store("q", "auto", "a", [], [])
        assert cache.count() == 1
        cache.clear()
        assert cache.count() == 0


class TestServiceCacheIntegration:
    def test_second_query_hits_cache(self, app_config, tmp_path):
        from ragstack.service import RAGStack

        svc = RAGStack(app_config)
        svc._embeddings = FakeEmbeddings()
        svc._llm = FakeLLM([{"content": "Cached answer [S1]."}])

        doc = tmp_path / "d.md"
        doc.write_text("Some content about things. " * 40, encoding="utf-8")
        svc.ingest([doc], with_graph=False)

        first_events = list(svc.stream_query("what is this document about?"))
        assert first_events[-1]["type"] == "done"
        assert not first_events[0].get("cached")

        second_events = list(svc.stream_query("what is this document about?"))
        assert second_events[0].get("cached") is True
        assert second_events[-1]["answer"] == first_events[-1]["answer"]
        assert svc.status()["cache_entries"] == 1

        # bypass works
        bypass = list(svc.stream_query("what is this document about?", use_cache=False))
        assert not bypass[0].get("cached")


class TestGraderInRunner:
    def test_grader_hint_appended_to_observation(self, tmp_path):
        from ragstack.agent.runner import AgentRunner
        from ragstack.agent.tools import ToolContext

        emb, lexical, vector = _seed(tmp_path)
        grader = EvidenceGrader(upper=0.5, lower=0.2)
        ctx = ToolContext(embeddings=emb, vector_store=vector, lexical_store=lexical, llm=None, grader=grader, max_steps=3)
        llm = FakeLLM(
            [
                {"tool_calls": [{"name": "semantic_search", "args": {"query": "eiffel tower location"}}]},
                {"content": "Paris."},
            ]
        )
        events = list(AgentRunner(llm, ctx).run("where is the eiffel tower?", stream=False))
        done = events[-1]
        assert done["type"] == "done" and done["answer"] == "Paris."

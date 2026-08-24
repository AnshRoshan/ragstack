"""Frontier pass: adaptive routing, verification/abstention, traces, contradiction tool."""

from __future__ import annotations

import json

from conftest import FakeEmbeddings, FakeLLM

from ragstack.agent.query_intel import _heuristic, classify, resolve_route
from ragstack.agent.verifier import maybe_abstain
from ragstack.service import RAGStack


def _cfg(app_config, **flags):
    app_config.agent.query_intelligence = flags.get("intel", True)
    app_config.generation.verify_answers = flags.get("verify", True)
    app_config.generation.abstain_threshold = flags.get("abstain_threshold", 0.5)
    return app_config


class TestHeuristicRouting:
    def test_compare_routes_agentic(self):
        assert _heuristic("compare the 2024 and 2026 agreements", False)["suggested_mode"] == "agentic"

    def test_aggregate_routes_sql_only_with_databases(self):
        assert _heuristic("how many invoices total?", True)["suggested_mode"] == "sql"
        assert _heuristic("how many invoices total?", False)["suggested_mode"] != "sql"

    def test_exact_identifier_routes_lexical(self):
        out = _heuristic("what does error CODE-4212 mean?", False)
        assert out["suggested_mode"] == "lexical"

    def test_resolve_route_user_wins(self):
        mode, k = resolve_route({"suggested_mode": "graph", "top_k": 12}, "vector", 8)
        assert (mode, k) == ("vector", 8)

    def test_classify_llm_parse(self):
        llm = FakeLLM([{"content": json.dumps({
            "intent": "research", "complexity": "broad", "ambiguity": "none",
            "needs_clarification": False, "clarifying_question": None,
            "suggested_mode": "agentic", "top_k": 12})}])
        out = classify(llm, "big research question")
        assert out["suggested_mode"] == "agentic" and out["top_k"] == 12

    def test_classify_falls_back_on_garbage(self):
        llm = FakeLLM([{"content": "cannot classify"}])
        out = classify(llm, "simple question?")
        assert out["suggested_mode"] in ("hybrid", "lexical", "vector", "agentic")


class TestAdaptiveServiceFlow:
    def test_route_event_emitted_and_mode_switched(self, tmp_path):
        from ragstack.config import AppConfig

        cfg = AppConfig(mode="local")
        cfg.index.root = str(tmp_path / "idx")
        cfg.graph.enabled = False
        cfg.agent.query_intelligence = True
        svc = RAGStack(cfg)
        svc._embeddings = FakeEmbeddings()
        # classifier says lexical; synthesis answers
        svc._llm = FakeLLM([
            {"content": json.dumps({"intent": "factual", "complexity": "simple",
                                    "ambiguity": "none", "needs_clarification": False,
                                    "clarifying_question": None, "suggested_mode": "lexical",
                                    "top_k": 5})},
            {"content": "Answer [S1]."},
        ])
        doc = tmp_path / "d.md"
        doc.write_text("Content about error CODE-7788 handling. " * 30, encoding="utf-8")
        svc.ingest([doc], with_graph=False)

        events = list(svc.stream_query("error CODE-7788?", mode="auto", use_cache=False))
        route_events = [e for e in events if e["type"] == "route"]
        assert route_events and route_events[0]["route"] == "lexical"
        done = events[-1]
        assert done["answer"].startswith("Answer")

    def test_clarify_route_short_circuits(self, tmp_path):
        from ragstack.config import AppConfig

        cfg = AppConfig(mode="local")
        cfg.index.root = str(tmp_path / "idx")
        cfg.graph.enabled = False
        cfg.agent.query_intelligence = True
        svc = RAGStack(cfg)
        svc._embeddings = FakeEmbeddings()
        svc._llm = FakeLLM([{"content": json.dumps({
            "intent": "factual", "complexity": "simple", "ambiguity": "high",
            "needs_clarification": True,
            "clarifying_question": "Which year's agreement?",
            "suggested_mode": "hybrid", "top_k": 5})}])
        events = list(svc.stream_query("termination period?", mode="auto"))
        kinds = [e["type"] for e in events]
        assert "clarify" in kinds and "done" not in kinds
        assert events[-1]["question"] == "Which year's agreement?"


class TestVerification:
    def test_manifest_and_confidence_blend(self, app_config, tmp_path):
        svc = RAGStack(_cfg(app_config))
        svc._embeddings = FakeEmbeddings()
        claims_json = json.dumps({"claims": [{"claim": "LanceDB stores vectors.", "refs": ["S1"]}]})
        verdicts_json = json.dumps({"results": [
            {"claim": "LanceDB stores vectors.", "verdict": "supported", "reason": "stated"}]})
        svc._llm = FakeLLM([
            {"content": "LanceDB stores vectors [S1]."},   # synthesis
            {"content": claims_json},                       # claim extraction
            {"content": verdicts_json},                     # verification
        ])
        doc = tmp_path / "d.md"
        doc.write_text("LanceDB is the vector store. " * 40, encoding="utf-8")
        svc.ingest([doc], with_graph=False)

        events = list(svc.stream_query("what stores vectors?", mode="vector", use_cache=False))
        verifs = [e for e in events if e["type"] == "verification"]
        assert verifs and verifs[0]["manifest"]["claims"][0]["verdict"] == "supported"
        done = events[-1]
        assert done["manifest"]["supported_ratio"] == 1.0
        assert not done.get("abstained")
        assert done["confidence"] >= 0.85

    def test_abstains_when_support_low(self, app_config, tmp_path):
        svc = RAGStack(_cfg(app_config))
        svc._embeddings = FakeEmbeddings()
        claims_json = json.dumps({"claims": [
            {"claim": "Claim one about cats.", "refs": ["S1"]},
            {"claim": "Claim two about dogs.", "refs": ["S1"]},
        ]})
        verdicts_json = json.dumps({"results": [
            {"claim": "Claim one about cats.", "verdict": "unsupported"},
            {"claim": "Claim two about dogs.", "verdict": "unsupported"},
        ]})
        svc._llm = FakeLLM([
            {"content": "Bold unsupported answer [S1]."},
            {"content": claims_json},
            {"content": verdicts_json},
        ])
        doc = tmp_path / "d.md"
        doc.write_text("Unrelated content entirely. " * 40, encoding="utf-8")
        svc.ingest([doc], with_graph=False)

        events = list(svc.stream_query("tell me about cats and dogs?", mode="hybrid", use_cache=False))
        done = events[-1]
        assert done.get("abstained") is True
        assert "Insufficient verified evidence" in done["answer"]
        assert done["confidence"] <= 0.25
        assert done["citations"]  # evidence still listed for the user

    def test_maybe_abstain_passes_good_ratio(self):
        manifest = {"claims": [{"verdict": "supported"}], "supported_ratio": 1.0}
        answer, abstained = maybe_abstain("fine answer", manifest, 0.5)
        assert (answer, abstained) == ("fine answer", False)


class TestTraces:
    def test_trace_persisted_with_ids(self, app_config, tmp_path):
        svc = RAGStack(_cfg(app_config, intel=False, verify=False))
        svc._embeddings = FakeEmbeddings()
        svc._llm = FakeLLM([{"content": "ans [S1]."}])
        doc = tmp_path / "d.md"
        doc.write_text("Traceable content. " * 40, encoding="utf-8")
        svc.ingest([doc], with_graph=False)

        events = list(svc.stream_query("q?", mode="vector", use_cache=False))
        tid = events[0]["trace_id"]
        assert events[-1]["trace_id"] == tid
        trace_file = Path(app_config.index.root) / "traces" / f"{tid}.json"
        data = json.loads(trace_file.read_text(encoding="utf-8"))
        assert data["trace_id"] == tid
        assert any(e["type"] == "done" for e in data["events"])

    def test_prune_keeps_recent(self, app_config, tmp_path):
        from pathlib import Path as P

        svc = RAGStack(_cfg(app_config, intel=False, verify=False))
        svc._embeddings = FakeEmbeddings()
        svc._llm = FakeLLM([{"content": "a."}])
        doc = tmp_path / "d.md"
        doc.write_text("Some content. " * 40, encoding="utf-8")
        svc.ingest([doc], with_graph=False)
        app_config.generation.trace_keep = 2
        for i in range(4):
            list(svc.stream_query(f"question {i}?", mode="vector", use_cache=False))
        files = list((P(app_config.index.root) / "traces").glob("*.json"))
        assert len(files) <= 2


from pathlib import Path  # noqa: E402


class TestContradictionTool:
    def test_returns_candidates(self, tmp_path):
        from ragstack.agent.tools import ToolContext, make_executor

        emb = FakeEmbeddings()
        lexical = LexicalStore(tmp_path / "lex")
        vector = VectorStore(tmp_path / "vec")

        texts = [
            "The vendor contract allows termination with thirty days notice.",
            "Under the amended rider, termination requires sixty days notice.",
            "Pasta water should be salted generously.",
        ]
        rows = []
        vecs = emb.embed(texts)
        for i, t in enumerate(texts):
            rows.append({"id": f"c{i}", "doc_id": f"d{i}", "ordinal": 0, "title": f"T{i}", "source": f"s{i}.md", "text": t, "context": "", "meta": "{}"})
            lexical.add([{"id": f"c{i}", "kind": "chunk", "doc_id": f"d{i}", "title": f"T{i}", "body": t, "source": f"s{i}.md"}])
        vector.add(rows, vecs)

        ctx = ToolContext(embeddings=emb, vector_store=vector, lexical_store=lexical)
        out = make_executor(ctx)("find_contradicting_evidence", {"claim": "termination takes thirty days"})
        hits = json.loads(out)
        assert isinstance(hits, list) and len(hits) >= 2  # both contract passages surfaced


from ragstack.stores.lexical import LexicalStore  # noqa: E402
from ragstack.stores.vector import VectorStore  # noqa: E402

"""Confidence scores, vertical presets, and the Cohere reranker."""

from __future__ import annotations

from conftest import FakeEmbeddings, FakeLLM

from ragstack.agent.prompts import build_system_prompt
from ragstack.config import AppConfig
from ragstack.providers.reranker import CohereReranker
from ragstack.service import RAGStack


class TestConfidence:
    def test_direct_mode_done_has_confidence(self, app_config, tmp_path):
        svc = RAGStack(app_config)
        svc._embeddings = FakeEmbeddings()
        svc._llm = FakeLLM([{"content": "Answer [S1]."}])
        doc = tmp_path / "d.md"
        doc.write_text("Content about vector stores. " * 40, encoding="utf-8")
        svc.ingest([doc], with_graph=False)

        events = list(svc.stream_query("vector stores?", mode="vector", use_cache=False))
        done = events[-1]
        assert isinstance(done["confidence"], float)
        assert 0.0 <= done["confidence"] <= 0.97
        retrieval = next(e for e in events if e["type"] == "retrieval")
        assert retrieval["verdict"] in ("correct", "ambiguous", "incorrect")

    def test_confidence_scales_with_citations(self):
        from ragstack.service import _confidence

        assert _confidence("correct", 4) > _confidence("correct", 0)
        assert _confidence("incorrect", 4) < _confidence("ambiguous", 2)
        assert 0.05 <= _confidence(None, 0) <= 0.97


class TestVerticalPresets:
    def test_legal_preset_adjusts_chunking(self):
        cfg = AppConfig(mode="local", vertical="legal").resolve_providers()
        assert cfg.chunking.size == 384 and cfg.chunking.overlap == 48

    def test_unknown_vertical_rejected(self):
        import pytest

        from ragstack.errors import ConfigError

        with pytest.raises(ConfigError):
            AppConfig(mode="local", vertical="cookbook").resolve_providers()

    def test_system_prompt_includes_domain(self):
        prompt = build_system_prompt("medical")
        assert "medical research" in prompt
        assert build_system_prompt(None) == build_system_prompt("").replace("", "")

    def test_agent_receives_vertical_prompt(self, app_config, tmp_path):
        app_config.vertical = "academic"
        svc = RAGStack(app_config)
        svc._embeddings = FakeEmbeddings()
        svc._llm = FakeLLM([{"content": "ans"}])
        doc = tmp_path / "d.md"
        doc.write_text("Paper text. " * 40, encoding="utf-8")
        svc.ingest([doc], with_graph=False)
        list(svc.stream_query("q?", mode="vector", use_cache=False))
        system_msg = svc._llm.calls[-1][0]["content"]
        assert "academic papers" in system_msg


class TestCohereReranker:
    def test_missing_key_raises(self, monkeypatch):
        import pytest

        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        with pytest.raises(ValueError):
            CohereReranker("rerank-v3.5")

    def test_sorts_by_api_scores(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-key")

        captured = {}

        class FakeResp:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.2},
                    ]
                }

        def fake_post(url, headers=None, json=None, timeout=None):
            captured.update({"url": url, "headers": headers, "payload": json})
            return FakeResp()

        import ragstack.providers.reranker as rr_mod

        assert rr_mod is not None  # module imports cleanly
        monkeypatch.setattr("httpx.post", fake_post)
        r = CohereReranker("rerank-v3.5")

        class Item:
            def __init__(self, t):
                self.text = t

        out = r.rerank("q", [Item("first"), Item("second")])
        assert out[0].text == "second"
        assert captured["payload"]["model"] == "rerank-v3.5"
        assert captured["headers"]["Authorization"] == "Bearer test-key"

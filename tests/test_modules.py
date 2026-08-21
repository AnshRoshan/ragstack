"""Config, enricher, crawler-extract, and eval harness tests."""

from __future__ import annotations

import yaml
from conftest import FakeEmbeddings, FakeLLM

from ragstack.config import AppConfig, load_config, save_config
from ragstack.ingestion.crawler import _extract
from ragstack.ingestion.enricher import Enricher
from ragstack.types import Chunk, Document


class TestConfig:
    def test_yaml_roundtrip(self, tmp_path):
        cfg = AppConfig(mode="local")
        cfg.index.root = str(tmp_path / "x")
        p = tmp_path / "ragstack.yaml"
        save_config(cfg, p)
        loaded = load_config(p)
        assert loaded.mode == "local"
        assert loaded.index.root == str(tmp_path / "x")

    def test_load_missing_explicit_file_raises(self):
        import pytest

        from ragstack.errors import ConfigError

        with pytest.raises(ConfigError):
            load_config("/nonexistent/path/definitely.yaml")

    def test_databases_config(self, tmp_path):
        data = {"mode": "local", "databases": {"main": "sqlite:///x.db"}, "index": {"root": str(tmp_path)}}
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(data), encoding="utf-8")
        cfg = load_config(p)
        assert cfg.databases["main"] == "sqlite:///x.db"


class TestEnricher:
    def test_enrich_adds_context_and_caches(self, tmp_path):
        from ragstack.config import IndexConfig

        llm = FakeLLM([{"content": f"This chunk explains topic {i}."} for i in range(10)])
        enricher = Enricher(llm, IndexConfig(root=str(tmp_path / "idx")))
        doc = Document(id="d1", source="s.md", title="T", text="body")
        chunks = [
            Chunk(id=f"c{i}", doc_id="d1", ordinal=i, text=f"chunk text {i}", n_tokens=10)
            for i in range(3)
        ]
        enriched = enricher.enrich(doc, chunks)
        assert all("explains topic" in c.context for c in enriched)
        assert len(llm.calls) == 3

        # second pass: all cache hits, no LLM calls
        llm2 = FakeLLM([])
        enricher2 = Enricher(llm2, IndexConfig(root=str(tmp_path / "idx")))
        enriched2 = enricher2.enrich(doc, [Chunk(id=c.id, doc_id=c.doc_id, ordinal=c.ordinal, text=c.text) for c in chunks])
        assert len(llm2.calls) == 0
        assert all("explains topic" in c.context for c in enriched2)

    def test_enrich_survives_llm_failure(self, tmp_path):
        from ragstack.config import IndexConfig

        class Boom(FakeLLM):
            def chat(self, *a, **k):
                raise RuntimeError("llm down")

        enricher = Enricher(Boom([]), IndexConfig(root=str(tmp_path / "idx")))
        doc = Document(id="d1", source="s.md", title="T", text="body")
        chunks = [Chunk(id="c0", doc_id="d1", ordinal=0, text="text here", n_tokens=5)]
        out = enricher.enrich(doc, chunks)
        assert out[0].context == ""  # no crash, context simply not added


class TestCrawlerExtract:
    def test_extract_html_offline(self):
        html = """
        <html><head><title>Test Page</title></head><body>
        <article><h1>Main</h1><p>This is the main content of the page for testing extraction.</p></article>
        <a href="/about">About</a><a href="https://external.com/x">Ext</a>
        </body></html>
        """
        title, text, links = _extract(html, "https://example.com/post")
        assert title == "Test Page"
        assert "main content" in text
        assert any(link.endswith("/about") for link in links)
        assert any("external.com" in link for link in links)


class TestEvalHarness:
    def test_golden_loading_and_metrics(self, tmp_path, app_config):
        from ragstack.eval.harness import run_eval
        from ragstack.service import RAGStack

        svc = RAGStack(app_config)
        svc._embeddings = FakeEmbeddings()
        svc._llm = FakeLLM([])

        doc = tmp_path / "kubernetes.md"
        doc.write_text("# Kubernetes\n\nKubernetes orchestrates containers in a cluster. " * 20, encoding="utf-8")
        svc.ingest([doc], with_graph=False)

        golden = tmp_path / "golden.yaml"
        golden.write_text(
            yaml.safe_dump(
                {
                    "cases": [
                        {"question": "What orchestrates containers?", "expected_docs": ["kubernetes"], "expected_keywords": ["containers"]},
                        {"question": "completely unrelated query about quantum physics", "expected_docs": ["does-not-exist"]},
                    ]
                }
            ),
            encoding="utf-8",
        )
        report = run_eval(svc, golden, k=5)
        assert report["cases"] == 2
        assert report["hit_rate"] == 0.5
        assert report["mrr"] > 0

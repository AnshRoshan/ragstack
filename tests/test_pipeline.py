"""End-to-end ingestion pipeline tests with fakes (no network, no models)."""

from __future__ import annotations

from pathlib import Path

from conftest import FakeEmbeddings, FakeLLM

from ragstack.ingestion.parsers import collect_files, parse_file
from ragstack.pipeline import IngestionPipeline
from ragstack.service import RAGStack
from ragstack.stores.lexical import LexicalStore
from ragstack.stores.vector import VectorStore


def _make_pipeline(tmp_path: Path, llm=None) -> tuple[IngestionPipeline, Path]:
    root = tmp_path / "idx"
    emb = FakeEmbeddings()
    lexical = LexicalStore(root)
    vector = VectorStore(root)
    cfg_llm = llm or FakeLLM([])
    from ragstack.config import AppConfig

    cfg = AppConfig(mode="local")
    cfg.index.root = str(root)
    cfg.graph.enabled = False
    return IngestionPipeline(cfg, emb, cfg_llm, lexical, vector, None), root


class TestIngestionE2E:
    def test_index_md_file_end_to_end(self, tmp_path):
        pipeline, root = _make_pipeline(tmp_path)
        doc_file = tmp_path / "note.md"
        doc_file.write_text(
            "# Title\n\n" + ("Some meaningful content about retrieval systems. " * 30),
            encoding="utf-8",
        )
        stats = pipeline.ingest_paths([doc_file])
        assert stats.indexed == 1
        assert stats.chunks >= 1
        assert stats.failed == 0

        lexical = LexicalStore(root)
        vector = VectorStore(root)
        assert lexical.count() >= 2  # 1 page + N chunks
        assert vector.count() == stats.chunks
        hits = lexical.search("retrieval systems", kind="chunk")
        assert hits

    def test_reingest_same_file_is_skipped(self, tmp_path):
        pipeline, _ = _make_pipeline(tmp_path)
        f = tmp_path / "a.md"
        f.write_text("Stable content " * 100, encoding="utf-8")
        first = pipeline.ingest_paths([f])
        second = pipeline.ingest_paths([f])
        assert first.indexed == 1
        assert second.indexed == 0
        assert second.skipped == 1

    def test_changed_file_is_reindexed(self, tmp_path):
        pipeline, root = _make_pipeline(tmp_path)
        f = tmp_path / "a.md"
        f.write_text("Version one content " * 50, encoding="utf-8")
        pipeline.ingest_paths([f])
        f.write_text("Version two content entirely different words " * 50, encoding="utf-8")
        again = pipeline.ingest_paths([f])
        assert again.indexed == 1

        # old chunks must be gone (no duplicates)
        vector = VectorStore(root)
        assert vector.count() == again.chunks

    def test_force_reindex(self, tmp_path):
        pipeline, _ = _make_pipeline(tmp_path)
        f = tmp_path / "a.md"
        f.write_text("Content " * 100, encoding="utf-8")
        pipeline.ingest_paths([f])
        forced = pipeline.ingest_paths([f], force=True)
        assert forced.indexed == 1

    def test_collect_files_ignores_junk_dirs(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "good.md").write_text("hello", encoding="utf-8")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config.md").write_text("junk", encoding="utf-8")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg.md").write_text("junk", encoding="utf-8")
        files = collect_files([tmp_path], recursive=True)
        names = [f.name for f in files]
        assert names == ["good.md"]

    def test_parse_dispatch(self, tmp_path):
        md = tmp_path / "x.md"
        md.write_text("# H\ntext", encoding="utf-8")
        doc = parse_file(md)
        assert doc.metadata["format"] == "md"

        code = tmp_path / "y.py"
        code.write_text("print('hi')", encoding="utf-8")
        dcode = parse_file(code)
        assert dcode.metadata["format"] == "code"
        assert "```py" in dcode.text

        csvf = tmp_path / "z.csv"
        csvf.write_text("id,name\n1,ansh\n", encoding="utf-8")
        dcsv = parse_file(csvf)
        assert "ansh" in dcsv.text and "rows: 1" in dcsv.text


class TestServiceWiring:
    def test_service_ingest_and_status(self, app_config, tmp_path):
        svc = RAGStack(app_config)
        svc._embeddings = FakeEmbeddings()
        svc._llm = FakeLLM([])

        f = tmp_path / "doc.md"
        f.write_text("# Doc\n\n" + "Body text about databases and graphs. " * 40, encoding="utf-8")
        stats = svc.ingest([f], with_graph=False)
        assert stats.indexed == 1

        info = svc.status()
        assert info["vector_chunks"] == stats.chunks
        assert info["lexical_docs"] >= 2

    def test_reset_clears_everything(self, app_config, tmp_path):
        svc = RAGStack(app_config)
        svc._embeddings = FakeEmbeddings()
        svc._llm = FakeLLM([])
        f = tmp_path / "doc.md"
        f.write_text("Some text " * 60, encoding="utf-8")
        svc.ingest([f], with_graph=False)
        assert svc.vector.count() > 0
        svc.reset()
        assert svc.vector.count() == 0
        assert svc.lexical.count() == 0

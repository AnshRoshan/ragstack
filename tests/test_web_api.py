"""Web API tests via FastAPI TestClient (hermetic: fake providers injected)."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import FakeEmbeddings, FakeLLM
from fastapi.testclient import TestClient

from ragstack.service import RAGStack
from ragstack.web.app import create_app


def _client(tmp_path: Path) -> TestClient:
    from ragstack.config import AppConfig

    cfg = AppConfig(mode="local")
    cfg.index.root = str(tmp_path / "idx")
    cfg.graph.enabled = False
    svc = RAGStack(cfg)
    svc._embeddings = FakeEmbeddings()
    svc._llm = FakeLLM([{"content": "Final answer [S1]."}])

    doc = tmp_path / "doc.md"
    doc.write_text("# Doc\n\n" + "Content about vector databases. " * 40, encoding="utf-8")
    svc.ingest([doc], with_graph=False)

    app = create_app(svc)
    return TestClient(app)


class TestWebAPI:
    def test_status_endpoint(self, tmp_path):
        c = _client(tmp_path)
        r = c.get("/api/status")
        assert r.status_code == 200
        body = r.json()
        assert body["vector_chunks"] > 0
        assert body["mode"] == "local"

    def test_ui_served(self, tmp_path):
        c = _client(tmp_path)
        r = c.get("/")
        assert r.status_code == 200
        assert b"RAG" in r.content and b"text/html" in r.headers["content-type"].encode()

    def test_openapi_docs(self, tmp_path):
        c = _client(tmp_path)
        assert c.get("/api/docs").status_code == 200

    def test_query_sse_stream(self, tmp_path):
        c = _client(tmp_path)
        with c.stream("POST", "/api/query", json={"question": "what stores vectors?", "mode": "auto"}) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]
            events = []
            for line in r.iter_lines():
                if line.startswith("data: "):
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    events.append(json.loads(payload))
        types = [e["type"] for e in events]
        assert types[0] == "start"
        assert types[-1] == "done"
        done = events[-1]
        assert done["answer"] == "Final answer [S1]."

    def test_index_endpoint(self, tmp_path):
        c = _client(tmp_path)
        doc2 = tmp_path / "doc2.md"
        doc2.write_text("Another document about graph databases. " * 30, encoding="utf-8")
        r = c.post("/api/index", json={"paths": [str(doc2)], "graph": False})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True and body["indexed"] == 1

    def test_query_validation_error(self, tmp_path):
        c = _client(tmp_path)
        r = c.post("/api/query", json={"question": ""})
        assert r.status_code == 422

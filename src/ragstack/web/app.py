"""FastAPI app: SSE query streaming, indexing, status; serves the chat UI."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..service import RAGStack
from ..utils import get_logger

log = get_logger("ragstack.web")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: str = "auto"
    top_k: int | None = None
    use_cache: bool = True
    session_id: str | None = Field(default=None, max_length=128)


class IndexRequest(BaseModel):
    paths: list[str]
    recursive: bool = True
    enrich: bool = False
    graph: bool = True
    force: bool = False


class CrawlRequest(BaseModel):
    url: str
    depth: int = 1
    max_pages: int = 10


def create_app(service: RAGStack | None = None) -> FastAPI:
    svc = service or RAGStack()
    from .. import __version__

    app = FastAPI(title="RAGStack", version=__version__, docs_url="/api/docs")
    # Allow the hosted landing page (and other local tools) to call a local server.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/modes")
    def modes() -> list[dict[str, Any]]:
        from ..service import MODE_CATALOG

        return MODE_CATALOG

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return svc.status()

    @app.post("/api/index")
    def index(req: IndexRequest) -> dict[str, Any]:
        try:
            stats = svc.ingest(
                req.paths,
                recursive=req.recursive,
                enrich=req.enrich,
                with_graph=req.graph,
                force=req.force,
            )
            return {"ok": stats.failed == 0, **asdict(stats)}
        except Exception as e:
            log.exception("index failed")
            raise HTTPException(500, str(e)) from e

    @app.post("/api/crawl")
    def crawl(req: CrawlRequest) -> dict[str, Any]:
        try:
            stats = svc.ingest_url(req.url, depth=req.depth, max_pages=req.max_pages)
            return {"ok": stats.failed == 0, **asdict(stats)}
        except Exception as e:
            log.exception("crawl failed")
            raise HTTPException(500, str(e)) from e

    @app.post("/api/query")
    def query(req: QueryRequest) -> StreamingResponse:
        def sse():
            try:
                for event in svc.stream_query(
                    req.question, mode=req.mode, top_k=req.top_k,
                    use_cache=req.use_cache, session_id=req.session_id,
                ):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as e:
                log.exception("query crashed")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    static_dir = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")
    return app

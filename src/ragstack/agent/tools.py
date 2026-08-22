"""Agent tool definitions (OpenAI function-calling schema) + execution."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..errors import ToolError
from ..types import Citation, RetrievedItem
from ..utils import truncate

log = logging.getLogger("ragstack.tools")

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_pages",
            "description": "Vectorless BM25 keyword search over whole indexed documents/pages. Best for broad lookups, exact terms, IDs, names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_chunks",
            "description": "BM25 keyword search over text chunks with reranking. Good for precise phrase/keyword matching inside passages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 8},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Dense embedding search over chunks. Best for meaning-based queries where exact words may not appear.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 8},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hybrid_search",
            "description": "Fused dense+keyword search (RRF) with reranking. Strong default for most questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 8},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "decomposed_search",
            "description": "For compound or multi-part questions: splits the question into sub-queries, runs hybrid retrieval for each in parallel, and fuses the results. Use when a question compares things, has multiple parts, or needs several facts combined.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "top_k": {"type": "integer", "default": 8},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_search",
            "description": "Knowledge-graph neighborhood: entities and relations related to a topic/name, plus their source chunks. Use for multi-hop 'who/what connects to what' questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_hint": {"type": "string", "description": "topic, name or entity to anchor on"},
                    "top_k": {"type": "integer", "default": 8},
                },
                "required": ["entity_hint"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "community_overview",
            "description": "Global GraphRAG: thematic community summaries of the whole corpus. Use for broad 'what are the main themes' questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "top_k": {"type": "integer", "default": 4},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": "Run a read-only SELECT against a registered external database. Use for numbers, aggregates, records. Schemas available via description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "database": {"type": "string", "description": "registered database name"},
                    "sql": {"type": "string", "description": "SELECT statement"},
                },
                "required": ["database", "sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch and extract readable text from a public web page URL. Use only if the user asks for live web content.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]


@dataclass
class ToolContext:
    embeddings: Any = None
    vector_store: Any = None
    lexical_store: Any = None
    graph_store: Any = None
    llm: Any = None
    sql_catalog: Any = None
    reranker: Any = None
    grader: Any = None
    top_k: int = 8
    max_hops: int = 2
    max_steps: int = 8
    strip_refinement: bool = True
    citations: dict[str, Citation] = field(default_factory=dict)
    _counter: int = 0

    def register(self, item: RetrievedItem) -> str:
        key = item.chunk_id or item.source
        existing = self.citations.get(f"@{key}")
        if existing:
            return existing.ref_id
        self._counter += 1
        ref = f"S{self._counter}"
        self.citations[f"@{key}"] = Citation(
            ref_id=ref,
            source=item.source,
            title=item.title,
            snippet=truncate(item.text, 300),
        )
        return ref


def _format_hits(ctx: ToolContext, items: list[RetrievedItem], limit: int) -> str:
    if not items:
        return "[]"
    out = []
    for it in items[:limit]:
        ref = ctx.register(it)
        out.append(
            {
                "ref": ref,
                "score": round(it.score, 4),
                "source": it.source,
                "title": it.title,
                "origin": it.origin,
                "text": truncate(it.text, 700),
            }
        )
    return json.dumps(out, ensure_ascii=False)


def make_executor(ctx: ToolContext) -> Callable[[str, dict], str]:
    from ..agent.planner import decomposed_search, format_decomposed
    from ..graphrag.search import global_search, local_search
    from ..retrieval.lexical_rag import search_chunks, search_pages
    from ..retrieval.vector_rag import hybrid_search, semantic_search

    def execute(name: str, args: dict[str, Any]) -> str:
        try:
            if name == "search_pages":
                items = search_pages(ctx.lexical_store, args["query"], int(args.get("top_k", 5)))
                return _format_hits(ctx, items, limit=int(args.get("top_k", 5)))

            if name == "search_chunks":
                items = search_chunks(ctx.lexical_store, args["query"], int(args.get("top_k", ctx.top_k)))
                return _format_hits(ctx, items, limit=len(items))

            if name == "semantic_search":
                items = semantic_search(ctx.embeddings, ctx.vector_store, args["query"], int(args.get("top_k", ctx.top_k)))
                return _format_hits(ctx, items, limit=len(items))

            if name == "hybrid_search":
                items = hybrid_search(
                    ctx.embeddings, ctx.vector_store, ctx.lexical_store, args["query"], int(args.get("top_k", ctx.top_k))
                )
                return _format_hits(ctx, items, limit=len(items))

            if name == "decomposed_search":
                subs, items = decomposed_search(
                    ctx.llm,
                    ctx.embeddings,
                    ctx.vector_store,
                    ctx.lexical_store,
                    args["question"],
                    top_k=int(args.get("top_k", ctx.top_k)),
                    reranker=ctx.reranker,
                )
                return format_decomposed(subs, items, ctx.register)

            if name == "graph_search":
                context, items = local_search(
                    ctx.graph_store, ctx.vector_store, args["entity_hint"], int(args.get("top_k", ctx.top_k)), ctx.max_hops
                )
                hits = _format_hits(ctx, items, limit=len(items))
                return f"{context}\n\nSOURCE CHUNKS:\n{hits}" if items else context

            if name == "community_overview":
                context, items = global_search(ctx.graph_store, ctx.llm, args["topic"], int(args.get("top_k", 4)))
                refs = [ctx.register(i) for i in items]
                return f"{context}\n\nCOMMUNITY REFS: {refs}"

            if name == "sql_query":
                if ctx.sql_catalog is None:
                    raise ToolError("no databases registered")
                return ctx.sql_catalog.query(args["database"], args["sql"])

            if name == "fetch_url":
                from ..ingestion.crawler import crawl

                docs = crawl(args["url"], depth=0, max_pages=1)
                if not docs:
                    return "fetch failed or empty page"
                doc = docs[0]
                item = RetrievedItem(chunk_id="", doc_id=doc.id, source=doc.source, title=doc.title, text=doc.text, origin="web")
                ref = ctx.register(item)
                return json.dumps({"ref": ref, "source": doc.source, "title": doc.title, "text": truncate(doc.text, 4000)}, ensure_ascii=False)

            raise ToolError(f"unknown tool {name}")
        except ToolError as e:
            return json.dumps({"error": str(e)})
        except Exception as e:
            log.exception("tool %s crashed", name)
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    return execute


def tools_for_mode(mode: str) -> list[dict]:
    by_name = {t["function"]["name"]: t for t in TOOL_SCHEMAS}
    mode = (mode or "auto").lower()
    if mode == "lexical":
        names = ["search_pages", "search_chunks"]
    elif mode == "vector":
        names = ["semantic_search", "hybrid_search"]
    elif mode == "hybrid":
        names = ["hybrid_search", "decomposed_search", "search_chunks", "semantic_search"]
    elif mode == "graph":
        names = ["graph_search", "community_overview", "semantic_search"]
    elif mode == "sql":
        names = ["sql_query", "search_chunks"]
    else:
        names = [t["function"]["name"] for t in TOOL_SCHEMAS]
    return [by_name[n] for n in names]


RETRIEVAL_TOOLS = {"search_pages", "search_chunks", "semantic_search", "hybrid_search", "decomposed_search"}

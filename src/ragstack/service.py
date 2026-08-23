"""RAGStack service facade: wires config → providers → stores → pipeline → agent."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .agent.runner import AgentRunner
from .agent.tools import ToolContext
from .config import AppConfig, load_config
from .errors import ConfigError
from .pipeline import IngestionPipeline, IngestStats
from .providers.embeddings import EmbeddingProvider, make_embedding_provider
from .providers.llm import LLMProvider, make_llm_provider
from .providers.reranker import Reranker, make_reranker
from .stores.graph.base import GraphStore
from .stores.lexical import LexicalStore
from .stores.sql_catalog import SQLCatalog
from .stores.vector import VectorStore
from .types import Answer
from .utils import get_logger

log = get_logger("ragstack.service")


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("ragstack")
    except Exception:
        from . import __version__

        return __version__

AGENTIC_MODES = {"auto", "agentic", "sql"}
DIRECT_MODES = {"vector", "lexical", "hybrid", "graph", "global"}
ALL_MODES = AGENTIC_MODES | DIRECT_MODES

MODE_CATALOG = [
    {
        "id": "auto",
        "label": "Auto — Agentic Router (recommended)",
        "kind": "agentic",
        "description": "The agent plans each hop and picks whichever tool fits: keyword, vector, hybrid, graph, communities, SQL or web fetch.",
        "tools": ["hybrid_search", "decomposed_search", "search_pages", "search_chunks", "semantic_search", "graph_search", "community_overview", "sql_query", "fetch_url"],
    },
    {
        "id": "agentic",
        "label": "Agentic RAG — full multi-hop agent",
        "kind": "agentic",
        "description": "Same full toolkit as Auto, made explicit: ReAct loop with up to max_steps tool calls, evidence grading and citation tracking.",
        "tools": ["hybrid_search", "decomposed_search", "search_pages", "search_chunks", "semantic_search", "graph_search", "community_overview", "sql_query", "fetch_url"],
    },
    {
        "id": "hybrid",
        "label": "Hybrid Retrieval — dense + keyword fusion",
        "kind": "direct",
        "description": "One fast pass: BM25 and embedding search run together, merged with Reciprocal Rank Fusion, then cross-encoder reranked. Best default for simple questions.",
        "pipeline": ["bm25", "dense", "rrf", "rerank"],
    },
    {
        "id": "vector",
        "label": "Vector RAG — embeddings only",
        "kind": "direct",
        "description": "Pure meaning-based search over chunk embeddings with reranking. Use when wording differs from the documents.",
        "pipeline": ["dense", "rerank"],
    },
    {
        "id": "lexical",
        "label": "Lexical RAG — vectorless keyword search",
        "kind": "direct",
        "description": "Pure BM25 over pages and chunks. No embeddings involved. Best for exact names, error codes, identifiers.",
        "pipeline": ["bm25", "rerank"],
    },
    {
        "id": "graph",
        "label": "GraphRAG — entity neighborhood",
        "kind": "direct",
        "description": "Anchors on entities in your question and walks their knowledge-graph connections, returning related facts and their source chunks.",
        "pipeline": ["entity_match", "graph_walk", "chunk_lookup"],
    },
    {
        "id": "global",
        "label": "Global GraphRAG — corpus themes",
        "kind": "direct",
        "description": "Answers broad questions ('what are the main themes?') from LLM-summarized entity communities covering the whole corpus.",
        "pipeline": ["community_rank", "map_reduce_synthesis"],
    },
    {
        "id": "sql",
        "label": "SQL — structured data agent",
        "kind": "agentic",
        "description": "Agent restricted to read-only text-to-SQL against your registered databases, plus chunk search for context.",
        "tools": ["sql_query", "search_chunks"],
    },
]


class RAGStack:
    def __init__(self, config: AppConfig | str | Path | None = None):
        if isinstance(config, AppConfig):
            self.cfg = config.resolve_providers()
        else:
            self.cfg = load_config(config).resolve_providers()
        self.root = self.cfg.resolved_root()
        self.root.mkdir(parents=True, exist_ok=True)

        self._embeddings: EmbeddingProvider | None = None
        self._llm: LLMProvider | None = None
        self._reranker: Reranker | None = None
        self._lexical: LexicalStore | None = None
        self._vector: VectorStore | None = None
        self._graph: GraphStore | None = None
        self._sql: SQLCatalog | None = None
        self._pipeline: IngestionPipeline | None = None
        self._cache = None
        self._memory = None
        self._recall = None

    # -- lazy components -------------------------------------------------------
    @property
    def embeddings(self) -> EmbeddingProvider:
        if self._embeddings is None:
            self._embeddings = make_embedding_provider(self.cfg.embedding)
        return self._embeddings

    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = make_llm_provider(self.cfg.llm)
        return self._llm

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = make_reranker(self.cfg.rerank.provider, self.cfg.rerank.model)
        return self._reranker

    @property
    def lexical(self) -> LexicalStore:
        if self._lexical is None:
            self._lexical = LexicalStore(self.root)
        return self._lexical

    @property
    def vector(self) -> VectorStore:
        if self._vector is None:
            self._vector = VectorStore(self.root)
        return self._vector

    @property
    def graph(self) -> GraphStore | None:
        if not self.cfg.graph.enabled:
            return None
        if self._graph is None:
            from .stores.graph import open_graph

            self._graph = open_graph(self.cfg.graph, self.root)
        return self._graph

    @property
    def sql(self) -> SQLCatalog:
        if self._sql is None:
            self._sql = SQLCatalog(self.cfg.databases)
        return self._sql

    @property
    def pipeline(self) -> IngestionPipeline:
        if self._pipeline is None:
            self._pipeline = IngestionPipeline(
                self.cfg, self.embeddings, self.llm, self.lexical, self.vector, self.graph
            )
        return self._pipeline

    # -- operations --------------------------------------------------------------
    def ingest(
        self,
        paths: list[str | Path],
        recursive: bool = False,
        enrich: bool = False,
        with_graph: bool | None = None,
        force: bool = False,
    ) -> IngestStats:
        return self.pipeline.ingest_paths(paths, recursive=recursive, enrich=enrich, with_graph=with_graph, force=force)

    def ingest_url(self, url: str, depth: int = 1, max_pages: int = 10, enrich: bool = False) -> IngestStats:
        return self.pipeline.ingest_url(url, depth=depth, max_pages=max_pages, enrich=enrich)

    def _tool_context(self, top_k: int | None) -> ToolContext:
        grader = None
        if self.cfg.agent.evidence_grading:
            from .retrieval.evaluator import EvidenceGrader

            grader = EvidenceGrader(
                grader=self.cfg.agent.evidence_grader,
                llm=self.llm,
                reranker=self.reranker if self.cfg.rerank.provider != "none" else None,
            )
        return ToolContext(
            embeddings=self.embeddings,
            vector_store=self.vector,
            lexical_store=self.lexical,
            graph_store=self.graph,
            llm=self.llm,
            sql_catalog=self.sql if self.cfg.databases else None,
            recall_store=self.recall,
            reranker=self.reranker if self.cfg.rerank.provider != "none" else None,
            grader=grader,
            top_k=top_k or self.cfg.agent.top_k,
            max_hops=self.cfg.graph.max_hops,
            max_steps=self.cfg.agent.max_steps,
            strip_refinement=self.cfg.agent.strip_refinement,
        )

    @property
    def cache(self):
        if not self.cfg.cache.enabled:
            return None
        if self._cache is None:
            from .cache import SemanticCache

            self._cache = SemanticCache(
                self.root,
                self.embeddings,
                threshold=self.cfg.cache.threshold,
                fuzzy_threshold=self.cfg.cache.fuzzy_threshold,
            )
        return self._cache

    @property
    def memory(self):
        if self.cfg.agent.memory_turns <= 0:
            return None
        if self._memory is None:
            from .memory import ConversationMemory

            self._memory = ConversationMemory(self.root)
        return self._memory

    @property
    def recall(self):
        if not self.cfg.agent.recall_enabled:
            return None
        if self._recall is None:
            from .memory import RecallStore

            self._recall = RecallStore(self.root, self.embeddings)
        return self._recall

    def stream_query(
        self,
        question: str,
        mode: str = "auto",
        top_k: int | None = None,
        use_cache: bool = True,
        session_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        mode = (mode or "auto").lower()
        if mode not in ALL_MODES:
            yield {"type": "error", "message": f"unknown mode {mode!r}; valid: {sorted(ALL_MODES)}"}
            return

        memory = self.memory
        history = (
            memory.history(session_id, limit=self.cfg.agent.memory_turns * 2)
            if (memory and session_id)
            else []
        )

        # Co-reference rewrite: turn follow-ups into standalone search queries.
        search_query = question
        if history:
            from .memory import rewrite_question

            search_query = rewrite_question(self.llm, question, history)
            yield {
                "type": "rewrite",
                "original": question,
                "standalone": search_query,
            }

        cache = self.cache if use_cache else None
        if cache is not None:
            hit = cache.lookup(search_query, mode=mode)
            if hit:
                yield {"type": "start", "mode": mode, "tools": [], "cached": True}
                yield {"type": "done", **hit, "error": None}
                self._remember(session_id, question, hit.get("answer", ""))
                return

        final_event: dict[str, Any] | None = None
        if mode in DIRECT_MODES:
            for event in self._stream_direct(question, mode, top_k, search_query, history):
                if event["type"] == "done":
                    final_event = event
                yield event
        else:
            ctx = self._tool_context(top_k)
            runner = AgentRunner(self.llm, ctx)
            for event in runner.run(
                question, mode=mode, stream=True, history=history, search_query=search_query
            ):
                if event["type"] == "done":
                    final_event = event
                yield event

        if cache is not None and final_event and not final_event.get("error") and final_event.get("answer"):
            cache.store(
                search_query,
                mode,
                final_event["answer"],
                final_event.get("citations", []),
                final_event.get("steps", []),
            )
        self._remember(session_id, question, final_event.get("answer", "") if final_event else "")

    def _remember(self, session_id: str | None, question: str, answer: str) -> None:
        if not session_id or self.memory is None or not answer:
            return
        self.memory.append(session_id, "user", question)
        self.memory.append(session_id, "assistant", answer)
        if self.recall is not None:
            self.recall.add(session_id, question, answer)

    # -- direct single-pass pipelines ----------------------------------------
    def _retrieve_direct(self, question: str, mode: str, top_k: int):
        from .graphrag.search import global_search, local_search
        from .retrieval.lexical_rag import search_chunks
        from .retrieval.vector_rag import hybrid_search, semantic_search

        k = top_k or self.cfg.agent.top_k
        extra_context: list[str] = []
        if mode == "vector":
            return semantic_search(self.embeddings, self.vector, question, k, reranker=self._direct_reranker()), extra_context
        if mode == "lexical":
            return search_chunks(self.lexical, question, k, reranker=self._direct_reranker()), extra_context
        if mode == "hybrid":
            return hybrid_search(self.embeddings, self.vector, self.lexical, question, k, reranker=self._direct_reranker()), extra_context
        if mode == "graph":
            context, items = local_search(self.graph, self.vector, question, top_k=k, max_hops=self.cfg.graph.max_hops)
            return items, [context]
        if mode == "global":
            context, items = global_search(self.graph, self.llm, question, top_k=4)
            return items, [context]
        raise ValueError(f"not a direct mode: {mode}")

    def _direct_reranker(self):
        return self.reranker if self.cfg.rerank.provider != "none" else None

    def _stream_direct(
        self,
        question: str,
        mode: str,
        top_k: int | None,
        search_query: str | None = None,
        history: list[dict] | None = None,
    ) -> Iterator[dict[str, Any]]:
        from .agent.tools import ToolContext

        yield {"type": "start", "mode": mode, "tools": [f"{mode}_retrieval"]}

        try:
            items, extra_context = self._retrieve_direct(search_query or question, mode, top_k)
        except Exception as e:
            log.exception("direct retrieval failed")
            yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
            yield {"type": "done", "answer": "", "citations": [], "steps": [], "error": str(e)}
            return

        registry = ToolContext()
        refs = [registry.register(it) for it in items]
        citations = list(registry.citations.values())
        yield {
            "type": "retrieval",
            "count": len(items),
            "refs": refs,
            "sub_queries": [],
        }

        context_blocks = [
            f"[{ref}] {it.title} ({it.source})\n{it.text[:2500]}"
            for ref, it in zip(refs, items, strict=True)
        ] + extra_context
        context_text = "\n\n".join(context_blocks) if context_blocks else "(nothing retrieved)"

        history_block = ""
        if history:
            from .memory import format_history

            history_block = f"\n\nRECENT CONVERSATION (for continuity, do not cite):\n{format_history(history)}"
        synthesis = (
            f"Answer the QUESTION using ONLY the CONTEXT. End every claim with its [Sn] citation. "
            f"If the context does not contain the answer, say exactly what is missing.\n\n"
            f"QUESTION: {question}{history_block}\n\nCONTEXT:\n{context_text}"
        )
        answer_parts: list[str] = []
        error: str | None = None
        try:
            for delta in self.llm.stream([{"role": "user", "content": synthesis}], temperature=self.cfg.llm.temperature, max_tokens=self.cfg.llm.max_tokens):
                if delta.kind == "text" and delta.text:
                    answer_parts.append(delta.text)
                    yield {"type": "answer", "delta": delta.text}
                elif delta.kind == "result" and delta.result is not None and delta.result.content:
                    if not answer_parts:
                        answer_parts.append(delta.result.content)
                        yield {"type": "answer", "delta": delta.result.content}
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            yield {"type": "error", "message": error}

        answer_text = "".join(answer_parts).strip() or "(no answer produced)"
        steps = [{"step": 1, "tool": f"{mode}_retrieval", "args": {"top_k": top_k or self.cfg.agent.top_k}}]
        yield {
            "type": "done",
            "answer": answer_text,
            "citations": [asdict(c) for c in citations],
            "steps": steps,
            "error": error,
        }

    def query(
        self, question: str, mode: str = "auto", top_k: int | None = None, session_id: str | None = None
    ) -> Answer:
        from .types import Answer, Citation

        mode = (mode or "auto").lower()
        answer_text, citations, steps = "", [], []
        for event in self.stream_query(question, mode=mode, top_k=top_k, session_id=session_id):
            if event["type"] == "done":
                answer_text = event.get("answer", "")
                citations = [Citation(**c) for c in event.get("citations", [])]
                steps = event.get("steps", [])
        return Answer(text=answer_text, citations=citations, steps=steps)

    def status(self) -> dict[str, Any]:
        graph_stats: dict[str, int] = {}
        if self.graph is not None:
            try:
                graph_stats = self.graph.stats()
            except Exception as e:
                log.warning("graph stats failed: %s", e)
                graph_stats = {}
        return {
            "version": _package_version(),
            "mode": self.cfg.mode,
            "modes": [m["id"] for m in MODE_CATALOG],
            "embedding": {"provider": self.cfg.embedding.provider, "model": self.cfg.embedding.model},
            "llm": {"provider": self.cfg.llm.provider, "model": self.cfg.llm.model},
            "rerank": {"provider": self.cfg.rerank.provider},
            "graph_backend": self.cfg.graph.backend if self.cfg.graph.enabled else "disabled",
            "lexical_docs": self.lexical.count(),
            "vector_chunks": self.vector.count(),
            "graph": graph_stats,
            "databases": list(self.cfg.databases.keys()),
            "cache_entries": self.cache.count() if self.cache is not None else 0,
            "memory": self.memory.stats() if self.memory is not None else {"sessions": 0, "turns": 0},
            "recall_entries": self.recall.count() if self.recall is not None else 0,
            "auth": bool(self.cfg.server.auth_token),
            "root": str(self.root),
        }

    def reset(self) -> None:
        self.lexical.clear()
        self.vector.clear()
        if self._graph is not None:
            self._graph.clear()
        if self._cache is not None:
            self._cache.clear()
        if self._memory is not None:
            self._memory.clear()
        if self._recall is not None:
            self._recall.clear()
        manifest = self.root / "manifest.jsonl"
        if manifest.exists():
            manifest.unlink()
        log.info("reset complete: %s", self.root)


def make_service(config_path: str | Path | None = None) -> RAGStack:
    try:
        return RAGStack(config_path)
    except ConfigError as e:
        log.error("configuration error: %s", e)
        raise

"""Query decomposition planner: split compound questions, retrieve in parallel, fuse."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from ..providers.llm import LLMProvider
from ..retrieval.vector_rag import hybrid_search, rrf
from ..types import RetrievedItem
from ..utils import get_logger

log = get_logger("ragstack.planner")

DECOMPOSE_PROMPT = """Break the question below into independent search queries (1-4).
Each sub-query must be answerable by searching a document corpus on its own.
Keep the original wording of key terms. If the question is already simple,
return just one query equal to the question.
Reply with ONLY JSON: {{"sub_queries": ["...", "..."]}}

QUESTION: {question}"""


def decompose(llm: LLMProvider, question: str) -> list[str]:
    try:
        result = llm.chat(
            [{"role": "user", "content": DECOMPOSE_PROMPT.format(question=question)}],
            temperature=0.0,
            max_tokens=300,
        )
        raw = result.content or ""
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return [question]
        data = json.loads(raw[start : end + 1])
        subs = [str(q).strip() for q in data.get("sub_queries", []) if str(q).strip()]
        if not subs:
            return [question]
        return subs[:4]
    except Exception as e:
        log.warning("decomposition failed (%s); using original question", e)
        return [question]


def decomposed_search(
    llm: LLMProvider,
    embeddings,
    vector_store,
    lexical_store,
    question: str,
    top_k: int = 8,
    reranker=None,
    workers: int = 3,
) -> tuple[list[str], list[RetrievedItem]]:
    """Returns (sub_queries_used, fused_items)."""
    sub_queries = decompose(llm, question)

    def run_one(sub: str) -> list[RetrievedItem]:
        return hybrid_search(embeddings, vector_store, lexical_store, sub, top_k=top_k * 2, reranker=None)

    if len(sub_queries) == 1:
        merged = run_one(sub_queries[0])
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(sub_queries))) as pool:
            lists = list(pool.map(run_one, sub_queries))
        merged = rrf(lists)

    if reranker is not None and merged:
        try:
            merged = reranker.rerank(question, merged, text_key="text")
        except Exception as e:
            log.warning("rerank after decomposition failed: %s", e)
    return sub_queries, merged[:top_k]


def format_decomposed(sub_queries: list[str], items: list[RetrievedItem], register) -> str:
    """register: callable RetrievedItem -> ref_id."""
    payload = {
        "sub_queries": sub_queries,
        "results": [
            {
                "ref": register(it),
                "score": round(it.score, 4),
                "source": it.source,
                "title": it.title,
                "text": it.text[:700],
            }
            for it in items
        ],
    }
    return json.dumps(payload, ensure_ascii=False, default=str)

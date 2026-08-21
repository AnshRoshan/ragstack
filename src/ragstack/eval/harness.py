"""Golden-set evaluation: retrieval metrics (hit@k, MRR) + LLM-judged faithfulness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from ..retrieval.vector_rag import hybrid_search
from ..service import RAGStack
from ..utils import get_logger

log = get_logger("ragstack.eval")
console = Console()

JUDGE_PROMPT = """Rate the faithfulness of the ANSWER to the CONTEXT on 0-5
(5 = every claim supported by context, 0 = contradicted or unsupported).
Reply with ONLY the integer.

CONTEXT:
{context}

ANSWER:
{answer}"""


def _load_golden(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cases", [])
    return [c for c in data if c.get("question")]


def run_eval(service: RAGStack, golden_path: str | Path, k: int = 8) -> dict[str, Any]:
    cases = _load_golden(Path(golden_path))
    if not cases:
        raise ValueError(f"no eval cases found in {golden_path}")

    hits = 0
    mrr_sum = 0.0
    faith_scores: list[int] = []
    rows = []

    for case in cases:
        q = case["question"]
        expected_docs = [e.lower() for e in case.get("expected_docs", [])]
        expected_keywords = [e.lower() for e in case.get("expected_keywords", [])]

        items = hybrid_search(
            service.embeddings, service.vector, service.lexical, q, top_k=k,
            reranker=service.reranker,
        )

        rank = None
        for i, item in enumerate(items, start=1):
            hay = f"{item.source} {item.title}".lower()
            if any(exp in hay for exp in expected_docs):
                rank = i
                break
        if expected_docs and rank is not None:
            hits += 1
        mrr_sum += (1.0 / rank) if rank else 0.0

        keyword_ok = True
        if expected_keywords:
            joined = " ".join(i.text.lower() for i in items)
            keyword_ok = all(kw in joined for kw in expected_keywords)

        faith = None
        if case.get("judge", False) and items:
            answer = service.query(q).text
            context = "\n\n".join(i.snippet(500) for i in items[:6])
            try:
                result = service.llm.chat(
                    [{"role": "user", "content": JUDGE_PROMPT.format(context=context, answer=answer)}],
                    temperature=0.0, max_tokens=8,
                )
                digits = "".join(ch for ch in (result.content or "") if ch.isdigit())
                faith = int(digits[0]) if digits else None
            except Exception as e:
                log.warning("judge failed: %s", e)
            if faith is not None:
                faith_scores.append(faith)

        rows.append((q, rank, keyword_ok, faith))

    n = len(cases)
    report = {
        "cases": n,
        "hit_rate": round(hits / max(1, len([c for c in cases if c.get("expected_docs")])), 3),
        "mrr": round(mrr_sum / n, 3),
        "keyword_pass_rate": round(sum(1 for _, _, ok, _ in rows if ok) / n, 3),
        "faithfulness_avg": round(sum(faith_scores) / len(faith_scores), 2) if faith_scores else None,
    }

    table = Table(title=f"Eval — {n} cases @ k={k}")
    table.add_column("question", max_width=50, overflow="fold")
    table.add_column("first-good-rank")
    table.add_column("keywords")
    table.add_column("faith")
    for q, rank, ok, faith in rows:
        table.add_row(q, str(rank) if rank else "-", "pass" if ok else "FAIL", str(faith) if faith is not None else "-")
    console.print(table)

    summary = Table(title="Summary")
    summary.add_column("metric", style="bold")
    summary.add_column("value")
    for key, value in report.items():
        summary.add_row(key, str(value))
    console.print(summary)
    return report

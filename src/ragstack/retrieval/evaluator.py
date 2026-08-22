"""CRAG-style evidence grader: judges whether retrieved evidence supports answering.

Actions (Yan et al., 2024 — Corrective RAG):
  correct     -> strong evidence present; proceed
  ambiguous   -> some possibly-relevant evidence; proceed with caution
  incorrect   -> nothing relevant; the agent should change strategy

Two graders:
  heuristic (default, free): uses reranker scores when available, else falls back
          to an honest "ambiguous" when it cannot judge.
  llm: one cheap LLM judgment over the concatenated snippets.

Also implements CRAG §4.4 knowledge-strip refinement: when a retrieval grades
"correct", chunks are split into sentence strips and only the query-relevant
strips are kept — fewer noise tokens reach the generator.
"""

from __future__ import annotations

import re
from typing import Any

from ..providers.llm import LLMProvider
from ..providers.reranker import Reranker
from ..types import RetrievedItem
from ..utils import get_logger, truncate

log = get_logger("ragstack.grader")

HINTS = {
    "correct": "",
    "ambiguous": (
        "EVIDENCE CHECK: ambiguous — results may be only partly relevant. "
        "Verify they actually answer the question before citing."
    ),
    "incorrect": (
        "EVIDENCE CHECK: insufficient — results look irrelevant. Try different "
        "keywords, another tool (graph_search, community_overview, sql_query), "
        "or tell the user what information is missing instead of guessing."
    ),
}

_LLM_JUDGE_PROMPT = """Rate how well the CONTEXT answers the QUESTION, 0-10.
0 = completely unrelated, 10 = directly and fully answers it.
Reply with ONLY the integer.

QUESTION: {question}

CONTEXT:
{context}"""

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class _Strip:
    """A single sentence strip carrying a pointer back to its parent item."""

    def __init__(self, parent: RetrievedItem, text: str):
        self.parent = parent
        self.text = text


class EvidenceGrader:
    def __init__(
        self,
        grader: str = "heuristic",
        llm: LLMProvider | None = None,
        reranker: Reranker | None = None,
        upper: float = 0.5,
        lower: float = 0.2,
        min_items: int = 1,
    ):
        self.grader = grader
        self.llm = llm
        self.reranker = reranker
        self.upper = upper
        self.lower = lower
        self.min_items = min_items

    def grade(self, question: str, items: list[RetrievedItem]) -> dict[str, Any]:
        if not items or len(items) < self.min_items:
            return {"action": "incorrect", "score": 0.0, "hint": HINTS["incorrect"]}

        if self.grader == "llm" and self.llm is not None:
            return self._grade_llm(question, items)
        return self._grade_heuristic(question, items)

    def refine(self, question: str, items: list[RetrievedItem], max_strips: int = 12) -> list[RetrievedItem]:
        """CRAG §4.4 knowledge-strip refinement. Returns items rebuilt from only
        their query-relevant sentences; falls back to the input unchanged when
        no cross-encoder is available."""
        if self.reranker is None or not items:
            return items
        try:
            strips: list[_Strip] = []
            for item in items[:6]:
                for sent in _SENTENCE_SPLIT.split(item.text):
                    if len(sent.strip()) > 30:
                        strips.append(_Strip(item, sent.strip()))
            if len(strips) <= max_strips:
                return items
            scored = self.reranker.rerank(question, strips, text_key="text")
            kept: dict[str, list[str]] = {}
            order: list[str] = []
            for strip in scored[:max_strips]:
                key = strip.parent.chunk_id or strip.parent.source
                if key not in kept:
                    kept[key] = []
                    order.append(key)
                kept[key].append(strip.text)
            refined: list[RetrievedItem] = []
            for item in items:
                key = item.chunk_id or item.source
                if kept.get(key):
                    refined.append(
                        RetrievedItem(
                            chunk_id=item.chunk_id,
                            doc_id=item.doc_id,
                            source=item.source,
                            title=item.title,
                            text=" … ".join(kept[key]),
                            score=item.score,
                            origin=item.origin,
                        )
                    )
            return refined or items
        except Exception as e:
            log.debug("strip refinement skipped (%s)", e)
            return items

    # -- heuristic -----------------------------------------------------------
    def _grade_heuristic(self, question: str, items: list[RetrievedItem]) -> dict[str, Any]:
        if self.reranker is not None:
            try:
                scored = self.reranker.rerank(question, items[:8], text_key="text")
                from sentence_transformers import CrossEncoder  # noqa: F401

                # re-score to get numbers: predict directly
                pairs = [[question, it.text[:4000]] for it in scored[:8]]
                scores = self.reranker._model.predict(pairs, show_progress_bar=False)
                top = float(max(scores))
            except Exception as e:
                log.debug("heuristic rerank scoring unavailable: %s", e)
                top = None
        else:
            top = None

        if top is None:
            # vector scores are cosine-ish [0,1]; BM25/RRF are unbounded — only trust vector
            vec_scores = [it.score for it in items if it.origin == "vector"]
            if vec_scores:
                top = max(vec_scores)
            else:
                return {
                    "action": "ambiguous",
                    "score": None,
                    "hint": HINTS["ambiguous"],
                }

        if top >= self.upper:
            action = "correct"
        elif top < self.lower:
            action = "incorrect"
        else:
            action = "ambiguous"
        return {"action": action, "score": round(top, 4), "hint": HINTS[action]}

    # -- llm -----------------------------------------------------------------
    def _grade_llm(self, question: str, items: list[RetrievedItem]) -> dict[str, Any]:
        context = "\n\n".join(f"[{i.snippet(400)}]" for i in items[:6])
        try:
            result = self.llm.chat(
                [{"role": "user", "content": _LLM_JUDGE_PROMPT.format(question=question, context=truncate(context, 6000))}],
                temperature=0.0,
                max_tokens=8,
            )
            digits = "".join(ch for ch in (result.content or "") if ch.isdigit())
            score = int(digits[0]) / 10.0 if digits else 0.5
        except Exception as e:
            log.warning("llm grader failed (%s); falling back to ambiguous", e)
            return {"action": "ambiguous", "score": None, "hint": HINTS["ambiguous"]}
        if score >= 0.7:
            action = "correct"
        elif score < 0.3:
            action = "incorrect"
        else:
            action = "ambiguous"
        return {"action": action, "score": score, "hint": HINTS[action]}

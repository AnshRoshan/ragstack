"""Query understanding + adaptive strategy routing (spec §7).

One cheap LLM call produces a structured query-understanding object that
selects the minimum sufficient retrieval strategy. Falls back to deterministic
heuristics whenever the LLM is unavailable or returns garbage.
"""

from __future__ import annotations

import json
import re

from ..config import AgentConfig
from ..providers.llm import LLMProvider
from ..utils import get_logger

log = get_logger("ragstack.intel")

VALID_ROUTES = {"vector", "lexical", "hybrid", "graph", "global", "sql", "agentic"}

CLASSIFY_PROMPT = """Classify this question for a retrieval system. Reply with ONLY JSON:

{{"intent": "factual|compare|aggregate|research|conversational|meta",
  "complexity": "simple|multi_hop|broad",
  "ambiguity": "none|low|high",
  "needs_clarification": true|false,
  "clarifying_question": null | "...",
  "suggested_mode": "vector|lexical|hybrid|graph|global|sql|agentic",
  "top_k": 5..14}}

Rules:
- compare/multi-part/dependent questions -> complexity multi_hop, mode agentic
- corpus-wide themes -> mode global; relationships between entities -> mode graph
- numbers/counts/statuses when databases are registered -> mode sql
- exact identifiers/error codes -> mode lexical; ordinary meaning questions -> hybrid
- high ambiguity about WHICH entity/document/time -> needs_clarification true
- top_k: 5 simple, 8 multi_hop, 12 broad

QUESTION: {question}
REGISTERED_DATABASES: {databases}"""

_AGGREGATE_RE = re.compile(r"\b(how many|total|sum|average|count|revenue|profit|number of)\b", re.I)
_COMPARE_RE = re.compile(r"\b(compare|versus|\bvs\b|difference between|which changed)\b", re.I)
_EXACT_RE = re.compile(r"(error code|[A-Z]{2,}-\d+|\bsku\b|section \d+|\bcite\b)", re.I)


def _heuristic(question: str, has_databases: bool) -> dict:
    q = question.strip()
    words = len(q.split())
    if _COMPARE_RE.search(q):
        return _pkg("compare", "multi_hop", "agentic", 8)
    if _AGGREGATE_RE.search(q) and has_databases:
        return _pkg("aggregate", "simple", "sql", 5)
    if _EXACT_RE.search(q):
        return _pkg("factual", "simple", "lexical", 5)
    if words <= 6:
        return _pkg("factual", "simple", "hybrid", 5)
    if words >= 22 or (q.count("?") >= 2 and words >= 14):
        return _pkg("research", "broad", "agentic", 12)
    return _pkg("factual", "simple", "hybrid", 8)


def _pkg(intent: str, complexity: str, mode: str, top_k: int, ambiguity: str = "none") -> dict:
    return {
        "intent": intent,
        "complexity": complexity,
        "ambiguity": ambiguity,
        "needs_clarification": False,
        "clarifying_question": None,
        "suggested_mode": mode,
        "top_k": top_k,
    }


def _parse_json(raw: str) -> dict | None:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def classify(
    llm: LLMProvider | None,
    question: str,
    databases: list[str] | None = None,
    cfg: AgentConfig | None = None,
) -> dict:
    """Structured query understanding. Never raises."""
    dbs = ", ".join(databases or []) or "(none)"
    if llm is not None:
        try:
            result = llm.chat(
                [{"role": "user", "content": CLASSIFY_PROMPT.format(question=question[:2000], databases=dbs)}],
                temperature=0.0,
                max_tokens=300,
            )
            data = _parse_json(result.content or "")
            if data:
                out = _heuristic(question, bool(databases))  # defaults for missing keys
                out.update({k: v for k, v in data.items() if v is not None})
                mode = str(out.get("suggested_mode", "hybrid")).lower()
                out["suggested_mode"] = mode if mode in VALID_ROUTES else "hybrid"
                try:
                    tk = int(out.get("top_k", 8))
                except (TypeError, ValueError):
                    tk = 8
                out["top_k"] = max(3, min(20, tk))
                out["needs_clarification"] = bool(out.get("needs_clarification"))
                return out
        except Exception as e:
            log.debug("classification failed (%s); heuristic fallback", e)
    return _heuristic(question, bool(databases))


def resolve_route(classification: dict, requested_mode: str | None, default_top_k: int) -> tuple[str, int]:
    """Returns (effective_mode, top_k). Explicit user mode always wins over suggestion."""
    if requested_mode and requested_mode != "auto":
        return requested_mode, default_top_k
    mode = classification.get("suggested_mode", "hybrid")
    if mode not in VALID_ROUTES:
        mode = "hybrid"
    return mode, int(classification.get("top_k") or default_top_k)

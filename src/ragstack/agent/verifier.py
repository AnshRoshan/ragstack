"""Atomic-claim verification and evidence manifests (spec §15-16).

After generation, the answer is decomposed into atomic claims, each claim is
checked against its cited evidence, and a machine-readable manifest is
attached to the response. Low support triggers abstention per policy.
"""

from __future__ import annotations

import json
import re

from ..types import Citation
from ..utils import get_logger

log = get_logger("ragstack.verify")

CLAIMS_PROMPT = """Decompose the answer below into its atomic factual claims.
Skip greetings, hedging, and statements about the answer itself.
For each claim list the citation refs [Sn] it depends on.
Reply with ONLY JSON: {{"claims": [{{"claim": "...", "refs": ["S1"]}}]}}

ANSWER:
{answer}"""

VERIFY_PROMPT = """For each claim, decide whether its cited evidence SUPPORTS it.
verdict: "supported" | "unsupported" | "partial"
Reply with ONLY JSON: {{"results": [{{"claim": "...", "verdict": "...", "reason": "..."}}]}}

EVIDENCE:
{evidence}

CLAIMS:
{claims}"""

_ABSTAIN_TEXT = (
    "**Insufficient verified evidence.**\n\n"
    "The retrieved sources did not sufficiently support the key claims in a draft answer "
    "({unsupported} of {total} claims unsupported), so rather than guess I am withholding it.\n\n"
    "Retrieved but inconclusive sources are listed below. Try narrowing the question, "
    "or ask me to show raw evidence for a specific sub-question."
)


def extract_claims(llm, answer: str) -> list[dict]:
    try:
        result = llm.chat(
            [{"role": "user", "content": CLAIMS_PROMPT.format(answer=answer[:6000])}],
            temperature=0.0,
            max_tokens=900,
        )
        data = _parse_json(result.content or "")
        claims = data.get("claims", []) if data else []
        return [
            {"claim": str(c.get("claim", "")).strip(), "refs": [str(r) for r in c.get("refs", [])]}
            for c in claims
            if isinstance(c, dict) and str(c.get("claim", "")).strip()
        ][:12]
    except Exception as e:
        log.debug("claim extraction failed (%s)", e)
        return []


def verify_claims(llm, answer: str, citations: list[Citation]) -> dict:
    """Returns {"claims": [...], "supported_ratio": float}. Empty manifest when unverifiable."""
    by_ref = {c.ref_id: c for c in citations}
    claims = extract_claims(llm, answer)
    if not claims or not citations:
        return {"claims": [], "supported_ratio": None}

    evidence_blocks = [
        f"[{ref}] {c.title} ({c.source})\n{c.snippet}"
        for ref, c in by_ref.items()
    ]
    try:
        result = llm.chat(
            [{
                "role": "user",
                "content": VERIFY_PROMPT.format(
                    evidence="\n\n".join(evidence_blocks)[:8000],
                    claims=json.dumps(claims, ensure_ascii=False),
                ),
            }],
            temperature=0.0,
            max_tokens=1200,
        )
        data = _parse_json(result.content or "")
        verdicts = {v.get("claim"): v for v in (data or {}).get("results", []) if isinstance(v, dict)}
    except Exception as e:
        log.warning("claim verification failed (%s); marking unverified", e)
        verdicts = {}

    manifest = []
    supported = 0
    judged = 0
    for c in claims:
        v = verdicts.get(c["claim"]) or {}
        verdict = str(v.get("verdict", "unverified")).lower()
        if verdict in ("supported", "partial", "unsupported"):
            judged += 1
            if verdict == "supported":
                supported += 1
        conf = {"supported": 0.9, "partial": 0.55, "unsupported": 0.1}.get(verdict, 0.5)
        manifest.append(
            {
                "claim": c["claim"],
                "refs": [r for r in c["refs"] if r in by_ref],
                "verdict": verdict,
                "confidence": conf,
                "reason": str(v.get("reason", ""))[:200],
            }
        )

    ratio = (supported / judged) if judged else None
    return {"claims": manifest, "supported_ratio": ratio}


def maybe_abstain(answer: str, manifest: dict, threshold: float) -> tuple[str, bool]:
    """Returns (final_answer, abstained)."""
    ratio = manifest.get("supported_ratio")
    if ratio is None or not manifest.get("claims"):
        return answer, False
    if ratio >= threshold:
        return answer, False
    unsupported = sum(1 for c in manifest["claims"] if c["verdict"] == "unsupported")
    total = len(manifest["claims"])
    return _ABSTAIN_TEXT.format(unsupported=unsupported, total=total), True


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

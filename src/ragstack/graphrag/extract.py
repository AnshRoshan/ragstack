"""LLM-based entity/relation extraction from chunks (GraphRAG indexing stage)."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from ..types import Chunk, ChunkExtraction, ExtractedEntity, ExtractedRelation
from ..utils import get_logger, sha1, truncate

log = get_logger("ragstack.graphrag.extract")

VALID_TYPES = {"person", "organization", "place", "event", "concept", "technology", "product", "date", "metric", "other"}

PROMPT = """Extract entities and relationships from the text below for a knowledge graph.

Rules:
- entities: 3-12 items. name is the canonical short name. type must be one of: {types}.
- relationships: 2-15 items connecting ONLY the extracted entities, using source/target names exactly as extracted.
- descriptions: one factual sentence each, grounded in the text.
- Reply with ONLY a JSON object, no markdown fences:

{{"entities": [{{"name": "...", "type": "...", "description": "..."}}],
  "relationships": [{{"source": "...", "target": "...", "relation": "verb_phrase", "description": "..."}}]}}

TITLE: {title}
TEXT:
{text}"""


def _parse_json_loose(raw: str) -> dict | None:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


def _normalize(data: dict) -> ChunkExtraction:
    out = ChunkExtraction()
    seen_names: set[str] = set()
    for e in data.get("entities", [])[:20]:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        if not name or len(name) > 120:
            continue
        etype = str(e.get("type", "other")).strip().lower()
        if etype not in VALID_TYPES:
            etype = "other"
        seen_names.add(name.lower())
        out.entities.append(
            ExtractedEntity(name=name, type=etype, description=str(e.get("description", "")).strip())
        )
    for r in data.get("relationships", [])[:25]:
        if not isinstance(r, dict):
            continue
        src, dst = str(r.get("source", "")).strip(), str(r.get("target", "")).strip()
        if not src or not dst or src.lower() not in seen_names or dst.lower() not in seen_names:
            continue
        if src.lower() == dst.lower():
            continue
        out.relationships.append(
            ExtractedRelation(
                source=src,
                target=dst,
                relation=str(r.get("relation", "related_to")).strip()[:80],
                description=str(r.get("description", "")).strip(),
            )
        )
    return out


def extract_from_chunk(llm, chunk: Chunk) -> ChunkExtraction:
    prompt = PROMPT.format(
        types=", ".join(sorted(VALID_TYPES)),
        title=chunk.metadata.get("title", ""),
        text=truncate(chunk.text, 6000),
    )
    try:
        result = llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=1500)
        data = _parse_json_loose(result.content or "")
        if data is None:
            return ChunkExtraction()
        return _normalize(data)
    except Exception as e:
        log.warning("extraction failed on chunk %s: %s", chunk.id, e)
        return ChunkExtraction()


def extraction_key(chunk: Chunk, model_name: str) -> str:
    return sha1(f"{chunk.text[:512]}|{model_name}|extract-v1")


def extract_chunks(llm, chunks: list[Chunk], cache=None, workers: int = 3, model_name: str = "?") -> list[ChunkExtraction]:
    keys = [extraction_key(c, model_name) for c in chunks]
    results: list[ChunkExtraction | None] = [None] * len(chunks)

    todo: list[int] = []
    for i, key in enumerate(keys):
        cached = cache.cache_get(key) if cache else None
        if cached:
            data = _parse_json_loose(cached)
            results[i] = _normalize(data) if data else ChunkExtraction()
        else:
            todo.append(i)

    def run(i: int) -> tuple[int, str]:
        ext = extract_from_chunk(llm, chunks[i])
        payload = json.dumps(
            {
                "entities": [asdict(e) for e in ext.entities],
                "relationships": [asdict(r) for r in ext.relationships],
            },
            ensure_ascii=False,
        )
        return i, payload

    if todo:
        log.info("extracting entities/relations from %d chunks (%d workers) …", len(todo), workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, payload in pool.map(run, todo):
                data = _parse_json_loose(payload)
                results[i] = _normalize(data) if data else ChunkExtraction()
                if cache:
                    cache.cache_put(keys[i], payload)

    return [r or ChunkExtraction() for r in results]

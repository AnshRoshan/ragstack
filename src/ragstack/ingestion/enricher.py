"""Contextual chunk enrichment: LLM writes a short situating context per chunk."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..config import IndexConfig
from ..types import Chunk, Document
from ..utils import get_logger, load_jsonl, sha1, truncate

log = get_logger("ragstack.enrich")

_PROMPT = (
    "You are preparing a document chunk for a retrieval index. Given the document title/metadata "
    "and the chunk, write ONE sentence (max 40 words) that situates the chunk within the whole "
    "document so it is findable by searchers who lack context. Reply with ONLY that sentence.\n\n"
    "DOCUMENT: {title}\nSOURCE: {source}\n\nCHUNK:\n{chunk}"
)


class Enricher:
    def __init__(self, llm, index_cfg: IndexConfig, workers: int = 3):
        self.llm = llm
        self.workers = workers
        self.cache_path = Path(index_cfg.root) / "enrich_cache.jsonl"
        self._cache: dict[str, str] = {}
        for row in load_jsonl(self.cache_path):
            self._cache[row["k"]] = row["v"]

    def _key(self, doc: Document, chunk: Chunk) -> str:
        return sha1(f"{doc.id}|{chunk.text[:512]}|enrich-v1")

    def enrich(self, doc: Document, chunks: list[Chunk]) -> list[Chunk]:
        keys = [self._key(doc, c) for c in chunks]
        todo = [(i, k) for i, k in enumerate(keys) if k not in self._cache]

        def run(item: tuple[int, str]) -> str:
            idx = item[0]
            c = chunks[idx]
            prompt = _PROMPT.format(
                title=doc.title, source=doc.source, chunk=truncate(c.text, 3000)
            )
            try:
                result = self.llm.chat(
                    [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=120
                )
                return (result.content or "").strip().strip('"')
            except Exception as e:
                log.warning("enrich failed for chunk %d: %s", idx, e)
                return ""

        if todo:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                results = list(pool.map(run, todo))
            new_rows = []
            for (_idx, key), value in zip(todo, results, strict=True):
                if value:
                    self._cache[key] = value
                    new_rows.append({"k": key, "v": value})
            if new_rows:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a", encoding="utf-8") as f:
                    for row in new_rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

        for c, key in zip(chunks, keys, strict=True):
            ctx = self._cache.get(key)
            if ctx:
                c.context = f"{c.context} {ctx}".strip()
        return chunks

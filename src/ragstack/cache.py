"""Semantic response cache: near-duplicate questions answered from a vector cache.

Design (per production practice, e.g. Higress 2026):
- store question embedding + full answer payload in LanceDB
- lookup: cosine similarity above threshold (default 0.95)
- dynamic thresholding: questions with uncertainty markers ("maybe", "possibly",
  "do you think") require a stricter match (default 0.98) or bypass the cache
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .providers.embeddings import EmbeddingProvider
from .utils import get_logger, now_iso

log = get_logger("ragstack.cache")

_UNCERTAINTY = re.compile(
    r"\b(maybe|possibly|perhaps|not sure|unsure|guess|think|might|could be|opinion)\b",
    re.IGNORECASE,
)


class SemanticCache:
    def __init__(
        self,
        root: Path,
        embeddings: EmbeddingProvider,
        threshold: float = 0.95,
        fuzzy_threshold: float = 0.98,
    ):
        self.dir = Path(root) / "qcache"
        self.embeddings = embeddings
        self.threshold = threshold
        self.fuzzy_threshold = fuzzy_threshold
        self._table = None

    def _ensure_table(self):
        if self._table is not None:
            return self._table
        import lancedb
        import pyarrow as pa

        db = lancedb.connect(str(self.dir))
        schema = pa.schema(
            [
                pa.field("question", pa.string()),
                pa.field("mode", pa.string()),
                pa.field("payload", pa.string()),
                pa.field("ts", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.embeddings.dim)),
            ]
        )
        self._table = db.create_table("answers", schema=schema, mode="create", exist_ok=True)
        return self._table

    def _threshold_for(self, question: str) -> float:
        return self.fuzzy_threshold if _UNCERTAINTY.search(question or "") else self.threshold

    def lookup(self, question: str, mode: str = "auto") -> dict[str, Any] | None:
        try:
            if self.count() == 0:
                return None
            qvec = np.asarray(self.embeddings.embed([question])[0], dtype=np.float32)
            rows = (
                self._ensure_table()
                .search(qvec)
                .where(f"mode = '{mode}'", prefilter=True)
                .limit(1)
                .to_list()
            )
            if not rows:
                return None
            row = rows[0]
            sim = 1.0 - float(row["_distance"]) / 2.0
            if sim < self._threshold_for(question):
                return None
            payload = json.loads(row["payload"])
            payload["cached"] = True
            payload["cache_similarity"] = round(sim, 4)
            log.info("semantic cache HIT sim=%.3f for %r", sim, question[:60])
            return payload
        except Exception as e:
            log.debug("cache lookup failed: %s", e)
            return None

    def store(
        self, question: str, mode: str, answer: str, citations: list[dict], steps: list[dict]
    ) -> None:
        try:
            qvec = np.asarray(self.embeddings.embed([question])[0], dtype=np.float32)
            payload = json.dumps(
                {"answer": answer, "citations": citations, "steps": steps}, ensure_ascii=False
            )
            self._ensure_table().add(
                [{"question": question, "mode": mode, "payload": payload, "ts": now_iso(), "vector": qvec}]
            )
        except Exception as e:
            log.debug("cache store failed: %s", e)

    def count(self) -> int:
        if self._table is None and not self.dir.exists():
            return 0
        try:
            return int(self._ensure_table().count_rows())
        except Exception:
            return 0

    def clear(self) -> None:
        self._table = None
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)

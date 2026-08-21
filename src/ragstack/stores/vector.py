"""Vector index over chunks — LanceDB (embedded, scales to millions)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import StoreError
from ..utils import get_logger

log = get_logger("ragstack.vector")

_SCHEMA_META = "meta.json"


class VectorStore:
    def __init__(self, root: Path):
        self.dir = Path(root) / "vector"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._table = None

    # -- meta ----------------------------------------------------------------
    @property
    def meta_path(self) -> Path:
        return self.dir / _SCHEMA_META

    def get_meta(self) -> dict[str, Any]:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        return {}

    def set_meta(self, **kwargs: Any) -> None:
        meta = self.get_meta()
        meta.update(kwargs)
        self.meta_path.write_text(json.dumps(meta), encoding="utf-8")

    # -- table ---------------------------------------------------------------
    def _ensure_table(self, dim: int):
        if self._table is not None:
            return self._table
        import lancedb
        import pyarrow as pa

        db = lancedb.connect(str(self.dir))
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("doc_id", pa.string()),
                pa.field("ordinal", pa.int32()),
                pa.field("title", pa.string()),
                pa.field("source", pa.string()),
                pa.field("text", pa.string()),
                pa.field("context", pa.string()),
                pa.field("meta", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ]
        )
        self._table = db.create_table("chunks", schema=schema, mode="create", exist_ok=True)
        stored = self.get_meta()
        if stored and stored.get("dim") not in (None, dim):
            raise StoreError(
                f"vector dim mismatch: store has {stored.get('dim')}, provider gives {dim}. "
                "Run `ragstack reset` or switch back to the original embedding model."
            )
        if not stored.get("dim"):
            self.set_meta(dim=dim)
        return self._table

    # -- ops -----------------------------------------------------------------
    def add(self, rows: list[dict[str, Any]], vectors: np.ndarray) -> None:
        if not rows:
            return
        table = self._ensure_table(int(vectors.shape[1]))
        payload = []
        for row, vec in zip(rows, vectors):
            payload.append({**row, "vector": np.asarray(vec, dtype=np.float32)})
        table.add(payload)

    def delete_doc(self, doc_id: str) -> None:
        if self._table is None and not (self.dir / "chunks.lance").exists():
            return
        try:
            self._table.delete(f"doc_id = '{doc_id}'")
        except Exception as e:
            log.debug("delete_doc %s: %s", doc_id, e)

    def search(self, vector: np.ndarray, top_k: int = 10) -> list[dict[str, Any]]:
        if self.count() == 0:
            return []
        vec = np.asarray(vector, dtype=np.float32)
        results = (
            self._table.search(vec)
            .limit(top_k)
            .to_list()
        )
        out = []
        for r in results:
            dist = float(r.pop("_distance"))
            r["score"] = max(0.0, 1.0 - dist / 2.0)
            out.append(r)
        return out

    def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if self.count() == 0 or not ids:
            return []
        found: dict[str, dict[str, Any]] = {}
        chunk_size = 200
        for i in range(0, len(ids), chunk_size):
            batch = ids[i : i + chunk_size]
            quoted = ", ".join(f"'{x}'" for x in batch)
            try:
                rows = self._table.search().where(f"id IN ({quoted})", prefilter=True).limit(len(batch)).to_list()
            except Exception as e:
                log.debug("get_by_ids batch failed: %s", e)
                continue
            for r in rows:
                r.pop("_distance", None)
                r.pop("vector", None)
                found[r["id"]] = r
        return [found[i] for i in ids if i in found]

    def count(self) -> int:
        if self._table is None:
            import lancedb

            db = lancedb.connect(str(self.dir))
            try:
                self._table = db.open_table("chunks")
            except Exception:
                return 0
        try:
            return int(self._table.count_rows())
        except Exception:
            return 0

    def clear(self) -> None:
        self._table = None
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)

"""Vectorless lexical index over pages and chunks — tantivy BM25."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import tantivy

from ..errors import StoreError
from ..utils import get_logger

log = get_logger("ragstack.lexical")

_SCHEMA_FIELDS = ("id", "kind", "doc_id", "title", "body", "source", "meta")
_BAD_QUERY_CHARS = re.compile(r'[\[\]{}()^"~*?:\\/&|!+-]')


class LexicalStore:
    def __init__(self, root: Path):
        self.dir = Path(root) / "lexical"
        self._index: tantivy.Index | None = None

    def _schema(self) -> tantivy.Schema:
        b = tantivy.SchemaBuilder()
        b.add_text_field("id", stored=True, tokenizer_name="raw")
        b.add_text_field("kind", stored=True, tokenizer_name="raw")
        b.add_text_field("doc_id", stored=True, tokenizer_name="raw")
        b.add_text_field("title", stored=True)
        b.add_text_field("body", stored=True)
        b.add_text_field("source", stored=True, tokenizer_name="raw")
        b.add_text_field("meta", stored=True, tokenizer_name="raw")
        return b.build()

    def _open(self) -> tantivy.Index:
        if self._index is not None:
            return self._index
        if self.dir.exists() and tantivy.Index.exists(str(self.dir)):
            try:
                self._index = tantivy.Index.open(str(self.dir))
                return self._index
            except Exception as e:
                raise StoreError(f"corrupt lexical index at {self.dir}; run `ragstack reset` ({e})") from e
        self.dir.mkdir(parents=True, exist_ok=True)
        self._index = tantivy.Index(self._schema(), path=str(self.dir))
        self._index.writer().commit()
        return self._index

    def add(self, docs: list[dict[str, Any]]) -> None:
        """Each doc: id, kind(page|chunk), doc_id, title, body, source, meta(json str)."""
        if not docs:
            return
        index = self._open()
        writer = index.writer(heap_size=80_000_000)
        for d in docs:
            writer.add_document(
                tantivy.Document(
                    id=[str(d["id"])],
                    kind=[str(d.get("kind", "chunk"))],
                    doc_id=[str(d.get("doc_id", ""))],
                    title=[str(d.get("title", ""))],
                    body=[str(d.get("body", ""))],
                    source=[str(d.get("source", ""))],
                    meta=[str(d.get("meta", "{}"))],
                )
            )
        writer.commit()
        writer.wait_merging_threads()
        index.reload()

    def delete_doc(self, doc_id: str) -> int:
        index = self._open()
        writer = index.writer()
        before = index.searcher().num_docs
        writer.delete_documents_by_term("doc_id", doc_id)
        writer.commit()
        writer.wait_merging_threads()
        index.reload()
        return before - index.searcher().num_docs

    def search(self, query: str, kind: str | None = None, top_k: int = 10) -> list[dict[str, Any]]:
        index = self._open()
        searcher = index.searcher()
        q = _BAD_QUERY_CHARS.sub(" ", query).strip()
        if not q:
            return []
        try:
            parsed = index.parse_query(q, ["title", "body"])
        except Exception:
            words = " ".join(re.findall(r"\w{3,}", q))
            if not words:
                return []
            parsed = index.parse_query(words, ["title", "body"])
        if kind:
            try:
                parsed = index.parse_query(f"kind:{kind} AND ({q})", ["title", "body", "kind"])
            except Exception as e:
                log.debug("kind filter fell back to unfiltered query (%s)", e)
        hits = searcher.search(parsed, top_k).hits
        out: list[dict[str, Any]] = []
        for score, addr in hits:
            doc = searcher.doc(addr).to_dict()
            out.append(
                {
                    "id": doc["id"][0],
                    "kind": doc["kind"][0],
                    "doc_id": doc["doc_id"][0],
                    "title": doc["title"][0] if doc.get("title") else "",
                    "text": doc["body"][0] if doc.get("body") else "",
                    "source": doc["source"][0] if doc.get("source") else "",
                    "score": float(score),
                }
            )
        return out

    def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        index = self._open()
        searcher = index.searcher()
        found: dict[str, dict[str, Any]] = {}
        for cid in ids:
            try:
                q = index.parse_query(f"id:{cid}", ["id"])
                hits = searcher.search(q, 1).hits
                if not hits:
                    continue
                doc = searcher.doc(hits[0][1]).to_dict()
                found[cid] = {
                    "id": cid,
                    "kind": doc["kind"][0],
                    "doc_id": doc["doc_id"][0],
                    "title": doc["title"][0] if doc.get("title") else "",
                    "text": doc["body"][0] if doc.get("body") else "",
                    "source": doc["source"][0] if doc.get("source") else "",
                    "score": 0.0,
                }
            except Exception as e:
                log.debug("get_by_ids missed %s (%s)", cid, e)
                continue
        return [found[i] for i in ids if i in found]

    def count(self) -> int:
        if self._index is None and not self.dir.exists():
            return 0
        return self._open().searcher().num_docs

    def clear(self) -> None:
        self._index = None
        if self.dir.exists():
            shutil.rmtree(self.dir, ignore_errors=True)

"""Embedded graph store on SQLite — zero infrastructure, recursive-CTE capable."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from ...types import ChunkExtraction
from ...utils import get_logger, norm_name, sha1
from .base import GraphStore

log = get_logger("ragstack.graph.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities(
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    name_norm TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'concept',
    description TEXT NOT NULL DEFAULT '',
    degree INTEGER NOT NULL DEFAULT 0,
    UNIQUE(name_norm, type)
);
CREATE TABLE IF NOT EXISTS relations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    rel TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    chunk_id TEXT NOT NULL DEFAULT '',
    doc_id TEXT NOT NULL DEFAULT '',
    weight REAL NOT NULL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_rel_src ON relations(src_id);
CREATE INDEX IF NOT EXISTS idx_rel_dst ON relations(dst_id);
CREATE TABLE IF NOT EXISTS chunk_entities(
    chunk_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY(chunk_id, entity_id)
);
CREATE TABLE IF NOT EXISTS communities(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    summary TEXT NOT NULL DEFAULT '',
    keywords TEXT NOT NULL DEFAULT '[]',
    member_ids TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS cache(
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);
"""


class SQLiteGraphStore(GraphStore):
    backend = "sqlite"

    def __init__(self, root: Path):
        self.path = Path(root) / "graph.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _entity_id(name: str, etype: str) -> str:
        return "e" + sha1(f"{norm_name(name)}|{etype}")[:20]

    # -- writes --------------------------------------------------------------
    def upsert_extraction(self, extraction: ChunkExtraction, chunk_id: str, doc_id: str) -> None:
        with self._lock:
            cur = self._conn.cursor()
            id_map: dict[str, str] = {}
            for ent in extraction.entities:
                eid = self._entity_id(ent.name, ent.type)
                id_map[norm_name(ent.name)] = eid
                cur.execute(
                    """
                    INSERT INTO entities(id, name, name_norm, type, description, degree)
                    VALUES(?,?,?,?,?,1)
                    ON CONFLICT(name_norm, type) DO UPDATE SET
                        degree = degree + 1,
                        description = CASE
                            WHEN length(excluded.description) > length(entities.description)
                            THEN excluded.description ELSE entities.description END
                    """,
                    (eid, ent.name.strip(), norm_name(ent.name), ent.type, ent.description or ""),
                )
            for rel in extraction.relationships:
                src = id_map.get(norm_name(rel.source))
                dst = id_map.get(norm_name(rel.target))
                if not src or not dst or src == dst:
                    continue
                cur.execute(
                    """
                    INSERT INTO relations(src_id, dst_id, rel, description, chunk_id, doc_id, weight)
                    VALUES(?,?,?,?,?,?,1.0)
                    """,
                    (src, dst, rel.relation, rel.description or "", chunk_id, doc_id),
                )
            for eid in set(id_map.values()):
                cur.execute(
                    "INSERT OR IGNORE INTO chunk_entities(chunk_id, entity_id) VALUES(?,?)",
                    (chunk_id, eid),
                )
            self._conn.commit()

    def save_communities(self, communities: list[dict[str, Any]]) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM communities")
            for c in communities:
                cur.execute(
                    "INSERT INTO communities(summary, keywords, member_ids) VALUES(?,?,?)",
                    (
                        c.get("summary", ""),
                        json.dumps(c.get("keywords", [])),
                        json.dumps(c.get("member_ids", [])),
                    ),
                )
            self._conn.commit()

    def cache_put(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache(k, v) VALUES(?,?)", (key, value)
            )
            self._conn.commit()

    # -- reads ---------------------------------------------------------------
    def neighborhood(self, query: str, limit: int = 25, max_hops: int = 2) -> dict[str, Any]:
        tokens = [t for t in norm_name(query).split() if len(t) >= 3][:6]
        with self._lock:
            cur = self._conn.cursor()
            seeds: list[dict] = []
            seen_ids: set[str] = set()
            for tok in tokens:
                rows = cur.execute(
                    """
                    SELECT id, name, type, description, degree FROM entities
                    WHERE name_norm LIKE ? ORDER BY degree DESC LIMIT ?
                    """,
                    (f"%{tok}%", limit // max(1, len(tokens))),
                ).fetchall()
                for r in rows:
                    if r[0] not in seen_ids:
                        seen_ids.add(r[0])
                        seeds.append({"id": r[0], "name": r[1], "type": r[2], "description": r[3], "degree": r[4]})
            if not seeds:
                return {"seeds": [], "entities": [], "relations": [], "chunk_ids": []}

            frontier = [s["id"] for s in seeds]
            entity_rows: dict[str, tuple] = {s["id"]: None for s in seeds}
            relation_rows: list[tuple] = []
            for _ in range(max(1, max_hops)):
                if not frontier:
                    break
                marks = ",".join("?" * len(frontier))
                rels = cur.execute(
                    f"SELECT id, src_id, dst_id, rel, description, chunk_id, weight "
                    f"FROM relations WHERE src_id IN ({marks}) OR dst_id IN ({marks}) LIMIT 400",
                    (*frontier, *frontier),
                ).fetchall()
                new_frontier: list[str] = []
                known = set(entity_rows.keys())
                for rid, src, dst, rel, desc, cid, w in rels:
                    relation_rows.append((rid, src, dst, rel, desc, cid, w))
                    for node in (dst if src in known else src,):
                        if node not in known:
                            known.add(node)
                            new_frontier.append(node)
                            entity_rows[node] = None
                frontier = new_frontier[:60]

            missing = [eid for eid, row in entity_rows.items() if row is None]
            if missing:
                marks = ",".join("?" * len(missing))
                for eid, name, etype, desc, deg in cur.execute(
                    f"SELECT id, name, type, description, degree FROM entities WHERE id IN ({marks})",
                    missing,
                ).fetchall():
                    entity_rows[eid] = (eid, name, etype, desc, deg)

            entities = [
                {"id": r[0], "name": r[1], "type": r[2], "description": r[3], "degree": r[4]}
                for r in entity_rows.values()
                if r is not None
            ]
            chunk_ids: list[str] = []
            for r in relation_rows:
                if r[5] and r[5] not in chunk_ids:
                    chunk_ids.append(r[5])
            seed_chunk_ids = [
                row[0]
                for row in cur.execute(
                    f"SELECT DISTINCT chunk_id FROM chunk_entities WHERE entity_id IN "
                    f"({','.join('?' * len(seeds))})",
                    [s["id"] for s in seeds],
                ).fetchall()
            ]
            for cid in seed_chunk_ids:
                if cid not in chunk_ids:
                    chunk_ids.append(cid)
            return {
                "seeds": seeds,
                "entities": entities,
                "relations": [
                    {"id": r[0], "src": r[1], "dst": r[2], "rel": r[3], "description": r[4], "chunk_id": r[5]}
                    for r in relation_rows
                ],
                "chunk_ids": chunk_ids[:40],
            }

    def all_entities(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, type, description, degree FROM entities"
            ).fetchall()
        return [{"id": r[0], "name": r[1], "type": r[2], "description": r[3], "degree": r[4]} for r in rows]

    def all_relations(self) -> list[tuple[str, str, float]]:
        with self._lock:
            rows = self._conn.execute("SELECT src_id, dst_id, SUM(weight) FROM relations GROUP BY src_id, dst_id").fetchall()
        return [(r[0], r[1], float(r[2])) for r in rows]

    def community_summaries(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT id, summary, keywords, member_ids FROM communities").fetchall()
        return [
            {
                "id": r[0],
                "summary": r[1],
                "keywords": json.loads(r[2]),
                "member_ids": json.loads(r[3]),
            }
            for r in rows
        ]

    def cache_get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
        return row[0] if row else None

    def stats(self) -> dict[str, int]:
        with self._lock:
            e = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            r = self._conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
            c = self._conn.execute("SELECT COUNT(*) FROM communities").fetchone()[0]
        return {"entities": e, "relations": r, "communities": c}

    def clear(self) -> None:
        with self._lock:
            self._conn.executescript(
                "DELETE FROM relations; DELETE FROM chunk_entities; DELETE FROM entities; DELETE FROM communities;"
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

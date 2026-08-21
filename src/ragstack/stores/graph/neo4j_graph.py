"""Optional Neo4j backend implementing the same GraphStore interface.

Requires: pip install ragstack[neo4j]  (or pip install neo4j)
Configure in ragstack.yaml:
    graph:
      backend: neo4j
      uri: bolt://localhost:7687
      user: neo4j
      password_env: NEO4J_PASSWORD
"""

from __future__ import annotations

import json
import os
from typing import Any

from ...errors import ProviderError, StoreError
from ...types import ChunkExtraction
from ...utils import get_logger, norm_name, sha1
from .base import GraphStore

log = get_logger("ragstack.graph.neo4j")


class Neo4jGraphStore(GraphStore):
    backend = "neo4j"

    def __init__(self, uri: str, user: str, password_env: str):
        try:
            from neo4j import GraphDatabase
        except ImportError as e:
            raise ProviderError(
                f"neo4j driver missing ({e}); install with: pip install neo4j"
            ) from e
        password = os.environ.get(password_env)
        if not password:
            raise StoreError(f"set {password_env} in the environment for Neo4j auth")
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()
        self._ensure_constraints()

    def _ensure_constraints(self) -> None:
        with self._driver.session() as s:
            s.run(
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE"
            ).consume()
            s.run("CREATE INDEX entity_norm IF NOT EXISTS FOR (e:Entity) ON (e.name_norm)").consume()

    @staticmethod
    def _entity_id(name: str, etype: str) -> str:
        return "e" + sha1(f"{norm_name(name)}|{etype}")[:20]

    def upsert_extraction(self, extraction: ChunkExtraction, chunk_id: str, doc_id: str) -> None:
        ents = [
            {
                "id": self._entity_id(e.name, e.type),
                "name": e.name.strip(),
                "name_norm": norm_name(e.name),
                "type": e.type,
                "description": e.description or "",
            }
            for e in extraction.entities
        ]
        name_to_id = {e["name_norm"]: e["id"] for e in ents}
        rels = [
            {
                "src": name_to_id[norm_name(r.source)],
                "dst": name_to_id[norm_name(r.target)],
                "rel": r.relation,
                "description": r.description or "",
                "chunk_id": chunk_id,
                "doc_id": doc_id,
            }
            for r in extraction.relationships
            if norm_name(r.source) in name_to_id and norm_name(r.target) in name_to_id
        ]
        with self._driver.session() as s:
            if ents:
                s.run(
                    """
                    UNWIND $ents AS e
                    MERGE (n:Entity {id: e.id})
                    SET n.name = e.name, n.name_norm = e.name_norm, n.type = e.type,
                        n.degree = coalesce(n.degree, 0) + 1,
                        n.description = CASE WHEN size(e.description) > size(coalesce(n.description,''))
                            THEN e.description ELSE n.description END
                    """,
                    ents=ents,
                ).consume()
            if rels:
                s.run(
                    """
                    UNWIND $rels AS r
                    MATCH (a:Entity {id: r.src}), (b:Entity {id: r.dst})
                    MERGE (a)-[k:RELATES {rel: r.rel}]->(b)
                    SET k.weight = coalesce(k.weight, 0) + 1,
                        k.description = CASE WHEN size(r.description) > size(coalesce(k.description,''))
                            THEN r.description ELSE k.description END,
                        k.last_chunk = r.chunk_id
                    WITH a, b, k
                    MERGE (a)-[:EVIDENCE]->(c:ChunkRef {id: r.chunk_id})
                    MERGE (b)-[:EVIDENCE]->(c)
                    """,
                    rels=rels,
                ).consume()

    def neighborhood(self, query: str, limit: int = 25, max_hops: int = 2) -> dict[str, Any]:
        tokens = [t for t in norm_name(query).split() if len(t) >= 3][:6]
        if not tokens:
            return {"seeds": [], "entities": [], "relations": [], "chunk_ids": []}
        with self._driver.session() as s:
            result = s.run(
                """
                UNWIND $tokens AS tok
                MATCH (e:Entity) WHERE e.name_norm CONTAINS tok
                WITH DISTINCT e LIMIT $limit
                RETURN e.id AS id
                """,
                tokens=tokens,
                limit=limit,
            )
            seed_ids = [r["id"] for r in result]
        if not seed_ids:
            return {"seeds": [], "entities": [], "relations": [], "chunk_ids": []}

        hops = max(1, min(int(max_hops), 3))
        entities_by_id: dict[str, dict] = {}
        relations: list[dict] = []
        chunk_ids: list[str] = []
        with self._driver.session() as s:
            record = s.run(
                f"""
                MATCH (seed:Entity) WHERE seed.id IN $seeds
                OPTIONAL MATCH (seed)-[rels:RELATES*1..{hops}]-(other:Entity)
                WITH seed, other, rels LIMIT 600
                RETURN collect(DISTINCT seed) AS seeds_n, collect(DISTINCT other) AS others,
                       collect(rels) AS relists
                """,
                seeds=seed_ids,
            ).single()
        if record:
            for n in (record["seeds_n"] or []) + (record["others"] or []):
                if n is None:
                    continue
                entities_by_id[n["id"]] = {
                    "id": n["id"],
                    "name": n["name"],
                    "type": n.get("type", ""),
                    "description": n.get("description", ""),
                    "degree": n.get("degree", 0),
                }
            seen_rel = set()
            for lst in record["relists"] or []:
                if not lst:
                    continue
                for r in lst:
                    key = r.element_id
                    if key in seen_rel:
                        continue
                    seen_rel.add(key)
                    cid = r.get("last_chunk", "")
                    relations.append(
                        {
                            "id": key,
                            "src": r.start_node["id"],
                            "dst": r.end_node["id"],
                            "rel": r["rel"],
                            "description": r.get("description", ""),
                            "chunk_id": cid,
                        }
                    )
                    if cid and cid not in chunk_ids:
                        chunk_ids.append(cid)
        entities = list(entities_by_id.values())
        return {"seeds": entities[:limit], "entities": entities, "relations": relations, "chunk_ids": chunk_ids[:40]}

    def all_entities(self) -> list[dict[str, Any]]:
        with self._driver.session() as s:
            rows = s.run("MATCH (e:Entity) RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description, e.degree AS degree").data()
        return rows

    def all_relations(self) -> list[tuple[str, str, float]]:
        with self._driver.session() as s:
            rows = s.run("MATCH (a:Entity)-[r:RELATES]->(b:Entity) RETURN a.id AS src, b.id AS dst, r.weight AS w").data()
        agg: dict[tuple[str, str], float] = {}
        for r in rows:
            key = (r["src"], r["dst"])
            agg[key] = agg.get(key, 0.0) + float(r.get("w") or 1.0)
        return [(a, b, w) for (a, b), w in agg.items()]

    def save_communities(self, communities: list[dict[str, Any]]) -> None:
        with self._driver.session() as s:
            s.run("MATCH (c:Community) DETACH DELETE c").consume()
            for i, c in enumerate(communities):
                s.run(
                    """
                    CREATE (cm:Community {id: $id, summary: $summary, keywords: $keywords})
                    WITH cm
                    UNWIND $members AS mid
                    MATCH (e:Entity {id: mid})
                    MERGE (cm)-[:MEMBER]->(e)
                    """,
                    id=i,
                    summary=c.get("summary", ""),
                    keywords=json.dumps(c.get("keywords", [])),
                    members=c.get("member_ids", []),
                ).consume()

    def community_summaries(self) -> list[dict[str, Any]]:
        with self._driver.session() as s:
            rows = s.run("MATCH (c:Community) RETURN c.id AS id, c.summary AS summary, c.keywords AS keywords, [(c)-[:MEMBER]->(e) | e.id] AS member_ids").data()
        return [{"id": r["id"], "summary": r["summary"], "keywords": json.loads(r["keywords"]), "member_ids": r["member_ids"]} for r in rows]

    def cache_get(self, key: str) -> str | None:
        with self._driver.session() as s:
            row = s.run("MATCH (c:Cache {k: $k}) RETURN c.v AS v", k=key).single()
        return row["v"] if row else None

    def cache_put(self, key: str, value: str) -> None:
        with self._driver.session() as s:
            s.run("MERGE (c:Cache {k: $k}) SET c.v = $v", k=key, v=value).consume()

    def stats(self) -> dict[str, int]:
        with self._driver.session() as s:
            e = s.run("MATCH (x:Entity) RETURN count(x) AS n").single()["n"]
            r = s.run("MATCH ()-[k:RELATES]->() RETURN count(k) AS n").single()["n"]
            c = s.run("MATCH (x:Community) RETURN count(x) AS n").single()["n"]
        return {"entities": e, "relations": r, "communities": c}

    def clear(self) -> None:
        with self._driver.session() as s:
            s.run("MATCH (n) WHERE n:Entity OR n:Community OR n:Cache OR n:ChunkRef DETACH DELETE n").consume()

    def close(self) -> None:
        self._driver.close()

"""GraphRAG query-time: local entity-neighborhood search and global community search."""

from __future__ import annotations

from ..types import RetrievedItem
from ..utils import get_logger, truncate

log = get_logger("ragstack.graphrag.search")


def format_local_context(neighborhood: dict) -> str:
    lines = ["KNOWLEDGE GRAPH CONTEXT:"]
    seeds = {s["id"] for s in neighborhood.get("seeds", [])}
    for e in neighborhood.get("entities", [])[:30]:
        mark = "*" if e["id"] in seeds else ""
        lines.append(f"ENTITY{mark}: {e['name']} [{e['type']}] — {truncate(e['description'], 220)}")
    seen_rel: set[tuple] = set()
    for r in neighborhood.get("relations", [])[:40]:
        key = (r["src"], r["rel"], r["dst"])
        if key in seen_rel:
            continue
        seen_rel.add(key)
        desc = f" — {truncate(r['description'], 160)}" if r.get("description") else ""
        lines.append(f"RELATION: {r['src']} -[{r['rel']}]-> {r['dst']}{desc}")
    return "\n".join(lines)


def local_search(graph_store, vector_store, query: str, top_k: int = 8, max_hops: int = 2) -> tuple[str, list[RetrievedItem]]:
    nb = graph_store.neighborhood(query, limit=25, max_hops=max_hops)
    context = format_local_context(nb)
    items: list[RetrievedItem] = []
    chunk_ids = nb.get("chunk_ids", [])
    if chunk_ids and vector_store is not None:
        rows = vector_store.get_by_ids(chunk_ids[: top_k * 2])
        for row in rows[:top_k]:
            items.append(
                RetrievedItem(
                    chunk_id=row["id"],
                    doc_id=row["doc_id"],
                    source=row["source"],
                    title=row["title"],
                    text=row["text"],
                    score=0.0,
                    origin="graph",
                )
            )
    return context, items


def global_search(graph_store, llm, query: str, top_k: int = 4) -> tuple[str, list[RetrievedItem]]:
    communities = graph_store.community_summaries()
    if not communities:
        return "", []

    q_tokens = set(query.lower().split())
    scored = []
    for c in communities:
        text = (c["summary"] + " " + " ".join(c["keywords"])).lower()
        overlap = sum(1 for t in q_tokens if t in text)
        scored.append((overlap, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for score, c in scored[:top_k]]

    map_parts = []
    items: list[RetrievedItem] = []
    for i, c in enumerate(top):
        map_parts.append(f"COMMUNITY {i + 1}:\n{c['summary']}")
        items.append(
            RetrievedItem(
                chunk_id=f"community:{c['id']}",
                doc_id="",
                source="knowledge-graph/community",
                title=f"Community {i + 1}",
                text=c["summary"],
                score=float(scored[i][0]),
                origin="community",
            )
        )

    reduce_prompt = (
        "Use the community summaries below to answer the question. Summarize relevant points only.\n\n"
        f"QUESTION: {query}\n\n" + "\n\n".join(map_parts)
    )
    try:
        result = llm.chat([{"role": "user", "content": reduce_prompt}], temperature=0.1, max_tokens=800)
        context = "GLOBAL GRAPH ANSWER:\n" + (result.content or "")
    except Exception as e:
        log.warning("global reduce failed: %s", e)
        context = "GLOBAL GRAPH SUMMARIES:\n" + "\n\n".join(map_parts)
    return context, items

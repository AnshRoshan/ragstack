"""Community detection (Louvain) + LLM community summaries for global search."""

from __future__ import annotations

import networkx as nx

from ..utils import get_logger, truncate
from .extract import _parse_json_loose

log = get_logger("ragstack.graphrag.communities")

SUMMARY_PROMPT = """These entities form a community in a knowledge graph. Write a dense summary
(3-6 sentences) of the shared theme, key facts and how members relate, so someone can answer
questions about this theme from the summary alone. Then list 5-10 keywords.
Reply with ONLY JSON: {{"summary": "...", "keywords": ["...", ...]}}

ENTITIES:
{entities}"""


def build_communities(graph_store, llm, min_size: int = 3) -> list[dict]:
    entities = graph_store.all_entities()
    relations = graph_store.all_relations()
    if not entities:
        log.info("graph empty; skipping communities")
        return []

    G = nx.Graph()
    by_id = {}
    for e in entities:
        G.add_node(e["id"])
        by_id[e["id"]] = e
    for src, dst, weight in relations:
        if src in G and dst in G and src != dst:
            if G.has_edge(src, dst):
                G[src][dst]["weight"] += weight
            else:
                G.add_edge(src, dst, weight=weight)

    try:
        components = nx.community.louvain_communities(G, weight="weight", seed=42)
    except Exception as e:
        log.warning("louvain failed (%s); single community fallback", e)
        components = [set(G.nodes)] if G.nodes else []

    communities = [c for c in components if len(c) >= min_size]
    log.info("detected %d communities (of %d nodes)", len(communities), len(entities))

    out: list[dict] = []
    for members in communities:
        member_ids = sorted(members)
        member_lines = []
        for mid in member_ids[:60]:
            e = by_id[mid]
            member_lines.append(f"- {e['name']} ({e['type']}): {truncate(e['description'], 200)}")
        summary, keywords = "", []
        prompt = SUMMARY_PROMPT.format(entities="\n".join(member_lines))
        try:
            result = llm.chat([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=500)
            data = _parse_json_loose(result.content or "")
            if data:
                summary = str(data.get("summary", "")).strip()
                keywords = [str(k) for k in data.get("keywords", [])][:10]
        except Exception as e:
            log.warning("community summary failed: %s", e)
        if not summary:
            summary = "; ".join(by_id[mid]["name"] for mid in member_ids[:20])
            keywords = [by_id[mid]["name"].lower() for mid in member_ids[:8]]
        out.append({"member_ids": member_ids, "summary": summary, "keywords": keywords})

    graph_store.save_communities(out)
    return out

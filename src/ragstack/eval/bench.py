"""Deterministic synthetic corpus generator for self-benchmarking."""

from __future__ import annotations

from ..types import Document
from ..utils import new_doc_id

FILLER = (
    "The team documented the decision after reviewing operational feedback. "
    "Measurements were recorded during the weekly maintenance window. "
    "Further evaluation confirmed the expected behaviour under load. "
)

TOPICS: list[tuple[str, str, list[str]]] = [
    ("kubernetes", "container orchestration across clusters", ["pods schedule automatically", "rolling updates keep services online", "horizontal scaling reacts to load"]),
    ("postgres", "relational database engine", ["btree indexes speed up lookups", "autovacuum reclaims dead tuples", "streaming replication warm standbys"]),
    ("redis", "in-memory data store", ["key expiry drives cache eviction", "pubsub channels fan out messages", "snapshot persistence rewrites dumps"]),
    ("rabbitmq", "message broker for services", ["queues buffer bursts of work", "topic exchanges route by pattern", "consumers acknowledge deliveries"]),
    ("nginx", "web server and reverse proxy", ["upstream pools balance requests", "tls termination offloads certificates", "rate limits protect backends"]),
]


def build_synthetic_corpus(n_docs: int) -> tuple[list[str], list[Document], list[tuple[str, str, str]]]:
    """Returns (topic_names, documents, goldens).

    goldens: (doc_id, question, expected_source_substring)
    """
    docs: list[Document] = []
    goldens: list[tuple[str, str, str]] = []
    made = 0
    while made < n_docs:
        topic, blurb, facts = TOPICS[made % len(TOPICS)]
        idx = made // len(TOPICS)
        doc_id = new_doc_id(f"/synthetic/{topic}-{idx}.md")
        parts = [
            f"# {topic.title()} deployment notes {idx}",
            f"This note covers {blurb}.",
        ]
        chosen = [facts[(idx + j) % len(facts)] for j in range(len(facts))]
        for fact in chosen:
            parts.append(f"Observation: {fact}. {FILLER}")
        text = "\n\n".join(parts)
        source = f"/synthetic/{topic}-{idx}.md"
        docs.append(
            Document(id=doc_id, source=source, title=f"{topic} {idx}", text=text,
                     metadata={"format": "md", "topic": topic})
        )
        keyword = chosen[0].split()[0]
        goldens.append((doc_id, f"where do we document {topic} {keyword}?", f"{topic}-{idx}"))
        made += 1
    return [t[0] for t in TOPICS], docs, goldens

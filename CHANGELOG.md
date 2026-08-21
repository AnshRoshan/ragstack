# Changelog

## 0.2.0 — 2026-08-21

Eight first-class retrieval modes with fast direct pipelines (`vector`, `lexical`,
`hybrid`, `graph`, `global`) alongside the agentic modes (`auto`, `agentic`, `sql`);
`GET /api/modes` catalog; web console v2 (mode picker, in-browser indexing and
crawling); CRAG-style evidence grading; query decomposition planner; semantic
response cache; CORS support; landing page in `website/`; packaging polish
(metadata, py.typed, CI, GitHub Pages workflow).

## 0.1.0 — 2026-08-20

Initial release: Docling ingestion, structure-aware chunking, contextual
enrichment, tantivy BM25 index, LanceDB vector index, SQLite/Neo4j knowledge
graph with Louvain communities, text-to-SQL catalog, ReAct agent with 9 tools,
SSE streaming, Typer CLI, FastAPI server, eval harness, 65 hermetic tests.

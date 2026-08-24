# RAGStack

**A hybrid Agentic RAG system.** It reads your documents (PDF, Word, PowerPoint, Excel, Markdown, code, web pages), builds four kinds of search indexes over them, and an AI **agent** decides which index(es) to use for each question â€” often chaining several in multiple steps â€” then answers with citations.

> New to RAG or the terms here? Read **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** â€” it explains the complete flow in simple English, with every technical term defined.

## The four ways it can find answers

| Index | Good at | Tech |
|---|---|---|
| **Keyword / "vectorless"** (pages + chunks) | exact names, error codes, IDs | BM25 via tantivy |
| **Vector** (meaning-based) | questions where your words don't appear verbatim | LanceDB + BGE embeddings |
| **Knowledge graph** | "how is X related to Y", multi-hop connections, corpus-wide themes | LLM-extracted entities/relations â†’ SQLite (default) or Neo4j, with Louvain communities + summaries |
| **Your SQL databases** | numbers, records, aggregates | read-only text-to-SQL |

The agent (a ReAct-style loop: think â†’ call tool â†’ read result â†’ repeat) picks tools per question, up to a hard step budget, and must answer with `[S1]`-style citations. Every answer carries a **confidence score** derived from evidence grading + citation coverage.

**Vertical presets** tune the whole stack for a domain â€” one line in `ragstack.yaml`:

```yaml
vertical: legal      # legal | medical | academic
```

Each preset adjusts chunking and injects domain answer discipline (verbatim clause quoting for legal, population/dosage caveats for medical, author-year attribution for academic).

**Eight retrieval modes** â€” pick per query with `--mode` / the UI dropdown / the API:

| mode | kind | what it does |
|---|---|---|
| `auto` | agentic | agent plans each hop, picks any of the 9 tools |
| `agentic` | agentic | explicit full multi-hop ReAct loop |
| `hybrid` | direct | BM25 + dense fused via RRF, reranked â€” fast default |
| `vector` | direct | embeddings only |
| `lexical` | direct | vectorless BM25 only â€” exact terms/IDs |
| `graph` | direct | entity neighborhood walk |
| `global` | direct | corpus-wide themes from community summaries |
| `sql` | agentic | read-only text-to-SQL + chunk search |

Inspect them any time: `ragstack modes`, `GET /api/modes`, or the UI dropdown.

Three quality layers on top (all research-backed, see docs/ARCHITECTURE.md):

- **Evidence grading (CRAG)** â€” after every retrieval the system checks whether the results actually look relevant; if not, the agent is told to change strategy instead of forcing an answer from bad evidence.
- **Query decomposition** â€” compound questions ("compare X and Y") are split into sub-queries, searched in parallel, and fused.
- **Semantic cache** â€” near-duplicate questions are answered instantly from a vector cache (0.95 similarity; 0.98 when your question sounds uncertain). `--no-cache` bypasses it.

## Install

```powershell
cd E:\RAG
uv sync                      # everything: docling, lancedb, tantivy, embeddings
uv sync --extra neo4j        # only if you want Neo4j instead of the built-in graph
```

Pick a mode in `ragstack.yaml` (copy from `config.example.yaml`):

- `local` â€” fully offline. Needs [Ollama](https://ollama.com) running for the answering LLM.
- `hybrid` â€” local search/indexing, cloud LLM for answers (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`). Best quality/cost balance.
- `cloud` â€” everything via API.

Any OpenAI-compatible server (vLLM, LM Studio, Ollama `/v1`) works via `llm.base_url`.

## Use

```powershell
ragstack index ./docs                 # index a folder (PDFs, DOCX, MD, code...)
ragstack index ./docs --enrich        # + LLM context sentences per chunk (better recall)
ragstack crawl https://example.com    # index a website (same-domain, robots-aware)

ragstack query "How does hybrid retrieval fuse results?"
ragstack query "error CODE-4212 meaning" --mode lexical --no-cache
ragstack query "and its rate limits?" --session demo   # follow-up memory: 'its' resolves
ragstack watch ./docs                                  # re-index on change
ragstack bench --docs 50                               # throughput/latency/recall self-benchmark
ragstack serve                        # web console at http://127.0.0.1:8000
ragstack sessions                     # list conversation sessions
ragstack forget demo                  # forget a session's memory
ragstack modes                        # the 8-mode catalog
ragstack status                       # what's indexed
ragstack eval my_golden.yaml          # measure hit-rate / MRR / faithfulness
```

The web console (`ragstack serve`) lets you index folders, crawl sites, switch modes and chat â€” no CLI needed. A standalone overview site lives in `website/index.html` (open it directly in a browser; fully offline, no CDN dependencies).

Register a database for the SQL tool:

```powershell
ragstack db analytics "sqlite:///D:/data/analytics.db"
```

## Verify everything works

```powershell
uv sync --extra dev
uv run pytest tests/ -q     # 65 hermetic tests (fake models, no network)
```

Covers: chunker, parsers, crawler extraction, lexical store, vector store, graph store round-trips, entity extraction + caching, communities, graph search, agent loop (tool calls, streaming, step budget, error surfacing), tool executor + citation registry, evidence grader, query decomposition, semantic cache (hit/miss/uncertainty/mode-scoping), config resolution, enricher caching, full ingestion pipeline (index â†’ skip unchanged â†’ re-index on change â†’ reset), web API (status, SSE query stream, indexing endpoint, UI serving).

## Deploy with Docker

```bash
docker compose up --build          # console on http://localhost:8000
docker compose --profile neo4j up  # + Neo4j sidecar; set graph.backend: neo4j
```

To require an access token on the API (recommended when not binding to localhost):

```yaml
server:
  auth_token: "your-secret"        # or env RAGSTACK_AUTH_TOKEN
```

## Project layout

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a guided tour of every file and both pipelines (document-in, question-out).

## Roadmap

- Knowledge-strip refinement of retrieved context (CRAG Â§4.4)
- Per-hop evaluation in the harness (attribute failures to agent steps)
- Late chunking / ColBERT-style multi-vector scoring

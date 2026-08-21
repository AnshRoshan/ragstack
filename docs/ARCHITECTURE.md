# RAGStack Architecture — The Complete Flow, Explained Simply

This document explains **everything that happens** when you (1) add documents to RAGStack and (2) ask a question. It is written in plain English. Every technical word is explained the first time it appears.

---

## Part 0: Words you need to know

| Term | Simple explanation |
|---|---|
| **RAG** (Retrieval-Augmented Generation) | A way for an AI to answer questions using *your* documents: first **retrieve** (find) relevant text, then **generate** (write) an answer based on that text. |
| **Chunk** | A small piece of a document (a few paragraphs). Documents are too big to search directly, so we cut them into chunks. |
| **Embedding** | A list of numbers (a "vector") that represents the *meaning* of a text. Texts with similar meanings get similar numbers, even if they use different words. Example: "car" and "automobile" end up close together. |
| **Vector store** | A database that stores these number-lists and can quickly find the ones closest to your question's numbers. We use **LanceDB** (a file-based database — no server needed). |
| **BM25** | A classic keyword-search formula (used by search engines). It finds chunks containing your exact words and ranks them by how rare and how frequent those words are. It does NOT understand meaning — only words. |
| **Lexical index** | The keyword-search index built with BM25. "Lexical" = about words. We use a library called **tantivy**. |
| **"Vectorless" index** | The same thing — searching by *words only*, no meaning-numbers involved. Great for exact names, error codes, part numbers. |
| **Hybrid search** | Running BOTH searches (meaning + keywords) and merging the results. Each method catches what the other misses. |
| **RRF** (Reciprocal Rank Fusion) | A simple way to merge two ranked lists: each item gets points based on its position in each list (higher position = more points), and we sum the points. |
| **Cross-encoder reranker** | A slow-but-smart AI model that reads (question, candidate chunk) pairs together and scores how well they match. We use it only on the top candidates because it is slow. |
| **Knowledge graph** | A network of **entities** (things like people, companies, products) connected by **relations** ("works_at", "made_by"). It stores facts and connections, not just text. |
| **GraphRAG** | Using that network to answer questions — e.g., "how is X connected to Y?" by walking the connections. |
| **Community detection** (Louvain) | An algorithm that groups tightly-connected entities into clusters ("communities"). We then write a summary of each cluster, so broad questions ("what are the main themes?") can be answered. |
| **Agent** | An AI model that can decide what to do next: which tool to call, whether the results are good enough, when to stop. |
| **ReAct loop** | The agent's rhythm: **Rea**son ("I should look up X") → **Act** (call a search tool) → look at results → repeat until ready to answer. |
| **Tool** | A function the agent may call: keyword search, meaning search, graph search, SQL query, fetch a web page. |
| **Multi-hop** | When answering needs several chained lookups. Example: find the project name → then search for who leads that project → then look up that person's team. |
| **Text-to-SQL** | The agent writes a database query (SQL) in response to your question, runs it read-only, and uses the results. |
| **SSE** (Server-Sent Events) | A way for a web server to stream updates to the browser while it works, so you see progress live. |
| **Citation** | A marker like `[S1]` in the answer pointing to the exact source it came from. |

---

## Part 1: What happens when you run `ragstack index ./docs`

```
your files
    │
    ▼
[1] PARSE          Docling/native loaders turn files into plain structured text
    │
    ▼
[2] CHUNK          split text into small pieces, keeping headings/code intact
    │
    ▼
[3] ENRICH   (opt) an LLM adds one "where this fits" sentence per chunk
    │
    ├──────────────────────────────┬─────────────────────────────┐
    ▼                              ▼                             ▼
[4a] EMBED + VECTOR STORE      [4b] LEXICAL INDEX            [4c] GRAPH EXTRACT
    meaning-numbers per chunk     BM25 index of pages           LLM finds entities +
    stored in LanceDB             AND chunks (tantivy)          relations → knowledge graph
                                                               (SQLite or Neo4j)
                                                               │
                                                               ▼
                                                          [5] COMMUNITIES
                                                          group related entities,
                                                          summarize each group
```

### Step 1 — Parse (`ingestion/parsers.py`)
Every file becomes one `Document`: `{id, source path, title, full text}`.

- **PDF, Word, PowerPoint, Excel, scanned images** → parsed by **Docling** (an IBM open-source library that understands page layout, tables, and can read text out of images via OCR — Optical Character Recognition, i.e., reading letters from pictures).
- **Markdown, TXT, logs** → read as-is.
- **Code files** → wrapped in a code block labeled with the file name.
- **HTML** → main content extracted with trafilatura (drops menus/ads).
- **CSV** → header + sample rows turned into text.
- **Web pages** (`ragstack crawl URL`) → fetched politely (checks robots.txt — the website's permission file), cleaned to main text.

Files already indexed and unchanged are **skipped** (we keep a manifest — a record book listing each file's content fingerprint, called a hash: a short unique code computed from the text; change one letter and the code changes).

### Step 2 — Chunk (`ingestion/chunker.py`)
The document text is cut into chunks (~512 "tokens"; a token ≈ ¾ of a word).

Why care? If a chunk is too big, search gets vague. Too small, and context is lost. Our cutter is **structure-aware**:

- It keeps **heading paths**: a chunk from "Setup > Database" carries that label, so search knows its context.
- Code blocks are never cut in half.
- Chunks overlap slightly (the last sentences are repeated at the start of the next chunk) so no idea is lost at a boundary.

### Step 3 — Enrich (optional, `--enrich`)
An LLM writes ONE sentence per chunk: *"this chunk explains X inside document Y"*. That sentence is glued to the chunk before indexing. This helps you find a chunk with words that appear nowhere in it but describe it well. Results are cached so re-indexing is free.

### Step 4a — Vector index (`stores/vector.py`)
Each chunk's text is converted to an embedding (meaning-numbers) and stored in LanceDB alongside its text and metadata.

### Step 4b — Lexical index (`stores/lexical.py`)
The SAME chunks plus the WHOLE PAGE are written into the tantivy BM25 index. Whole pages are included because sometimes the best match for "error CODE-1234" is the exact page, not a fragment.

### Step 4c — Graph extraction (`graphrag/extract.py`)
For each chunk, the LLM answers with strict JSON:

```json
{"entities": [{"name": "Acme", "type": "organization", "description": "..."}],
 "relationships": [{"source": "Alice", "target": "Acme", "relation": "works_at"}]}
```

We validate it hard (unknown types → "other", relations must reference extracted entities, duplicates merged). Entities and relations go into the graph store:

- **Default: SQLite** — a single file database, zero setup. Relations are rows; finding neighbors is a fast indexed lookup.
- **Optional: Neo4j** — a standalone graph server, same interface, switch with one config line.

Every extraction is cached by chunk-fingerprint, so interrupted indexing resumes instead of restarting.

### Step 5 — Communities (`graphrag/communities.py`)
All entities become a network; the Louvain algorithm groups densely-connected entities into clusters; the LLM writes a summary per cluster ("these 12 entities are all about payment processing"). Stored for global questions.

---

## Part 2: What happens when you ask a question

```
your question
    │
    ▼
[1] AGENT STARTS  (agent/runner.py)
    system prompt = role + strategy + citation rules
    │
    ▼
[2] THINK → CALL TOOL(S) ──────────────────────────────┐
    │               │                                  │
    │   available tools:                                │
    │   hybrid_search     (both, fused + reranked)      │
    │   decomposed_search (splits compound questions,   │
    │                      searches parts in parallel)  │
    │   graph_search      (entity neighborhood)         │
    │   community_overview(thematic summaries)          │
    │   sql_query         (read-only DB queries)        │
    │   fetch_url         (live web page)               │
    │                                                  │
    ▼                                                  │
[3] READ RESULTS → EVIDENCE CHECK → good? ──no──► refine & repeat ┘
    │ yes                    (hard cap: max_steps, then MUST answer)
    ▼
[4] WRITE ANSWER with [S1][S2] citations
    │
    ▼
[5] STREAM to you: thoughts, tool calls, answer, source list
```

### How a tool call actually works (example)

1. The LLM replies not with text but with a structured request: `hybrid_search({"query": "rate limits", "top_k": 8})`.
2. `retrieval/vector_rag.py::hybrid_search` runs:
   - dense leg: embed the question → LanceDB top-24 by meaning-similarity;
   - sparse leg: BM25 top-24 by keywords;
   - **RRF merge** both lists;
   - **cross-encoder rerank** the merged candidates, keep top-8.
3. Each result gets a reference id (`S1`, `S2`, …) recorded in a shared citation registry, and a compact preview goes back to the LLM as the tool result.
4. The LLM sees the previews and either calls another tool (multi-hop) or writes the final answer citing `[S1]`.

### Graph tools at query time

- `graph_search("Acme")` → finds Acme in the graph, walks up to `max_hops` connections, returns related entities, their relationships, AND the original text chunks that mentioned them (so citations still point to real sources).
- `community_overview("payments")` → picks the most relevant cluster summaries and has the LLM synthesize them — this answers "what are the big themes?" style questions that no single chunk contains.

### Safety rails (learned from production agentic-RAG practice)

| Rail | Where |
|---|---|
| Hard step budget; on the LAST step tools are removed so the model must answer with what it has | `runner.py` |
| **Evidence grading (CRAG)**: after each retrieval, results are judged *correct / ambiguous / incorrect*; on "incorrect" the agent is explicitly told to change strategy instead of forcing an answer | `retrieval/evaluator.py` |
| **Query decomposition**: compound questions split into sub-queries, searched in parallel, fused with RRF | `agent/planner.py` |
| **Semantic cache**: near-duplicate questions answered instantly (similarity ≥ 0.95; ≥ 0.98 if your question sounds uncertain — "maybe", "possibly"...) | `cache.py` |
| Every claim must carry a `[Sn]` marker; sources listed at the end | prompt + registry |
| SQL tool refuses anything not starting with SELECT/WITH/EXPLAIN and blocks write keywords | `sql_catalog.py` |
| Crawler respects robots.txt, stays on one domain, caps pages | `crawler.py` |
| Provider errors surface as visible error events, never silent wrong answers | `runner.py` |

**Why these three extras matter** (in simple words): research papers and production systems in 2024-2026 all converged on the same lessons. *Corrective RAG* showed that blindly trusting whatever the search returns is the #1 cause of confident-but-wrong answers — so you grade the evidence first. *Query decomposition* handles questions that are really several questions glued together. *Semantic caching* makes repeated questions ~50× faster and cheaper, but must be strict when a question sounds uncertain, or you get a confidently cached answer to slightly the wrong question.

---

## Part 3: The three stack modes

One config line switches where intelligence runs:

| mode | meaning-numbers made by | answer written by | rerank by | internet needed |
|---|---|---|---|---|
| `local` | BGE model on your machine | Ollama (local LLM server) | local cross-encoder | no |
| `hybrid` | BGE model on your machine | OpenAI/Anthropic API | local cross-encoder | yes (only for the LLM) |
| `cloud` | OpenAI API | OpenAI/Anthropic API | none | yes |

"Auto" resolution fills in sensible models per mode; any OpenAI-compatible endpoint (vLLM, LM Studio, Ollama) works via `llm.base_url`.

---

## Part 4: File map

```
src/ragstack/
├── config.py            settings + auto-selection of providers
├── pipeline.py          the Part-1 conveyor belt
├── service.py           one object wiring everything (used by CLI/API)
├── cli.py               terminal commands
├── ingestion/
│   ├── parsers.py       files → Documents        (Step 1)
│   ├── chunker.py       Documents → chunks       (Step 2)
│   ├── enricher.py      context sentences        (Step 3)
│   └── crawler.py       websites → Documents     (Step 1, web)
├── stores/
│   ├── vector.py        LanceDB meaning-index    (Step 4a)
│   ├── lexical.py       tantivy keyword-index    (Step 4b)
│   ├── graph/           SQLite or Neo4j graph    (Step 4c)
│   └── sql_catalog.py   registered databases
├── graphrag/
│   ├── extract.py       entities + relations     (Step 4c)
│   ├── communities.py   clustering + summaries   (Step 5)
│   └── search.py        local/global graph tools
├── retrieval/
│   ├── lexical_rag.py   BM25 search helpers
│   ├── vector_rag.py    semantic + hybrid + RRF
│   └── evaluator.py     CRAG evidence grader (correct/ambiguous/incorrect)
├── agent/
│   ├── tools.py         the 9 tools + executor
│   ├── planner.py       query decomposition (parallel sub-searches)
│   ├── prompts.py       system prompt
│   └── runner.py        the ReAct loop           (Part 2)
├── cache.py             semantic response cache
├── web/                 FastAPI server + chat UI
└── eval/harness.py      hit-rate / MRR / faithfulness report
```

## Part 5: How to verify it all works

```powershell
uv sync --extra dev
uv run pytest tests/ -q          # 65 tests, no network needed
uv run ragstack status           # see store counts
uv run ragstack index ./examples/sample_docs --no-graph
uv run ragstack serve            # open http://127.0.0.1:8000
```

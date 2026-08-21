# RAGStack Demo Corpus

RAGStack is a hybrid Agentic Retrieval-Augmented Generation system.

## Architecture

The ingestion pipeline parses documents with Docling, splits them with
structure-aware chunking, embeds chunks into LanceDB, and indexes pages and
chunks into a tantivy BM25 index. Entity and relation extraction builds a
knowledge graph used for GraphRAG local and global search.

## Retrieval Modes

- lexical: BM25 keyword search over pages and chunks. No vectors involved.
- vector: dense embedding search with optional cross-encoder reranking.
- hybrid: reciprocal rank fusion of dense and sparse results, then rerank.
- graph: entity neighborhood traversal plus community summaries.
- sql: read-only text-to-SQL against registered databases.

The agent chains these tools across multiple hops, citing sources as [S1], [S2].

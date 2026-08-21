"""RAGStack — hybrid Agentic RAG.

Vectorless lexical index + vector RAG + GraphRAG + text-to-SQL,
unified under one tool-calling agent.

Quick start:
    from ragstack import RAGStack

    svc = RAGStack("ragstack.yaml")
    svc.ingest(["./docs"])
    answer = svc.query("What does the spec say about rate limits?")
"""

__version__ = "0.2.0"

__all__ = [
    "AppConfig",
    "RAGStack",
    "__version__",
    "load_config",
]


def __getattr__(name: str):
    # Lazy re-exports keep `import ragstack` fast and dependency-light.
    if name == "RAGStack":
        from .service import RAGStack

        return RAGStack
    if name in ("AppConfig", "load_config"):
        from . import config

        return getattr(config, name)
    raise AttributeError(f"module 'ragstack' has no attribute {name!r}")

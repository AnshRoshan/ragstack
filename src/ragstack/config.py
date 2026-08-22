"""Configuration: YAML + env, with provider resolution per stack mode."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import BaseModel, Field

from .errors import ConfigError
from .utils import get_logger

log = get_logger("ragstack.config")

DEFAULT_OLLAMA_BASE = "http://localhost:11434/v1"
DEFAULT_LOCAL_EMBED = "BAAI/bge-small-en-v1.5"
DEFAULT_CLOUD_EMBED = "text-embedding-3-small"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class EmbeddingConfig(BaseModel):
    provider: str = "auto"
    model: str = "auto"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    batch_size: int = 64


class LLMConfig(BaseModel):
    provider: str = "auto"
    model: str = "auto"
    base_url: str | None = None
    temperature: float = 0.2
    max_tokens: int = 2048


class RerankConfig(BaseModel):
    provider: str = "auto"
    model: str = DEFAULT_RERANK_MODEL


class ChunkingConfig(BaseModel):
    size: int = 512
    overlap: int = 64
    min_size: int = 80


class GraphConfig(BaseModel):
    enabled: bool = True
    backend: str = "sqlite"
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password_env: str = "NEO4J_PASSWORD"
    community_summaries: bool = True
    max_hops: int = 2
    workers: int = 3


class AgentConfig(BaseModel):
    max_steps: int = 8
    top_k: int = 8
    evidence_grading: bool = True  # CRAG-style check after each retrieval tool
    evidence_grader: str = "heuristic"  # heuristic | llm
    strip_refinement: bool = True  # keep only query-relevant sentences from graded evidence
    memory_turns: int = 3  # conversation turns remembered per session (0 disables memory)


class CacheConfig(BaseModel):
    enabled: bool = True
    threshold: float = 0.95
    fuzzy_threshold: float = 0.98


class IndexConfig(BaseModel):
    root: str = ".ragstack"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class AppConfig(BaseModel):
    mode: str = "hybrid"
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    databases: dict[str, str] = Field(default_factory=dict)
    index: IndexConfig = Field(default_factory=IndexConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    def resolved_root(self, base: Path | None = None) -> Path:
        p = Path(self.index.root)
        if not p.is_absolute():
            p = (base or Path.cwd()) / p
        return p

    def resolve_providers(self) -> AppConfig:
        """Fill in 'auto' choices based on mode + environment."""
        mode = self.mode.lower()
        if mode not in {"local", "cloud", "hybrid"}:
            raise ConfigError(f"mode must be local|cloud|hybrid, got {self.mode!r}")

        emb = self.embedding.model_copy()
        llm = self.llm.model_copy()
        rr = self.rerank.model_copy()

        openai_key = bool(os.environ.get("OPENAI_API_KEY"))
        anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

        if mode == "local":
            if emb.provider == "auto":
                emb.provider = "sentence-transformers"
            if emb.model == "auto":
                emb.model = DEFAULT_LOCAL_EMBED
            if rr.provider == "auto":
                rr.provider = "cross-encoder"
            if llm.provider == "auto":
                llm.provider = "ollama"
                llm.base_url = llm.base_url or DEFAULT_OLLAMA_BASE
                if llm.model == "auto":
                    llm.model = _first_ollama_model(llm.base_url) or "llama3.1"
        elif mode == "cloud":
            if emb.provider == "auto":
                emb.provider = "openai-compatible"
                emb.base_url = emb.base_url or "https://api.openai.com/v1"
            if emb.model == "auto":
                emb.model = DEFAULT_CLOUD_EMBED
            if rr.provider == "auto":
                rr.provider = "none"
            if llm.provider == "auto":
                if openai_key:
                    llm.provider, llm.model = "openai", (
                        llm.model if llm.model != "auto" else "gpt-4o-mini"
                    )
                elif anthropic_key:
                    llm.provider, llm.model = "anthropic", (
                        llm.model if llm.model != "auto" else "claude-sonnet-4-20250514"
                    )
                else:
                    raise ConfigError(
                        "cloud mode needs OPENAI_API_KEY or ANTHROPIC_API_KEY in the environment"
                    )
        else:
            if emb.provider == "auto":
                emb.provider = "sentence-transformers"
            if emb.model == "auto":
                emb.model = DEFAULT_LOCAL_EMBED
            if rr.provider == "auto":
                rr.provider = "cross-encoder"
            if llm.provider == "auto":
                if openai_key:
                    llm.provider, llm.model = "openai", (
                        llm.model if llm.model != "auto" else "gpt-4o-mini"
                    )
                elif anthropic_key:
                    llm.provider, llm.model = "anthropic", (
                        llm.model if llm.model != "auto" else "claude-sonnet-4-20250514"
                    )
                else:
                    llm.provider = "ollama"
                    llm.base_url = llm.base_url or DEFAULT_OLLAMA_BASE
                    if llm.model == "auto":
                        llm.model = _first_ollama_model(llm.base_url) or "llama3.1"

        out = self.model_copy()
        out.embedding, out.llm, out.rerank = emb, llm, rr
        return out


def _first_ollama_model(base_url: str) -> str | None:
    tags_url = base_url.replace("/v1", "") + "/api/tags"
    try:
        r = httpx.get(tags_url, timeout=0.6)
        models = r.json().get("models") or []
        for m in models:
            name = m.get("name", "")
            if name and not any(t in name for t in ("embed", "bge", "nomic-embed", "minilm")):
                return name.split(":")[0]
    except Exception:
        return None
    return None


def load_config(path: str | Path | None = None) -> AppConfig:
    data: dict[str, Any] = {}
    if path is None:
        for candidate in ("ragstack.yaml", "ragstack.yml"):
            p = Path(candidate)
            if p.exists():
                path = p
                break
    if path is not None and Path(path).exists():
        try:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"invalid YAML in {path}: {e}") from e
    elif path is not None:
        raise ConfigError(f"config file not found: {path}")
    return AppConfig(**data)


def save_config(cfg: AppConfig, path: str | Path) -> None:
    Path(path).write_text(
        yaml.safe_dump(cfg.model_dump(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

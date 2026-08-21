"""Graph store factory."""

from __future__ import annotations

from pathlib import Path

from ...config import GraphConfig
from ...errors import ConfigError
from .base import GraphStore


def open_graph(cfg: GraphConfig, root: Path) -> GraphStore:
    if cfg.backend == "sqlite":
        from .sqlite_graph import SQLiteGraphStore

        return SQLiteGraphStore(root)
    if cfg.backend == "neo4j":
        from .neo4j_graph import Neo4jGraphStore

        return Neo4jGraphStore(cfg.uri, cfg.user, cfg.password_env)
    raise ConfigError(f"unknown graph backend: {cfg.backend} (use sqlite|neo4j)")


__all__ = ["GraphStore", "open_graph"]

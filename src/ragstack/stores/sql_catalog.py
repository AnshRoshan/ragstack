"""SQL catalog: register external databases, expose read-only text-to-SQL tool."""

from __future__ import annotations

import re
from functools import lru_cache

from sqlalchemy import create_engine, inspect, text

from ..errors import ToolError
from ..utils import get_logger, truncate

log = get_logger("ragstack.sql")

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|grant|revoke|pragma|vacuum|replace|merge|copy|truncate)\b",
    re.IGNORECASE,
)
_MAX_ROWS = 200


class SQLCatalog:
    def __init__(self, databases: dict[str, str]):
        self.databases = dict(databases)

    @lru_cache(maxsize=16)
    def _engine(self, name: str):
        if name not in self.databases:
            raise ToolError(f"unknown database {name!r}; registered: {list(self.databases)}")
        try:
            return create_engine(self.databases[name], pool_pre_ping=True)
        except Exception as e:
            raise ToolError(f"cannot connect to {name!r}: {e}") from e

    def schema_digest(self, name: str) -> str:
        engine = self._engine(name)
        insp = inspect(engine)
        lines = [f"DATABASE {name}:"]
        for table in insp.get_table_names()[:80]:
            cols = []
            for col in insp.get_columns(table)[:40]:
                cols.append(f"{col['name']} {col.get('type', '')}")
            lines.append(f"  {table}({', '.join(cols)})")
        return "\n".join(lines)

    def all_schemas(self) -> str:
        if not self.databases:
            return ""
        parts = []
        for name in self.databases:
            try:
                parts.append(self.schema_digest(name))
            except Exception as e:
                parts.append(f"DATABASE {name}: unavailable ({e})")
        return "\n\n".join(parts)

    def query(self, name: str, sql: str) -> str:
        sql = sql.strip().rstrip(";").strip()
        if not sql:
            raise ToolError("empty SQL")
        first_word = sql.split()[0].lower()
        if first_word not in ("select", "with", "explain"):
            raise ToolError("only SELECT/WITH/EXPLAIN queries are allowed")
        if _FORBIDDEN.search(sql):
            raise ToolError("write keywords detected — read-only access only")
        if "limit" not in sql.lower():
            sql += f" LIMIT {_MAX_ROWS}"
        engine = self._engine(name)
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchmany(_MAX_ROWS + 1)
        truncated = len(rows) > _MAX_ROWS
        rows = rows[:_MAX_ROWS]
        cols = list(result.keys())
        out = [f"QUERY on {name}: {truncate(sql, 300)}", f"columns: {', '.join(cols)}"]
        for row in rows:
            out.append(" | ".join("" if v is None else truncate(str(v), 120) for v in row))
        if truncated:
            out.append(f"(showing first {_MAX_ROWS} rows)")
        return "\n".join(out)

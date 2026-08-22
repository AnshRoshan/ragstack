"""Session conversation memory (SQLite) + co-reference query rewriting."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from .providers.llm import LLMProvider
from .utils import get_logger, now_iso

log = get_logger("ragstack.memory")

REWRITE_PROMPT = """Given the chat history and a follow-up question, rewrite the follow-up as a
standalone search query that can be understood without the history.
Resolve pronouns and references ("it", "that", "he", "there") using the history.
Do NOT answer the question. If it is already standalone, return it unchanged.
Reply with ONLY the rewritten question and nothing else.

HISTORY:
{history}

FOLLOW-UP QUESTION: {question}

Standalone question:"""


class ConversationMemory:
    def __init__(self, root: Path):
        self.path = Path(root) / "memory.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS turns(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts TEXT NOT NULL)"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session)")
        self._conn.commit()

    def history(self, session: str, limit: int = 6) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM turns WHERE session = ? ORDER BY id DESC LIMIT ?",
                (session, limit),
            ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def append(self, session: str, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns(session, role, content, ts) VALUES(?,?,?,?)",
                (session, role, content, now_iso()),
            )
            self._conn.commit()

    def clear(self, session: str | None = None) -> None:
        with self._lock:
            if session is None:
                self._conn.execute("DELETE FROM turns")
            else:
                self._conn.execute("DELETE FROM turns WHERE session = ?", (session,))
            self._conn.commit()

    def stats(self) -> dict:
        with self._lock:
            sessions = self._conn.execute("SELECT COUNT(DISTINCT session) FROM turns").fetchone()[0]
            turns = self._conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        return {"sessions": sessions, "turns": turns}


def format_history(history: list[dict], max_chars: int = 2400) -> str:
    lines = []
    for m in history:
        who = "USER" if m["role"] == "user" else "ASSISTANT"
        content = m["content"][:600]
        lines.append(f"{who}: {content}")
    out = "\n".join(lines)
    return out[-max_chars:]


def rewrite_question(llm: LLMProvider | None, question: str, history: list[dict]) -> str:
    """Resolve co-references in a follow-up using recent turns; falls back to input."""
    if not history or llm is None or not question:
        return question
    try:
        result = llm.chat(
            [
                {
                    "role": "user",
                    "content": REWRITE_PROMPT.format(
                        history=format_history(history), question=question
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=200,
        )
        rewritten = (result.content or "").strip().strip('"').removeprefix("Standalone question:")
        rewritten = rewritten.strip()
        if rewritten and len(rewritten) < 4 * len(question) + 80:
            return rewritten
        return question
    except Exception as e:
        log.warning("query rewrite failed (%s); using original", e)
        return question

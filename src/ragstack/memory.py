"""Session conversation memory (SQLite) + co-reference query rewriting."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np

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

    def list_sessions(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT session, COUNT(*) AS turns, MAX(ts) AS last
                   FROM turns GROUP BY session ORDER BY last DESC"""
            ).fetchall()
        return [{"session": r[0], "turns": r[1], "last": r[2]} for r in rows]


class RecallStore:
    """Cross-session semantic memory: past Q&A pairs, searchable by meaning.

    Backed by a LanceDB table inside the index root. Lets later sessions
    reuse earlier answers through the recall_memory agent tool.
    """

    def __init__(self, root: Path, embeddings):
        self.dir = Path(root) / "recall"
        self.embeddings = embeddings
        self._table = None

    def _ensure_table(self):
        if self._table is not None:
            return self._table
        import lancedb
        import pyarrow as pa

        db = lancedb.connect(str(self.dir))
        schema = pa.schema(
            [
                pa.field("question", pa.string()),
                pa.field("answer", pa.string()),
                pa.field("session", pa.string()),
                pa.field("ts", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.embeddings.dim)),
            ]
        )
        self._table = db.create_table("qa", schema=schema, mode="create", exist_ok=True)
        return self._table

    def add(self, session: str, question: str, answer: str) -> None:
        try:
            vec = np.asarray(self.embeddings.embed([question])[0], dtype=np.float32)
            self._ensure_table().add(
                [{"question": question[:2000], "answer": answer[:8000], "session": session,
                  "ts": now_iso(), "vector": vec}]
            )
        except Exception as e:
            log.debug("recall add failed: %s", e)

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if self.count() == 0:
            return []
        try:
            qvec = np.asarray(self.embeddings.embed([query])[0], dtype=np.float32)
            rows = self._ensure_table().search(qvec).limit(top_k).to_list()
            out = []
            for r in rows:
                sim = max(0.0, 1.0 - float(r["_distance"]) / 2.0)
                out.append(
                    {"question": r["question"], "answer": r["answer"],
                     "session": r["session"], "similarity": round(sim, 3)}
                )
            return out
        except Exception as e:
            log.debug("recall search failed: %s", e)
            return []

    def count(self) -> int:
        if self._table is None and not self.dir.exists():
            return 0
        try:
            return int(self._ensure_table().count_rows())
        except Exception:
            return 0

    def clear(self) -> None:
        self._table = None
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)



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

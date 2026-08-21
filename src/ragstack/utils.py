"""Small shared helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

_log_initialized = False


def get_logger(name: str = "ragstack") -> logging.Logger:
    global _log_initialized
    log = logging.getLogger(name)
    if not _log_initialized:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        _log_initialized = True
    return log


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def new_doc_id(source: str) -> str:
    return "d" + sha1(str(Path(source).resolve()))[:20]


def chunk_id(doc_id: str, ordinal: int, text: str) -> str:
    return "c" + sha1(f"{doc_id}|{ordinal}|{text[:512]}")[:24]


def norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def truncate(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[:n] + " …"


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


_TOKENIZER = None


def token_count(text: str) -> int:
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            import tiktoken

            _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            class _Fallback:
                @staticmethod
                def encode(t: str) -> list[int]:
                    return [0] * max(1, len(t) // 4)

            _TOKENIZER = _Fallback()
    try:
        return len(_TOKENIZER.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def read_text_safe(path: Path) -> str:
    from charset_normalizer import from_path

    try:
        result = from_path(path)
        best = result.best()
        if best is not None:
            return str(best)
    except Exception:
        pass
    return path.read_text(encoding="utf-8", errors="replace")


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

"""Structure-aware chunking: heading paths, code fences, token budgets, overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import ChunkingConfig
from ..types import Chunk, Document
from ..utils import chunk_id, get_logger, token_count

log = get_logger("ragstack.chunk")

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class Block:
    text: str
    heading: str


def split_blocks(text: str) -> list[Block]:
    blocks: list[Block] = []
    headings: dict[int, str] = {}
    fence = False
    para: list[str] = []
    cur_heading = ""

    def flush_para():
        nonlocal para
        joined = "\n".join(para).strip()
        if joined:
            blocks.append(Block(text=joined, heading=cur_heading))
        para = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            fence = not fence
            para.append(line)
            continue
        if fence:
            para.append(line)
            continue
        m = _HEADING.match(stripped)
        if m:
            flush_para()
            level = len(m.group(1))
            headings[level] = m.group(2).strip()
            for deeper in [k for k in headings if k > level]:
                del headings[deeper]
            cur_heading = " > ".join(headings[k] for k in sorted(headings))
            continue
        if not stripped:
            flush_para()
            continue
        para.append(line)
    flush_para()
    return blocks


def _hard_split(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    sentences = _SENTENCE.split(text)
    parts: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for s in sentences:
        t = token_count(s)
        if buf_tokens + t > max_tokens and buf:
            parts.append(" ".join(buf))
            keep: list[str] = []
            keep_tokens = 0
            for prev in reversed(buf):
                pt = token_count(prev)
                if keep_tokens + pt > overlap_tokens:
                    break
                keep.insert(0, prev)
                keep_tokens += pt
            buf = keep[:]
            buf_tokens = keep_tokens
        buf.append(s)
        buf_tokens += t
    if buf:
        parts.append(" ".join(buf))
    return parts


def chunk_document(doc: Document, cfg: ChunkingConfig) -> list[Chunk]:
    blocks = split_blocks(doc.text)
    chunks: list[Chunk] = []

    def emit(body: str, heading: str):
        body = body.strip()
        if not body:
            return
        prefix = f"[{doc.title} › {heading}]\n" if heading else f"[{doc.title}]\n"
        chunks.append(
            Chunk(
                id="",
                doc_id=doc.id,
                ordinal=len(chunks),
                text=body,
                context=prefix.strip(),
                metadata={"source": doc.source, "title": doc.title, "heading": heading, **doc.metadata},
                n_tokens=token_count(prefix + body),
            )
        )

    buf: list[str] = []
    buf_heading = ""
    buf_tokens = 0
    for block in blocks:
        btokens = token_count(block.text)
        if btokens > cfg.size:
            if buf:
                emit("\n\n".join(buf), buf_heading)
                buf, buf_tokens = [], 0
            pieces = _hard_split(block.text, cfg.size, cfg.overlap)
            for piece in pieces:
                emit(piece, block.heading)
            continue
        heading_changed = bool(buf) and block.heading != buf_heading
        if (buf_tokens + btokens > cfg.size or heading_changed) and buf:
            emit("\n\n".join(buf), buf_heading)
            if heading_changed:
                buf, buf_tokens = [], 0
            else:
                tail: list[str] = []
                tail_tokens = 0
                for prev in reversed(buf):
                    pt = token_count(prev)
                    if tail_tokens + pt > cfg.overlap:
                        break
                    tail.insert(0, prev)
                    tail_tokens += pt
                buf, buf_tokens = tail, tail_tokens
        if not buf:
            buf_heading = block.heading
        buf.append(block.text)
        buf_tokens += btokens
    if buf:
        emit("\n\n".join(buf), buf_heading)

    for c in chunks:
        c.id = chunk_id(c.doc_id, c.ordinal, c.text)

    merged: list[Chunk] = []
    for c in chunks:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and c.n_tokens < cfg.min_size
            and prev.metadata.get("heading") == c.metadata.get("heading")
            and prev.n_tokens + c.n_tokens <= int(cfg.size * 1.25) + cfg.overlap
        ):
            prev.text = prev.text + "\n\n" + c.text
            prev.n_tokens = token_count(prev.context + prev.text)
            prev.id = chunk_id(prev.doc_id, prev.ordinal, prev.text)
        else:
            merged.append(c)
    log.debug("chunked %s into %d chunks", doc.source, len(merged))
    return merged

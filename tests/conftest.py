"""Shared test fixtures: deterministic fakes for embeddings and LLM (no network)."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragstack.config import AppConfig
from ragstack.providers.embeddings import EmbeddingProvider
from ragstack.providers.llm import LLMProvider, StreamDelta
from ragstack.types import LLMResult, ToolCall


class FakeEmbeddings(EmbeddingProvider):
    """Deterministic bag-of-words hashing embeddings. Same text -> same vector."""

    name = "fake"

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                h = int(hashlib.sha1(word.encode()).hexdigest(), 16)  # noqa: S324
                out[i, h % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


class FakeLLM(LLMProvider):
    """Scripted LLM: pops one LLMResult per chat() call.

    Each script entry:
      {"content": "..."}                          -> plain answer
      {"tool_calls": [{"name": ..., "args": ...}]} -> tool call turn
    """

    name = "fake"

    def __init__(self, script: list[dict]):
        self.script = list(script)
        self.calls: list[list[dict]] = []
        self.model = "fake-1"

    def _next(self) -> LLMResult:
        if not self.script:
            return LLMResult(content="done", finish_reason="stop")
        step = self.script.pop(0)
        calls = [
            ToolCall(id=f"call_{i}", name=c["name"], arguments=c.get("args", {}))
            for i, c in enumerate(step.get("tool_calls", []))
        ]
        return LLMResult(content=step.get("content"), tool_calls=calls)

    def chat(self, messages, tools=None, temperature=0.2, max_tokens=2048) -> LLMResult:
        self.calls.append(messages)
        return self._next()

    def stream(self, messages, tools=None, temperature=0.2, max_tokens=2048):
        self.calls.append(messages)
        result = self._next()
        if result.content:
            yield StreamDelta(kind="text", text=result.content)
        yield StreamDelta(kind="result", result=result)


@pytest.fixture()
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture()
def app_config(tmp_path):
    cfg = AppConfig(mode="local")
    cfg.index.root = str(tmp_path / "idx")
    cfg.graph.enabled = False
    return cfg

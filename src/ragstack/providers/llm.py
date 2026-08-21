"""LLM providers with a unified chat/stream + tool-calling interface.

OpenAICompatLLM covers OpenAI, Ollama (/v1), LM Studio, vLLM, any OpenAI-style server.
AnthropicLLM covers Claude with tool use.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ..errors import ProviderError
from ..types import LLMResult, ToolCall
from ..utils import get_logger

log = get_logger("ragstack.llm")


@dataclass(slots=True)
class StreamDelta:
    kind: str  # "text" | "result"
    text: str = ""
    result: LLMResult | None = None


def tools_to_openai(tools: list[dict]) -> list[dict]:
    return tools


def tools_to_anthropic(tools: list[dict]) -> list[dict]:
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}}),
        }
        for t in tools
    ]


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResult: ...

    @abstractmethod
    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> Iterator[StreamDelta]: ...


class OpenAICompatLLM(LLMProvider):
    def __init__(self, model: str, base_url: str | None, api_key: str, provider_name: str):
        from openai import OpenAI

        self.name = provider_name
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")

    def chat(self, messages, tools=None, temperature=0.2, max_tokens=2048) -> LLMResult:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or None,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise ProviderError(f"{self.name} chat failed: {e}") from e
        msg = resp.choices[0].message
        calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=_safe_json(c.function.arguments))
            for c in (msg.tool_calls or [])
        ]
        return LLMResult(content=msg.content, tool_calls=calls, finish_reason=resp.choices[0].finish_reason or "stop")

    def stream(self, messages, tools=None, temperature=0.2, max_tokens=2048) -> Iterator[StreamDelta]:
        try:
            flow = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or None,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
        except Exception as e:
            raise ProviderError(f"{self.name} stream failed: {e}") from e

        content_parts: list[str] = []
        calls_by_index: dict[int, dict] = {}
        finish = "stop"
        for chunk in flow:
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            if ch.finish_reason:
                finish = ch.finish_reason
            delta = ch.delta
            if delta is None:
                continue
            if delta.content:
                content_parts.append(delta.content)
                yield StreamDelta(kind="text", text=delta.content)
            for tc in delta.tool_calls or []:
                slot = calls_by_index.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments

        calls = [
            ToolCall(id=s["id"] or f"call_{i}", name=s["name"], arguments=_safe_json(s["args"]))
            for i, s in sorted(calls_by_index.items())
        ]
        yield StreamDelta(
            kind="result",
            result=LLMResult(content="".join(content_parts) or None, tool_calls=calls, finish_reason=finish),
        )


class AnthropicLLM(LLMProvider):
    def __init__(self, model: str, api_key_env: str = "ANTHROPIC_API_KEY"):
        import os

        from anthropic import Anthropic

        self.name = "anthropic"
        self.model = model
        key = os.environ.get(api_key_env)
        if not key:
            raise ProviderError(f"missing {api_key_env} for anthropic")
        self.client = Anthropic(api_key=key)

    # -- message conversion -------------------------------------------------
    @staticmethod
    def _convert(messages: list[dict]) -> tuple[str | None, list[dict]]:
        system_parts: list[str] = []
        out: list[dict] = []
        for m in messages:
            role, content = m["role"], m.get("content")
            if role == "system":
                if content:
                    system_parts.append(content)
                continue
            if role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m["tool_call_id"],
                                "content": str(content),
                            }
                        ],
                    }
                )
                continue
            if role == "assistant" and m.get("tool_calls"):
                blocks: list[dict] = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    blocks.append(
                        {"type": "tool_use", "id": tc["id"], "name": fn.get("name", ""), "input": args}
                    )
                out.append({"role": "assistant", "content": blocks})
                continue
            out.append({"role": role, "content": content or ""})
        return ("\n\n".join(system_parts) or None), out

    @staticmethod
    def _parse_response(resp) -> LLMResult:
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        stop = {"end_turn": "stop", "tool_use": "tool_calls", "stop_sequence": "stop"}
        return LLMResult(
            content="".join(texts) or None,
            tool_calls=calls,
            finish_reason=stop.get(resp.stop_reason, resp.stop_reason or "stop"),
        )

    # -- API ----------------------------------------------------------------
    def chat(self, messages, tools=None, temperature=0.2, max_tokens=2048) -> LLMResult:
        system, msgs = self._convert(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": msgs,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools_to_anthropic(tools)
        try:
            resp = self.client.messages.create(**kwargs)
        except Exception as e:
            raise ProviderError(f"anthropic chat failed: {e}") from e
        return self._parse_response(resp)

    def stream(self, messages, tools=None, temperature=0.2, max_tokens=2048) -> Iterator[StreamDelta]:
        result = self.chat(messages, tools, temperature, max_tokens)
        if result.content:
            yield StreamDelta(kind="text", text=result.content)
        yield StreamDelta(kind="result", result=result)


def _safe_json(s: str | None) -> dict:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        return {}


def make_llm_provider(cfg) -> LLMProvider:
    """cfg: resolved LLMConfig."""
    import os

    if cfg.provider in ("openai", "openai-compatible"):
        base = cfg.base_url or ("https://api.openai.com/v1" if cfg.provider == "openai" else None)
        key = os.environ.get("OPENAI_API_KEY") if (cfg.provider == "openai" or "api.openai.com" in (base or "")) else os.environ.get("OPENAI_API_KEY", "not-needed")
        return OpenAICompatLLM(cfg.model, base, key, provider_name=cfg.provider)
    if cfg.provider == "ollama":
        return OpenAICompatLLM(cfg.model, cfg.base_url or "http://localhost:11434/v1", "ollama", provider_name="ollama")
    if cfg.provider == "anthropic":
        return AnthropicLLM(cfg.model)
    raise ProviderError(f"unknown llm provider: {cfg.provider}")

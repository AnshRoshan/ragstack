"""Agentic runner: ReAct-style iterative tool-calling loop with streaming events.

Design notes (informed by current agentic-RAG practice):
- Hard step budget; on the FINAL step tools are withheld so the model must
  answer from evidence gathered so far instead of looping forever.
- Every assistant turn streams as 'thought' events; tool executions emit
  'tool_start'/'tool_end' events so UIs can show live progress.
- Citations accumulate across all hops in a shared ToolContext registry.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

from ..errors import ProviderError
from ..types import Answer, Citation, ToolCall
from ..utils import get_logger, truncate
from .prompts import SYSTEM_PROMPT
from .tools import RETRIEVAL_TOOLS, ToolContext, make_executor, tools_for_mode

log = get_logger("ragstack.agent")


class AgentRunner:
    def __init__(self, llm, ctx: ToolContext):
        self.llm = llm
        self.ctx = ctx

    def _execute(self, call: ToolCall) -> str:
        observation = make_executor(self.ctx)(call.name, call.arguments)
        if self.ctx.grader is not None and call.name in RETRIEVAL_TOOLS:
            try:
                parsed = json.loads(observation)
                if isinstance(parsed, list):
                    from ..types import RetrievedItem

                    items = [
                        RetrievedItem(
                            chunk_id=r.get("ref", ""),
                            doc_id="",
                            source=r.get("source", ""),
                            title=r.get("title", ""),
                            text=r.get("text", ""),
                            score=float(r.get("score", 0.0)),
                            origin="tool",
                        )
                        for r in parsed
                        if isinstance(r, dict)
                    ]
                    query_text = call.arguments.get("query") or call.arguments.get("question") or ""
                    verdict = self.ctx.grader.grade(query_text, items)
                    if verdict.get("hint"):
                        observation = f"{observation}\n\n{verdict['hint']}"
            except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
                pass
        return observation

    def run(
        self, question: str, mode: str = "auto", stream: bool = True
    ) -> Iterator[dict[str, Any]]:
        all_tools = tools_for_mode(mode)
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        steps: list[dict] = []
        answer_text = ""
        error: str | None = None

        yield {
            "type": "start",
            "mode": mode,
            "tools": [t["function"]["name"] for t in all_tools],
            "max_steps": self.ctx.max_steps,
        }

        try:
            step = 0
            while step < self.ctx.max_steps:
                step += 1
                final_step = step >= self.ctx.max_steps
                tools = [] if final_step else all_tools

                final_text = ""
                pending_calls: list[ToolCall] = []
                if stream:
                    for delta in self.llm.stream(messages, tools=tools or None):
                        if delta.kind == "text" and delta.text:
                            yield {"type": "thought", "delta": delta.text}
                            final_text += delta.text
                        elif delta.kind == "result" and delta.result is not None:
                            pending_calls = delta.result.tool_calls
                            if delta.result.content and not final_text:
                                final_text = delta.result.content
                else:
                    result = self.llm.chat(messages, tools=tools or None)
                    final_text = result.content or ""
                    pending_calls = result.tool_calls

                if not pending_calls:
                    answer_text = final_text
                    break

                messages.append(
                    {
                        "role": "assistant",
                        "content": final_text or None,
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {
                                    "name": c.name,
                                    "arguments": json.dumps(c.arguments, ensure_ascii=False),
                                },
                            }
                            for c in pending_calls
                        ],
                    }
                )
                for call in pending_calls:
                    yield {
                        "type": "tool_start",
                        "step": step,
                        "name": call.name,
                        "args": call.arguments,
                    }
                    observation = self._execute(call)
                    steps.append({"step": step, "tool": call.name, "args": call.arguments})
                    yield {
                        "type": "tool_end",
                        "step": step,
                        "name": call.name,
                        "preview": truncate(observation, 240),
                    }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": truncate(observation, 12000),
                        }
                    )
            else:
                if not answer_text:
                    error = f"step budget exhausted ({self.ctx.max_steps})"

            if step >= self.ctx.max_steps and not answer_text and not error:
                error = f"step budget exhausted ({self.ctx.max_steps})"
        except ProviderError as e:
            error = str(e)
        except Exception as e:
            log.exception("agent crashed")
            error = f"{type(e).__name__}: {e}"

        citations = list(self.ctx.citations.values())
        if not answer_text and not error:
            answer_text = "(no answer produced)"
        if error:
            yield {"type": "error", "message": error}

        yield {
            "type": "done",
            "answer": answer_text,
            "citations": [asdict(c) for c in citations],
            "steps": steps,
            "error": error,
        }


def collect_answer(llm, ctx: ToolContext, question: str, mode: str = "auto") -> Answer:
    runner = AgentRunner(llm, ctx)
    text, cites, steps = "", [], []
    for event in runner.run(question, mode=mode, stream=False):
        if event["type"] == "done":
            text = event.get("answer", "")
            cites = [Citation(**c) for c in event.get("citations", [])]
            steps = event.get("steps", [])
    return Answer(text=text, citations=cites, steps=steps)

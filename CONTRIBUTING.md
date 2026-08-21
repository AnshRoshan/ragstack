# Contributing to RAGStack

Thanks for your interest! The whole architecture is documented in plain English
in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — read that first.

## Setup

```bash
git clone https://github.com/AnshRoshan/ragstack.git
cd ragstack
uv sync --extra dev
uv run pytest tests/ -q     # must be all green before any PR
uv run ruff check src/ tests/
```

Tests are hermetic (fake embeddings/LLM, no network, no model downloads) — keep
them that way. Add tests for every behavior change.

## Ground rules

- Plain, typed Python; no comments unless the code is genuinely surprising.
- New retrieval tools: add the schema + executor branch in `agent/tools.py`,
  register it in `tools_for_mode()`, cover it in `tests/test_agent.py`.
- New providers (embedding/rerank/LLM): implement the existing ABC in
  `providers/`, wire resolution in `config.py::resolve_providers`.
- Keep the local-first promise: nothing may phone home unless the user
  explicitly configures a cloud provider.

## PRs

Small and focused. Update `CHANGELOG.md` and, when behavior changes,
`docs/ARCHITECTURE.md`. CI runs lint + tests on Linux and Windows across
Python 3.11 and 3.13 — it must pass.

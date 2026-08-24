# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.6.x   | ✅ |
| < 0.6   | ❌ |

## Reporting a vulnerability

**Do not open a public issue for security vulnerabilities.**

Email **ianshroshan@gmail.com** with:

- Description and impact
- Reproduction steps or PoC
- Affected version/commit

You will receive an acknowledgment within 72 hours. Please allow up to 90 days
for a fix before public disclosure; coordinated disclosure is appreciated.

## Scope notes

RAGStack is designed local-first. Areas of particular interest:

- The HTTP API auth bypass for loopback callers (threat model: same-machine processes)
- Indirect prompt injection through ingested documents (see `agent/prompts.py` defenses)
- SQL tool escaping (`stores/sql_catalog.py`)
- Semantic cache key collisions across permission boundaries

## Hardening defaults

- Server binds to 127.0.0.1 unless explicitly overridden
- SQL tool accepts only SELECT/WITH/EXPLAIN with row caps
- Agent system prompt forbids executing instructions found in retrieved content
- No telemetry of any kind leaves the machine

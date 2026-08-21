"""RAGStack CLI."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="ragstack",
    help="Hybrid Agentic RAG: lexical + vector + GraphRAG + SQL under one agent.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _service(config: str | None):
    from .service import RAGStack

    return RAGStack(config)


@app.callback()
def cli_root(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to ragstack.yaml"),
):
    _service(str(config) if config else None)


@app.command()
def index(
    paths: list[str] = typer.Argument(..., help="Files or directories to index"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r"),
    enrich: bool = typer.Option(
        False, "--enrich", help="LLM contextual chunk enrichment (slower, better recall)"
    ),
    graph: bool = typer.Option(True, "--graph/--no-graph", help="Entity/relation extraction for GraphRAG"),
    force: bool = typer.Option(False, "--force", help="Re-index even if unchanged"),
    config: Path | None = typer.Option(None, "--config", "-c", hidden=True),
):
    """Index documents (PDF/DOCX/PPTX/MD/code/web-saved HTML...)."""
    svc = _service(config)
    with console.status("[bold]indexing…"):
        stats = svc.ingest(paths, recursive=recursive, enrich=enrich, with_graph=graph, force=force)
    color = "red" if stats.failed else "green"
    console.print(f"[{color}]{stats.summary()}[/{color}]")
    if stats.errors:
        for err in stats.errors[:10]:
            console.print(f"  [dim]! {err}[/dim]")
        raise typer.Exit(1)


@app.command()
def crawl(
    url: str = typer.Argument(...),
    depth: int = typer.Option(1, help="Link-follow depth"),
    max_pages: int = typer.Option(10),
    enrich: bool = typer.Option(False, "--enrich"),
    graph: bool = typer.Option(True, "--graph/--no-graph"),
    config: Path | None = typer.Option(None, "--config", "-c", hidden=True),
):
    """Crawl a website (same-domain) and index the pages."""
    svc = _service(config)
    with console.status("[bold]crawling…"):
        stats = svc.ingest_url(url, depth=depth, max_pages=max_pages, enrich=enrich)
    console.print(f"[green]{stats.summary()}[/green]")


@app.command()
def query(
    question: list[str] = typer.Argument(...),
    mode: str = typer.Option(
        "auto",
        help="auto|agentic|hybrid|vector|lexical|graph|global|sql — see `ragstack modes`",
    ),
    top_k: int = typer.Option(None, help="Override top_k"),
    stream: bool = typer.Option(True, "--stream/--no-stream"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the semantic response cache"),
    show_citations: bool = typer.Option(True, "--citations/--no-citations"),
    config: Path | None = typer.Option(None, "--config", "-c", hidden=True),
):
    """Ask a question through the selected pipeline."""
    svc = _service(config)
    q = " ".join(question)

    if not stream:
        answer = svc.query(q, mode=mode, top_k=top_k)
        console.print(answer.text)
        if show_citations and answer.citations:
            _print_citations(answer.citations)
        return

    citations = []
    for event in svc.stream_query(q, mode=mode, top_k=top_k, use_cache=not no_cache):
        etype = event["type"]
        if etype == "start":
            cached_note = " [yellow](cached)[/yellow]" if event.get("cached") else ""
            console.print(
                f"[dim]mode={event['mode']} tools={', '.join(event['tools'])}{cached_note}[/dim]"
            )
        elif etype == "tool_start":
            args_preview = json.dumps(event.get("args", {}))[:120]
            console.print(f"[cyan]→ {event['name']}[/cyan] [dim]{args_preview}[/dim]")
        elif etype == "tool_end":
            console.print(f"[cyan]✓ {event['name']}[/cyan] [dim]{event.get('preview', '')[:160]}[/dim]")
        elif etype == "error":
            console.print(f"[red]error: {event['message']}[/red]")
        elif etype == "done":
            text = event.get("answer", "")
            if text:
                console.print()
                console.print(text)
            citations = event.get("citations", [])

    if show_citations and citations:
        _print_citations(citations)


def _print_citations(citations):
    table = Table(title="Sources", show_lines=False)
    table.add_column("ref", style="bold cyan")
    table.add_column("title", max_width=40)
    table.add_column("source", max_width=60, overflow="fold")
    for c in citations:
        title = c["title"] if isinstance(c, dict) else c.title
        source = c["source"] if isinstance(c, dict) else c.source
        ref_id = c["ref_id"] if isinstance(c, dict) else c.ref_id
        table.add_row(ref_id, title, source)
    console.print(table)


@app.command()
def serve(
    host: str | None = typer.Option(None),
    port: int | None = typer.Option(None),
    config: Path | None = typer.Option(None, "--config", "-c", hidden=True),
):
    """Start the web UI + API server."""
    import uvicorn

    from .web.app import create_app

    svc = _service(config)
    host = host or svc.cfg.server.host
    port = port or svc.cfg.server.port
    web_app = create_app(svc)
    console.print(f"[bold green]RAGStack UI → http://{host}:{port}[/bold green]")
    uvicorn.run(web_app, host=host, port=port, log_level="info")


@app.command("modes")
def modes_list():
    """List every retrieval mode and what it does."""
    from .service import MODE_CATALOG

    table = Table(title="RAGStack retrieval modes")
    table.add_column("mode", style="bold cyan")
    table.add_column("kind")
    table.add_column("what it does", max_width=70, overflow="fold")
    table.add_column("tools / pipeline")
    for m in MODE_CATALOG:
        detail = ", ".join(m.get("tools", m.get("pipeline", [])))
        table.add_row(m["id"], m["kind"], m["description"], detail)
    console.print(table)


@app.command()
def status(config: Path | None = typer.Option(None, "--config", "-c", hidden=True)):
    """Show index/store/provider status."""
    svc = _service(config)
    info = svc.status()
    table = Table(title="RAGStack status")
    table.add_column("key", style="bold")
    table.add_column("value")
    for k, v in info.items():
        table.add_row(k, json.dumps(v) if isinstance(v, dict) else str(v))
    console.print(table)


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", prompt=False),
    config: Path | None = typer.Option(None, "--config", "-c", hidden=True),
):
    """Delete all indexes."""
    if not yes:
        confirm = typer.confirm("Delete ALL indexes?")
        if not confirm:
            raise typer.Abort()
    svc = _service(config)
    svc.reset()
    console.print("[green]indexes cleared[/green]")


@app.command("db")
def db_register(
    name: str = typer.Argument(...),
    url: str = typer.Argument(..., help="SQLAlchemy URL, e.g. sqlite:///D:/data/app.db"),
    config: Path | None = typer.Option(None, "--config", "-c", hidden=True),
):
    """Register a database for the sql_query tool."""
    from .config import load_config, save_config

    cfg_path = config or (Path("ragstack.yaml") if Path("ragstack.yaml").exists() else None)
    cfg = load_config(cfg_path)
    cfg.databases[name] = url
    target = cfg_path or "ragstack.yaml"
    save_config(cfg, target)
    console.print(f"[green]registered {name} → {url} in {target}[/green]")


@app.command()
def eval(
    golden: Path = typer.Argument(..., help="YAML golden set"),
    k: int = typer.Option(8, help="Retrieval cutoff"),
    config: Path | None = typer.Option(None, "--config", "-c", hidden=True),
):
    """Run retrieval/generation evaluation against a golden set."""
    from .eval.harness import run_eval

    svc = _service(config)
    report = run_eval(svc, golden, k=k)
    sys.exit(0 if report["hit_rate"] >= 0.5 else 2)


def main() -> None:
    import sys

    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception as e:  # pragma: no cover - very old streams
            logging.getLogger("ragstack.cli").debug("stream reconfigure failed: %s", e)
    app()


if __name__ == "__main__":
    main()

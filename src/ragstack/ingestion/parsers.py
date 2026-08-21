"""Document parsing: Docling for PDF/DOCX/PPTX/XLSX/images, native loaders for the rest."""

from __future__ import annotations

import csv
from pathlib import Path

from ..errors import RagStackError
from ..types import Document
from ..utils import get_logger, new_doc_id, read_text_safe

log = get_logger("ragstack.parse")

CODE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".rb", ".php", ".sh", ".bash", ".ps1", ".sql", ".json", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".xml", ".css", ".scss", ".html", ".htm", ".kt", ".swift", ".scala",
}
TEXT_EXTS = {".md", ".markdown", ".txt", ".log", ".rst", ".tex"}
DOCLING_EXTS = {".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


class ParseError(RagStackError):
    pass


_docling_converter = None


def _get_docling():
    global _docling_converter
    if _docling_converter is None:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as e:
            raise ParseError(
                f"docling not available ({e}); install with: pip install docling"
            ) from e
        log.info("initializing Docling converter (first run downloads models) …")
        _docling_converter = DocumentConverter()
    return _docling_converter


def _make(path: Path, text: str, fmt: str) -> Document:
    return Document(
        id=new_doc_id(str(path)),
        source=str(path),
        title=path.stem,
        text=text,
        metadata={"format": fmt, "size": path.stat().st_size},
    )


def _parse_code(path: Path) -> Document:
    text = read_text_safe(path)
    lang = path.suffix.lstrip(".")
    wrapped = f"# {path.name}\n```{lang}\n{text}\n```"
    return _make(path, wrapped, "code")


def _parse_html(path: Path) -> Document:
    import trafilatura

    html = read_text_safe(path)
    text = trafilatura.extract(html, include_tables=True, include_links=False) or ""
    if not text:
        from bs4 import BeautifulSoup

        text = BeautifulSoup(html, "lxml").get_text(separator="\n")
    return _make(path, text, "html")


def _parse_csv(path: Path) -> Document:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return _make(path, "", "csv")
    header, body = rows[0], rows[1:]
    lines = [f"CSV file: {path.name} | columns: {', '.join(header)} | rows: {len(body)}"]
    for row in body[:100]:
        lines.append(" | ".join(row))
    return _make(path, "\n".join(lines), "csv")


def parse_file(path: str | Path) -> Document:
    p = Path(path)
    if not p.exists():
        raise ParseError(f"not found: {p}")
    ext = p.suffix.lower()

    if ext in DOCLING_EXTS:
        try:
            result = _get_docling().convert(str(p))
            text = result.document.export_to_markdown()
            return _make(p, text, ext.lstrip("."))
        except ParseError:
            raise
        except Exception as e:
            log.warning("docling failed on %s (%s); falling back to raw text", p.name, e)
            return _make(p, read_text_safe(p), ext.lstrip("."))

    if ext == ".csv":
        return _parse_csv(p)
    if ext in CODE_EXTS:
        return _parse_code(p)
    if ext == ".pdf":
        return _make(p, read_text_safe(p), "txt")
    return _make(p, read_text_safe(p), ext.lstrip(".") or "txt")


IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", ".ragstack", "__pycache__", ".idea", ".vscode", "dist", "build", ".pytest_cache", ".mypy_cache"}
MAX_FILE_BYTES = 50 * 1024 * 1024


def collect_files(paths: list[str | Path], recursive: bool = False) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            for f in p.glob(pattern):
                if not f.is_file() or f.name.startswith("."):
                    continue
                if any(part in IGNORED_DIRS for part in f.parts):
                    continue
                try:
                    if f.stat().st_size > MAX_FILE_BYTES:
                        log.warning("skipping huge file (%s)", f)
                        continue
                except OSError:
                    continue
                out.append(f)
        elif p.is_file():
            out.append(p)
        else:
            log.warning("path not found, skipping: %s", p)
    seen: set[str] = set()
    unique = []
    for f in sorted(out):
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique

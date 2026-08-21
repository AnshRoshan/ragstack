"""Polite same-domain web crawler producing Documents."""

from __future__ import annotations

import time
from collections import deque
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from ..types import Document
from ..utils import get_logger, new_doc_id

log = get_logger("ragstack.crawl")

UA = "RAGStack/0.1 (+local research crawler)"


def _normalize(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", "", ""))


def _can_fetch(url: str, cache: dict[str, RobotFileParser | None]) -> bool:
    parts = urlsplit(url)
    host = f"{parts.scheme}://{parts.netloc}"
    if host not in cache:
        rp = RobotFileParser()
        try:
            rp.set_url(host + "/robots.txt")
            rp.read()
            cache[host] = rp
        except Exception:
            cache[host] = None
    rp = cache[host]
    return True if rp is None else rp.can_fetch(UA, url)


def _extract(html: str, url: str) -> tuple[str, str, list[str]]:
    import trafilatura

    text = trafilatura.extract(html, url=url, include_tables=True, include_links=False) or ""
    title = ""
    links: list[str] = []
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not text:
        text = soup.get_text(separator="\n")
    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"]).split("#")[0]
        if href.startswith(("http://", "https://")):
            links.append(href)
    return title, text.strip(), links


def crawl(start_url: str, depth: int = 1, max_pages: int = 10, delay: float = 0.5) -> list[Document]:
    docs: list[Document] = []
    seen: set[str] = set({_normalize(start_url)})
    queue: deque[tuple[str, int]] = deque([(start_url, 0)])
    robots: dict[str, RobotFileParser | None] = {}
    allowed_netloc = urlsplit(start_url).netloc

    with httpx.Client(follow_redirects=True, headers={"User-Agent": UA}, timeout=20.0) as client:
        while queue and len(docs) < max_pages:
            url, d = queue.popleft()
            if not _can_fetch(url, robots):
                log.info("robots.txt disallows %s", url)
                continue
            try:
                resp = client.get(url)
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "")
                if "text/html" not in ctype and "text/plain" not in ctype:
                    continue
                title, text, links = _extract(resp.text, url)
            except Exception as e:
                log.warning("fetch failed %s: %s", url, e)
                continue
            if len(text) > 200:
                docs.append(
                    Document(
                        id=new_doc_id(url),
                        source=url,
                        title=title or url,
                        text=text,
                        metadata={"format": "web", "depth": d},
                    )
                )
                log.info("crawled [%d] %s (%d chars)", len(docs), url, len(text))
            if d < depth:
                for link in links:
                    norm = _normalize(link)
                    if urlsplit(norm).netloc != allowed_netloc:
                        continue
                    if norm in seen:
                        continue
                    seen.add(norm)
                    queue.append((norm, d + 1))
            time.sleep(delay)
    return docs

"""arXiv's API, faked: an httpx transport serving Atom feeds, for
``ArxivSearchSource(transport=...)``. Nothing reaches arXiv.

``ArxivAPI`` answers ``search_query`` and ``id_list`` requests from the
papers it holds, after ``delay`` seconds, and records each request: when it
started (``starts``, ``time.monotonic()``), its URL (``requests``) and the
most that were in flight at once (``max_in_flight``). ``refuse(status, n,
retry_after=)`` makes the next ``n`` requests answer ``status`` (429, say),
with a ``Retry-After`` header when given; ``garble(n)`` makes them answer
200 with a body that is no feed; ``fail(exc, n)`` makes them raise (an
``httpx.ReadTimeout``, say).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlsplit
from xml.sax.saxutils import escape

import httpx


@dataclass
class Paper:
    """One paper the fake serves."""

    arxiv_id: str
    title: str = "A paper"
    abstract: str = "What it found."
    authors: tuple[str, ...] = ("A. Author",)
    published: str = "2024-01-01T00:00:00Z"
    categories: tuple[str, ...] = ("cs.AI",)


def atom_feed(papers: list[Paper]) -> str:
    """The Atom document arXiv answers with, holding ``papers``."""
    entries = []
    for paper in papers:
        authors = "".join(f"<author><name>{escape(a)}</name></author>" for a in paper.authors)
        categories = "".join(f'<category term="{escape(c)}"/>' for c in paper.categories)
        entries.append(
            "<entry>"
            f"<id>http://arxiv.org/abs/{paper.arxiv_id}v1</id>"
            f"<title>{escape(paper.title)}</title>"
            f"<summary>{escape(paper.abstract)}</summary>"
            f"{authors}"
            f"<published>{paper.published}</published>"
            f"<updated>{paper.published}</updated>"
            f'<link href="http://arxiv.org/abs/{paper.arxiv_id}v1" rel="alternate" type="text/html"/>'
            f'<link title="pdf" href="http://arxiv.org/pdf/{paper.arxiv_id}v1" rel="related" '
            'type="application/pdf"/>'
            f'<arxiv:primary_category term="{escape(paper.categories[0])}"/>'
            f"{categories}"
            "</entry>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">'
        "<title>ArXiv Query</title>" + "".join(entries) + "</feed>"
    )


@dataclass
class ArxivAPI:
    """The fake API; ``transport`` is what the source is given."""

    papers: list[Paper] = field(default_factory=list)
    delay: float = 0.0
    requests: list[str] = field(default_factory=list)
    starts: list[float] = field(default_factory=list)
    max_in_flight: int = 0
    _in_flight: int = 0
    _script: list[httpx.Response | Exception] = field(default_factory=list)

    def refuse(self, status: int, times: int = 1, *, retry_after: str | None = None) -> None:
        headers = {"Retry-After": retry_after} if retry_after is not None else {}
        self._script.extend(httpx.Response(status, headers=headers) for _ in range(times))

    def garble(self, times: int = 1) -> None:
        self._script.extend(httpx.Response(200, text="not a feed") for _ in range(times))

    def fail(self, error: Exception, times: int = 1) -> None:
        self._script.extend(error for _ in range(times))

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        self.starts.append(time.monotonic())
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self._script:
                scripted = self._script.pop(0)
                if isinstance(scripted, Exception):
                    raise scripted
                return scripted
            return httpx.Response(200, text=atom_feed(self._answer(request)))
        finally:
            self._in_flight -= 1

    def _answer(self, request: httpx.Request) -> list[Paper]:
        query = parse_qs(urlsplit(str(request.url)).query)
        if "id_list" in query:
            wanted = set(query["id_list"][0].split(","))
            return [paper for paper in self.papers if paper.arxiv_id in wanted]
        limit = int(query.get("max_results", ["10"])[0])
        return self.papers[:limit]

"""What ``web_fetch``'s summary is written from: the whole page, converted, then capped.

``webfetch_max_content_bytes`` (100 KB) cut the raw HTML before html2text
converted it, so on a page whose head, scripts and navigation fill the first
100 KB the summarizer never saw the article. Markdown, XHTML, Atom and
``+json`` pages, which ``web_fetch`` saves as text, reached the summarizer as
``[Binary content: …]``, and a PDF's text reached it uncapped (1.87 million
characters for a 300-page PDF), extracted on the event loop. Now the whole
body is converted in a worker thread, as the knowledge base converts a page
(``convert_document``), and the text is then cut at the limit. Offline: DNS
is stubbed and the transport is an ``httpx.MockTransport`` behind the real
PinnedTransport.
"""

from __future__ import annotations

import asyncio
import socket
import sys
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from agentic_cli.config import BaseSettings, SettingsContext
from agentic_cli.tools.webfetch import saved
from tests.memory.kb_extra import require_kb_module
from tests.memory.pdf_bytes import pdf_with_text

URL = "https://example.com/page"
ARTICLE = "Magma rises through the crust and erupts as lava."


class _Summarizer:
    def __init__(self) -> None:
        self.contents: list[str] = []

    async def summarize(self, content: str, prompt: str) -> str:
        self.contents.append(content)
        return "summary"


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch):
    """Pages are saved under the working directory: a temporary one."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def no_trafilatura(monkeypatch):
    """html2text only: the converter without the ``kb`` extra."""
    monkeypatch.setitem(sys.modules, "trafilatura", None)


def _fetcher(monkeypatch, body: bytes, content_type: str, settings: BaseSettings | None = None):
    """A fetcher serving ``body``, with ``settings``' limits as ``get_or_create_fetcher`` sets them."""
    from agentic_cli.tools.webfetch.fetcher import ContentFetcher
    from agentic_cli.tools.webfetch.robots import RobotsTxtChecker
    from agentic_cli.tools.webfetch.transport import PinnedTransport
    from agentic_cli.tools.webfetch.validator import URLValidator

    def _gai(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(socket, "getaddrinfo", _gai)
    validator = URLValidator()
    transport = PinnedTransport(validator, inner=httpx.MockTransport(
        lambda req: httpx.Response(200, content=body, headers={"content-type": content_type})))
    settings = settings or BaseSettings()
    fetcher = ContentFetcher(
        validator=validator, robots_checker=RobotsTxtChecker(), transport=transport,
        max_content_bytes=settings.webfetch_max_content_bytes,
        max_pdf_bytes=settings.webfetch_max_pdf_bytes,
        max_download_bytes=settings.webfetch_max_download_bytes,
    )
    fetcher._robots.can_fetch = AsyncMock(return_value=True)
    return fetcher


async def _fetch(monkeypatch, body: bytes, content_type: str, **settings):
    """``fetch_and_summarize`` of ``body``; the result and what the summarizer got."""
    from agentic_cli.tools.webfetch_tool import fetch_and_summarize

    summarizer = _Summarizer()
    settings = BaseSettings(**settings)
    fetcher = _fetcher(monkeypatch, body, content_type, settings)
    with SettingsContext(settings), \
            patch("agentic_cli.tools.webfetch_tool.get_or_create_fetcher", return_value=fetcher):
        result = await fetch_and_summarize(URL, "What is this page about?", 30, summarizer)
    return result, summarizer.contents


def _pdf(pages: list[str]) -> bytes:
    """A PDF with one line of text on each page, which pypdf extracts."""
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    objects: dict[int, bytes] = {3: font}
    kids = []
    number = 4
    for text in pages:
        stream = f"BT /F1 8 Tf 20 760 Td ({text}) Tj ET".encode()
        page, contents = number, number + 1
        number += 2
        kids.append(page)
        objects[page] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {contents} 0 R >>"
        ).encode()
        objects[contents] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = (
        "<< /Type /Pages /Kids [" + " ".join(f"{k} 0 R" for k in kids) + f"] /Count {len(kids)} >>"
    ).encode()
    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for n in sorted(objects):
        offsets[n] = len(out)
        out += b"%d 0 obj\n" % n + objects[n] + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % number
    for n in range(1, number):
        out += b"%010d 00000 n \n" % offsets[n]
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (number, xref)
    return bytes(out)


async def test_an_article_after_100_kb_of_markup_reaches_the_summarizer(monkeypatch, no_trafilatura):
    """Scripts and a long menu fill the first 100 KB of the HTML; converted,
    the page is a few kilobytes of text, the article in it."""
    head = "<html><head><title>Volcanoes</title><script>" + "var x = 1;\n" * 8000 + "</script></head>"
    menu = "".join(f'<a href="/section/{i}/a-long-path-segment">S{i}</a> ' for i in range(1500))
    html = (
        f"{head}<body><nav>{menu}</nav>"
        f"<article><h1>Volcanoes</h1><p>{ARTICLE * 20}</p></article></body></html>"
    ).encode()
    assert html.index(ARTICLE.encode()) > 102400

    result, contents = await _fetch(monkeypatch, html, "text/html; charset=utf-8")

    assert result["success"] is True
    assert ARTICLE in contents[0]
    assert len(contents[0].encode()) < 102400
    assert result["truncated"] is False


def _body(content_type: str) -> bytes:
    """A page of ``content_type`` whose text says "Needle in the page"."""
    extension = saved.saved_page_extension(content_type)
    if extension == ".pdf":
        return pdf_with_text("Needle in the page")
    if extension == ".html":
        return b"<html><body><p>Needle in the page</p></body></html>"
    if extension == ".json":
        return b'{"note": "Needle in the page"}'
    if extension == ".xml":
        return b"<feed><entry><title>Needle in the page</title></entry></feed>"
    return b"# Notes\n\nNeedle in the page\n"


@pytest.mark.parametrize(
    "content_type",
    [
        *saved._EXTENSIONS,
        "application/atom+xml",
        "application/rss+xml",
        "application/problem+json",
        "application/ld+json",
    ],
)
async def test_each_type_web_fetch_saves_reaches_the_summarizer_as_text(monkeypatch, content_type):
    result, contents = await _fetch(monkeypatch, _body(content_type), content_type)

    assert result["success"] is True
    assert "Needle in the page" in contents[0]
    assert "[Binary content" not in contents[0]


async def test_a_type_that_is_not_text_is_not_summarized(monkeypatch):
    result, contents = await _fetch(monkeypatch, b"\x89PNG\r\n\x1a\n" + b"z" * 100, "image/png")

    assert result["success"] is False
    assert "image/png" in result["error"]
    assert "not a text page" in result["error"].lower()
    assert contents == []


async def test_a_pdf_over_the_limit_reaches_the_summarizer_capped(monkeypatch):
    limit = 4000
    pdf = _pdf([f"Page {n} of the report: " + "lava flows and tephra layers " * 8 for n in range(60)])

    result, contents = await _fetch(
        monkeypatch, pdf, "application/pdf", webfetch_max_content_bytes=limit,
    )

    marker = f"\n\n[Content truncated at {limit} bytes of text]"
    assert result["success"] is True
    assert result["truncated"] is True
    assert contents[0].endswith(marker)
    assert len(contents[0].encode()) <= limit + len(marker.encode())
    assert contents[0].startswith("--- Page 1 ---\nPage 0 of the report")


async def test_a_long_text_is_cut_on_a_character_boundary(monkeypatch):
    """The limit counts bytes of UTF-8; a cut never splits a character."""
    body = ("é" * 3000).encode()  # two bytes each

    result, contents = await _fetch(
        monkeypatch, body, "text/plain; charset=utf-8", webfetch_max_content_bytes=1001,
    )

    assert result["truncated"] is True
    assert contents[0] == "é" * 500 + "\n\n[Content truncated at 1001 bytes of text]"


async def test_converting_a_pdf_does_not_stall_the_event_loop(monkeypatch):
    """pypdf runs in a worker thread: a slow PDF holds its own call only."""
    import pypdf

    reader = pypdf.PdfReader

    def slow_reader(*args, **kwargs):
        time.sleep(0.3)
        return reader(*args, **kwargs)

    monkeypatch.setattr(pypdf, "PdfReader", slow_reader)
    ticks: list[float] = []

    async def tick():
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.01)

    ticker = asyncio.create_task(tick())
    await asyncio.sleep(0.03)
    try:
        result, contents = await _fetch(monkeypatch, pdf_with_text("Slow to read"), "application/pdf")
        await asyncio.sleep(0.03)
    finally:
        ticker.cancel()

    assert "Slow to read" in contents[0]
    assert max(b - a for a, b in zip(ticks, ticks[1:])) < 0.1


LINK = '<a href="https://docs.example.org/eruptions">the eruption records</a>'


@pytest.mark.parametrize("converter", ["trafilatura", "html2text"])
async def test_the_summarizer_sees_where_a_page_links(monkeypatch, converter):
    """A link's URL stays in the text a summary is written from, so the agent
    can follow it (the knowledge base leaves links out of what it stores)."""
    if converter == "trafilatura":
        require_kb_module("trafilatura")
    else:
        monkeypatch.setitem(sys.modules, "trafilatura", None)
    html = f"<html><body><article><h1>Volcanoes</h1><p>{ARTICLE * 8} See {LINK}.</p></article></body></html>"

    result, contents = await _fetch(monkeypatch, html.encode(), "text/html")

    assert result["success"] is True
    assert "https://docs.example.org/eruptions" in contents[0]


async def test_a_pdf_s_pages_are_marked(monkeypatch):
    """Page markers, as web_fetch gave them before, so a summary can cite a page."""
    result, contents = await _fetch(
        monkeypatch, _pdf(["First page text", "Second page text"]), "application/pdf",
    )

    assert "--- Page 1 ---\nFirst page text" in contents[0]
    assert "--- Page 2 ---\nSecond page text" in contents[0]


async def test_web_fetch_reads_no_page_metadata(monkeypatch):
    """trafilatura's metadata pass took most of a large page's time (17.7 s of
    18.8 s at 2 MB), for a title, author and date web_fetch never uses."""
    trafilatura = require_kb_module("trafilatura")
    calls = []
    monkeypatch.setattr(trafilatura, "extract_metadata", lambda *a, **k: calls.append(a))
    html = f"<html><body><article><h1>Volcanoes</h1><p>{ARTICLE * 20}</p></article></body></html>"

    result, contents = await _fetch(monkeypatch, html.encode(), "text/html")

    assert ARTICLE in contents[0]
    assert calls == []


async def test_a_page_over_a_megabyte_is_converted_by_html2text(monkeypatch):
    """trafilatura's time grows faster than the page (5 MB of link-dense text:
    87 s) and a worker thread cannot be stopped: a page over 1 MB is
    converted whole by html2text (1.4 s for 5 MB)."""
    trafilatura = require_kb_module("trafilatura")
    extracted = []
    extract = trafilatura.extract

    def recording(*args, **kwargs):
        extracted.append(1)
        return extract(*args, **kwargs)

    monkeypatch.setattr(trafilatura, "extract", recording)
    small = f"<html><body><article><p>{ARTICLE * 20}</p></article></body></html>".encode()
    large = ("<html><body>" + f"<p>{ARTICLE}</p>\n" * 22_000 + "</body></html>").encode()
    assert len(small) < 1024 * 1024 < len(large)

    await _fetch(monkeypatch, small, "text/html")
    result, contents = await _fetch(monkeypatch, large, "text/html")

    assert extracted == [1]  # the small page only
    assert contents[0].startswith(ARTICLE)
    assert result["truncated"] is True


async def test_a_pdf_cut_at_the_download_limit_is_an_error(monkeypatch):
    """A cut PDF cannot be read: the error names the limit and its setting,
    rather than a summary of no text."""
    pdf = _pdf([f"Page {n}: lava flows and tephra layers" for n in range(200)])
    limit = len(pdf) // 2

    result, contents = await _fetch(monkeypatch, pdf, "application/pdf", webfetch_max_pdf_bytes=limit)

    assert result["success"] is False
    assert "webfetch_max_pdf_bytes" in result["error"]
    assert f"{limit:,} bytes" in result["error"]
    assert contents == []


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (b"", "text/html"),
        (b"<html><head><script>document.write('x')</script></head><body><div id=app></div></body></html>",
         "text/html"),
        (b"   \n", "text/plain"),
        (_pdf([""]), "application/pdf"),
    ],
    ids=["empty", "script-only", "blank-text", "pdf-without-text"],
)
async def test_a_page_with_no_text_is_an_error(monkeypatch, body, content_type):
    result, contents = await _fetch(monkeypatch, body, content_type)

    assert result["success"] is False
    assert "no text" in result["error"].lower()
    assert contents == []


@pytest.mark.parametrize(
    "content_type", ["text/csv", "text/javascript", "text/x-python", "application/javascript"],
)
async def test_any_text_type_is_summarized_as_text(monkeypatch, content_type):
    result, contents = await _fetch(monkeypatch, b"Needle in the page,1,2\n", content_type)

    assert result["success"] is True
    assert "Needle in the page" in contents[0]

"""The ingestion converter (``tools/kb_convert.py``).

Ingestion has its own converter, separate from ``web_fetch``'s. A web page
goes in as its main text, not its HTML: trafilatura (the ``kb`` extra) keeps
the main text and reads the page's title, author, date and description;
html2text, which keeps the whole page, is the fallback.
"""

from __future__ import annotations

import sys

import pytest

from agentic_cli.tools.kb_convert import UnsupportedDocument, convert_document

ARTICLE = (
    "<html><head><title>Volcano notes</title>"
    '<meta name="author" content="Ada Lovelace; Mary Somerville">'
    '<meta name="description" content="How magma reaches the surface.">'
    '<meta property="article:published_time" content="2025-06-01">'
    "</head><body>"
    "<nav><a href='/'>Home</a> <a href='/about'>About us</a></nav>"
    "<article><h1>Volcanoes</h1><p>"
    + "Magma rises through the crust and erupts as lava. " * 12
    + "</p><table><tr><th>Volcano</th><th>Height</th></tr>"
    "<tr><td>Etna</td><td>3357 m</td></tr></table>"
    "</article><footer>Copyright Example Press</footer></body></html>"
)


@pytest.fixture
def no_trafilatura(monkeypatch):
    monkeypatch.setitem(sys.modules, "trafilatura", None)


class _Log:
    """Records log calls. structlog's level filter would drop info events
    once logging is configured, so tests record through this instead."""

    def __init__(self):
        self.events: list[dict] = []

    def _record(self, event, **kw):
        self.events.append({"event": event, **kw})

    debug = info = warning = _record


@pytest.fixture
def log(monkeypatch):
    from agentic_cli.tools import kb_convert

    recorder = _Log()
    monkeypatch.setattr(kb_convert, "logger", recorder)
    return recorder


def test_html_main_text_with_trafilatura():
    pytest.importorskip("trafilatura")
    doc = convert_document(ARTICLE.encode(), content_type="text/html", extension=".html",
                           charset="utf-8", url="https://example.com/volcanoes")

    assert doc.extractor == "trafilatura"
    assert "Magma rises through the crust" in doc.text
    assert "Etna" in doc.text and "3357 m" in doc.text
    assert "Home" not in doc.text and "About us" not in doc.text
    assert "Copyright" not in doc.text
    assert "<p>" not in doc.text
    assert doc.title == "Volcanoes"
    assert doc.authors == ("Ada Lovelace", "Mary Somerville")
    assert doc.published == "2025-06-01"
    assert doc.description == "How magma reaches the surface."


def test_html_falls_back_to_html2text_without_trafilatura(no_trafilatura, log):
    doc = convert_document(ARTICLE.encode(), content_type=None, extension=".html")

    assert doc.extractor == "html2text"
    assert "Magma rises through the crust" in doc.text
    assert "<p>" not in doc.text
    assert doc.title == "Volcano notes"
    assert doc.authors == ()
    assert {"event": "kb_convert_html", "extractor": "html2text", "reason": "not_installed"} in log.events


def test_html_falls_back_when_trafilatura_finds_no_main_text(log):
    pytest.importorskip("trafilatura")
    shell = b"<html><head><title>App</title></head><body><div id='root'></div><script>start()</script></body></html>"
    doc = convert_document(shell, content_type="text/html", extension=".html", charset="utf-8")

    assert doc.extractor == "html2text"
    assert doc.title == "App"
    assert {"event": "kb_convert_html", "extractor": "html2text", "reason": "no_main_text"} in log.events


def test_a_page_is_decoded_with_its_charset(no_trafilatura):
    page = "<html><body><p>Магма поднимается сквозь кору.</p></body></html>".encode("windows-1251")
    doc = convert_document(page, content_type="text/html; charset=windows-1251",
                           extension=".html", charset="windows-1251")
    assert "Магма поднимается сквозь кору." in doc.text


def test_a_pdf_goes_through_pypdf(monkeypatch):
    from agentic_cli.tools import pdf_utils

    seen = []
    monkeypatch.setattr(pdf_utils, "extract_pdf_text", lambda data, **kw: seen.append(data) or "pdf text")
    doc = convert_document(b"%PDF-1.4", content_type=None, extension=".pdf")

    assert doc.text == "pdf text"
    assert doc.extractor == "pypdf"
    assert seen == [b"%PDF-1.4"]


@pytest.mark.parametrize("extension", [".md", ".txt", ".json", ".xml", ".py"])
def test_text_files_go_in_as_they_are(extension):
    text = '# Notes\n\n{"key": "café"}\n'
    doc = convert_document(text.encode(), content_type=None, extension=extension)
    assert doc.text == text
    assert doc.extractor == "text"


def test_a_local_file_that_is_not_utf8_is_refused():
    with pytest.raises(UnsupportedDocument, match="not a PDF or a UTF-8 text file"):
        convert_document(b"\x89PNG\r\n\x1a\n\xff\xfe", content_type=None, extension=".png")


def test_a_fetched_text_page_is_decoded_leniently():
    doc = convert_document(b"caf\xe9", content_type="text/plain", extension=".txt", charset="latin-1")
    assert doc.text == "café"

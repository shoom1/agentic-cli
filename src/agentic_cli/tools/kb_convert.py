"""Turn a file's bytes into the text the knowledge base stores.

Ingestion owns this converter; ``web_fetch`` keeps its own for summaries.

- HTML goes through trafilatura when it is installed (the ``kb`` extra): it
  keeps a page's main text as Markdown, drops menus, sidebars, footers, links
  and images, and reads the page's title, author, date and description.
  Without trafilatura, or when it finds no main text, html2text converts the
  whole page and the title comes from the ``<title>`` element.
- PDFs go through pypdf (``pdf_utils.extract_pdf_text``).
- Anything else is decoded as text: leniently with the given ``charset`` for a
  fetched page, strictly as UTF-8 for a local file.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from html.parser import HTMLParser

import structlog

from agentic_cli.tools._charset import text_codec

logger = structlog.get_logger(__name__)

_HTML_TYPES = ("text/html", "application/xhtml+xml")
_HTML_EXTENSIONS = (".html", ".htm", ".xhtml")


class UnsupportedDocument(ValueError):
    """The bytes are not a format the knowledge base can ingest."""


@dataclass(frozen=True)
class ConvertedDocument:
    """A document's text plus what the page says about itself."""

    text: str
    title: str | None = None
    authors: tuple[str, ...] = ()
    published: str | None = None
    description: str | None = None
    extractor: str = "text"


def convert_document(
    data: bytes,
    *,
    content_type: str | None,
    extension: str,
    charset: str | None = None,
    url: str | None = None,
) -> ConvertedDocument:
    """Convert ``data`` to the text the knowledge base stores.

    Args:
        data: The file's bytes.
        content_type: The page's Content-Type, or None for a local file.
        extension: The file's extension, for example ``.html``.
        charset: The fetched page's charset. Given, the bytes are decoded
            with it and undecodable bytes are replaced. None means a local
            file, which must be UTF-8 unless it is a PDF.
        url: The page's URL, which helps trafilatura.

    Raises:
        UnsupportedDocument: a local file that is neither a PDF nor UTF-8.
    """
    media = (content_type or "").split(";", 1)[0].strip().lower()
    ext = extension.lower()
    if media == "application/pdf" or ext == ".pdf":
        from agentic_cli.tools import pdf_utils

        return ConvertedDocument(text=pdf_utils.extract_pdf_text(data), extractor="pypdf")

    text = _decode(data, charset)
    if media in _HTML_TYPES or ext in _HTML_EXTENSIONS:
        return _convert_html(text, url)
    return ConvertedDocument(text=text, extractor="text")


def _decode(data: bytes, charset: str | None) -> str:
    if charset is None:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            raise UnsupportedDocument("not a PDF or a UTF-8 text file") from None
    try:
        return data.decode(text_codec(charset), errors="replace")
    except (LookupError, UnicodeError):
        return data.decode("utf-8", errors="replace")


def _convert_html(html: str, url: str | None) -> ConvertedDocument:
    try:
        import trafilatura
    except ImportError:
        reason = "not_installed"
    else:
        try:
            # trafilatura's own loader, not lxml.html.fromstring directly: it
            # handles an XML declaration (typical for XHTML, which
            # lxml.html.fromstring refuses with "Unicode strings with
            # encoding declaration are not supported") and keeps text after
            # an inline comment, which lxml.html.fromstring silently drops.
            tree = trafilatura.load_html(html)
            if tree is None:
                # An empty or unparseable page: no main text, and nothing
                # here to call extract() on.
                main, meta, reason = None, None, "no_main_text"
            else:
                # Spans are removed, not preserved: trafilatura 2.3.0's table
                # handling is pathological on colspan/rowspan (quadratic in
                # row count: a 10x20-cell table with colspan=100 rowspan=100
                # took ~1.25s and produced ~330K chars of output). Stripping
                # the span attributes turns that into ~0s / ~1K chars, at the
                # cost that a legitimate rowspan/colspan shifts which cells
                # land in which row (accepted trade-off — trafilatura's span
                # expansion is unbounded).
                for cell in tree.iter("td", "th"):
                    cell.attrib.pop("colspan", None)
                    cell.attrib.pop("rowspan", None)
                # trafilatura may modify the tree it is given, so metadata
                # extraction gets its own copy taken before extract() runs.
                tree_for_meta = copy.deepcopy(tree)
                main = trafilatura.extract(
                    tree,
                    url=url,
                    output_format="markdown",
                    include_links=False,
                    include_images=False,
                    include_tables=True,
                    include_formatting=True,
                    include_comments=False,
                )
                meta = (
                    trafilatura.extract_metadata(tree_for_meta, default_url=url)
                    if main and main.strip() else None
                )
                reason = "no_main_text"
        except Exception as e:  # a third-party parser; fall back rather than fail ingestion
            logger.warning("kb_convert_trafilatura_failed", error=str(e))
            main, meta, reason = None, None, "error"
        if main and main.strip():
            logger.debug("kb_convert_html", extractor="trafilatura")
            return ConvertedDocument(
                text=main.strip(),
                title=_clean(getattr(meta, "title", None)),
                authors=_split_authors(getattr(meta, "author", None)),
                published=_clean(getattr(meta, "date", None)),
                description=_clean(getattr(meta, "description", None)),
                extractor="trafilatura",
            )
    logger.info("kb_convert_html", extractor="html2text", reason=reason)
    try:
        text = _html2text(html)
    except Exception as e:  # html2text is a third-party parser; never raise out of ingestion
        logger.warning("kb_convert_html_failed", error=type(e).__name__)
        raise UnsupportedDocument("could not convert this HTML page") from e
    return ConvertedDocument(text=text, title=_html_title(html), extractor="html2text")


def _clean(value: str | None) -> str | None:
    value = " ".join((value or "").split())
    return value or None


def _split_authors(value: str | None) -> tuple[str, ...]:
    return tuple(name for name in (_clean(part) for part in (value or "").split(";")) if name)


def _html2text(html: str) -> str:
    import html2text

    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    converter.body_width = 0
    converter.single_line_break = True
    return converter.handle(html).strip()


class _TitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._inside = False
        self._done = False

    def handle_starttag(self, tag, attrs):
        if tag == "title" and not self._done:
            self._inside = True

    def handle_endtag(self, tag):
        if tag == "title" and self._inside:
            self._inside = False
            self._done = True

    def handle_data(self, data):
        if self._inside:
            self.parts.append(data)


def _html_title(html: str) -> str | None:
    parser = _TitleParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # html.parser is lenient; a title is optional
        pass
    return _clean("".join(parser.parts))

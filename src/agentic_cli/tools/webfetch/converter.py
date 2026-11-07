"""A fetched page as the text its summary is written from.

``page_text`` is what ``web_fetch`` sends its summarizer: the whole body
converted to text (``convert_document``, as the knowledge base converts a
page, but with links kept and no metadata read; a PDF with its page
markers), then cut at ``webfetch_max_content_bytes``. ``HTMLToMarkdown`` is
the converter ``web_fetch`` used before, kept for applications that import it.
"""

from __future__ import annotations

import html2text

from agentic_cli.tools.webfetch.saved import saved_page_extension

#: Ends a page's text cut at ``webfetch_max_content_bytes``.
TRUNCATED = "\n\n[Content truncated at {limit} bytes of text]"
#: The largest HTML page whose main text trafilatura extracts; a larger one is
#: converted whole by html2text. trafilatura's time grows faster than the page
#: (synthetic pages, links kept: 1 MB of link-dense paragraphs 2.7 s, 5 MB
#: 87 s), html2text's with it (5 MB: 0.5 to 1.4 s), and a worker thread
#: cannot be stopped: Ctrl+C would leave it running and holding the exit.
MAIN_TEXT_MAX_BYTES = 1024 * 1024
#: Script types read as text, beside every ``text/*`` type.
_SCRIPT_TYPES = frozenset({
    "application/javascript", "application/x-javascript", "application/ecmascript",
})


def text_extension(content_type: str | None) -> str | None:
    """The extension a page of ``content_type`` is converted as; None when it
    is not text (an image, an archive, ``application/octet-stream``).

    The type it is saved as (``saved_page_extension``), else ``.txt`` for any
    other ``text/*`` or script type: a CSV file or a script is read as it is.
    """
    extension = saved_page_extension(content_type)
    if extension is not None:
        return extension
    media = (content_type or "").split(";", 1)[0].strip().lower()
    if media.startswith("text/") or media in _SCRIPT_TYPES:
        return ".txt"
    return None


def page_text(
    data: bytes,
    *,
    content_type: str,
    extension: str,
    charset: str | None,
    url: str,
    max_bytes: int,
) -> tuple[str, bool]:
    """A fetched page's text, and whether it was cut at ``max_bytes``.

    The whole body is converted, so the limit counts text rather than markup
    a converter drops. An HTML page's main text, its links kept (trafilatura
    with the ``kb`` extra, up to ``MAIN_TEXT_MAX_BYTES``; else html2text of
    the whole page); a PDF's text, each page marked (``--- Page N ---``) for
    a summary to cite; any other text type (``extension``, from
    ``text_extension``) as it is. A text longer than ``max_bytes`` of UTF-8
    is cut on a character boundary and ends with a marker naming the limit.
    No text at all is ``("", False)``.

    Runs pypdf, trafilatura or html2text over up to the download limit:
    call it off the event loop.

    Args:
        data: The body as received.
        content_type: The page's Content-Type.
        extension: The extension the page is converted as.
        charset: The page's charset; None reads it as UTF-8. Undecodable
            bytes are replaced.
        url: The URL the page was read from, which helps trafilatura.
        max_bytes: The most text the summarizer gets, in bytes of UTF-8.

    Raises:
        UnsupportedDocument: html2text failed on the page.
    """
    if extension == ".pdf":
        from agentic_cli.tools.pdf_utils import extract_pdf_text

        text = extract_pdf_text(data, page_markers=True)
    else:
        from agentic_cli.memory.kb import convert_document

        text = convert_document(
            data, content_type=content_type, extension=extension,
            charset=charset or "utf-8", url=url,
            links=True, metadata=False, main_text=len(data) <= MAIN_TEXT_MAX_BYTES,
        ).text
    text = text.strip()
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text, False
    kept = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return kept + TRUNCATED.format(limit=max_bytes), True


class HTMLToMarkdown:
    """Converts HTML content to Markdown.

    Uses html2text for conversion with configuration optimized
    for LLM consumption.
    """

    def __init__(self) -> None:
        """Initialize the converter with optimal settings."""
        self._converter = html2text.HTML2Text()
        self._converter.ignore_links = False
        self._converter.ignore_images = True
        self._converter.ignore_emphasis = False
        self._converter.body_width = 0  # Don't wrap lines
        self._converter.ignore_tables = False
        self._converter.bypass_tables = False
        self._converter.single_line_break = True

    def convert(self, content: str | bytes, content_type: str) -> str:
        """Convert content to markdown based on content type.

        Args:
            content: The content to convert.
            content_type: The MIME type of the content.

        Returns:
            Markdown string.
        """
        content_type_lower = content_type.lower()

        # Handle PDF before any text decoding (PDF is binary)
        if "application/pdf" in content_type_lower:
            if isinstance(content, bytes):
                return self._convert_pdf(content)
            return "[PDF content received as text — binary data was likely corrupted]"

        # Handle bytes
        if isinstance(content, bytes):
            try:
                content = content.decode("utf-8")
            except UnicodeDecodeError:
                return f"[Binary content: {content_type}]"

        # Route by content type
        if "text/html" in content_type_lower:
            return self._convert_html(content)
        elif "text/plain" in content_type_lower:
            return content
        elif "application/json" in content_type_lower:
            return f"```json\n{content}\n```"
        elif "text/xml" in content_type_lower or "application/xml" in content_type_lower:
            return f"```xml\n{content}\n```"
        else:
            return f"[Binary content: {content_type}]"

    def _convert_pdf(self, content: bytes) -> str:
        """Extract text from PDF bytes.

        Args:
            content: Raw PDF bytes.

        Returns:
            Extracted text with page markers, or fallback message.
        """
        from agentic_cli.tools.pdf_utils import extract_pdf_text

        text = extract_pdf_text(content, page_markers=True)
        if not text:
            return "[PDF contained no extractable text]"
        return text

    def _convert_html(self, html: str) -> str:
        """Convert HTML to markdown.

        Args:
            html: HTML string to convert.

        Returns:
            Markdown string.
        """
        return self._converter.handle(html).strip()

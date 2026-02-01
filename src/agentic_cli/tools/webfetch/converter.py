"""HTML to Markdown conversion."""

from __future__ import annotations

import html2text


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
        # Handle bytes
        if isinstance(content, bytes):
            try:
                content = content.decode("utf-8")
            except UnicodeDecodeError:
                return f"[Binary content: {content_type}]"

        # Route by content type
        content_type_lower = content_type.lower()

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

    def _convert_html(self, html: str) -> str:
        """Convert HTML to markdown.

        Args:
            html: HTML string to convert.

        Returns:
            Markdown string.
        """
        return self._converter.handle(html).strip()

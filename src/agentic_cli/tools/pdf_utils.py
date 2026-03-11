"""Shared PDF text extraction utility.

Consolidates the three separate PDF extraction implementations
(knowledge_base/manager, knowledge_tools, webfetch/converter)
into a single function.
"""

from __future__ import annotations

import io
from pathlib import Path

from agentic_cli.logging import Loggers

logger = Loggers.tools()


def extract_pdf_text(
    source: Path | bytes,
    *,
    page_markers: bool = False,
) -> str:
    """Extract text from a PDF file or bytes.

    Args:
        source: File path or raw PDF bytes.
        page_markers: If True, prefix each page with ``--- Page N ---``.

    Returns:
        Extracted text, or empty string on failure.
    """
    try:
        import pypdf
    except ImportError:
        logger.warning("pypdf_not_installed")
        return ""

    try:
        if isinstance(source, (str, Path)):
            reader = pypdf.PdfReader(str(source))
        else:
            reader = pypdf.PdfReader(io.BytesIO(source))

        pages: list[str] = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                if page_markers:
                    pages.append(f"--- Page {i} ---\n{text.strip()}")
                else:
                    pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        logger.warning("pdf_text_extraction_failed", error=str(e))
        return ""

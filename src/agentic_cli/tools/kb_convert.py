"""Deprecated: the ingestion converter moved to ``agentic_cli.memory.kb``.

Import ``convert_document``, ``ConvertedDocument`` and ``UnsupportedDocument``
from ``agentic_cli.memory.kb``. This module is removed in 0.7.0.
"""

from __future__ import annotations

import warnings

from agentic_cli.memory.kb import ConvertedDocument, UnsupportedDocument, convert_document

__all__ = ["ConvertedDocument", "UnsupportedDocument", "convert_document"]

warnings.warn(
    "agentic_cli.tools.kb_convert is deprecated and will be removed in 0.7.0; import "
    "convert_document, ConvertedDocument and UnsupportedDocument from agentic_cli.memory.kb",
    DeprecationWarning,
    stacklevel=2,
)

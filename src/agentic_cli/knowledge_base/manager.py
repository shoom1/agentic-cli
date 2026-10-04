"""Deprecated: moved to ``agentic_cli.memory.kb``. Removed in 0.7.0.

``KnowledgeBaseManager`` here keeps the old signature: it reads its directory
and embedding model from settings, finds its summarizer in the turn in
progress, and keeps ``extract_text_from_pdf``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_cli.memory.kb import manager as _moved
from agentic_cli.memory.kb.manager import *  # noqa: F401,F403


class _TurnRegistrySummarizer:
    """The old lookup: the workflow's summarizer, from the turn in progress."""

    async def summarize(self, content: str, prompt: str) -> str:
        from agentic_cli.workflow.service_registry import LLM_SUMMARIZER, get_service

        summarizer = get_service(LLM_SUMMARIZER)
        if summarizer is None:
            return ""  # the knowledge base falls back to the preview
        return await summarizer.summarize(content, prompt)


# Default for ``summarizer=``: use a fresh ``_TurnRegistrySummarizer()``, as
# before. An explicit ``None`` (or any other summarizer) is passed through.
_TURN_REGISTRY: Any = object()


class KnowledgeBaseManager(_moved.KnowledgeBaseManager):
    """``agentic_cli.memory.kb.KnowledgeBaseManager`` with the pre-0.6.3 signature."""

    def __init__(
        self,
        settings: Any = None,
        use_mock: bool = False,
        base_dir: Path | None = None,
        embedding_service: Any = None,
        vector_store: Any = None,
        *,
        summarizer: Any = _TURN_REGISTRY,
    ) -> None:
        from agentic_cli.workflow.memory_services import embedding_config

        if base_dir is None:
            base_dir = (
                settings.knowledge_base_dir
                if settings is not None
                else Path.home() / ".agentic" / "knowledge_base"
            )
        super().__init__(
            base_dir,
            embedding=embedding_config(settings) if settings is not None else None,
            summarizer=_TurnRegistrySummarizer() if summarizer is _TURN_REGISTRY else summarizer,
            use_mock=use_mock,
            embedding_service=embedding_service,
            vector_store=vector_store,
        )

    @staticmethod
    def extract_text_from_pdf(file_path: Path) -> str:
        """Extract text from a PDF file (empty on failure)."""
        from agentic_cli.tools.pdf_utils import extract_pdf_text

        return extract_pdf_text(file_path)

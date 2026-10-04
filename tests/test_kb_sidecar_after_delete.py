"""A summary is not written for a document deleted while it was being built.

``kb_read`` builds a missing summary file (``documents/{id}.md``), and so does
``backfill_sidecars``, by awaiting the LLM summarizer and then writing the
result. A ``delete_document`` that ran during that call removed the document,
and the summary, which is derived from the document's text, was then written
back, and ``kb_read`` returned it as if the document still existed. The write
now happens only if the document is still there, checked under the lock that
deletion holds.
"""

from __future__ import annotations

from agentic_cli.memory.kb.manager import KnowledgeBaseManager
from agentic_cli.memory.kb.models import SourceType
from agentic_cli.tools.knowledge_tools import kb_read
from agentic_cli.workflow.service_registry import set_service_registry


class _DeletingSummarizer:
    """Summarizes, but the document is deleted while it does."""

    def __init__(self, kb: KnowledgeBaseManager, doc_id: str) -> None:
        self._kb = kb
        self._doc_id = doc_id

    async def summarize(self, content: str, prompt: str) -> str:
        self._kb.delete_document(self._doc_id)
        return "SUMMARY: a plan for airships\nCLAIMS:\n- airships float\nENTITIES:\n"


def _document_without_summary(tmp_path) -> tuple[KnowledgeBaseManager, str]:
    kb = KnowledgeBaseManager(base_dir=tmp_path / "kb", use_mock=True)
    doc = kb.ingest_document(
        content="a plan for airships", title="Plan", source_type=SourceType.USER
    )
    kb._sidecar_path(doc.id).unlink()  # as a document from before summaries
    return kb, doc.id


async def test_kb_read_does_not_bring_a_deleted_document_back(tmp_path):
    kb, doc_id = _document_without_summary(tmp_path)
    token = set_service_registry(
        {"kb_manager": kb, "llm_summarizer": _DeletingSummarizer(kb, doc_id)}
    )
    try:
        result = await kb_read(doc_id)
    finally:
        token.var.reset(token)

    assert result["success"] is False
    assert "not found" in result["error"].lower()
    assert not kb._sidecar_path(doc_id).exists()


async def test_backfill_does_not_bring_a_deleted_document_back(tmp_path):
    kb, doc_id = _document_without_summary(tmp_path)
    token = set_service_registry({"llm_summarizer": _DeletingSummarizer(kb, doc_id)})
    try:
        written = await kb.backfill_sidecars()
    finally:
        token.var.reset(token)

    assert written == 0
    assert not kb._sidecar_path(doc_id).exists()

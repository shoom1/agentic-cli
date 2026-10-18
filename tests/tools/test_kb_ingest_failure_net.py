"""An error nobody planned for ends one ingest call, not the turn."""

from __future__ import annotations

from agentic_cli.memory.kb import KnowledgeBase, KnowledgeBaseManager


class _FullDisk(KnowledgeBaseManager):
    def ingest_document(self, *args, **kwargs):
        raise OSError("disk full")


async def test_kb_ingest_text_reports_an_unexpected_error(tmp_path):
    from agentic_cli.tools.knowledge_tools import _ingest_text

    kb = KnowledgeBase({"project": _FullDisk(tmp_path / "kb", use_mock=True)})

    result = await _ingest_text(kb, "Some text.", title="T")

    assert result == {"success": False, "error": "Ingestion failed: disk full"}

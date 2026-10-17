"""The sidecar methods a host needs for lazy sidecars are public."""

from __future__ import annotations

from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType


def test_sidecar_path_names_the_documents_markdown_file(tmp_path):
    kb = KnowledgeBaseManager(tmp_path / "kb", use_mock=True)
    doc = kb.ingest_document(content="Body.", title="T", source_type=SourceType.USER)

    assert kb.sidecar_path(doc.id) == tmp_path / "kb" / "documents" / f"{doc.id}.md"
    assert kb.sidecar_path(doc.id).is_file()


def test_a_sidecar_is_not_written_for_a_deleted_document(tmp_path):
    kb = KnowledgeBaseManager(tmp_path / "kb", use_mock=True)
    doc = kb.ingest_document(content="Body.", title="T", source_type=SourceType.USER)
    kb.delete_document(doc.id)

    payload = {"summary": "Late.", "claims": [], "entities": {}}
    assert kb.write_sidecar_if_present(doc, payload) is False
    assert not kb.sidecar_path(doc.id).exists()

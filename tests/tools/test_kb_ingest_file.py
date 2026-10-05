"""``kb_ingest_file`` ingests text files and reports a file it cannot read.

Only a PDF had its text extracted: any other file was stored with empty
content, so a Markdown or plain-text file was reported as ingested yet had no
chunks and never matched a search. UTF-8 text files are now ingested as text,
and a binary file that is not a PDF is refused instead of stored unsearchable.

A path that could not be read (no permission, a directory that cannot be
searched, a NUL byte) raised instead of returning an error.
"""

from __future__ import annotations

import os

import pytest

from agentic_cli.memory.kb.manager import KnowledgeBaseManager
from agentic_cli.tools.knowledge_tools import kb_ingest_file
from agentic_cli.workflow.service_registry import set_service_registry

needs_non_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root reads files whatever their permissions",
)


@pytest.fixture
def kb(tmp_path):
    manager = KnowledgeBaseManager(base_dir=tmp_path / "kb", use_mock=True)
    token = set_service_registry({"kb_manager": manager})
    try:
        yield manager
    finally:
        token.var.reset(token)


def _stored_files(kb) -> list:
    return list(kb.files_dir.iterdir())


async def test_a_text_file_is_ingested_as_text(kb, tmp_path):
    notes = tmp_path / "notes.md"
    notes.write_text("# Mooring\n\nHow to moor a zeppelin in a crosswind.\n")

    result = await kb_ingest_file(path=str(notes))

    assert result["success"] is True
    assert result["chunks_created"] > 0
    doc = kb.get_document(result["document_id"])
    assert doc.content == notes.read_text()
    assert [r["document_title"] for r in kb.search("zeppelin")["results"]] == ["notes"]


async def test_a_binary_file_is_refused(kb, tmp_path):
    image = tmp_path / "diagram.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe")

    result = await kb_ingest_file(path=str(image))

    assert result["success"] is False
    assert "diagram.png" in result["error"]
    assert kb.list_documents() == []
    assert _stored_files(kb) == []


@needs_non_root
async def test_an_unreadable_file_returns_an_error(kb, tmp_path):
    secret = tmp_path / "locked.txt"
    secret.write_text("text")
    secret.chmod(0)
    try:
        result = await kb_ingest_file(path=str(secret))
    finally:
        secret.chmod(0o600)

    assert result["success"] is False
    assert "locked.txt" in result["error"]


@needs_non_root
async def test_a_file_in_an_unsearchable_directory_returns_an_error(kb, tmp_path):
    folder = tmp_path / "closed"
    folder.mkdir()
    (folder / "inside.txt").write_text("text")
    folder.chmod(0)
    try:
        result = await kb_ingest_file(path=str(folder / "inside.txt"))
    finally:
        folder.chmod(0o700)

    assert result["success"] is False
    assert "inside.txt" in result["error"]


async def test_a_path_that_cannot_name_a_file_returns_an_error(kb):
    result = await kb_ingest_file(path="notes\x00.md")

    assert result["success"] is False


async def test_hostile_html_is_refused_not_raised(kb, tmp_path, monkeypatch):
    """I-1: html2text fails an assert on ``<ol start>`` (an attribute with no
    value); that must come back as a refusal, not an exception out of the
    tool, and nothing must be stored."""
    import sys

    monkeypatch.setitem(sys.modules, "trafilatura", None)
    evil = tmp_path / "evil.html"
    evil.write_text("<html><body><ol start><li>x</li></ol></body></html>")

    result = await kb_ingest_file(path=str(evil))

    assert result["success"] is False
    assert "evil.html" in result["error"]
    assert kb.list_documents() == []


async def test_any_other_conversion_failure_is_also_refused(kb, tmp_path, monkeypatch):
    """Defensive: whatever a future converter raises besides
    ``UnsupportedDocument`` must still come back as a refusal, not an
    exception out of the tool."""
    from agentic_cli.tools import knowledge_tools

    def _boom(*a, **kw):
        raise RuntimeError("converter exploded")

    monkeypatch.setattr(knowledge_tools, "convert_document", _boom)
    notes = tmp_path / "notes.md"
    notes.write_text("# Notes\n")

    result = await kb_ingest_file(path=str(notes))

    assert result["success"] is False
    assert "notes.md" in result["error"]
    assert "RuntimeError" in result["error"]
    assert kb.list_documents() == []


async def test_a_cwd_that_cannot_be_statted_is_treated_as_a_local_file(kb, tmp_path, monkeypatch):
    """M-1: ``saved_pages_dir`` reads ``Path.cwd()``; with a deleted working
    directory an absolute-path ``kb_ingest_file`` must still ingest the file
    as local, not raise."""
    from agentic_cli.tools import knowledge_tools

    def _raise_cwd_gone(app_name):
        raise FileNotFoundError("cwd deleted")

    monkeypatch.setattr(knowledge_tools, "saved_pages_dir", _raise_cwd_gone)
    notes = tmp_path / "notes.md"
    notes.write_text("# Notes\n\nSome text about dirigibles.\n")

    result = await kb_ingest_file(path=str(notes))

    assert result["success"] is True
    assert "dirigibles" in kb.get_document(result["document_id"]).content

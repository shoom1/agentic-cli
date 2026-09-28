"""A document ID from ``metadata.json`` is a plain name, never a path.

Each document's ID names its files: ``documents/{id}.json`` (its text, read when
the knowledge base opens) and ``documents/{id}.md`` (its summary, which
``kb_read`` returns, or builds and writes when it is missing). The project
knowledge base lives in the project, so a cloned repository can ship
``metadata.json``, and an ID such as ``../../elsewhere/notes`` made ``kb_read``
return the text of a Markdown or JSON file elsewhere on disk and create new
files outside the knowledge base; deleting that document would have removed
them. A document whose ID is not 1-64 letters, digits, ``-`` or ``_`` (every ID
the framework has ever assigned is a UUID) is now skipped when loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_cli.knowledge_base._mocks import MockEmbeddingService, MockVectorStore
from agentic_cli.knowledge_base.manager import KnowledgeBaseManager
from agentic_cli.knowledge_base.models import SourceType
from agentic_cli.tools.knowledge_tools import kb_read
from agentic_cli.workflow.service_registry import set_service_registry

OUTSIDE = "outside notes"  # only in files outside the knowledge base
CRAFTED = "../../../../elsewhere/notes"  # from documents/ up to tmp_path


@pytest.fixture
def kb_dir(tmp_path) -> Path:
    return tmp_path / "project" / ".app" / "knowledge_base"


@pytest.fixture
def elsewhere(tmp_path) -> Path:
    folder = tmp_path / "elsewhere"
    folder.mkdir()
    return folder


def _open_kb(kb_dir: Path) -> KnowledgeBaseManager:
    return KnowledgeBaseManager(
        base_dir=kb_dir,
        embedding_service=MockEmbeddingService(),
        vector_store=MockVectorStore(index_path=kb_dir / "embeddings" / "index.mock"),
    )


def _ship_document_with_id(kb_dir: Path, doc_id: str, *, version: int | None = None) -> None:
    """Write a knowledge base whose metadata.json gives a document ``doc_id``,
    as a repository could ship it, next to an ordinary document."""
    kb = _open_kb(kb_dir)
    crafted = kb.ingest_document(content="a plan", title="Plan", source_type=SourceType.USER)
    kb.ingest_document(content="a sensible plan", title="Balloons", source_type=SourceType.USER)
    for leftover in (kb_dir / "documents").glob(f"{crafted.id}.*"):
        leftover.unlink()
    metadata = json.loads((kb_dir / "metadata.json").read_text())
    for entry in metadata["documents"]:
        if entry["id"] == crafted.id:
            entry["id"] = doc_id
            if version == 1:
                entry["content"] = "a plan"
    if version is not None:
        metadata["version"] = version
    (kb_dir / "metadata.json").write_text(json.dumps(metadata))


async def _read(kb: KnowledgeBaseManager, title: str, **kwargs) -> str:
    token = set_service_registry({"kb_manager": kb})
    try:
        return json.dumps(await kb_read(title, **kwargs))
    finally:
        token.var.reset(token)


async def test_an_outside_markdown_file_is_not_returned_as_the_summary(kb_dir, elsewhere):
    (elsewhere / "notes.md").write_text(OUTSIDE)
    _ship_document_with_id(kb_dir, CRAFTED)

    assert OUTSIDE not in await _read(_open_kb(kb_dir), "Plan")


async def test_a_summary_is_not_written_outside(kb_dir, elsewhere):
    _ship_document_with_id(kb_dir, CRAFTED)

    await _read(_open_kb(kb_dir), "Plan")

    assert list(elsewhere.iterdir()) == []


async def test_an_outside_json_file_is_not_loaded_as_the_text(kb_dir, elsewhere):
    (elsewhere / "notes.json").write_text(json.dumps({"content": OUTSIDE, "chunks": {}}))
    _ship_document_with_id(kb_dir, CRAFTED)

    assert OUTSIDE not in await _read(_open_kb(kb_dir), "Plan", full=True)


def test_migrating_an_old_format_does_not_write_outside(kb_dir, elsewhere):
    _ship_document_with_id(kb_dir, CRAFTED, version=1)

    _open_kb(kb_dir)

    assert list(elsewhere.iterdir()) == []


def test_deleting_does_not_remove_outside_files(kb_dir, elsewhere):
    (elsewhere / "notes.md").write_text(OUTSIDE)
    (elsewhere / "notes.json").write_text("{}")
    _ship_document_with_id(kb_dir, CRAFTED)
    kb = _open_kb(kb_dir)

    kb.delete_document(CRAFTED)
    kb.clear()

    assert sorted(p.name for p in elsewhere.iterdir()) == ["notes.json", "notes.md"]


@pytest.mark.parametrize("doc_id", [CRAFTED, "/abs/notes", "notes\x00", "", "a.b"])
def test_the_other_documents_still_load(kb_dir, doc_id):
    _ship_document_with_id(kb_dir, doc_id)

    kb = _open_kb(kb_dir)

    assert [d.title for d in kb.list_documents()] == ["Balloons"]

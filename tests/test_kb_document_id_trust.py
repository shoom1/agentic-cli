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

A name can lead elsewhere without a ``..`` too: git keeps symlinks, so the
repository can ship ``documents/{id}.md`` linked to a file elsewhere (which
``kb_read`` returned), ``ingest_log.md`` linked to a shell rc (each ingest and
delete appended a line holding the title there), or a whole ``documents/``
folder linked elsewhere. The knowledge base now reads and appends its files
only when they are regular files, never through a symlink, and does not open a
scope whose own folders (or lock file) are symlinks.
"""

from __future__ import annotations

from tests.kb_registry import kb_services

import contextlib
import json
import shutil
from pathlib import Path

import pytest

from agentic_cli.memory._core.mock_embeddings import MockEmbeddingService
from agentic_cli.memory.kb import KnowledgeBase
from agentic_cli.memory.kb._mock_vector_store import MockVectorStore
from agentic_cli.memory.kb.manager import KnowledgeBaseManager
from agentic_cli.memory.kb.models import SourceType
from agentic_cli.tools.knowledge_tools import kb_list, kb_read
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
    token = set_service_registry(kb_services(kb))
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


# -- symlinks: a name in the knowledge base never reaches a file elsewhere ----

FOLDERS = ["documents", "embeddings", "files", "concepts"]


def _ship(kb_dir: Path) -> str:
    """A knowledge base holding one document, "Plan", as a repository could ship it; its ID."""
    doc = _open_kb(kb_dir).ingest_document(content="a plan", title="Plan", source_type=SourceType.USER)
    return doc.id


def _link(path: Path, target: Path) -> None:
    """Replace ``path`` (a file or a folder, or nothing) with a symlink to ``target``."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.is_symlink() or path.exists():
        path.unlink()
    path.symlink_to(target, target_is_directory=target.is_dir())


def _open_scopes(kb_dir: Path, tmp_path: Path) -> KnowledgeBase:
    """The project and user scopes, as the workflow opens them."""
    return KnowledgeBase.open({"project": kb_dir, "user": tmp_path / "user_kb"}, use_mock=True)


async def _read_from(kb: KnowledgeBase, title: str, **kwargs) -> str:
    token = set_service_registry({"knowledge_base": kb})
    try:
        return json.dumps(await kb_read(title, **kwargs))
    finally:
        token.var.reset(token)


async def test_a_summary_file_that_links_elsewhere_is_not_returned(kb_dir, elsewhere):
    (elsewhere / "notes.md").write_text(OUTSIDE)
    doc_id = _ship(kb_dir)
    _link(kb_dir / "documents" / f"{doc_id}.md", elsewhere / "notes.md")

    assert OUTSIDE not in await _read(_open_kb(kb_dir), "Plan")
    assert (elsewhere / "notes.md").read_text() == OUTSIDE


async def test_a_text_file_that_links_elsewhere_is_not_loaded(kb_dir, elsewhere):
    (elsewhere / "notes.json").write_text(json.dumps({"content": OUTSIDE, "chunks": {}}))
    doc_id = _ship(kb_dir)
    _link(kb_dir / "documents" / f"{doc_id}.json", elsewhere / "notes.json")

    assert OUTSIDE not in await _read(_open_kb(kb_dir), "Plan", full=True)


def test_a_metadata_file_that_links_elsewhere_is_not_read(kb_dir, elsewhere):
    _ship(kb_dir)
    shipped = elsewhere / "metadata.json"
    shutil.move(kb_dir / "metadata.json", shipped)
    before = shipped.read_bytes()
    (kb_dir / "metadata.json").symlink_to(shipped)

    kb = _open_kb(kb_dir)
    listed = [d.title for d in kb.list_documents()]
    kb.ingest_document(content="a sensible plan", title="Balloons", source_type=SourceType.USER)

    assert listed == []
    assert shipped.read_bytes() == before


def test_an_ingest_log_that_links_elsewhere_is_not_appended_to(kb_dir, elsewhere):
    shell_rc = elsewhere / "shellrc"
    shell_rc.write_bytes(b"# the user's shell rc\n")
    kb = _open_kb(kb_dir)
    (kb_dir / "ingest_log.md").symlink_to(shell_rc)

    doc = kb.ingest_document(
        content="a plan", title="$(touch elsewhere/ran) payload", source_type=SourceType.USER
    )
    kb.delete_document(doc.id)

    assert shell_rc.read_bytes() == b"# the user's shell rc\n"


def test_a_title_is_logged_as_one_clean_line(kb_dir):
    kb = _open_kb(kb_dir)

    kb.ingest_document(
        content="a plan",
        title="first\nsecond\r\x1b[31mred\x00 end\t",
        source_type=SourceType.ARXIV,
        metadata={"arxiv_id": "2401.00001\n- forged line"},
    )

    log = (kb_dir / "ingest_log.md").read_text(encoding="utf-8")
    assert log.endswith("\n") and log.count("\n") == 1
    assert all(ch.isprintable() for ch in log[:-1])
    assert '"first second [31mred end"' in log
    assert "2401.00001 - forged line" in log


async def test_a_documents_folder_that_links_elsewhere_is_not_read(kb_dir, elsewhere, tmp_path):
    doc_id = _ship(kb_dir)
    (elsewhere / f"{doc_id}.md").write_text(OUTSIDE)
    (elsewhere / f"{doc_id}.json").write_text(json.dumps({"content": OUTSIDE, "chunks": {}}))
    _link(kb_dir / "documents", elsewhere)

    kb = _open_scopes(kb_dir, tmp_path)

    assert OUTSIDE not in await _read_from(kb, "Plan")
    assert OUTSIDE not in await _read_from(kb, "Plan", full=True)


@pytest.mark.parametrize("folder", FOLDERS)
async def test_a_folder_that_links_elsewhere_leaves_its_scope_out(kb_dir, elsewhere, tmp_path, folder):
    kb_dir.mkdir(parents=True)
    (kb_dir / folder).symlink_to(elsewhere, target_is_directory=True)

    kb = _open_scopes(kb_dir, tmp_path)
    stored = await kb.ingest_bytes(b"a plan", extension=".txt", title="Plan")
    kb.write_concept("Idea", "a body", [stored.document.id])

    assert list(elsewhere.iterdir()) == []
    assert kb.scopes == ("user",)


@pytest.mark.parametrize("folder", FOLDERS)
def test_a_knowledge_base_whose_folder_links_elsewhere_does_not_open(kb_dir, elsewhere, folder):
    kb_dir.mkdir(parents=True)
    (kb_dir / folder).symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(OSError, match="symbolic link"):
        _open_kb(kb_dir)


def test_a_lock_file_that_links_elsewhere_is_not_created(kb_dir, elsewhere, tmp_path):
    kb_dir.mkdir(parents=True)
    (kb_dir / ".lock").symlink_to(elsewhere / "created")

    kb = _open_scopes(kb_dir, tmp_path)

    assert list(elsewhere.iterdir()) == []
    assert kb.scopes == ("user",)


async def test_a_documents_folder_linked_after_opening_is_not_used(kb_dir, elsewhere):
    """A ``git pull`` can replace a folder while a session runs."""
    doc_id = _ship(kb_dir)
    kb = _open_kb(kb_dir)
    (elsewhere / f"{doc_id}.md").write_text(OUTSIDE)
    _link(kb_dir / "documents", elsewhere)

    read = await _read(kb, "Plan")
    with contextlib.suppress(OSError):
        kb.ingest_document(content="more", title="More", source_type=SourceType.USER)
    token = set_service_registry(kb_services(kb))
    try:
        listed = kb_list()
    finally:
        token.var.reset(token)

    assert OUTSIDE not in read
    assert sorted(p.name for p in elsewhere.iterdir()) == [f"{doc_id}.md"]
    assert listed["success"] is False and "symbolic link" in listed["error"]


def test_deleting_removes_nothing_through_a_files_folder_linked_elsewhere(kb_dir, elsewhere):
    kb = _open_kb(kb_dir)
    doc = kb.ingest_document(
        content="a plan", title="Plan", source_type=SourceType.USER, file_bytes=b"%PDF", file_extension=".pdf"
    )
    (elsewhere / f"{doc.id}.pdf").write_bytes(b"theirs")
    _link(kb_dir / "files", elsewhere)

    with contextlib.suppress(OSError):
        kb.delete_document(doc.id)

    assert (elsewhere / f"{doc.id}.pdf").read_bytes() == b"theirs"


def test_a_concept_page_that_links_elsewhere_is_not_read(kb_dir, elsewhere):
    page = elsewhere / "notes.md"
    page.write_text(f"---\nslug: notes\ntitle: Notes\n---\n{OUTSIDE}\n")
    doc_id = _ship(kb_dir)
    kb = KnowledgeBase.open({"project": kb_dir}, use_mock=True)
    kb.write_concept("Idea", "a body", [doc_id])
    (kb_dir / "concepts" / "notes.md").symlink_to(page)

    found = kb.search_concepts("outside")
    listed = kb.manager("project").concepts.list()

    assert OUTSIDE not in json.dumps(found)
    assert [c["slug"] for c in listed] == ["idea"]


def test_a_keyword_index_that_links_elsewhere_is_not_loaded(kb_dir, elsewhere):
    KnowledgeBaseManager(kb_dir, use_mock=True).ingest_document(
        content="a plan", title="Plan", source_type=SourceType.USER
    )
    index = kb_dir / "embeddings" / "bm25_index.json"
    shipped = json.loads(index.read_text())
    shipped["documents"] = [["zeppelin"] for _ in shipped["documents"]]
    (elsewhere / "index.json").write_text(json.dumps(shipped))
    _link(index, elsewhere / "index.json")

    reopened = KnowledgeBaseManager(kb_dir, use_mock=True)

    assert reopened.search("zeppelin")["results"] == []


@pytest.mark.parametrize("base", ["elsewhere/knowledge_base", "project/.app/../../elsewhere/knowledge_base"])
def test_a_knowledge_base_outside_the_folder_it_must_be_within_does_not_open(tmp_path, base):
    with pytest.raises(ValueError):
        KnowledgeBaseManager(tmp_path / base, use_mock=True, within=tmp_path / "project")

    assert not (tmp_path / "elsewhere").exists()

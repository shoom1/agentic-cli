"""A knowledge base opens the scopes it can, and the memory store opens without its model.

Opening was all or nothing: one scope that could not open (a read-only
working directory, a database that is not one), an embedding model that did
not load (an offline first run, the model not cached) or an index that could
not be read (a git-LFS pointer) made ``KnowledgeBase.open`` raise, and with
it the workflow's initialization. A scope that fails is now left out with a
warning, a model that does not load leaves the knowledge base (and the
memory store) searching by keyword, with one warning, and a row that does
not parse is left out of what the scope lists, the scope open.
"""

from __future__ import annotations

import os

import pytest

from agentic_cli.memory import EmbeddingService
from agentic_cli.memory.kb import KnowledgeBase, KnowledgeBaseManager, SourceType
from agentic_cli.memory.kb import _db as db_module
from agentic_cli.memory.kb import knowledge_base as knowledge_base_module
from agentic_cli.memory.kb import manager as manager_module
from tests.memory.fake_embeddings import FakeEmbeddingService
from tests.memory.kb_storage import raw_write, read_rows


class _Log:
    """Records what a module logs: (level, event, fields)."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict]] = []

    def _at(level: str):
        def record(self, event, **fields) -> None:
            self.records.append((level, event, fields))

        return record

    debug, info, warning, error = _at("debug"), _at("info"), _at("warning"), _at("error")
    del _at

    def warnings(self, event: str) -> list[dict]:
        return [fields for level, name, fields in self.records if level == "warning" and name == event]


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    """No test here may build a real embedding service (it would download a model)."""
    monkeypatch.setattr(EmbeddingService, "is_available", staticmethod(lambda: False))


@pytest.fixture
def log(monkeypatch) -> _Log:
    recorder = _Log()
    for module in (manager_module, knowledge_base_module, db_module):
        monkeypatch.setattr(module, "logger", recorder, raising=False)
    return recorder


class _OfflineModel:
    """An embedding service whose model cannot load: an offline first run."""

    model_name = "all-MiniLM-L6-v2"
    batch_size = 32

    def _fail(self):
        raise OSError("We couldn't connect to 'https://huggingface.co' to load this model")

    @property
    def embedding_dim(self) -> int:
        self._fail()

    def embed_text(self, text):
        self._fail()

    def embed_batch(self, texts):
        self._fail()

    def chunk_document(self, content, chunk_size=512, overlap=50):
        return FakeEmbeddingService().chunk_document(content, chunk_size, overlap)


# -- the embedding model -------------------------------------------------------


async def test_a_model_that_cannot_load_opens_the_knowledge_base_keyword_only(tmp_path, log):
    kb = KnowledgeBase.open(
        {"project": tmp_path / "project", "user": tmp_path / "user"}, embedding_service=_OfflineModel()
    )
    await kb.ingest_text("Otters nest in burrows along the river.", title="Otters")

    assert kb.scopes == ("project", "user")
    assert [hit["document_title"] for hit in kb.search("otters")["results"]] == ["Otters"]
    [warning] = log.warnings("kb_keyword_search_only")  # one for both scopes
    assert "huggingface" in warning["error"]


def test_a_manager_whose_model_cannot_load_searches_by_keyword(tmp_path, log):
    manager = KnowledgeBaseManager(tmp_path / "kb", embedding_service=_OfflineModel())
    manager.ingest_document(content="Herons wade at dawn.", title="Herons", source_type=SourceType.USER)

    assert [hit["document_title"] for hit in manager.search("herons")["results"]] == ["Herons"]
    assert len(log.warnings("kb_keyword_search_only")) == 1


# -- a scope that cannot open --------------------------------------------------


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a folder the user cannot write")
async def test_a_read_only_working_directory_leaves_the_user_scope_working(tmp_path, log):
    read_only = tmp_path / "read-only"
    read_only.mkdir()
    read_only.chmod(0o555)
    try:
        kb = KnowledgeBase.open(
            {"project": read_only / ".app" / "knowledge_base", "user": tmp_path / "user"},
            use_mock=True,
            within={"project": read_only},
        )
        found = await kb.ingest_text("Basalt forms from lava.", title="Basalt")
    finally:
        read_only.chmod(0o755)

    assert kb.scopes == ("user",)
    assert found.scope == "user"
    assert [hit["document_title"] for hit in kb.search("basalt")["results"]] == ["Basalt"]
    [warning] = log.warnings("kb_scope_unavailable")
    assert warning["scope"] == "project" and "PermissionError" in warning["error"]


async def test_a_row_that_does_not_parse_is_left_out_and_its_scope_stays_open(tmp_path, log):
    project, user = tmp_path / "project", tmp_path / "user"
    granite = KnowledgeBaseManager(project, use_mock=True).ingest_document(
        content="Granite cools slowly.", title="Granite", source_type=SourceType.USER
    )
    raw_write(project, ("UPDATE documents SET source_type = 'podcast' WHERE id = ?", [granite.id]))

    kb = KnowledgeBase.open({"project": project, "user": user}, use_mock=True)
    found = await kb.ingest_text("Basalt forms from lava.", title="Basalt")

    assert kb.scopes == ("project", "user")
    assert found.scope == "project"
    assert [d.document.title for d in kb.list_documents()] == ["Basalt"]
    assert read_rows(project, "SELECT source_type FROM documents WHERE id = ?", [granite.id]) == [("podcast",)]
    [warning] = log.warnings("kb_row_unreadable")
    assert "podcast" in warning["error"]


def test_when_no_scope_opens_the_first_scopes_error_is_raised(tmp_path, log):
    KnowledgeBaseManager(tmp_path / "project", use_mock=True).close()
    raw_write(tmp_path / "project", "UPDATE info SET value = '9' WHERE key = 'format'")
    (tmp_path / "user").mkdir()
    (tmp_path / "user" / "kb.sqlite").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")

    with pytest.raises(ValueError, match="project.*newer version.*format 9"):
        KnowledgeBase.open({"project": tmp_path / "project", "user": tmp_path / "user"}, use_mock=True)


# -- a database that cannot be read --------------------------------------------


@pytest.mark.parametrize(
    "contents",
    [b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1234\n", b"\x89PNG\r\n\x1a\n" + bytes(56)],
    ids=["git-lfs-pointer", "png"],
)
async def test_a_database_that_is_not_one_leaves_its_scope_out_unchanged(tmp_path, log, contents):
    project = tmp_path / "project"
    project.mkdir()
    (project / "kb.sqlite").write_bytes(contents)

    kb = KnowledgeBase.open({"project": project, "user": tmp_path / "user"}, use_mock=True)
    found = await kb.ingest_text("Basalt forms from lava.", title="Basalt")

    assert kb.scopes == ("user",)
    assert found.scope == "user"
    assert (project / "kb.sqlite").read_bytes() == contents  # never replaced or rewritten
    [warning] = log.warnings("kb_scope_unavailable")
    assert warning["scope"] == "project" and "DatabaseError" in warning["error"]


async def test_documents_ingested_without_the_model_are_embedded_when_it_loads_again(tmp_path, log):
    """A session whose model did not load stores its chunks without vectors;
    the next open with a working model embeds them."""
    kb_dir = tmp_path / "kb"
    await KnowledgeBase.open({"project": kb_dir}, embedding_service=FakeEmbeddingService()).ingest_text(
        "Otters nest in burrows.", title="Otters"
    )
    offline = KnowledgeBase.open({"project": kb_dir}, embedding_service=_OfflineModel())
    await offline.ingest_text("Herons wade at dawn.", title="Herons")
    assert read_rows(kb_dir, "SELECT count(*) FROM chunks WHERE vector IS NULL") == [(1,)]

    service = FakeEmbeddingService()
    KnowledgeBase.open({"project": kb_dir}, embedding_service=service)

    assert service.calls == 1
    assert read_rows(kb_dir, "SELECT count(*), count(vector) FROM chunks") == [(2, 2)]

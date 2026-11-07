"""A knowledge base opens the scopes it can, and the memory store opens without its model.

Opening was all or nothing: one scope that could not open (a read-only
working directory, a document header it cannot parse), an embedding model
that did not load (an offline first run, the model not cached) or an index
FAISS cannot read (a git-LFS pointer) made ``KnowledgeBase.open`` raise, and
with it the workflow's initialization. A scope that fails is now left out
with a warning, a model that does not load leaves the knowledge base (and
the memory store) searching by keyword, with one warning, and an unreadable
index counts as no index: the stored chunks are embedded again.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentic_cli.memory import EmbeddingService, MemoryStore
from agentic_cli.memory import store as store_module
from agentic_cli.memory.kb import KnowledgeBase, KnowledgeBaseManager, SourceType
from agentic_cli.memory.kb import knowledge_base as knowledge_base_module
from agentic_cli.memory.kb import manager as manager_module
from agentic_cli.memory.kb import vector_store as vector_store_module
from tests.memory.fake_embeddings import FakeEmbeddingService
from tests.memory.kb_extra import require_kb_module


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
    for module in (manager_module, knowledge_base_module, vector_store_module, store_module):
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


def _files(folder: Path) -> dict[str, bytes]:
    """Every file under ``folder``, by relative path, with its bytes."""
    return {
        str(path.relative_to(folder)): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


# -- the embedding model -------------------------------------------------------


async def test_a_model_that_cannot_load_opens_the_knowledge_base_keyword_only(tmp_path, log):
    require_kb_module("faiss")  # without FAISS the model is never loaded

    kb = KnowledgeBase.open(
        {"project": tmp_path / "project", "user": tmp_path / "user"}, embedding_service=_OfflineModel()
    )
    await kb.ingest_text("Otters nest in burrows along the river.", title="Otters")

    assert kb.scopes == ("project", "user")
    assert [hit["document_title"] for hit in kb.search("otters")["results"]] == ["Otters"]
    [warning] = log.warnings("kb_keyword_search_only")  # one for both scopes
    assert "huggingface" in warning["error"]


def test_a_manager_whose_model_cannot_load_searches_by_keyword(tmp_path, log):
    require_kb_module("faiss")

    manager = KnowledgeBaseManager(tmp_path / "kb", embedding_service=_OfflineModel())
    manager.ingest_document(content="Herons wade at dawn.", title="Herons", source_type=SourceType.USER)

    assert [hit["document_title"] for hit in manager.search("herons")["results"]] == ["Herons"]
    assert len(log.warnings("kb_keyword_search_only")) == 1


def test_a_memory_store_whose_model_cannot_load_searches_by_substring(tmp_path, log):
    store = MemoryStore(tmp_path / "memory", embedding_service=_OfflineModel())
    store.store("The user prefers metric units")

    assert [item.content for item in store.search("metric")] == ["The user prefers metric units"]
    assert len(log.warnings("memory_substring_search_only")) == 1


def test_memories_embedded_earlier_are_searched_by_substring_without_the_model(tmp_path, log):
    MemoryStore(tmp_path / "memory", embedding_service=FakeEmbeddingService()).store("The mascot is an otter")

    store = MemoryStore(tmp_path / "memory", embedding_service=_OfflineModel())

    assert [item.content for item in store.search("otter")] == ["The mascot is an otter"]


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


async def test_a_header_with_an_unknown_source_type_leaves_its_scope_out_and_unwritten(tmp_path, log):
    project, user = tmp_path / "project", tmp_path / "user"
    KnowledgeBaseManager(project, use_mock=True).ingest_document(
        content="Granite cools slowly.", title="Granite", source_type=SourceType.USER
    )
    metadata = json.loads((project / "metadata.json").read_text())
    metadata["documents"][0]["source_type"] = "podcast"
    (project / "metadata.json").write_text(json.dumps(metadata))
    before = _files(project)

    kb = KnowledgeBase.open({"project": project, "user": user}, use_mock=True)
    found = await kb.ingest_text("Basalt forms from lava.", title="Basalt")

    assert kb.scopes == ("user",)
    assert found.scope == "user"
    assert [hit["document_title"] for hit in kb.search("basalt")["results"]] == ["Basalt"]
    assert _files(project) == before
    [warning] = log.warnings("kb_scope_unavailable")
    assert warning["scope"] == "project" and "podcast" in warning["error"]


def test_when_no_scope_opens_the_first_scopes_error_is_raised(tmp_path, log):
    for name in ("project", "user"):
        KnowledgeBaseManager(tmp_path / name, use_mock=True).ingest_document(
            content="Granite cools slowly.", title="Granite", source_type=SourceType.USER
        )
        path = tmp_path / name / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["documents"][0]["source_type"] = f"{name}-podcast"
        path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="project-podcast"):
        KnowledgeBase.open({"project": tmp_path / "project", "user": tmp_path / "user"}, use_mock=True)


# -- an index that cannot be read ----------------------------------------------


def _kb_with_one_document(kb_dir: Path) -> str:
    """A knowledge base with a FAISS index of one chunk; the chunk's text."""
    manager = KnowledgeBaseManager(kb_dir, embedding_service=FakeEmbeddingService())
    doc = manager.ingest_document(
        content="Otters nest in burrows.", title="Otters", source_type=SourceType.USER
    )
    return doc.chunks[0].content


@pytest.mark.parametrize(
    "broken, contents",
    [
        ("index.faiss", b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1234\n"),
        ("index.faiss", b"\x89PNG\r\n\x1a\n" + bytes(56)),
        ("index.mappings.json", b'{"id_map": []}'),
        ("index.mappings.json", b"not json"),
        ("index.mappings.json", b'{"id_map": {"zero": "c"}, "chunk_to_faiss": {}, "next_id": 1}'),
    ],
)
async def test_an_unreadable_index_is_rebuilt_from_the_stored_text(tmp_path, log, broken, contents):
    require_kb_module("faiss")
    project = tmp_path / "project"
    chunk_text = _kb_with_one_document(project)
    (project / "embeddings" / broken).write_bytes(contents)
    service = FakeEmbeddingService()

    kb = KnowledgeBase.open({"project": project, "user": tmp_path / "user"}, embedding_service=service)

    assert kb.scopes == ("project", "user")
    assert service.calls == 1  # the stored chunk, embedded again
    store = kb.manager("project")._vector_store
    [(chunk_id, _)] = store.search(service.vector(chunk_text), 1)
    assert kb.manager("project")._chunks[chunk_id].content == chunk_text
    assert log.warnings("kb_vector_index_unreadable")
    # The rebuilt index is the one on disk: the next open reads it as it is.
    reopened = FakeEmbeddingService()
    KnowledgeBase.open({"project": project}, embedding_service=reopened)
    assert reopened.calls == 0


async def test_documents_ingested_without_the_model_are_embedded_when_it_loads_again(tmp_path, log):
    """Keyword mode leaves index.faiss and its model record as they were; a
    record that still named the model made the next open with a working model
    keep that index, and the documents ingested meanwhile were never embedded."""
    require_kb_module("faiss")
    kb_dir = tmp_path / "kb"
    await KnowledgeBase.open({"project": kb_dir}, embedding_service=FakeEmbeddingService()).ingest_text(
        "Otters nest in burrows.", title="Otters"
    )
    offline = KnowledgeBase.open({"project": kb_dir}, embedding_service=_OfflineModel())
    await offline.ingest_text("Herons wade at dawn.", title="Herons")

    manager = KnowledgeBase.open({"project": kb_dir}, embedding_service=FakeEmbeddingService()).manager("project")

    assert len(manager._chunks) == 2
    assert manager._vector_store.chunk_ids() == set(manager._chunks)
    assert manager._vector_store.size == 2

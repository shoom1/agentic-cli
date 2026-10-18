"""``KnowledgeBase``: the project and user knowledge bases as one.

Each scope is a ``KnowledgeBaseManager`` (one directory). ``KnowledgeBase``
does what spans them, which the tools did by hand: search merged by
reciprocal rank fusion, lookups in scope order, one catalogue, sidecars,
concept pages that may cite any scope, and a backfill of every scope.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from agentic_cli.memory import EmbeddingService, MockEmbeddingService
from agentic_cli.memory.kb import (
    BackfillAlreadyRunning,
    BackfillResult,
    DocumentNotFound,
    KnowledgeBase,
    KnowledgeBaseManager,
    ScopedDocument,
    SourceType,
)
from tests.memory.pdf_bytes import pdf_with_text

EPOCH = datetime(2026, 1, 1)


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    monkeypatch.setattr(EmbeddingService, "is_available", staticmethod(lambda: False))


def _manager(path, summarizer=None) -> KnowledgeBaseManager:
    return KnowledgeBaseManager(path, use_mock=True, summarizer=summarizer)


def _add(manager, title, minute=0, content=None, source=SourceType.USER, **kwargs):
    doc = manager.ingest_document(
        content=f"Text of {title}." if content is None else content, title=title, source_type=source, **kwargs
    )
    doc.updated_at = EPOCH + timedelta(minutes=minute)
    return doc


@pytest.fixture
def two(tmp_path) -> KnowledgeBase:
    return KnowledgeBase({"project": _manager(tmp_path / "project"), "user": _manager(tmp_path / "user")})


# -- scopes ------------------------------------------------------------------


def test_the_scopes_keep_their_order(two):
    assert two.scopes == ("project", "user")


def test_one_manager_under_two_names_is_one_scope(tmp_path):
    shared = _manager(tmp_path / "shared")

    kb = KnowledgeBase({"project": shared, "user": shared})

    assert kb.scopes == ("project",)
    assert kb.manager("user") is shared


def test_two_managers_on_one_directory_are_one_scope(tmp_path):
    first = _manager(tmp_path / "kb")
    second = _manager(tmp_path / "other" / ".." / "kb")

    kb = KnowledgeBase({"project": first, "user": second})

    assert kb.scopes == ("project",)
    assert kb.manager("user") is first


def test_one_scope_is_fine(tmp_path):
    kb = KnowledgeBase({"user": _manager(tmp_path / "user")})

    assert kb.scopes == ("user",)


def test_an_unknown_scope_is_a_key_error(two):
    with pytest.raises(KeyError, match="team"):
        two.manager("team")


@pytest.mark.parametrize("managers", [{}, {"": None}, {"project": None}])
def test_a_knowledge_base_needs_real_scopes(managers):
    with pytest.raises(ValueError):
        KnowledgeBase(managers)


def test_open_builds_a_manager_per_directory(tmp_path):
    kb = KnowledgeBase.open({"project": tmp_path / "project", "user": tmp_path / "user"}, use_mock=True)

    assert kb.scopes == ("project", "user")
    assert kb.manager("project").kb_dir == tmp_path / "project"
    assert kb.manager("user").kb_dir == tmp_path / "user"


def test_open_makes_one_scope_of_one_directory(tmp_path):
    (tmp_path / "kb").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "kb")

    kb = KnowledgeBase.open({"project": tmp_path / "kb", "user": tmp_path / "link"}, use_mock=True)

    assert kb.scopes == ("project",)
    assert kb.manager("user") is kb.manager("project")


def test_open_shares_the_given_service_across_scopes(tmp_path):
    service = MockEmbeddingService(embedding_dim=16)

    kb = KnowledgeBase.open(
        {"project": tmp_path / "project", "user": tmp_path / "user"}, embedding_service=service
    )

    assert kb.manager("project")._embedding_service is service
    assert kb.manager("user")._embedding_service is service


def test_open_builds_one_service_for_all_scopes(tmp_path, monkeypatch):
    built = []

    class _Service(MockEmbeddingService):
        """Stands in for EmbeddingService: built like it, mock to the manager."""

        is_available = staticmethod(lambda: True)

        def __init__(self, model_name, batch_size, device):
            super().__init__(model_name=model_name, batch_size=batch_size)
            built.append(self)

    monkeypatch.setattr("agentic_cli.memory.kb.knowledge_base.EmbeddingService", _Service)

    kb = KnowledgeBase.open({"project": tmp_path / "project", "user": tmp_path / "user"})

    assert len(built) == 1
    assert kb.manager("project")._embedding_service is built[0]
    assert kb.manager("user")._embedding_service is built[0]


def test_open_passes_the_summarizer_to_every_scope(tmp_path):
    class _Summarizer:
        async def summarize(self, content, prompt):
            return "SUMMARY: s"

    summarizer = _Summarizer()

    kb = KnowledgeBase.open(
        {"project": tmp_path / "project", "user": tmp_path / "user"}, summarizer=summarizer, use_mock=True
    )

    assert kb.manager("project")._summarizer is summarizer
    assert kb.manager("user")._summarizer is summarizer


# -- search ------------------------------------------------------------------


def test_search_tags_every_hit_with_its_scope(two):
    _add(two.manager("project"), "Basalt", content="Basalt columns from lava.")
    _add(two.manager("user"), "Glacier", content="A glacier carved the lava field.")

    result = two.search("lava")

    assert {(r["document_title"], r["scope"]) for r in result["results"]} == {
        ("Basalt", "project"),
        ("Glacier", "user"),
    }
    assert result["total_matches"] == 2


def test_search_merges_by_rank_not_by_score(tmp_path):
    """Scores from separate indexes are not comparable; ranks are."""

    class _Fixed:
        def __init__(self, hits):
            self.hits = hits

        def search(self, query, filters=None, top_k=10):
            return {"results": [dict(h) for h in self.hits], "total_matches": len(self.hits)}

    kb = KnowledgeBase({
        "project": _Fixed([{"document_id": "p1", "score": 0.5}, {"document_id": "p2", "score": 0.4}]),
        "user": _Fixed([{"document_id": "u1", "score": 0.9}, {"document_id": "u2", "score": 0.8}]),
    })

    merged = [r["document_id"] for r in kb.search("q")["results"]]

    assert max(merged.index("p1"), merged.index("u1")) < min(merged.index("p2"), merged.index("u2"))
    assert all(r["score"] == round(r["score"], 4) and r["score"] < 0.1 for r in kb.search("q")["results"])


def test_a_document_in_two_scopes_keeps_the_first_scopes_hit(tmp_path):
    class _Fixed:
        def __init__(self, hits):
            self.hits = hits

        def search(self, query, filters=None, top_k=10):
            return {"results": [dict(h) for h in self.hits], "total_matches": len(self.hits)}

    kb = KnowledgeBase({
        "project": _Fixed([{"document_id": "d", "chunk_content": "project copy"}, {"document_id": "p"}]),
        "user": _Fixed([{"document_id": "d", "chunk_content": "user copy"}]),
    })

    results = kb.search("q")["results"]

    assert results[0]["document_id"] == "d"
    assert results[0]["chunk_content"] == "project copy"
    assert results[0]["scope"] == "project"
    assert results[0]["score"] == round(1 / 61 + 1 / 61, 4)  # both ranks add up


def test_search_cuts_the_merged_list_at_top_k(two):
    for i in range(4):
        _add(two.manager("project"), f"Lava {i}", content=f"Lava note {i}.")
        _add(two.manager("user"), f"Lava user {i}", content=f"Lava user note {i}.")

    assert len(two.search("lava", top_k=3)["results"]) == 3


def test_a_failing_later_scope_is_skipped(tmp_path):
    class _Broken:
        def search(self, *args, **kwargs):
            raise RuntimeError("user knowledge base broken")

    project = _manager(tmp_path / "project")
    _add(project, "Basalt", content="Basalt from lava.")
    kb = KnowledgeBase({"project": project, "user": _Broken()})

    result = kb.search("lava")

    assert [r["document_title"] for r in result["results"]] == ["Basalt"]


def test_a_failing_first_scope_raises(tmp_path):
    class _Broken:
        def search(self, *args, **kwargs):
            raise RuntimeError("project knowledge base broken")

    kb = KnowledgeBase({"project": _Broken(), "user": _manager(tmp_path / "user")})

    with pytest.raises(RuntimeError, match="project knowledge base broken"):
        kb.search("lava")


def test_with_one_scope_the_scores_are_the_managers(tmp_path):
    project = _manager(tmp_path / "project")
    _add(project, "Basalt", content="Basalt from lava.")
    kb = KnowledgeBase({"project": project})

    assert kb.search("lava")["results"][0]["score"] == project.search("lava")["results"][0]["score"]


def test_search_passes_the_filters_to_every_scope(two):
    _add(two.manager("project"), "Paper", content="Lava paper.", source=SourceType.ARXIV)
    _add(two.manager("user"), "Page", content="Lava page.", source=SourceType.WEB)

    result = two.search("lava", filters={"source_type": "arxiv"})

    assert [r["document_title"] for r in result["results"]] == ["Paper"]


# -- lookups -----------------------------------------------------------------


def test_find_document_looks_in_the_project_first(two):
    _add(two.manager("project"), "Shared title", content="project version")
    _add(two.manager("user"), "Shared title", content="user version")

    found = two.find_document("Shared title")

    assert found.scope == "project"
    assert found.document.content == "project version"


def test_find_document_falls_back_to_later_scopes(two):
    doc = _add(two.manager("user"), "Glacier survey")

    found = two.find_document("glacier")

    assert found == ScopedDocument(doc, "user")


def test_find_document_tries_the_whole_chain_in_each_scope_first(two):
    """A title match in the project wins over an exact ID in the user scope (as 0.6.3)."""
    user_doc = _add(two.manager("user"), "Elsewhere")
    project_doc = _add(two.manager("project"), f"Notes on {user_doc.id}")

    assert two.find_document(user_doc.id).document is project_doc


def test_find_document_by_id_prefix(two):
    doc = _add(two.manager("user"), "Glacier survey")

    assert two.find_document(doc.id[:8]).document is doc


def test_an_unknown_document_is_none(two):
    assert two.find_document("nothing like it") is None
    assert two.get_document("no-such-id") is None


def test_get_document_takes_exact_ids_only(two):
    doc = _add(two.manager("user"), "Glacier survey")

    assert two.get_document(doc.id) == ScopedDocument(doc, "user")
    assert two.get_document(doc.id[:8]) is None


def test_a_shared_directory_is_looked_up_once(tmp_path):
    calls = []
    shared = _manager(tmp_path / "shared")
    real = shared.find_document
    shared.find_document = lambda ref: calls.append(ref) or real(ref)

    assert KnowledgeBase({"project": shared, "user": shared}).find_document("x") is None
    assert calls == ["x"]


# -- the catalogue -------------------------------------------------------------


def test_the_catalogue_is_newest_first_across_scopes(two):
    _add(two.manager("project"), "Old", minute=1)
    _add(two.manager("user"), "Middle", minute=2)
    _add(two.manager("project"), "New", minute=3)

    listed = two.list_documents()

    assert [(s.document.title, s.scope) for s in listed] == [
        ("New", "project"),
        ("Middle", "user"),
        ("Old", "project"),
    ]


def test_the_catalogue_filters_before_it_cuts(two):
    _add(two.manager("user"), "Zeppelin mooring", minute=0)
    for n in range(1, 26):
        _add(two.manager("project"), f"Note {n}", minute=n)

    assert [s.document.title for s in two.list_documents(title_or_author="zeppelin")] == ["Zeppelin mooring"]


def test_a_scope_is_filtered_before_it_is_cut(two):
    """A match behind the scope's newest documents is still listed."""
    _add(two.manager("project"), "Zeppelin mooring", minute=0)
    for n in range(1, 26):
        _add(two.manager("project"), f"Note {n}", minute=n)

    assert [s.document.title for s in two.list_documents(title_or_author="zeppelin")] == ["Zeppelin mooring"]


def test_the_catalogue_matches_titles_and_authors_case_insensitively(two):
    _add(two.manager("project"), "Volcano notes", metadata={"authors": ["Ada Lovelace"]})
    _add(two.manager("user"), "Glacier survey", metadata={"authors": ["Mary Somerville"]})

    assert [s.document.title for s in two.list_documents(title_or_author="SOMERVILLE")] == ["Glacier survey"]
    assert [s.document.title for s in two.list_documents(title_or_author="volcano")] == ["Volcano notes"]


def test_the_catalogue_filters_by_source_type(two):
    _add(two.manager("project"), "Paper", source=SourceType.ARXIV)
    _add(two.manager("user"), "Page", source=SourceType.WEB)

    assert [s.document.title for s in two.list_documents(source_type=SourceType.WEB)] == ["Page"]


def test_the_catalogue_stays_within_limit(two):
    for n in range(15):
        _add(two.manager("project"), f"Project {n}", minute=2 * n)
        _add(two.manager("user"), f"User {n}", minute=2 * n + 1)

    assert len(two.list_documents(limit=20)) == 20
    assert len(two.list_documents(limit=None)) == 30


def test_the_catalogue_of_one_scope(two):
    _add(two.manager("project"), "Mine")
    _add(two.manager("user"), "Theirs")

    assert [s.document.title for s in two.list_documents(scope="user")] == ["Theirs"]


def test_the_catalogue_of_an_alias_lists_its_scope(tmp_path):
    shared = _manager(tmp_path / "shared")
    _add(shared, "Only")
    kb = KnowledgeBase({"project": shared, "user": shared})

    assert [(s.document.title, s.scope) for s in kb.list_documents(scope="user")] == [("Only", "project")]


def test_the_catalogue_never_lists_a_document_twice(tmp_path):
    class _Same:
        def __init__(self, docs):
            self.docs = docs

        def list_documents(self, source_type=None, limit=100):
            return list(self.docs)

    doc = _add(_manager(tmp_path / "kb"), "Twice")
    kb = KnowledgeBase({"project": _Same([doc]), "user": _Same([doc])})

    assert [s.document.title for s in kb.list_documents()] == ["Twice"]


def test_a_failing_later_scope_is_left_out_of_the_catalogue(tmp_path):
    class _Broken:
        def list_documents(self, *args, **kwargs):
            raise RuntimeError("broken")

    project = _manager(tmp_path / "project")
    _add(project, "Mine")

    kb = KnowledgeBase({"project": project, "user": _Broken()})

    assert [s.document.title for s in kb.list_documents()] == ["Mine"]


# -- text ----------------------------------------------------------------------


def test_read_text_returns_the_stored_text(two):
    doc = _add(two.manager("user"), "Glacier", content="The glacier retreated.")

    assert two.read_text(ScopedDocument(doc, "user")) == ("The glacier retreated.", False)


def test_read_text_cuts_at_max_chars(two):
    doc = _add(two.manager("project"), "Long", content="x" * 50)

    assert two.read_text(ScopedDocument(doc, "project"), max_chars=10) == ("x" * 10, True)


def test_an_older_pdf_without_text_is_read_from_its_file(two):
    doc = _add(
        two.manager("user"), "Ledger", content="", file_bytes=pdf_with_text("Magma ledger entries"),
        file_extension=".pdf",
    )

    assert two.read_text(ScopedDocument(doc, "user")) == ("Magma ledger entries", False)


# -- sidecars ------------------------------------------------------------------


class _Counting:
    def __init__(self, delay: float = 0.0, delete=None) -> None:
        self.calls: list[str] = []
        self.delay = delay
        self.delete = delete

    async def summarize(self, content, prompt):
        self.calls.append(content)
        await asyncio.sleep(self.delay)
        if self.delete is not None:
            self.delete()
        return "SUMMARY: A lazy summary.\nCLAIMS:\n- One claim.\n"


def _without_sidecar(manager, title="Doc", **kwargs):
    doc = _add(manager, title, **kwargs)
    manager.sidecar_path(doc.id).unlink()
    return doc


async def test_sidecar_returns_the_existing_sidecar(two):
    doc = _add(two.manager("project"), "Doc")

    text = await two.sidecar(ScopedDocument(doc, "project"))

    assert text == two.manager("project").sidecar_path(doc.id).read_text()
    assert "## Summary" in text


async def test_a_missing_sidecar_is_written_from_the_summarizer(tmp_path):
    summarizer = _Counting()
    manager = _manager(tmp_path / "kb", summarizer)
    doc = _without_sidecar(manager, content="raw body")
    kb = KnowledgeBase({"project": manager})

    text = await kb.sidecar(ScopedDocument(doc, "project"))

    assert "A lazy summary." in text and "- One claim." in text
    assert manager.sidecar_path(doc.id).exists()
    assert summarizer.calls == ["raw body"]


async def test_two_first_reads_make_one_summarizer_call(tmp_path):
    summarizer = _Counting(delay=0.05)
    manager = _manager(tmp_path / "kb", summarizer)
    doc = _without_sidecar(manager)
    kb = KnowledgeBase({"project": manager})

    await asyncio.gather(kb.sidecar(ScopedDocument(doc, "project")), kb.sidecar(ScopedDocument(doc, "project")))

    assert len(summarizer.calls) == 1


async def test_the_sidecar_lock_comes_from_the_managers_accessor(tmp_path):
    manager = _manager(tmp_path / "kb", _Counting())
    doc = _without_sidecar(manager)
    seen = []
    original = manager.get_or_create_sidecar_lock
    manager.get_or_create_sidecar_lock = lambda doc_id: seen.append(doc_id) or original(doc_id)

    await KnowledgeBase({"project": manager}).sidecar(ScopedDocument(doc, "project"))

    assert seen == [doc.id]


async def test_an_older_pdfs_sidecar_is_written_from_its_files_text(tmp_path):
    summarizer = _Counting()
    manager = _manager(tmp_path / "kb", summarizer)
    doc = _without_sidecar(
        manager, "Ledger", content="", file_bytes=pdf_with_text("Magma ledger entries"), file_extension=".pdf"
    )

    await KnowledgeBase({"project": manager}).sidecar(ScopedDocument(doc, "project"))

    assert summarizer.calls == ["Magma ledger entries"]


async def test_a_document_deleted_during_the_summarizer_call_is_not_found(tmp_path):
    manager = _manager(tmp_path / "kb")
    doc = _without_sidecar(manager)
    manager._summarizer = _Counting(delete=lambda: manager.delete_document(doc.id))

    with pytest.raises(DocumentNotFound):
        await KnowledgeBase({"project": manager}).sidecar(ScopedDocument(doc, "project"))

    assert not manager.sidecar_path(doc.id).exists()


async def test_deleting_a_document_drops_its_sidecar_lock(tmp_path):
    manager = _manager(tmp_path / "kb", _Counting())
    doc = _without_sidecar(manager)
    await KnowledgeBase({"project": manager}).sidecar(ScopedDocument(doc, "project"))
    assert doc.id in manager._sidecar_locks

    manager.delete_document(doc.id)

    assert doc.id not in manager._sidecar_locks


# -- concepts ------------------------------------------------------------------


def test_a_concept_cites_documents_of_any_scope(two):
    project_doc = _add(two.manager("project"), "Basalt")
    user_doc = _add(two.manager("user"), "Glacier")

    result = two.write_concept(
        "Volcanic rocks", "What the knowledge base knows.", [project_doc.id, user_doc.id, "not-a-real-id"]
    )

    assert result["success"] is True
    assert result["slug"] == "volcanic-rocks"
    assert result["invalid_sources"] == ["not-a-real-id"]
    assert (two.manager("project").kb_dir / "concepts" / "volcanic-rocks.md").exists()
    assert not (two.manager("user").kb_dir / "concepts").exists()


def test_a_concept_with_no_valid_source_fails(two):
    result = two.write_concept("X", "b", ["fake-1", "fake-2"])

    assert result["success"] is False
    assert "at least one valid source" in result["error"]


def test_a_concept_with_no_sources_fails(two):
    assert two.write_concept("X", "b", [])["success"] is False


def test_an_explicit_slug_updates_the_concept(two):
    doc = _add(two.manager("project"), "P")
    two.write_concept("T", "v1", [doc.id], slug="x")

    assert two.write_concept("T", "v2", [doc.id], slug="x")["action"] == "updated"


def test_concepts_are_searched_in_the_first_scope(two):
    doc = _add(two.manager("project"), "P")
    for i in range(5):
        two.write_concept(f"Diffusion {i}", "Synthesis.", [doc.id], slug=f"c{i}")

    assert [hit["slug"] for hit in two.search_concepts("diffusion 3")] == ["c3"]
    assert len(two.search_concepts("diffusion", limit=2)) == 2
    assert two.search_concepts("nothing") == []


# -- maintenance ---------------------------------------------------------------


async def test_backfill_runs_over_every_scope(tmp_path):
    project = _manager(tmp_path / "project", _Counting())
    user = _manager(tmp_path / "user", _Counting())
    _without_sidecar(project, "P")
    _without_sidecar(user, "U1")
    _without_sidecar(user, "U2")
    seen = []

    results = await KnowledgeBase({"project": project, "user": user}).backfill_sidecars(
        progress=lambda scope, done, total, doc: seen.append((scope, done, total, doc.title))
    )

    assert results == [BackfillResult("project", 1), BackfillResult("user", 2)]
    assert seen == [("project", 0, 1, "P"), ("user", 0, 2, "U1"), ("user", 1, 2, "U2")]


async def test_one_scopes_failure_does_not_stop_the_next(tmp_path):
    project = _manager(tmp_path / "project", _Counting())
    user = _manager(tmp_path / "user", _Counting())
    _without_sidecar(user, "U")
    project._backfill_running = True  # another backfill holds the project scope

    results = await KnowledgeBase({"project": project, "user": user}).backfill_sidecars()

    assert results[0].scope == "project" and isinstance(results[0].error, BackfillAlreadyRunning)
    assert results[1] == BackfillResult("user", 1)


async def test_a_shared_directory_is_backfilled_once(tmp_path):
    shared = _manager(tmp_path / "shared", _Counting())
    _without_sidecar(shared)

    results = await KnowledgeBase({"project": shared, "user": shared}).backfill_sidecars()

    assert results == [BackfillResult("project", 1)]


def test_stats_per_scope(two):
    _add(two.manager("user"), "U")

    stats = two.stats()

    assert list(stats) == ["project", "user"]
    assert stats["project"]["document_count"] == 0
    assert stats["user"]["document_count"] == 1


# -- review fixes (PR 5 fix wave) -----------------------------------------------


def _case_insensitive(tmp_path) -> bool:
    (tmp_path / "KB").mkdir()
    return (tmp_path / "kb").exists()


def test_open_makes_one_scope_of_two_spellings_of_a_directory(tmp_path):
    if not _case_insensitive(tmp_path):
        pytest.skip("case-sensitive filesystem")

    kb = KnowledgeBase.open({"project": tmp_path / "KB", "user": tmp_path / "kb"}, use_mock=True)

    assert kb.scopes == ("project",)
    assert kb.manager("user") is kb.manager("project")


def test_two_managers_on_two_spellings_of_a_directory_are_one_scope(tmp_path):
    if not _case_insensitive(tmp_path):
        pytest.skip("case-sensitive filesystem")

    kb = KnowledgeBase({"project": _manager(tmp_path / "KB"), "user": _manager(tmp_path / "kb")})

    assert kb.scopes == ("project",)
    assert kb.manager("user") is kb.manager("project")


def test_a_later_scopes_document_without_authors_does_not_break_the_catalogue(two):
    _add(two.manager("project"), "Mine", metadata={"authors": ["Ada"]})
    _add(two.manager("user"), "Theirs", metadata={"authors": None})

    assert {s.document.title for s in two.list_documents()} == {"Mine", "Theirs"}
    assert [s.document.title for s in two.list_documents(title_or_author="ada")] == ["Mine"]


def test_an_author_list_with_a_non_string_still_matches(two):
    doc = _add(two.manager("project"), "Mixed")
    doc.metadata["authors"] = [7, "Ada"]  # set after ingestion: the sidecar writer wants strings

    assert [s.document.title for s in two.list_documents(title_or_author="ada")] == ["Mixed"]


async def test_a_document_deleted_before_the_sidecar_call_makes_no_summarizer_call(tmp_path):
    summarizer = _Counting()
    manager = _manager(tmp_path / "kb", summarizer)
    doc = _without_sidecar(manager)
    manager.delete_document(doc.id)

    with pytest.raises(DocumentNotFound):
        await KnowledgeBase({"project": manager}).sidecar(ScopedDocument(doc, "project"))

    assert summarizer.calls == []


async def test_a_copy_of_a_live_document_gets_its_sidecar(tmp_path):
    import dataclasses

    manager = _manager(tmp_path / "kb", _Counting())
    doc = _without_sidecar(manager)

    text = await KnowledgeBase({"project": manager}).sidecar(ScopedDocument(dataclasses.replace(doc), "project"))

    assert "A lazy summary." in text
    assert manager.sidecar_path(doc.id).exists()


def test_open_validates_the_scopes_before_creating_a_directory(tmp_path):
    with pytest.raises(ValueError):
        KnowledgeBase.open({"project": tmp_path / "a", "": tmp_path / "b"}, use_mock=True)

    assert not (tmp_path / "a").exists()
    assert not (tmp_path / "b").exists()


def _recording_service(monkeypatch):
    built = []

    class _Service(MockEmbeddingService):
        is_available = staticmethod(lambda: True)

        def __init__(self, model_name, batch_size, device):
            super().__init__(model_name=model_name, batch_size=batch_size)
            built.append(self)

    monkeypatch.setattr("agentic_cli.memory.kb.knowledge_base.EmbeddingService", _Service)
    return built


def test_open_builds_no_service_in_mock_mode(tmp_path, monkeypatch):
    built = _recording_service(monkeypatch)

    KnowledgeBase.open({"project": tmp_path / "project"}, use_mock=True)

    assert built == []


def test_open_builds_no_service_without_the_extra(tmp_path, monkeypatch):
    built = _recording_service(monkeypatch)
    monkeypatch.setattr(
        "agentic_cli.memory.kb.knowledge_base.EmbeddingService.is_available", staticmethod(lambda: False)
    )

    KnowledgeBase.open({"project": tmp_path / "project"})

    assert built == []


def test_open_constructs_one_manager_for_two_spellings_of_a_directory(tmp_path, monkeypatch):
    constructed = []

    class _Recording(KnowledgeBaseManager):
        def __init__(self, *args, **kwargs):
            constructed.append(args)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("agentic_cli.memory.kb.knowledge_base.KnowledgeBaseManager", _Recording)
    (tmp_path / "kb").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "kb")

    KnowledgeBase.open({"project": tmp_path / "kb", "user": tmp_path / "link"}, use_mock=True)

    assert len(constructed) == 1

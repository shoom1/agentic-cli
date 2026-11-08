"""workflow/memory_services.py is where settings become memory-package arguments."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.memory import EmbeddingService, MemoryStore, MockEmbeddingService
from agentic_cli.tools.knowledge_tools import kb_read
from agentic_cli.workflow import memory_services
from agentic_cli.workflow.service_registry import set_service_registry


class _Summarizer:
    async def summarize(self, content: str, prompt: str) -> str:
        return "SUMMARY: From the summarizer.\n"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A temp home (no real user settings) and a temp project as the cwd."""
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("AGENTIC_KNOWLEDGE_BASE_SUMMARIZE", raising=False)
    monkeypatch.chdir(tmp_path / "project")


@pytest.fixture
def settings(tmp_path):
    return BaseSettings(workspace_dir=tmp_path / "workspace", knowledge_base_use_mock=True)


def test_the_knowledge_base_uses_the_settings_embedding_model(tmp_path):
    settings = BaseSettings(
        workspace_dir=tmp_path / "w", embedding_model="m-x", knowledge_base_use_mock=True
    )

    kb = memory_services.open_knowledge_base(settings, summarizer=None)

    assert kb.manager("project").get_stats()["embedding_model"] == "m-x"


def test_the_memory_store_lives_in_the_workspace(settings):
    store = memory_services.open_memory_store(settings)
    store.store("a fact")

    assert isinstance(store, MemoryStore)
    assert (settings.workspace_dir / "memory" / "memories.json").is_file()


def test_the_mock_setting_gives_the_memory_store_no_embedder(settings):
    """Hash embeddings carry no meaning, so the store searches by substring."""
    store = memory_services.open_memory_store(settings)

    assert store._embedding_service is None


def test_memory_search_in_mock_mode_finds_the_matching_memory(settings):
    store = memory_services.open_memory_store(settings)
    for text in ("Project uses Python 3.12", "User prefers dark mode", "The team mascot is an otter"):
        store.store(text)

    assert [item.content for item in store.search("otter", limit=2)] == ["The team mascot is an otter"]


def test_the_memory_store_uses_the_service_it_is_given(settings):
    service = MockEmbeddingService(embedding_dim=16)

    store = memory_services.open_memory_store(settings, embedding_service=service)

    assert store._embedding_service is service


def test_no_embedding_service_in_mock_mode(settings, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("mock mode must not build an embedding service")

    monkeypatch.setattr(EmbeddingService, "is_available", staticmethod(lambda: True))
    monkeypatch.setattr(EmbeddingService, "__init__", refuse)

    assert memory_services.build_embedding_service(settings) is None


def test_no_embedding_service_without_sentence_transformers(tmp_path, monkeypatch):
    monkeypatch.setattr(EmbeddingService, "is_available", staticmethod(lambda: False))
    settings = BaseSettings(workspace_dir=tmp_path / "w", knowledge_base_use_mock=False)

    assert memory_services.build_embedding_service(settings) is None


def test_the_embedding_service_is_the_one_the_settings_name(tmp_path, monkeypatch):
    monkeypatch.setattr(EmbeddingService, "is_available", staticmethod(lambda: True))
    settings = BaseSettings(
        workspace_dir=tmp_path / "w", knowledge_base_use_mock=False,
        embedding_model="m-x", embedding_batch_size=7, embedding_device="cpu",
    )

    service = memory_services.build_embedding_service(settings)

    assert isinstance(service, EmbeddingService)
    assert (service.model_name, service.batch_size, service.device) == ("m-x", 7, "cpu")
    assert service._model is None  # built, not loaded


def test_the_knowledge_base_has_both_scopes_where_they_were(settings, tmp_path):
    kb = memory_services.open_knowledge_base(settings, summarizer=None)

    assert kb.scopes == ("project", "user")
    assert kb.manager("project").kb_dir == tmp_path / "project" / f".{settings.app_name}" / "knowledge_base"
    assert kb.manager("user").kb_dir == settings.knowledge_base_dir


def test_one_scope_when_both_resolve_to_the_same_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = BaseSettings(workspace_dir=tmp_path / ".agentic_cli", knowledge_base_use_mock=True)

    kb = memory_services.open_knowledge_base(settings, summarizer=None)

    assert kb.scopes == ("project",)
    assert kb.manager("user") is kb.manager("project")


async def test_the_setting_decides_whether_the_knowledge_base_summarizes(tmp_path):
    on = BaseSettings(workspace_dir=tmp_path / "on", knowledge_base_use_mock=True)
    off = BaseSettings(
        workspace_dir=tmp_path / "off", knowledge_base_use_mock=True, knowledge_base_summarize=False
    )

    kb_on = memory_services.open_knowledge_base(on, summarizer=_Summarizer())
    kb_off = memory_services.open_knowledge_base(off, summarizer=_Summarizer())

    assert (await kb_on.manager("user").generate_sidecar_payload("Body.", title="T"))["summary"] == "From the summarizer."
    assert (await kb_off.manager("user").generate_sidecar_payload("Body.", title="T"))["summary"] == "Body."


def test_the_knowledge_base_shares_the_given_service(tmp_path):
    service = MockEmbeddingService(embedding_dim=16)
    settings = BaseSettings(workspace_dir=tmp_path / "w", knowledge_base_use_mock=False)

    kb = memory_services.open_knowledge_base(settings, summarizer=None, embedding_service=service)

    assert kb.manager("project")._service is service
    assert kb.manager("user")._service is service


def test_one_scope_can_be_opened_alone(settings):
    kb = memory_services.open_knowledge_base(settings, summarizer=None, scopes=("user",))

    assert kb.scopes == ("user",)
    assert not memory_services.project_kb_dir(settings).exists()


def test_an_unknown_scope_is_refused(settings):
    with pytest.raises(ValueError, match="team"):
        memory_services.open_knowledge_base(settings, summarizer=None, scopes=("team",))


# -- the project scope's own folders are the project's ------------------------
# A cloned repository can ship ./.{app} or ./.{app}/knowledge_base as a
# symlink (git keeps them): the project knowledge base would then create,
# replace and read files in the folder it points to (another repository's
# index.md, another project's knowledge base). The project scope is opened only
# when every folder between the working directory and it is a real folder; the
# user scope's own folder may be a link (to another disk, say).

OUTSIDE = "outside notes"


def _seeded(folder: Path) -> Path:
    """A folder of someone else's, holding files with the knowledge base's names."""
    folder.mkdir(parents=True)
    (folder / "index.md").write_text("THEIR INDEX")
    (folder / "metadata.json").write_text("{}")
    return folder


def _snapshot(folder: Path) -> list[tuple[str, bytes | None]]:
    return sorted(
        (str(p.relative_to(folder)), None if p.is_dir() else p.read_bytes()) for p in folder.rglob("*")
    )


@pytest.mark.parametrize("linked", ["knowledge_base", "app"])
async def test_a_project_knowledge_base_that_links_elsewhere_is_left_out(settings, tmp_path, linked):
    target = _seeded(tmp_path / "elsewhere")
    before = _snapshot(target)
    kb_dir = memory_services.project_kb_dir(settings)
    if linked == "knowledge_base":
        kb_dir.parent.mkdir()
        kb_dir.symlink_to(target, target_is_directory=True)
    else:
        kb_dir.parent.symlink_to(target, target_is_directory=True)

    kb = memory_services.open_knowledge_base(settings, summarizer=None)
    stored = await kb.ingest_text("a plan", title="Plan")

    assert _snapshot(target) == before
    assert kb.scopes == ("user",) and stored.scope == "user"


@pytest.mark.parametrize("shape", ["a file at knowledge_base", "a file at .app", "a dangling link at .app"])
def test_something_else_where_the_project_knowledge_base_belongs_leaves_it_out(settings, tmp_path, shape):
    kb_dir = memory_services.project_kb_dir(settings)
    if shape == "a file at knowledge_base":
        kb_dir.parent.mkdir()
        kb_dir.write_text("not a folder")
    elif shape == "a file at .app":
        kb_dir.parent.write_text("not a folder")
    else:
        kb_dir.parent.symlink_to(tmp_path / "nowhere", target_is_directory=True)

    kb = memory_services.open_knowledge_base(settings, summarizer=None)

    assert kb.scopes == ("user",)
    assert not (tmp_path / "nowhere").exists()


async def test_a_project_knowledge_base_swapped_for_a_link_while_open_is_not_read(settings, tmp_path):
    """A ``git pull`` can replace the folder while a session runs."""
    kb = memory_services.open_knowledge_base(settings, summarizer=None)
    stored = await kb.ingest_text("a plan", title="Plan")
    kb_dir = memory_services.project_kb_dir(settings)
    other = tmp_path / "elsewhere" / "knowledge_base"
    shutil.copytree(kb_dir, other)
    (other / "documents" / f"{stored.document.id}.md").write_text(OUTSIDE)
    shutil.rmtree(kb_dir)
    kb_dir.symlink_to(other, target_is_directory=True)

    token = set_service_registry({"knowledge_base": kb})
    try:
        result = await kb_read("Plan")
    finally:
        token.var.reset(token)

    assert OUTSIDE not in json.dumps(result)
    assert result["success"] is False


def test_a_user_knowledge_base_that_is_a_link_still_opens(settings, tmp_path):
    disk = tmp_path / "other-disk" / "knowledge_base"
    disk.mkdir(parents=True)
    settings.knowledge_base_dir.parent.mkdir(parents=True)
    settings.knowledge_base_dir.symlink_to(disk, target_is_directory=True)

    kb = memory_services.open_knowledge_base(settings, summarizer=None)

    assert kb.scopes == ("project", "user")

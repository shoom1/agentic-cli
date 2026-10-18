"""workflow/memory_services.py is where settings become memory-package arguments."""

from __future__ import annotations

import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.memory import EmbeddingService, MemoryStore, MockEmbeddingService
from agentic_cli.workflow import memory_services


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

    assert kb.manager("project")._embedding_service is service
    assert kb.manager("user")._embedding_service is service


def test_one_scope_can_be_opened_alone(settings):
    kb = memory_services.open_knowledge_base(settings, summarizer=None, scopes=("user",))

    assert kb.scopes == ("user",)
    assert not memory_services.project_kb_dir(settings).exists()


def test_an_unknown_scope_is_refused(settings):
    with pytest.raises(ValueError, match="team"):
        memory_services.open_knowledge_base(settings, summarizer=None, scopes=("team",))

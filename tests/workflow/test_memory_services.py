"""workflow/memory_services.py is where settings become memory-package arguments."""

from __future__ import annotations

import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.memory import MemoryStore
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


def test_the_project_and_user_knowledge_bases_live_where_they_did(settings, tmp_path):
    project, user = memory_services.build_knowledge_bases(settings, summarizer=None)

    assert project.kb_dir == tmp_path / "project" / f".{settings.app_name}" / "knowledge_base"
    assert user.kb_dir == settings.knowledge_base_dir


def test_one_object_when_both_resolve_to_the_same_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = BaseSettings(workspace_dir=tmp_path / ".agentic_cli", knowledge_base_use_mock=True)
    assert memory_services.project_kb_dir(settings).resolve() == settings.knowledge_base_dir.resolve()

    project, user = memory_services.build_knowledge_bases(settings, summarizer=None)

    assert project is user


async def test_the_setting_decides_whether_the_summarizer_is_passed(tmp_path):
    on = BaseSettings(workspace_dir=tmp_path / "on", knowledge_base_use_mock=True)
    off = BaseSettings(
        workspace_dir=tmp_path / "off", knowledge_base_use_mock=True, knowledge_base_summarize=False
    )

    _, user_on = memory_services.build_knowledge_bases(on, summarizer=_Summarizer())
    _, user_off = memory_services.build_knowledge_bases(off, summarizer=_Summarizer())

    assert (await user_on.generate_sidecar_payload("Body.", title="T"))["summary"] == "From the summarizer."
    assert (await user_off.generate_sidecar_payload("Body.", title="T"))["summary"] == "Body."


def test_build_knowledge_base_uses_the_settings_embedding_model(tmp_path):
    settings = BaseSettings(
        workspace_dir=tmp_path / "w", embedding_model="m-x", knowledge_base_use_mock=True
    )

    kb = memory_services.build_knowledge_base(settings, tmp_path / "kb")

    assert kb.get_stats()["embedding_model"] == "m-x"


def test_the_memory_store_lives_in_the_workspace(settings):
    store = memory_services.build_memory_store(settings)
    store.store("a fact")

    assert isinstance(store, MemoryStore)
    assert (settings.workspace_dir / "memory" / "memories.json").is_file()


def test_the_mock_setting_gives_the_memory_store_no_embedder(settings):
    """Hash embeddings carry no meaning, so the store searches by substring."""
    store = memory_services.build_memory_store(settings)

    assert store._embedding_service is None


def test_memory_search_in_mock_mode_finds_the_matching_memory(settings):
    store = memory_services.build_memory_store(settings)
    for text in ("Project uses Python 3.12", "User prefers dark mode", "The team mascot is an otter"):
        store.store(text)

    assert [item.content for item in store.search("otter", limit=2)] == ["The team mascot is an otter"]

"""Tests for on_session_end auto-fact-extraction wiring.

Covers BaseWorkflowManager.on_session_end, including sourcing messages from the
live session when the caller passes none (so the CLI can call it argument-free
on exit).
"""

from typing import AsyncGenerator

import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.service_registry import MEMORY_STORE


# ---------------------------------------------------------------------------
# Minimal concrete subclass + fakes
# ---------------------------------------------------------------------------


class _FakeManager(BaseWorkflowManager):
    """Concrete BaseWorkflowManager for unit-testing base-class methods."""

    @property
    def backend_type(self) -> str:
        return "test"

    async def _do_initialize(self) -> None:
        pass

    async def process(
        self, message: str, user_id: str, session_id: str | None = None
    ) -> AsyncGenerator[WorkflowEvent, None]:
        if False:
            yield  # type: ignore[misc]

    async def reinitialize(self, model: str | None = None, preserve_sessions: bool = True) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    async def _extract_session_data(self, session_id: str) -> tuple[list[dict], str | None]:
        return getattr(self, "_fake_messages", []), None

    async def _inject_session_messages(
        self, session_id: str, messages: list[dict], current_agent: str | None = None
    ) -> None:
        pass

    def _get_state_tools(self) -> list:
        return []


class _FakeMemoryStore:
    def __init__(self) -> None:
        self.stored: list[tuple[str, list[str] | None]] = []

    def store(self, content: str, tags: list[str] | None = None, importance: int = 5) -> str:
        self.stored.append((content, tags))
        return "fake-id"


def _settings(tmp_path, **kw) -> BaseSettings:
    return BaseSettings(workspace_dir=tmp_path, **kw)


# ---------------------------------------------------------------------------
# on_session_end fact extraction
# ---------------------------------------------------------------------------


class TestOnSessionEnd:
    async def test_disabled_returns_empty(self, tmp_path):
        settings = _settings(tmp_path, auto_extract_session_facts=False)
        mgr = _FakeManager(agent_configs=[], settings=settings)
        mgr._services[MEMORY_STORE] = _FakeMemoryStore()
        assert await mgr.on_session_end([{"role": "user", "content": "hi"}]) == []

    async def test_enabled_without_memory_store_returns_empty(self, tmp_path):
        settings = _settings(tmp_path, auto_extract_session_facts=True)
        mgr = _FakeManager(agent_configs=[], settings=settings)
        assert await mgr.on_session_end([{"role": "user", "content": "hi"}]) == []

    async def test_extracts_and_stores_facts(self, tmp_path, monkeypatch):
        settings = _settings(tmp_path, auto_extract_session_facts=True)
        mgr = _FakeManager(agent_configs=[], settings=settings)
        store = _FakeMemoryStore()
        mgr._services[MEMORY_STORE] = store

        async def fake_generate(prompt, max_tokens=500):
            return "User prefers dark mode\nProject uses Python 3.12"

        monkeypatch.setattr(mgr, "generate_simple", fake_generate)

        facts = await mgr.on_session_end([{"role": "user", "content": "..."}])
        assert facts == ["User prefers dark mode", "Project uses Python 3.12"]
        assert [c for c, _ in store.stored] == facts
        assert store.stored[0][1] == ["auto-extracted", "session"]

    async def test_sources_messages_from_session_when_none_passed(self, tmp_path, monkeypatch):
        settings = _settings(tmp_path, auto_extract_session_facts=True)
        mgr = _FakeManager(agent_configs=[], settings=settings)
        mgr._fake_messages = [{"role": "user", "content": "please remember PROJECT_X"}]
        mgr._services[MEMORY_STORE] = _FakeMemoryStore()

        captured = {}

        async def fake_generate(prompt, max_tokens=500):
            captured["prompt"] = prompt
            return "Fact one"

        monkeypatch.setattr(mgr, "generate_simple", fake_generate)

        facts = await mgr.on_session_end()  # no messages -> source from session
        assert facts == ["Fact one"]
        assert "PROJECT_X" in captured["prompt"]

    async def test_empty_extraction_stores_nothing(self, tmp_path, monkeypatch):
        settings = _settings(tmp_path, auto_extract_session_facts=True)
        mgr = _FakeManager(agent_configs=[], settings=settings)
        store = _FakeMemoryStore()
        mgr._services[MEMORY_STORE] = store

        async def fake_generate(prompt, max_tokens=500):
            return "   \n  "

        monkeypatch.setattr(mgr, "generate_simple", fake_generate)

        facts = await mgr.on_session_end([{"role": "user", "content": "..."}])
        assert facts == []
        assert store.stored == []

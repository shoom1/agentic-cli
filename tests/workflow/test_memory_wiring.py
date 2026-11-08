"""BaseWorkflowManager.on_session_end proposes memories for review (spec §8.1).

It writes no memory: every candidate goes to MemoryStore.propose, which files
it in the inbox or drops it. Messages come from the live conversation when
the caller passes none (the CLI calls it argument-free on exit).
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.memory import Candidate, MemoryStore
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.service_registry import MEMORY_STORE


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

    async def recent_messages(self, session_id: str, limit: int = 20) -> list[dict]:
        return getattr(self, "_fake_messages", [])

    def _get_state_tools(self) -> list:
        return []


ANSWER = json.dumps([
    {"action": "new", "description": "Prefers dark mode", "content": "The user prefers dark mode.",
     "type": "user", "scope": "user"},
    {"action": "new", "description": "Runs on Python 3.12", "content": "The project runs on Python 3.12.",
     "type": "project", "scope": "project"},
])


def _manager(tmp_path, *, extract: bool = True) -> tuple[_FakeManager, MemoryStore]:
    mgr = _FakeManager(
        agent_configs=[], settings=BaseSettings(workspace_dir=tmp_path, auto_extract_session_facts=extract)
    )
    store = MemoryStore(tmp_path / "user", tmp_path / "project")
    mgr._services[MEMORY_STORE] = store
    return mgr, store


def _answering(mgr, monkeypatch, answer: str, prompts: list[str] | None = None) -> None:
    async def fake_generate(prompt, max_tokens=500):
        if prompts is not None:
            prompts.append(prompt)
        return answer

    monkeypatch.setattr(mgr, "generate_simple", fake_generate)


class TestOnSessionEnd:
    async def test_disabled_returns_empty(self, tmp_path):
        mgr, _ = _manager(tmp_path, extract=False)
        assert await mgr.on_session_end([{"role": "user", "content": "hi"}]) == []

    async def test_enabled_without_memory_store_returns_empty(self, tmp_path):
        mgr, _ = _manager(tmp_path)
        del mgr._services[MEMORY_STORE]
        assert await mgr.on_session_end([{"role": "user", "content": "hi"}]) == []

    async def test_candidates_become_proposals_not_memories(self, tmp_path, monkeypatch):
        mgr, store = _manager(tmp_path)
        _answering(mgr, monkeypatch, ANSWER)
        with mgr._workflow_context(session_id="conv-1", user_id="u"):
            proposals = await mgr.on_session_end([{"role": "user", "content": "..."}])
        assert sorted(p.description for p in proposals) == ["Prefers dark mode", "Runs on Python 3.12"]
        assert {p.conversation for p in proposals} == {"conv-1"}
        assert {p.outside_content for p in proposals} == {None}
        assert store.memories() == []
        assert sorted(p.id for p in store.proposals()) == sorted(p.id for p in proposals)

    async def test_the_prompt_holds_the_index_and_the_discards(self, tmp_path, monkeypatch):
        mgr, store = _manager(tmp_path)
        store.save("Prefers spaces", "Spaces.", "feedback", scope="user")
        store.discard(store.propose(Candidate("new", None, "Prefers emacs", "Emacs.", "user", "user")).id)
        prompts: list[str] = []
        _answering(mgr, monkeypatch, "[]", prompts)
        await mgr.on_session_end([{"role": "user", "content": "..."}])
        assert "feedback: Prefers spaces" in prompts[0]
        assert "- prefers emacs" in prompts[0]

    async def test_sources_messages_from_session_when_none_passed(self, tmp_path, monkeypatch):
        mgr, _ = _manager(tmp_path)
        mgr._fake_messages = [{"role": "user", "content": "please remember PROJECT_X"}]
        prompts: list[str] = []
        _answering(mgr, monkeypatch, "[]", prompts)
        assert await mgr.on_session_end() == []
        assert "PROJECT_X" in prompts[0]

    @pytest.mark.parametrize("answer", ["   \n  ", "[]", "Sure! Nothing to keep."])
    async def test_an_answer_with_no_candidates_files_nothing(self, tmp_path, monkeypatch, answer):
        mgr, store = _manager(tmp_path)
        _answering(mgr, monkeypatch, answer)
        assert await mgr.on_session_end([{"role": "user", "content": "..."}]) == []
        assert store.proposals() == []

    async def test_a_failed_call_files_nothing(self, tmp_path, monkeypatch):
        mgr, store = _manager(tmp_path)

        async def broken(prompt, max_tokens=500):
            raise RuntimeError("provider down")

        monkeypatch.setattr(mgr, "generate_simple", broken)
        assert await mgr.on_session_end([{"role": "user", "content": "..."}]) == []

    async def test_a_candidate_the_filters_drop_is_not_returned(self, tmp_path, monkeypatch):
        mgr, store = _manager(tmp_path)
        store.save("Prefers dark mode", "Dark.", "user", scope="user")
        _answering(mgr, monkeypatch, ANSWER)
        proposals = await mgr.on_session_end([{"role": "user", "content": "..."}])
        assert [p.description for p in proposals] == ["Runs on Python 3.12"]

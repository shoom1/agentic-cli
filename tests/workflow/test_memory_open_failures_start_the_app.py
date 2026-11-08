"""A workflow starts when the memory folders cannot open.

A user memory folder that is a link (a repository cannot ship one, but a person
can make one), or one the process cannot write, raised out of service
construction, so an app whose agents use the memory tools did not start
("Initialization failed: …"). The project scope is now left out when it is the
one that fails (the user scope keeps working); when the user scope fails no
memory store is published and the tools answer that it is not available. A
link is still refused, never followed.
"""

from __future__ import annotations

import os
from typing import AsyncGenerator

import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.tools.memory_tools import read_memory, save_memory, search_memory
from agentic_cli.workflow import memory_services
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.service_registry import set_service_registry
from tests.memory.kb_storage import LogRecorder


class _Manager(BaseWorkflowManager):
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

    def _get_state_tools(self) -> list:
        return []


@pytest.fixture
def started(tmp_path, monkeypatch):
    """A manager factory: the app's cwd is ``tmp_path/project``, its workspace ``tmp_path/workspace``."""
    (tmp_path / "project").mkdir()
    monkeypatch.chdir(tmp_path / "project")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")  # initialization resolves a model
    settings = BaseSettings(workspace_dir=tmp_path / "workspace", knowledge_base_use_mock=True)
    config = AgentConfig(name="a", prompt="p", tools=[save_memory, read_memory, search_memory])
    return _Manager(agent_configs=[config], settings=settings), settings


def _broken(folder, cause: str, tmp_path) -> None:
    """Make ``folder`` (a memory scope's folder) unusable: a link elsewhere, or read-only."""
    (tmp_path / "elsewhere").mkdir()
    if cause == "link":
        folder.parent.mkdir(parents=True, exist_ok=True)
        folder.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    else:
        folder.mkdir(parents=True)
        folder.chmod(0o500)


def _repaired(folder) -> None:
    """Let pytest clean up a read-only folder."""
    if folder.exists() and not folder.is_symlink():
        folder.chmod(0o700)


CAUSES = [
    "link",
    pytest.param("read-only", marks=pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")),
]


def _call(tool, services, *args, **kwargs):
    token = set_service_registry(dict(services))
    try:
        return tool(*args, **kwargs)
    finally:
        token.var.reset(token)


@pytest.mark.parametrize("cause", CAUSES)
async def test_a_user_scope_that_does_not_open_leaves_no_memory_store(started, tmp_path, monkeypatch, cause):
    mgr, settings = started
    log = LogRecorder().install(monkeypatch, memory_services)
    folder = settings.workspace_dir / "memory"
    _broken(folder, cause, tmp_path)
    try:
        await mgr.initialize_services(validate=False)  # no "Initialization failed"

        assert mgr.memory_store is None
        [warning] = log.warnings("memory_store_unavailable")
        assert warning["error"]
        for result in (
            _call(save_memory, mgr._services, "d", "c", "user", scope="user"),
            _call(search_memory, mgr._services, "d"),
            _call(read_memory, mgr._services, "abcd1234"),
        ):
            assert result == {"success": False, "error": "memory store not available"}
        assert list((tmp_path / "elsewhere").glob("*")) == []  # a link is never followed
    finally:
        _repaired(folder)


@pytest.mark.parametrize("cause", CAUSES)
async def test_a_project_scope_that_does_not_open_leaves_the_user_scope(started, tmp_path, monkeypatch, cause):
    mgr, settings = started
    log = LogRecorder().install(monkeypatch, memory_services)
    folder = memory_services.project_memory_dir(settings)
    _broken(folder, cause, tmp_path)
    try:
        await mgr.initialize_services(validate=False)

        store = mgr.memory_store
        assert store is not None and not store.has_project_scope
        [warning] = log.warnings("memory_project_scope_unavailable")
        assert warning["error"]
        saved = _call(save_memory, mgr._services, "Prefers metric units", "The user prefers metric units", "user", scope="user")
        assert saved["success"] is True, saved
        found = _call(search_memory, mgr._services, "metric")
        assert [item["description"] for item in found["items"]] == ["Prefers metric units"]
        assert _call(save_memory, mgr._services, "p", "c", "project")["success"] is False
        assert list((tmp_path / "elsewhere").glob("*")) == []  # a link is never followed
    finally:
        _repaired(folder)

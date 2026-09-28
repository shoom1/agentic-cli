"""A model switch keeps what belongs to the conversation, not to the model.

``reinitialize`` released every service and built new ones, so switching the
model mid-session (``/settings``) lost the permission engine's session grants
(everything allowed "for this session" was asked again), shut the sandbox
kernels down (their variables were gone), and closed the job manager. None of
them depends on the model. With ``preserve_sessions=True``, which the CLI
uses, they are now carried over, also through a switch that fails, so that
restoring the previous model keeps them. ``preserve_sessions=False`` still
starts afresh.

These run real ADK and LangGraph managers; only the network model listing
is stubbed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("google.adk")

from agentic_cli.tools import job_list  # noqa: E402
from agentic_cli.tools.registry import Capability  # noqa: E402
from agentic_cli.tools.sandbox import sandbox_execute  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.permissions.prompt import ALLOW_SESSION_CHOICE  # noqa: E402
from agentic_cli.workflow.service_registry import (  # noqa: E402
    JOB_MANAGER,
    PERMISSION_ENGINE,
    SANDBOX_MANAGER,
)
from tests.conftest import MockContext  # noqa: E402

MODELS = ("gemini-2.5-flash", "gemini-2.5-pro")
READ = [Capability("filesystem.read", target_arg="path")]


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    with MockContext(
        google_api_key="test-key",
        session_store="memory",
        stateful_executor_backend="local",
    ) as context:
        yield context


def _langgraph_manager_cls():
    pytest.importorskip("langgraph")
    from agentic_cli.workflow.langgraph.manager import LangGraphWorkflowManager

    return LangGraphWorkflowManager


@pytest.fixture(params=["adk", "langgraph"])
async def manager(request, ctx):
    manager_cls = (
        GoogleADKWorkflowManager if request.param == "adk" else _langgraph_manager_cls()
    )
    mgr = manager_cls(
        agent_configs=[
            AgentConfig(name="a", prompt="p", tools=[sandbox_execute, job_list])
        ],
        settings=ctx.settings,
        model=MODELS[0],
    )
    mgr._model_registry.refresh = AsyncMock()  # the built-in list, no network
    await mgr.initialize_services()
    try:
        yield mgr
    finally:
        await mgr.cleanup()


class _Approver:
    """Answers every permission prompt with "Allow for this session"."""

    def __init__(self) -> None:
        self.asked = 0

    async def __call__(self, request) -> str:
        self.asked += 1
        return ALLOW_SESSION_CHOICE


async def _read_allowed(mgr, path: Path) -> bool:
    engine = mgr.services[PERMISSION_ENGINE]
    result = await engine.check("read_file", READ, {"path": str(path)})
    return result.allowed


def _count_closes(monkeypatch, jobs) -> list[None]:
    closes: list[None] = []
    real_close = jobs.close

    def _close() -> None:
        closes.append(None)
        real_close()

    monkeypatch.setattr(jobs, "close", _close)
    return closes


def _other_model(mgr) -> str:
    return MODELS[1] if mgr.model == MODELS[0] else MODELS[0]


async def test_session_grants_survive_a_model_switch(manager, tmp_path):
    approver = _Approver()
    manager.set_input_callback(approver)
    data = tmp_path / "data" / "notes.txt"  # outside the project: asks
    assert await _read_allowed(manager, data)
    assert approver.asked == 1

    await manager.reinitialize(model=_other_model(manager), preserve_sessions=True)

    assert await _read_allowed(manager, data)
    assert approver.asked == 1, "the session grant was lost; the user was asked again"


async def test_sandbox_state_survives_a_model_switch(manager):
    sandbox = manager.services[SANDBOX_MANAGER]
    assert sandbox.execute("answer = 41", session_id="s").success

    await manager.reinitialize(model=_other_model(manager), preserve_sessions=True)

    result = manager.services[SANDBOX_MANAGER].execute("print(answer + 1)", session_id="s")
    assert result.success, result
    assert "42" in result.stdout


async def test_the_job_manager_survives_a_model_switch(manager, monkeypatch):
    jobs = manager.services[JOB_MANAGER]
    closes = _count_closes(monkeypatch, jobs)

    await manager.reinitialize(model=_other_model(manager), preserve_sessions=True)

    assert manager.services[JOB_MANAGER] is jobs
    assert closes == []


async def test_a_failed_switch_keeps_them_for_the_restore(manager, tmp_path, monkeypatch):
    """/settings restores the previous model after a failed switch; the
    grants and the kernel must still be there when it does."""
    approver = _Approver()
    manager.set_input_callback(approver)
    data = tmp_path / "data" / "notes.txt"
    assert await _read_allowed(manager, data)
    manager.services[SANDBOX_MANAGER].execute("answer = 41", session_id="s")
    previous = manager.model

    real_init = manager._do_initialize

    async def _broken_init():
        raise RuntimeError("backend init failed")

    monkeypatch.setattr(manager, "_do_initialize", _broken_init)
    with pytest.raises(RuntimeError, match="backend init failed"):
        await manager.reinitialize(model=_other_model(manager), preserve_sessions=True)
    assert manager.is_initialized is False

    monkeypatch.setattr(manager, "_do_initialize", real_init)
    await manager.reinitialize(model=previous, preserve_sessions=True)

    assert await _read_allowed(manager, data)
    assert approver.asked == 1
    result = manager.services[SANDBOX_MANAGER].execute("print(answer + 1)", session_id="s")
    assert "42" in result.stdout


async def test_cleanup_after_a_failed_switch_still_releases_them(manager, monkeypatch):
    sandbox = manager.services[SANDBOX_MANAGER]
    closes = _count_closes(monkeypatch, manager.services[JOB_MANAGER])
    sandbox.execute("answer = 41", session_id="s")

    async def _broken_init():
        raise RuntimeError("backend init failed")

    monkeypatch.setattr(manager, "_do_initialize", _broken_init)
    with pytest.raises(RuntimeError):
        await manager.reinitialize(model=_other_model(manager), preserve_sessions=True)

    await manager.cleanup()

    assert len(closes) == 1
    assert sandbox.list_sessions() == []
    assert manager.services == {}


async def test_preserve_sessions_false_starts_afresh(manager, tmp_path, monkeypatch):
    approver = _Approver()
    manager.set_input_callback(approver)
    data = tmp_path / "data" / "notes.txt"
    assert await _read_allowed(manager, data)
    jobs = manager.services[JOB_MANAGER]
    closes = _count_closes(monkeypatch, jobs)

    await manager.reinitialize(model=_other_model(manager), preserve_sessions=False)

    assert await _read_allowed(manager, data)
    assert approver.asked == 2
    assert len(closes) == 1
    assert manager.services[JOB_MANAGER] is not jobs

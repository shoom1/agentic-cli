"""The Deep Agents manager's lifecycle: locks, the checkpointer it owns, model switches.

The manager follows ADK's lifecycle contract. ``cleanup``, ``reinitialize``
and initialization take the lifecycle lock and then the turn lock, so they
wait for a running turn. The checkpointer's connection is owned: a failed
initialization closes the one it opened, and ``cleanup`` closes it once. A
model switch (``reinitialize(preserve_sessions=True)``) keeps the same saver
and connection and rebuilds only the agents; if it fails, the manager is left
uninitialized with the checkpointer kept, so a retry continues the same
conversations. ``preserve_sessions=False`` closes it and opens a fresh one.

Real ``InMemorySaver`` and ``AsyncSqliteSaver`` checkpointers, scripted
models; nothing reaches a provider.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("deepagents")

from agentic_cli.tools.registry import register_tool  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents import manager as manager_module  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from tests.conftest import MockContext  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedModels, ai  # noqa: E402

#: What ``hold_the_turn`` waits on, set by the test that calls it.
_GATE: dict[str, asyncio.Event] = {}


@register_tool(capabilities=EXEMPT)
async def hold_the_turn() -> dict:
    """Waits until the test lets it finish (an exempt test tool)."""
    _GATE["started"].set()
    await _GATE["release"].wait()
    return {"success": True}


@pytest.fixture
def hermetic(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


@pytest.fixture(params=["memory", "sqlite"])
def store(request, hermetic):
    """Settings on each store; the manager's own checkpointer."""
    with MockContext(google_api_key="test-key", session_store=request.param) as context:
        yield context.settings


@pytest.fixture
def sqlite(hermetic):
    with MockContext(google_api_key="test-key", session_store="sqlite") as context:
        yield context.settings


@pytest.fixture
def opened(monkeypatch) -> list[SimpleNamespace]:
    """Every checkpointer the manager opens, and how often its connection was closed."""
    records: list[SimpleNamespace] = []
    real = manager_module.open_checkpointer

    async def spy(*args, **kwargs):
        saver, cm = await real(*args, **kwargs)
        record = SimpleNamespace(saver=saver, cm=cm, closes=0)
        if cm is not None:
            real_exit = cm.__aexit__

            async def counted_exit(*exc):
                record.closes += 1
                return await real_exit(*exc)

            cm.__aexit__ = counted_exit
        records.append(record)
        return saver, cm

    monkeypatch.setattr(manager_module, "open_checkpointer", spy)
    return records


@pytest.fixture
def models(monkeypatch) -> ScriptedModels:
    models = ScriptedModels(solo=[ai("One."), ai("Two."), ai("Three.")])
    monkeypatch.setattr(manager_module, "build_chat_model", models)
    return models


def _failing_build(monkeypatch, times: int = 1) -> None:
    """The agent tree fails to build ``times`` times, then builds."""
    real = manager_module.build_agent_tree
    left = {"failures": times}

    def build(*args, **kwargs):
        if left["failures"]:
            left["failures"] -= 1
            raise RuntimeError("agent build failed")
        return real(*args, **kwargs)

    monkeypatch.setattr(manager_module, "build_agent_tree", build)


def _manager(settings, configs=None) -> DeepAgentsWorkflowManager:
    return DeepAgentsWorkflowManager(
        agent_configs=configs or [AgentConfig(name="solo", prompt="p")],
        settings=settings,
        model="gemini-2.5-flash",
    )


async def turn(manager, text: str) -> list:
    return [event async for event in manager.process(text, "u", session_id="s")]


async def is_open(saver) -> bool:
    """Whether a sqlite saver's connection still answers."""
    try:
        await saver.conn.execute("SELECT 1")
    except ValueError:  # aiosqlite: no active connection
        return False
    return True


class TestLocks:
    async def test_a_cleanup_waits_for_a_running_turn(self, store, monkeypatch):
        monkeypatch.setattr(manager_module, "build_chat_model", ScriptedModels(
            solo=[ai(calls=[("hold_the_turn", {})]), ai("Released.")]
        ))
        _GATE.update(started=asyncio.Event(), release=asyncio.Event())
        manager = _manager(store, [AgentConfig(name="solo", prompt="p", tools=[hold_the_turn])])

        running = asyncio.create_task(turn(manager, "Go."))
        await asyncio.wait_for(_GATE["started"].wait(), timeout=5)
        cleaning = asyncio.create_task(manager.cleanup())
        await asyncio.sleep(0.05)

        assert not cleaning.done(), "cleanup released the backend under a running turn"
        _GATE["release"].set()
        events = await asyncio.wait_for(running, timeout=5)
        await asyncio.wait_for(cleaning, timeout=5)

        assert [e.content for e in events if e.type is EventType.TEXT] == ["Released."]
        assert manager.is_initialized is False


class TestOwnedCheckpointer:
    async def test_a_failed_init_closes_the_checkpointer_it_opened(
        self, sqlite, models, opened, monkeypatch
    ):
        _failing_build(monkeypatch)
        manager = _manager(sqlite)

        with pytest.raises(RuntimeError, match="agent build failed"):
            await manager.initialize_services()

        (record,) = opened
        assert record.closes == 1
        assert not await is_open(record.saver)
        assert manager._checkpointer is None
        assert manager.is_initialized is False

        await manager.initialize_services()  # the next attempt opens its own
        try:
            assert len(opened) == 2 and await is_open(opened[1].saver)
        finally:
            await manager.cleanup()

    async def test_cleanup_closes_it_once(self, sqlite, models, opened):
        manager = _manager(sqlite)
        await turn(manager, "Hello.")

        await manager.cleanup()
        await manager.cleanup()

        (record,) = opened
        assert record.closes == 1
        assert not await is_open(record.saver)
        assert await manager.session_exists("s", user_id="u") is False


class TestModelSwitch:
    async def test_it_keeps_the_saver_and_rebuilds_the_agents(self, store, models, opened):
        manager = _manager(store)
        try:
            await turn(manager, "Hello.")
            saver, root = manager._checkpointer, manager._root
            connection = getattr(saver, "conn", None)

            await manager.reinitialize(model="gemini-2.5-pro", preserve_sessions=True)

            assert manager._checkpointer is saver
            assert len(opened) == 1 and opened[0].closes == 0
            if connection is not None:
                assert saver.conn is connection and await is_open(saver)
            assert manager._root is not root
            assert str(manager.agent_models["solo"].ref) == "google:gemini-2.5-pro"

            await turn(manager, "Again.")
            assert [m.text for m in models.chats["solo"].conversation(1)] == [
                "Hello.", "One.", "Again."
            ]
        finally:
            await manager.cleanup()
        assert opened[0].closes == (1 if opened[0].cm is not None else 0)

    async def test_a_failed_switch_keeps_the_checkpointer_for_a_retry(
        self, store, models, opened, monkeypatch
    ):
        manager = _manager(store)
        try:
            await turn(manager, "Hello.")
            saver = manager._checkpointer
            real_build = manager_module.build_agent_tree
            _failing_build(monkeypatch)

            with pytest.raises(RuntimeError, match="agent build failed"):
                await manager.reinitialize(model="gemini-2.5-pro", preserve_sessions=True)

            assert manager.is_initialized is False
            assert manager._checkpointer is saver
            assert opened[0].closes == 0
            if opened[0].cm is not None:
                assert await is_open(saver)
            assert await manager.session_exists("s", user_id="u") is True

            monkeypatch.setattr(manager_module, "build_agent_tree", real_build)
            await manager.reinitialize(model="gemini-2.5-flash", preserve_sessions=True)
            await turn(manager, "Again.")

            assert len(opened) == 1
            assert [m.text for m in models.chats["solo"].conversation(1)] == [
                "Hello.", "One.", "Again."
            ]
        finally:
            await manager.cleanup()

    async def test_cleanup_after_a_failed_switch_closes_the_kept_checkpointer(
        self, sqlite, models, opened, monkeypatch
    ):
        manager = _manager(sqlite)
        await turn(manager, "Hello.")
        _failing_build(monkeypatch)
        with pytest.raises(RuntimeError):
            await manager.reinitialize(preserve_sessions=True)

        await manager.cleanup()

        assert opened[0].closes == 1
        assert not await is_open(opened[0].saver)

    async def test_not_preserving_sessions_opens_a_fresh_checkpointer(
        self, store, models, opened
    ):
        manager = _manager(store)
        try:
            await turn(manager, "Hello.")
            saver = manager._checkpointer

            await manager.reinitialize(preserve_sessions=False)

            assert len(opened) == 2
            assert manager._checkpointer is opened[1].saver is not saver
            if opened[0].cm is not None:
                # The old connection is closed; the conversation is still in the file.
                assert opened[0].closes == 1 and not await is_open(saver)
                assert await manager.session_exists("s", user_id="u") is True
            else:
                # A fresh in-memory saver holds nothing.
                assert await manager.session_exists("s", user_id="u") is False
        finally:
            await manager.cleanup()

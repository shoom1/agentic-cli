"""The memory index each conversation's main agent gets: one snapshot per
conversation, taken once, dropped where the prompt prefix changes anyway (spec §11.1)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_cli.config import BaseSettings
from agentic_cli.memory import MemoryStore
from agentic_cli.tools import memory_tools
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.service_registry import MEMORY_STORE
from agentic_cli.workflow.sessions import SessionRef, reset_active_turn, set_active_turn
from tests.workflow.test_memory_wiring import _FakeManager

ROOT = AgentConfig(name="root", prompt="p", tools=[memory_tools.read_memory, memory_tools.save_memory])


def _manager(tmp_path, configs=(ROOT,), **values) -> tuple[_FakeManager, MemoryStore]:
    mgr = _FakeManager(agent_configs=list(configs), settings=BaseSettings(workspace_dir=tmp_path, **values))
    mgr._root_agent_name = configs[0].name
    store = MemoryStore(tmp_path / "user", tmp_path / "project")
    mgr._services[MEMORY_STORE] = store
    return mgr, store


@pytest.fixture
def ref(tmp_path):
    ref = SessionRef(app_name="agentic_cli", user_id="u", session_id="s")
    token = set_active_turn(ref)
    yield ref
    reset_active_turn(token)


def test_the_settings_defaults_and_range():
    settings = BaseSettings()
    assert (settings.memory_index_enabled, settings.memory_index_max_bytes) == (True, 25_000)
    for bad in (999, 100_001):
        with pytest.raises(ValidationError):
            BaseSettings(memory_index_max_bytes=bad)


def test_the_root_s_memory_tools_by_name(tmp_path):
    mgr, _ = _manager(tmp_path)
    assert mgr._root_memory_tools() == ("read_memory", "save_memory")
    assert mgr._memory_index_wanted("root")
    plain, _ = _manager(tmp_path, configs=(AgentConfig(name="root", prompt="p"),))
    assert not plain._memory_index_wanted("root")
    off, _ = _manager(tmp_path, memory_index_enabled=False)
    assert not off._memory_index_wanted("root")


def test_the_snapshot_is_taken_once_per_conversation(tmp_path, ref):
    mgr, store = _manager(tmp_path)
    store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    first = mgr._memory_index_now()
    assert first.startswith("<memory-index>") and "Prefers tabs" in first
    assert "Update or delete" not in first  # the root has no update/delete tool
    store.save("Prefers dark mode", "Dark.", "user", scope="user")
    assert mgr._memory_index_now() == first
    assert mgr._memory_index_cached() == first


def test_forgetting_it_takes_a_new_one(tmp_path, ref):
    mgr, store = _manager(tmp_path)
    store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    mgr._memory_index_now()
    store.save("Prefers dark mode", "Dark.", "user", scope="user")
    mgr._forget_memory_index()
    assert mgr._memory_index_cached() == ""
    assert "Prefers dark mode" in mgr._memory_index_now()


async def test_resuming_a_conversation_drops_its_snapshot(tmp_path, ref, monkeypatch):
    mgr, store = _manager(tmp_path)
    store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    mgr._memory_index_now()

    async def exists(*args, **kwargs):
        return True

    monkeypatch.setattr(mgr, "session_exists", exists)
    await mgr.load_session("s", user_id="u")
    assert mgr._memory_index_cached() == ""


@pytest.mark.parametrize("change", ["disabled", "no tools", "empty", "no store"])
def test_no_index(tmp_path, ref, change):
    configs = (AgentConfig(name="root", prompt="p"),) if change == "no tools" else (ROOT,)
    values = {"memory_index_enabled": False} if change == "disabled" else {}
    mgr, store = _manager(tmp_path, configs=configs, **values)
    if change == "no store":
        del mgr._services[MEMORY_STORE]
    if change != "empty":
        store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    assert mgr._memory_index_now() == ""


def test_the_cap_comes_from_the_setting(tmp_path, ref):
    mgr, store = _manager(tmp_path, memory_index_max_bytes=1_000)
    for n in range(40):
        store.save(f"fact number {n} " + "x" * 40, "c", "user", scope="user")
    assert len(mgr._memory_index_now().encode()) <= 1_000


def test_without_a_turn_there_is_no_index(tmp_path):
    mgr, store = _manager(tmp_path)
    store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    assert mgr._memory_index_now() == ""


def test_a_store_that_fails_gives_no_index_and_raises_nothing(tmp_path, ref, monkeypatch):
    mgr, store = _manager(tmp_path)
    store.save("Prefers tabs", "Tabs.", "feedback", scope="user")

    def broken(**kwargs):
        raise OSError("disk went away")

    monkeypatch.setattr(store, "index", broken)
    assert mgr._memory_index_now() == ""

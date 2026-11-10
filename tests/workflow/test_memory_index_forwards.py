"""The memory index is one collaborator a manager holds (``workflow/memory_index.py``).

``BaseWorkflowManager`` kept the snapshots, the code that took them and the
cap itself. It now holds one ``MemoryIndex`` and its release clears it. The
protected names the backends, the summarizers and the tests use stay on the
manager as forwards (``_memory_index_now``, ``_memory_index_cached``,
``_forget_memory_index``, ``_memory_index_cap``, and ``_index_snapshots``,
which a test harness builds a manager around), and a subclass that overrides
one is still the one asked: ADK's plugin and summarizer and Deep Agents'
middleware take the manager's methods, so an override reaches them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("deepagents")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.memory import MemoryStore  # noqa: E402
from agentic_cli.tools import memory_tools  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents import manager as deepagents_module  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.service_registry import MEMORY_STORE  # noqa: E402
from agentic_cli.workflow.sessions import SessionRef, reset_active_turn, set_active_turn  # noqa: E402
from tests.conftest import MockContext  # noqa: E402
from tests.workflow.compacting import compact_now  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedModels, ai  # noqa: E402
from tests.workflow.deepagents.test_compact import THREE, answers  # noqa: E402
from tests.workflow.deepagents.test_context import (  # noqa: E402,F401
    SMALL,
    hermetic,
    small,
    turn as deepagents_turn,
)
from tests.workflow.test_memory_wiring import _FakeManager  # noqa: E402

ROOT = AgentConfig(name="root", prompt="p", tools=[memory_tools.read_memory, memory_tools.save_memory])
REF = SessionRef(app_name="agentic_cli", user_id="u", session_id="s")


@pytest.fixture
def turn():
    token = set_active_turn(REF)
    yield REF
    reset_active_turn(token)


def _manager(tmp_path, cls=_FakeManager, **values):
    mgr = cls(agent_configs=[ROOT], settings=BaseSettings(workspace_dir=tmp_path, **values))
    mgr._root_agent_name = "root"
    store = MemoryStore(tmp_path / "user", tmp_path / "project")
    store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    mgr._services[MEMORY_STORE] = store
    return mgr, store


# ---- the manager holds one ----


def test_a_manager_holds_one_memory_index_whose_snapshots_are_its_index_snapshots(tmp_path, turn):
    from agentic_cli.workflow.memory_index import MemoryIndex

    mgr, _ = _manager(tmp_path)
    index = mgr._memory_index
    assert isinstance(index, MemoryIndex) and mgr._memory_index is index
    assert mgr._index_snapshots is index.snapshots

    taken = mgr._memory_index_now()
    assert taken and index.snapshots == {REF: taken}
    assert mgr._memory_index_cached() == taken == index.cached()


def test_a_manager_built_without_its_constructor_has_one(tmp_path, turn):
    """A harness builds managers with ``__new__`` and assigns what it needs."""
    from agentic_cli.workflow.memory_index import MemoryIndex

    mgr = _FakeManager.__new__(_FakeManager)
    mgr._index_snapshots = {}
    assert isinstance(mgr._memory_index, MemoryIndex)
    mgr._index_snapshots[REF] = "the harness's"
    assert mgr._memory_index_cached() == "the harness's"
    assert mgr._memory_index.snapshots == {REF: "the harness's"}

    mgr._index_snapshots = {REF: "replaced"}  # a test assigns a fresh dict
    assert mgr._memory_index_cached() == "replaced"
    assert mgr._memory_index.snapshots is mgr._index_snapshots


def test_what_the_index_reads_is_the_manager_s_as_it_is_when_a_snapshot_is_taken(tmp_path, turn):
    """The manager's settings, store, root and budgets are replaced after it is
    built (a settings change, a service, a resolved model): the index reads them live."""
    mgr, store = _manager(tmp_path)
    mgr._settings = BaseSettings(workspace_dir=tmp_path, memory_index_max_bytes=1_000)
    other = MemoryStore(tmp_path / "other-user", tmp_path / "other-project")
    other.save("Prefers spaces", "Spaces.", "feedback", scope="user")
    mgr._services[MEMORY_STORE] = other
    asked: list[dict] = []
    real = other.index
    other.index = lambda **kwargs: asked.append(kwargs) or real(**kwargs)

    mgr._memory_index_now()
    assert asked == [{"max_bytes": 1_000, "tools": ("read_memory", "save_memory")}]


# ---- the release ----


async def test_the_release_clears_it_and_keeps_it(tmp_path, turn, monkeypatch):
    from agentic_cli.workflow.memory_index import MemoryIndex

    mgr, _ = _manager(tmp_path)
    index = mgr._memory_index
    mgr._memory_index_now()
    cleared: list[MemoryIndex] = []
    real = MemoryIndex.clear
    monkeypatch.setattr(MemoryIndex, "clear", lambda self: (cleared.append(self), real(self))[1])

    await mgr._release_resources()

    assert cleared == [index]
    assert mgr._memory_index is index and index.snapshots == {}
    assert mgr._memory_index_cached() == ""


# ---- a subclass that overrides a forward is the one asked ----


def test_a_subclass_s_cap_is_the_cap_a_snapshot_is_taken_at(tmp_path, turn):
    class Capped(_FakeManager):
        def _memory_index_cap(self) -> int:
            return 1_234

    mgr, store = _manager(tmp_path, Capped)
    asked: list[dict] = []
    real = store.index
    store.index = lambda **kwargs: asked.append(kwargs) or real(**kwargs)

    mgr._memory_index_now()

    assert [call["max_bytes"] for call in asked] == [1_234]


def test_the_cap_forward_is_the_index_s_cap(tmp_path):
    mgr, _ = _manager(tmp_path, memory_index_max_bytes=7_000)
    assert mgr._memory_index_cap() == mgr._memory_index.cap() == 7_000


def test_the_cap_reads_the_root_s_budget_and_model(tmp_path):
    from agentic_cli.models import ProviderKind
    from agentic_cli.workflow.context import ContextBudget

    mgr, _ = _manager(tmp_path)
    mgr._budgets = {"root": ContextBudget(window=32_000, window_known=True, budget=16_000,
                                          trigger=13_600, keep=1_600)}
    assert mgr._memory_index_cap() == 25_000  # no resolved model: the setting
    mgr._resolved_models = {"root": SimpleNamespace(record=SimpleNamespace(kind=ProviderKind.GOOGLE))}
    assert mgr._memory_index_cap() == 5_818


async def test_a_subclass_s_forget_is_what_a_resumed_conversation_calls(tmp_path):
    forgotten: list[SessionRef | None] = []

    class Forgetting(_FakeManager):
        def _forget_memory_index(self, ref=None) -> None:
            forgotten.append(ref)

    mgr, _ = _manager(tmp_path, Forgetting)

    assert await mgr.load_session("s", user_id="u") is False
    assert forgotten == [SessionRef(app_name=mgr.app_name, user_id="u", session_id="s")]


async def test_adk_asks_a_subclass_for_the_index_and_for_a_new_one_after_a_summary(hermetic):
    forgotten: list[str] = []

    class Overriding(GoogleADKWorkflowManager):
        def _memory_index_now(self) -> str:
            return "the subclass's index"

        def _forget_memory_index(self, ref=None) -> None:
            forgotten.append("forgotten")

    mgr = Overriding(
        agent_configs=[AgentConfig(name="solo", prompt="Help.", tools=[memory_tools.search_memory])],
        settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6"),
    )
    try:
        await mgr.initialize_services()
        plugin = next(p for p in mgr._runner.app.plugins if p.name == "memory_index")
        assert plugin._index_for() == "the subclass's index"
        mgr._summarizer()._on_written()
        assert forgotten == ["forgotten"]
    finally:
        await mgr.cleanup()


async def test_deep_agents_asks_a_subclass_for_the_index(hermetic, monkeypatch):
    forgotten: list[str] = []

    class Overriding(DeepAgentsWorkflowManager):
        def _memory_index_now(self) -> str:
            return "the subclass's index"

        def _memory_index_cached(self) -> str:
            return "the subclass's snapshot"

        def _forget_memory_index(self, ref=None) -> None:
            forgotten.append("forgotten")

    monkeypatch.setattr(deepagents_module, "build_chat_model", ScriptedModels(solo=[ai("One.")]))
    built: list[dict] = []
    real = deepagents_module.build_agent_tree

    def spy(*args, **kwargs):
        built.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(deepagents_module, "build_agent_tree", spy)
    with MockContext(google_api_key="test-key", session_store="memory") as context:
        mgr = Overriding(
            agent_configs=[AgentConfig(name="solo", prompt="p", tools=[memory_tools.search_memory])],
            settings=context.settings,
            model="gemini-2.5-flash",
        )
        try:
            await mgr.initialize_services()
            (kwargs,) = built
            assert kwargs["memory_index"]() == "the subclass's index"
            assert kwargs["memory_index_cached"]() == "the subclass's snapshot"
            kwargs["memory_index_forget"]()
            assert forgotten == ["forgotten"]
        finally:
            await mgr.cleanup()


async def test_deep_agents_compact_drops_the_snapshot_through_the_forward(small, monkeypatch):
    """``/compact`` changes the prompt prefix, so the conversation takes a new
    index: the manager's ``_forget_memory_index``, whichever class it is."""
    forgotten: list[SessionRef | None] = []

    class Forgetting(DeepAgentsWorkflowManager):
        def _forget_memory_index(self, ref=None) -> None:
            forgotten.append(ref)
            super()._forget_memory_index(ref)

    monkeypatch.setattr(
        deepagents_module, "build_chat_model",
        ScriptedModels(solo=[*answers(3), ai("A summary."), ai("Answer 4.")]),
    )
    mgr = Forgetting(
        agent_configs=[AgentConfig(
            name="solo", prompt="Answer.", include_state_tools=False,
            tools=[memory_tools.search_memory],
        )],
        settings=small,
    )
    conversation = mgr.session_ref("s", "u")
    try:
        for text in THREE:
            await deepagents_turn(mgr, text)
        assert conversation in mgr._index_snapshots  # taken with its first request

        await compact_now(mgr)

        assert forgotten == [conversation]
        assert conversation not in mgr._index_snapshots
        await deepagents_turn(mgr, "Turn 4?")
        assert conversation in mgr._index_snapshots  # a new one
    finally:
        await mgr.cleanup()


async def test_a_model_switch_drops_the_snapshots_and_the_next_is_taken_at_the_new_cap(
    hermetic, turn
):
    """The cap follows the root's budget, so a snapshot taken for the old model
    is dropped by the switch (the services and the store are kept)."""
    from agentic_cli.workflow.service_registry import MEMORY_STORE

    settings = BaseSettings(
        session_store="memory", model_providers=SMALL, default_model="local:small",
        models={"local:small": {"context_window": 20_000}, "local:big": {"context_window": 200_000}},
    )
    mgr = DeepAgentsWorkflowManager(
        agent_configs=[AgentConfig(
            name="solo", prompt="Answer.", include_state_tools=False,
            tools=[memory_tools.search_memory],
        )],
        settings=settings,
    )
    try:
        await mgr.initialize_services()
        store = mgr._services[MEMORY_STORE]
        asked: list[int] = []
        real = store.index
        store.index = lambda **kwargs: asked.append(kwargs["max_bytes"]) or real(**kwargs)
        index = mgr._memory_index

        mgr._memory_index_now()
        assert list(mgr._index_snapshots) == [turn] and len(asked) == 1
        assert asked[0] < 25_000  # a tenth of the small model's budget

        await mgr.reinitialize(model="local:big", preserve_sessions=True)

        assert mgr._services[MEMORY_STORE] is store  # the switch keeps the services
        assert mgr._memory_index is index and mgr._index_snapshots == {}
        mgr._memory_index_now()
        assert asked == [asked[0], 25_000]  # the new model's roomy budget: the setting
    finally:
        await mgr.cleanup()

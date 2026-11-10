"""The memory index, alone (``workflow/memory_index.py``).

The main agent's requests carry an index of the saved memories: one snapshot
per conversation, taken the first time the conversation asks for it, so the
prompt prefix stays the same (and cached) for the conversation's life. A
manager held the snapshots and the code that took them. ``MemoryIndex`` is
that, given the four things it reads of a manager (its settings, its memory
store, the root's memory tools and the root's room) as callables, so these
tests need no manager, and it imports no backend.

``tests/workflow/test_memory_index_snapshot.py`` covers the same index through
a manager; ``test_memory_index_forwards.py`` the manager's side of the move.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import structlog

from agentic_cli.models import ProviderKind
from agentic_cli.workflow.sessions import SessionRef, reset_active_turn, set_active_turn


class Store:
    """A memory store whose ``index`` numbers its answers and records what it was asked."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def index(self, *, max_bytes: int, tools: tuple[str, ...]) -> str:
        self.calls.append({"max_bytes": max_bytes, "tools": tools})
        return f"index {len(self.calls)}"


def make(**held):
    """A ``MemoryIndex`` over ``world``, which a test may change as it goes.

    ``world`` is what the manager would hold: its settings, its memory store,
    the root's tools and its room (the input budget in tokens and the
    provider's kind, or None).
    """
    from agentic_cli.workflow.memory_index import MemoryIndex

    world = SimpleNamespace(
        settings=SimpleNamespace(memory_index_enabled=True, memory_index_max_bytes=25_000),
        store=Store(),
        tools=("read_memory", "save_memory"),
        room=None,
    )
    for name, value in held.items():
        setattr(world, name, value)
    index = MemoryIndex(
        settings=lambda: world.settings,
        store=lambda: world.store,
        tools=lambda: world.tools,
        room=lambda: world.room,
    )
    return index, world


def ref(session: str = "s", user: str = "u") -> SessionRef:
    return SessionRef(app_name="agentic_cli", user_id=user, session_id=session)


@contextmanager
def turn(of: SessionRef):
    token = set_active_turn(of)
    try:
        yield
    finally:
        reset_active_turn(token)


# ---- one snapshot per conversation ----


def test_without_a_turn_there_is_no_index_and_nothing_is_read():
    index, world = make()
    assert index.now() == ""
    assert index.cached() == ""
    assert world.store.calls == []


def test_a_conversation_reads_the_store_once():
    index, world = make()
    with turn(ref()):
        first = index.now()
        assert first == "index 1"
        assert index.now() == first
        assert index.now() == first
    assert len(world.store.calls) == 1


def test_each_conversation_has_its_own_snapshot():
    index, world = make()
    with turn(ref("a")):
        assert index.now() == "index 1"
    with turn(ref("b")):
        assert index.now() == "index 2"
    with turn(ref("a", user="other")):
        assert index.now() == "index 3"
    with turn(ref("a")):
        assert index.now() == "index 1"
    assert len(world.store.calls) == 3


def test_the_store_is_asked_for_the_cap_and_the_root_s_tools():
    index, world = make(tools=("read_memory",))
    with turn(ref()):
        index.now()
    assert world.store.calls == [{"max_bytes": 25_000, "tools": ("read_memory",)}]


def test_what_it_reads_is_read_when_a_snapshot_is_taken():
    """Not when the index is made: a manager's settings, services and root change."""
    index, world = make()
    world.store = Store()
    world.settings = SimpleNamespace(memory_index_enabled=True, memory_index_max_bytes=3_000)
    world.tools = ("search_memory",)
    with turn(ref()):
        index.now()
    assert world.store.calls == [{"max_bytes": 3_000, "tools": ("search_memory",)}]


def test_cached_reads_nothing():
    index, world = make()
    with turn(ref()):
        assert index.cached() == ""
        assert world.store.calls == []
        taken = index.now()
        assert index.cached() == taken
    assert len(world.store.calls) == 1
    with turn(ref("another")):
        assert index.cached() == ""
    assert len(world.store.calls) == 1


# ---- where a conversation takes a new one ----


def test_forgetting_makes_the_next_request_take_a_new_snapshot():
    index, world = make()
    with turn(ref()):
        assert index.now() == "index 1"
        index.forget()
        assert index.cached() == ""
        assert index.now() == "index 2"


def test_forget_can_name_another_conversation_and_leaves_the_rest():
    index, world = make()
    for session in ("a", "b"):
        with turn(ref(session)):
            index.now()
    index.forget(ref("a"))
    with turn(ref("a")):
        assert index.cached() == ""
    with turn(ref("b")):
        assert index.cached() == "index 2"


def test_forget_with_no_turn_and_no_conversation_drops_nothing():
    index, world = make()
    with turn(ref()):
        index.now()
    index.forget()
    with turn(ref()):
        assert index.cached() == "index 1"


def test_clearing_drops_every_snapshot():
    index, world = make()
    for session in ("a", "b"):
        with turn(ref(session)):
            index.now()
    index.clear()
    assert index.snapshots == {}
    for session in ("a", "b"):
        with turn(ref(session)):
            assert index.cached() == ""
    with turn(ref("a")):
        assert index.now() == "index 3"


def test_the_snapshots_are_the_index_s_to_hold_and_replace():
    index, world = make()
    with turn(ref()):
        index.now()
    assert index.snapshots == {ref(): "index 1"}
    index.snapshots = {ref(): "the harness's"}
    with turn(ref()):
        assert index.cached() == "the harness's"
        assert index.now() == "the harness's"


# ---- no index ----


@pytest.mark.parametrize("change", ["disabled", "no tools", "no store"])
def test_no_index_and_no_read(change):
    store = Store()
    index, world = make(store=store)
    if change == "disabled":
        world.settings = SimpleNamespace(memory_index_enabled=False, memory_index_max_bytes=25_000)
    elif change == "no tools":
        world.tools = ()
    else:
        world.store = None
    with turn(ref()):
        assert index.now() == ""
    assert store.calls == []


def test_a_settings_object_that_never_says_is_enabled():
    """The setting is read with a default: an app's own settings class may lack it."""
    index, world = make(settings=SimpleNamespace(memory_index_max_bytes=25_000))
    with turn(ref()):
        assert index.now() == "index 1"


# ---- a store that fails ----


def test_a_store_that_fails_gives_no_index_and_raises_nothing():
    class Failing:
        calls = 0

        def index(self, **kwargs):
            Failing.calls += 1
            raise OSError("disk went away")

    index, world = make(store=Failing())
    with structlog.testing.capture_logs() as logs, turn(ref()):
        assert index.now() == ""
    warned = [entry for entry in logs if entry["event"] == "memory_index_unavailable"]
    assert [entry["error"] for entry in warned] == ["OSError: disk went away"]
    assert warned[0]["log_level"] == "warning"


def test_a_failed_read_is_the_conversation_s_snapshot_and_is_not_retried():
    class Failing:
        calls = 0

        def index(self, **kwargs):
            Failing.calls += 1
            raise OSError("disk went away")

    index, world = make(store=Failing())
    with turn(ref()):
        assert index.now() == ""
        assert index.now() == ""
        assert index.cached() == ""
    assert Failing.calls == 1
    with turn(ref("another")):
        index.now()
    assert Failing.calls == 2


def test_a_cap_that_fails_is_the_same_failure():
    index, world = make(settings=SimpleNamespace(memory_index_enabled=True))  # no max bytes
    with structlog.testing.capture_logs() as logs, turn(ref()):
        assert index.now() == ""
    assert [entry["event"] for entry in logs] == ["memory_index_unavailable"]
    assert world.store.calls == []


# ---- the cap ----


def test_the_cap_is_the_setting_where_the_root_has_no_room():
    index, world = make(
        settings=SimpleNamespace(memory_index_enabled=True, memory_index_max_bytes=7_000)
    )
    assert index.cap() == 7_000


@pytest.mark.parametrize(
    ("kind", "cap"), [(ProviderKind.GOOGLE, 5_818), (ProviderKind.ANTHROPIC, 4_800)]
)
def test_a_small_budget_gives_a_tenth_of_it_in_characters(kind, cap):
    """A tenth of the root's input budget, as the fit guard estimates a request:
    4 characters a token (3.3 on Claude) and 10% on top. 1,600 tokens here."""
    index, world = make(room=(16_000, kind))
    assert index.cap() == cap


def test_a_roomy_budget_keeps_the_setting():
    index, world = make(room=(150_000, ProviderKind.GOOGLE))  # a tenth: 54,000 bytes
    assert index.cap() == 25_000


def test_a_budget_over_a_smaller_setting_keeps_the_setting():
    index, world = make(
        settings=SimpleNamespace(memory_index_enabled=True, memory_index_max_bytes=2_000),
        room=(150_000, ProviderKind.GOOGLE),
    )
    assert index.cap() == 2_000


def test_never_under_a_thousand_bytes():
    index, world = make(room=(1_000, ProviderKind.GOOGLE))  # a tenth is 363 bytes
    assert index.cap() == 1_000


def test_the_setting_below_the_floor_is_the_floor_only_under_a_budget():
    """With no room the setting stands as it is; under a budget the floor applies."""
    index, world = make(
        settings=SimpleNamespace(memory_index_enabled=True, memory_index_max_bytes=500)
    )
    assert index.cap() == 500
    world.room = (150_000, ProviderKind.GOOGLE)
    assert index.cap() == 1_000


def test_a_snapshot_is_taken_at_the_cap():
    index, world = make(room=(16_000, ProviderKind.GOOGLE))
    with turn(ref()):
        index.now()
    assert world.store.calls[0]["max_bytes"] == 5_818


def test_a_snapshot_is_taken_at_the_limit_it_is_given():
    """A manager gives its own ``_memory_index_cap``, so a subclass that
    overrides that one is still asked; the index's ``cap`` is the default."""
    from agentic_cli.workflow.memory_index import MemoryIndex

    store = Store()
    index = MemoryIndex(
        settings=lambda: SimpleNamespace(memory_index_enabled=True, memory_index_max_bytes=25_000),
        store=lambda: store,
        tools=lambda: ("read_memory",),
        room=lambda: None,
        limit=lambda: 1_234,
    )
    with turn(ref()):
        index.now()
    assert store.calls == [{"max_bytes": 1_234, "tools": ("read_memory",)}]
    assert index.cap() == 25_000


# ---- no backend ----


def test_it_imports_no_backend_and_not_the_manager():
    """A collaborator the manager composes: it reaches the manager only through
    what it is given, and no backend through anything."""
    source = Path(__import__("agentic_cli.workflow.memory_index", fromlist=["x"]).__file__).read_text()
    imported = re.findall(r"^\s*(?:from|import)\s+(agentic_cli[\w.]*)", source, re.MULTILINE)
    forbidden = [
        name for name in imported
        if re.match(
            r"agentic_cli\.workflow\.(adk|deepagents|langgraph|base_manager)\b", name
        )
    ]
    assert forbidden == []

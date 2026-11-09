"""One turn protocol, in the base.

``BaseWorkflowManager._turn(session_id, user_id)`` is how every turn-shaped
entry point starts: it names the conversation in the log context, enters
turn admission (initialize outside the turn lock, take it, re-check the
backend, retry once) and the workflow context (the settings, the services
and the active turn), and yields the turn's ``SessionRef``. Each backend
wrote that skeleton itself, six times over, so a change to how a turn starts
had to be made in every copy. A backend written to the base's hooks gets it
whole, and ADK's and Deep Agents' turns enter through it.

Offline: catalogs never list (``tests/conftest.py``), and no model is called.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

import pytest
import structlog

from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.config import AgentConfig, AgentGraph, validate_agent_graph
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.sessions import SessionRef, get_active_turn
from tests.conftest import MockContext
from tests.workflow.test_turn_serialization import turn_harness

# ---------------------------------------------------------------------------
# A backend written to the turn's hooks alone
# ---------------------------------------------------------------------------


class _Stub(BaseWorkflowManager):
    """A backend whose every turn holds until the test lets it go.

    Each turn records, as it starts and as it ends, what it is and the active
    turn it sees, and the log context it runs under.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.engine: object | None = None
        self.builds = 0
        self.seen: list[tuple[str, SessionRef | None]] = []
        self.logged_as: list[dict[str, Any]] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    @property
    def backend_type(self) -> str:
        return "stub"

    def _get_state_tools(self) -> list[Callable]:
        return []

    def _validate_agent_graph(self) -> AgentGraph:
        return validate_agent_graph(self._agent_configs, backend=self.backend_type)

    async def _do_initialize(self) -> None:
        self.builds += 1
        self.engine = object()

    def _backend_ready(self) -> bool:
        return self._initialized and self.engine is not None

    def _detach_backend(self):
        self.engine = None
        return []

    async def _hold(self, what: str) -> None:
        self.seen.append((f"start {what}", get_active_turn()))
        self.logged_as.append(structlog.contextvars.get_contextvars())
        self.entered.set()
        await self.release.wait()
        self.seen.append((f"end {what}", get_active_turn()))

    async def process(self, message, user_id, session_id=None):
        async with self._turn(session_id, user_id) as ref:
            await self._hold(f"turn {message}")
            yield WorkflowEvent.text(message, ref.session_id)


@pytest.fixture
def ctx():
    with MockContext(google_api_key="test-key") as context:
        yield context


@pytest.fixture
def stub(ctx):
    return _Stub([AgentConfig(name="a", prompt="p")], settings=ctx.settings)


async def _reply(manager, message: str = "hi", user: str = "u", session: str | None = "s"):
    return [event.content async for event in manager.process(message, user, session)]


def _ref(manager: BaseWorkflowManager, user: str = "u", session: str = "s") -> SessionRef:
    return SessionRef(manager.app_name, user, session)


class TestTheTurnEntry:
    async def test_a_turn_runs_as_its_conversation(self, stub):
        assert await _reply(stub, user="alice", session="s1") == ["hi"]

        ref = _ref(stub, "alice", "s1")
        assert stub.seen == [("start turn hi", ref), ("end turn hi", ref)]
        assert get_active_turn() is None

    async def test_without_a_session_it_is_the_managers_current_one(self, stub):
        stub.session_id = "current"

        await _reply(stub, user="alice", session=None)

        assert stub.seen[0] == ("start turn hi", _ref(stub, "alice", "current"))

    async def test_its_log_lines_name_the_conversation(self, stub):
        await _reply(stub, user="alice", session="s1")

        assert stub.logged_as[0]["session_id"] == "s1"
        assert stub.logged_as[0]["user_id"] == "alice"

    async def test_it_initializes_the_manager_and_again_after_a_cleanup(self, stub):
        await _reply(stub)
        assert stub.is_initialized and stub.builds == 1

        await stub.cleanup()
        await _reply(stub)

        assert stub.builds == 2
        await stub.cleanup()

    async def test_turns_run_one_at_a_time(self, stub):
        stub.release.clear()
        first = asyncio.create_task(_reply(stub, "first"))
        await asyncio.wait_for(stub.entered.wait(), timeout=2)

        second = asyncio.create_task(_reply(stub, "second"))
        await asyncio.sleep(0.05)
        assert [what for what, _ in stub.seen] == ["start turn first"], "turns overlapped"

        stub.release.set()
        assert await first == ["first"] and await second == ["second"]
        assert [what for what, _ in stub.seen] == [
            "start turn first", "end turn first", "start turn second", "end turn second",
        ]

    async def test_a_cancelled_turn_frees_the_manager(self, stub):
        stub.release.clear()
        first = asyncio.create_task(_reply(stub, "first"))
        await asyncio.wait_for(stub.entered.wait(), timeout=2)

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert get_active_turn() is None

        stub.release.set()
        assert await asyncio.wait_for(_reply(stub, "second"), timeout=2) == ["second"]


# ---------------------------------------------------------------------------
# ADK and Deep Agents enter through it
# ---------------------------------------------------------------------------


@pytest.fixture(params=["adk", "deepagents"])
def harness(request):
    h, context = turn_harness(request.param)
    h.release.set()
    try:
        yield h
    finally:
        context.__exit__(None, None, None)


def _record_turns(manager: BaseWorkflowManager) -> list[SessionRef]:
    """Each turn ``manager`` enters through ``_turn``, by its conversation."""
    entered: list[SessionRef] = []
    turn = manager._turn

    @contextlib.asynccontextmanager
    async def recorded(session_id=None, user_id=None):
        async with turn(session_id, user_id) as ref:
            entered.append(ref)
            yield ref

    manager._turn = recorded
    return entered


class TestEachBackendEntersThroughIt:
    async def test_a_turn(self, harness):
        entered = _record_turns(harness.manager)

        assert await harness.drain("sess-a") == ["sess-a"]

        assert entered == [_ref(harness.manager, "u", "sess-a")]

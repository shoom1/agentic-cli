"""The services a manager builds are one table (``workflow/services.py``).

Adding a service meant editing four places: the declarable keys, the build
function, the closers and the service-tool map, and a test read the build
function's source text to keep the first two in step. Each service is now one
row: what a tool may declare and what a release closes are read from it.

A backend outside the framework that adds a service of its own, by
overriding ``_build_services_into``, ``_get_service_tool_map`` or
``_SYNC_SERVICE_CLOSERS``, keeps working: the base still calls them.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import WorkflowEvent
from tests.conftest import MockContext


def test_what_a_release_closes_is_read_from_the_table():
    from agentic_cli.workflow.services import SERVICES

    closers = {(spec.key, spec.close) for spec in SERVICES if spec.close is not None}

    assert set(BaseWorkflowManager._SYNC_SERVICE_CLOSERS) == closers
    assert closers == {
        ("knowledge_base", "close"),
        ("sandbox_manager", "cleanup"),
        ("job_manager", "close"),
    }


def test_every_service_has_one_row():
    from agentic_cli.workflow.services import SERVICES

    keys = [spec.key for spec in SERVICES]
    assert len(keys) == len(set(keys))
    assert all(callable(spec.build) for spec in SERVICES)


# ---------------------------------------------------------------------------
# A backend that adds a service of its own
# ---------------------------------------------------------------------------


class _Own:
    """A service of the backend's own, which holds something to release."""

    def __init__(self) -> None:
        self.closes = 0

    def shutdown(self) -> None:
        self.closes += 1


class _OwnServiceBackend(BaseWorkflowManager):
    """Builds one more service, gives one more tool, and closes its service."""

    _SYNC_SERVICE_CLOSERS = BaseWorkflowManager._SYNC_SERVICE_CLOSERS + (("own", "shutdown"),)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tool_maps: list[dict[str, Callable]] = []
        self.built_tools: list[list] = []

    @property
    def backend_type(self) -> str:
        return "own"

    def _get_state_tools(self) -> list[Callable]:
        return []

    def _build_services_into(self, s: dict[str, Any], existing: frozenset[str]) -> None:
        super()._build_services_into(s, existing)
        if "own" not in existing:
            s["own"] = _Own()

    def _get_service_tool_map(self) -> dict[str, Callable]:
        tools = super()._get_service_tool_map()

        def own_tool() -> dict:
            """The backend's tool."""
            return {"success": True}

        tools["own_tool"] = own_tool
        self.tool_maps.append(tools)
        return tools

    async def _do_initialize(self) -> None:
        self.built_tools = [self._build_tools(config) for config in self._agent_configs]

    async def process(self, message, user_id, session_id=None):
        async with self._turn(session_id, user_id) as ref:
            yield WorkflowEvent.text("ok", ref.session_id)


@pytest.fixture
def ctx():
    with MockContext(google_api_key="test-key") as context:
        yield context


def _own_backend(ctx) -> _OwnServiceBackend:
    return _OwnServiceBackend([AgentConfig(name="a", prompt="p")], settings=ctx.settings)


class TestABackendThatAddsItsOwnService:
    async def test_its_service_is_built_published_and_closed_once(self, ctx):
        manager = _own_backend(ctx)
        await manager.initialize_services()
        own = manager.services["own"]

        assert isinstance(own, _Own)
        assert "permission_engine" in manager.services
        await manager.cleanup()
        await manager.cleanup()

        assert own.closes == 1
        assert manager.services == {}

    async def test_a_model_switch_carries_it(self, ctx):
        manager = _own_backend(ctx)
        await manager.initialize_services()
        own = manager.services["own"]

        await manager.reinitialize(preserve_sessions=True)

        assert manager.services["own"] is own
        assert own.closes == 0
        await manager.cleanup()
        assert own.closes == 1

    async def test_its_tool_map_is_the_one_the_agents_are_built_with(self, ctx):
        manager = _own_backend(ctx)
        await manager.initialize_services()

        assert len(manager.tool_maps) == 1
        assert "own_tool" in manager.tool_maps[0]
        assert "ask_clarification" in manager.tool_maps[0]
        await manager.cleanup()

    def test_a_build_that_fails_after_it_closes_it(self, ctx):
        """``_build_services`` rolls back what the override built too."""
        built: list[_Own] = []

        class _Fails(_OwnServiceBackend):
            def _build_services_into(self, s, existing):
                s["own"] = _Own()
                built.append(s["own"])
                raise RuntimeError("the next service failed")

        manager = _Fails([AgentConfig(name="a", prompt="p")], settings=ctx.settings)

        with pytest.raises(RuntimeError, match="the next service failed"):
            manager._build_services()

        assert built[0].closes == 1


def test_the_tool_map_has_the_interaction_tools_without_any_service(ctx):
    """The manager is always there for ``ask_clarification``."""
    manager = _own_backend(ctx)
    manager._services = {}

    assert "ask_clarification" in BaseWorkflowManager._get_service_tool_map(manager)


def test_services_assigned_to_the_manager_are_the_ones_it_publishes(ctx):
    """Harnesses assign ``_services`` and mutate it in place."""
    manager = _own_backend(ctx)
    services = {"job_manager": SimpleNamespace(close=lambda: None)}

    manager._services = services
    manager._services["sandbox_manager"] = SimpleNamespace(cleanup=lambda: None)

    assert manager.services is services
    assert set(manager.services) == {"job_manager", "sandbox_manager"}


class TestWhatASetOwns:
    """A set holds services it does not own (another manager's, borrowed or
    given away): no release of it closes them."""

    def test_a_release_drops_what_it_does_not_own_unclosed(self):
        from agentic_cli.workflow.services import ServiceSet

        own, theirs = SimpleNamespace(), SimpleNamespace()
        held = ServiceSet({"job_manager": own})
        held.borrow({"sandbox_manager": theirs, "job_manager": SimpleNamespace()})

        assert held.services == {"job_manager": own, "sandbox_manager": theirs}
        assert held.release() == {"job_manager": own}
        assert held.services == {} and held.not_owned == {}

    def test_a_carried_service_stays_whoever_owns_it(self):
        from agentic_cli.workflow.services import ServiceSet

        own, theirs = SimpleNamespace(), SimpleNamespace()
        held = ServiceSet({"job_manager": own})
        held.borrow({"sandbox_manager": theirs})
        held.carried = frozenset(held.services)

        assert held.release() == {}
        assert held.services == {"job_manager": own, "sandbox_manager": theirs}
        assert held.owns("job_manager", own) and not held.owns("sandbox_manager", theirs)

    def test_it_lends_the_table_s_services_it_owns_but_the_manager(self):
        from agentic_cli.workflow.services import ServiceSet

        manager, own, theirs = object(), SimpleNamespace(), SimpleNamespace()
        held = ServiceSet({"workflow": manager, "job_manager": own, "own": SimpleNamespace()})
        held.borrow({"sandbox_manager": theirs})

        assert held.lendable(manager) == {"job_manager": own}

    def test_given_away_it_closes_them_no_more(self):
        from agentic_cli.workflow.services import ServiceSet

        lent = SimpleNamespace()
        lender, borrower = ServiceSet({"job_manager": lent}), ServiceSet()
        borrower.borrow({"job_manager": lent})

        lender.give(borrower.borrowed({"job_manager": lent}))
        borrower.adopt({"job_manager": lent})

        assert lender.release() == {}
        assert borrower.release() == {"job_manager": lent}

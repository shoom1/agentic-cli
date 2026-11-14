"""A question or a permission prompt names the sub-agent run that asks.

``UserInputRequest.run`` is the ``RunRef`` of the sub-agent run whose call
asks, None for the root's own: ``request_user_input`` fills it from
``get_active_run()`` when the caller left it None, so every request is
covered, whoever built it (both permission gates, ``ask_clarification``, an
app's own tool). ``run_label`` is how a dialog names that run.
"""

from __future__ import annotations

import asyncio

import pytest

from agentic_cli.workflow.agents import run_label
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.events import UserInputRequest, UserInputUnavailable
from agentic_cli.workflow.sessions import (
    RunRef,
    close_turn_scope,
    open_turn_scope,
    reset_active_run,
    root_run,
    set_active_run,
)
from tests.conftest import settings_double

RESEARCHER = RunRef(
    agent="researcher", instance="researcher-2", run="researcher-2#1", parent_run=None, depth=1
)
LEAF = RunRef(agent="leaf", instance="leaf-1", run="leaf-1#3", parent_run="middle-1#1", depth=2)


class _Asking(BaseWorkflowManager):
    """The abstract hooks, and a consumer that keeps each request it is asked."""

    def __init__(self) -> None:
        super().__init__(agent_configs=[], settings=settings_double(default_user="user"))
        self.asked: list[UserInputRequest] = []

        async def answer(request: UserInputRequest) -> str:
            self.asked.append(request)
            return "yes"

        self.set_input_callback(answer)

    @property
    def backend_type(self) -> str:
        return "asking"

    def _get_state_tools(self):
        return []

    async def _do_initialize(self) -> None:
        pass

    async def process(self, message, user_id, session_id=None):
        raise NotImplementedError

    async def reinitialize(self, model=None, preserve_sessions=True):
        pass

    async def cleanup(self):
        pass


def _request(**fields) -> UserInputRequest:
    return UserInputRequest(request_id="r", tool_name="ask_clarification", prompt="Which?", **fields)


def test_a_request_names_no_run_unless_given_one():
    """The field is last, with a default: a request built as before is unchanged."""
    request = UserInputRequest("r", "t", "p")

    assert request.run is None


class TestTheRequestNamesItsRun:
    async def test_a_sub_agents_request_is_given_its_run(self):
        manager = _Asking()
        token = set_active_run(RESEARCHER)
        try:
            await manager.request_user_input(_request())
        finally:
            reset_active_run(token)

        (asked,) = manager.asked
        assert asked.run == RESEARCHER

    async def test_the_roots_request_names_no_run(self):
        manager = _Asking()

        await manager.request_user_input(_request())

        (asked,) = manager.asked
        assert asked.run is None

    async def test_a_run_the_caller_gave_is_kept(self):
        """A request that already names its run (a permission prompt, built
        with it) keeps it, whatever run is active where it is asked."""
        manager = _Asking()
        token = set_active_run(RESEARCHER)
        try:
            await manager.request_user_input(_request(run=LEAF))
        finally:
            reset_active_run(token)

        (asked,) = manager.asked
        assert asked.run == LEAF

    @pytest.mark.parametrize("refused_by", ["no callback", "its turn has ended"])
    async def test_a_refused_request_is_given_its_run_first(self, refused_by):
        """The run is filled before the request can be refused, so whoever
        catches the refusal can still say which run asked."""
        manager = _Asking()
        request = _request()
        token = set_active_run(RESEARCHER)
        try:
            if refused_by == "no callback":
                manager.clear_input_callback()
                asking = manager.request_user_input(request)
            else:
                go = asyncio.Event()

                async def later() -> str:
                    await go.wait()
                    return await manager.request_user_input(request)

                span = open_turn_scope()
                asking = asyncio.create_task(later())  # copies the turn's context
                close_turn_scope(span)
                go.set()
            with pytest.raises(UserInputUnavailable):
                await asking
        finally:
            reset_active_run(token)

        assert manager.asked == []
        assert request.run == RESEARCHER


class TestRunLabel:
    def test_a_sub_agent_run_is_its_instance_its_agent_and_its_number(self):
        assert run_label(RESEARCHER) == "researcher-2 (researcher, run 1)"
        assert run_label(LEAF) == "leaf-1 (leaf, run 3)"

    def test_the_root_has_none(self):
        assert run_label(root_run("coordinator")) is None
        assert run_label(None) is None

    def test_a_depth_0_agent_has_none(self):
        """An agent the root transfers to (a native ADK YAML app's) replies
        as the root does: it is no sub-agent run (``run`` None), so no label."""
        assert run_label(RunRef("helper", "helper", None, None, 0)) is None

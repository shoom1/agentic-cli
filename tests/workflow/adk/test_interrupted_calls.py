"""On ADK each call is recorded, sealed and answered as the run's that made it.

ADK runs a sub-agent on the branch ``<agent>@<call id>``, and a provider that
numbers its call ids per agent gives the sub-agent's first call the id of
the delegation that started it. The plugin records a call that finished
under the run in progress (``MainAgentPlugin``'s: the call's own), answers a
stopped turn's seal from the record of the seal's own branch's run; and
``left_open`` pairs a result with a call on its own branch only, and answers
a call a stopped turn left open as the user's.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("google.adk")

from google.adk.events import Event  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_cli.workflow.adk.branches import branch_run  # noqa: E402
from agentic_cli.workflow.adk.interrupted_calls_plugin import (  # noqa: E402
    ABORTED,
    InterruptedCallsPlugin,
    left_open,
)
from agentic_cli.workflow.agents import AgentRuns  # noqa: E402
from agentic_cli.workflow.interrupts import (  # noqa: E402
    INTERRUPTED,
    INTERRUPTED_BY_USER,
    OpenCalls,
    interrupted,
)
from agentic_cli.workflow.sessions import (  # noqa: E402
    SessionRef,
    reset_active_run,
    reset_active_turn,
    set_active_run,
    set_active_turn,
)

CONVERSATION = SessionRef("app", "u", "s")


@contextmanager
def turn_of(conversation: SessionRef) -> Iterator[None]:
    token = set_active_turn(conversation)
    try:
        yield
    finally:
        reset_active_turn(token)


def _tool(name: str) -> Any:
    return SimpleNamespace(name=name, is_long_running=False)


def _context(call_id: str, *, branch: str | None) -> Any:
    return SimpleNamespace(
        function_call_id=call_id, branch=branch,
        actions=SimpleNamespace(state_delta={}, artifact_delta={}),
    )


def _calls(author: str, branch: str | None, *made: tuple[str, str]) -> Event:
    return Event(
        author=author, invocation_id="i1", branch=branch,
        content=types.Content(role="model", parts=[
            types.Part(function_call=types.FunctionCall(id=call_id, name=name, args={}))
            for name, call_id in made
        ]),
    )


def _results(
    author: str, branch: str | None, *made: tuple[str, str, dict], error_code: str | None = None
) -> Event:
    return Event(
        author=author, invocation_id="i1", branch=branch, error_code=error_code,
        content=types.Content(role="user", parts=[
            types.Part(function_response=types.FunctionResponse(id=call_id, name=name, response=response))
            for name, call_id, response in made
        ]),
    )


def _seal(author: str, branch: str | None, *made: tuple[str, str]) -> Event:
    """ADK's seal of a stopped turn's open calls (``_abort_events``)."""
    return _results(
        author, branch,
        *((name, call_id, {"error": "Invocation was aborted by client."}) for name, call_id in made),
        error_code=ABORTED,
    )


def _answers(event: Event) -> dict[str, Any]:
    return {response.id: response.response for response in event.get_function_responses()}


class TestSeals:
    """ADK writes a stopped turn's seals outside the runs whose calls they
    answer, so each seal is answered from the record of its own branch's run."""

    async def test_a_seal_is_answered_from_its_own_branchs_record(self):
        calls, runs = OpenCalls(), AgentRuns()
        plugin = InterruptedCallsPlugin(calls, runs=runs)

        with turn_of(CONVERSATION):
            # The researcher's call finishes in its run (MainAgentPlugin's).
            token = set_active_run(branch_run(runs, CONVERSATION, "researcher@c1"))
            await plugin.after_tool_callback(
                tool=_tool("contract_lookup"), tool_args={},
                tool_context=_context("c1", branch="researcher@c1"),
                result={"success": True, "value": 42},
            )
            reset_active_run(token)
            parents = await plugin.on_event_callback(
                invocation_context=None, event=_seal("coordinator", None, ("researcher", "c1")),
            )
            its_own = await plugin.on_event_callback(
                invocation_context=None,
                event=_seal("researcher", "researcher@c1", ("contract_lookup", "c1")),
            )

        assert _answers(parents) == {"c1": interrupted(INTERRUPTED_BY_USER)}
        assert _answers(its_own) == {"c1": {"success": True, "value": 42}}


class TestLeftOpen:
    def test_a_sub_agents_result_never_answers_its_parents_call(self):
        events = [
            _calls("coordinator", None, ("researcher", "c1")),
            _calls("researcher", "researcher@c1", ("contract_lookup", "c1")),
            _results("researcher", "researcher@c1", ("contract_lookup", "c1", {"success": True})),
        ]

        with turn_of(CONVERSATION):
            (answer,) = left_open(events, OpenCalls())

        assert (answer.author, answer.branch) == ("coordinator", None)
        assert _answers(answer) == {"c1": interrupted(INTERRUPTED)}

    def test_after_a_stopped_turn_a_call_left_open_is_the_users(self):
        """Asked once: the next turn's calls left open are a crash's again."""
        calls = OpenCalls()
        calls.cancelled(CONVERSATION)
        events = [_calls("coordinator", None, ("researcher", "c1"))]

        with turn_of(CONVERSATION):
            (first,) = left_open(events, calls)
            (again,) = left_open(events, calls)

        assert _answers(first) == {"c1": interrupted(INTERRUPTED_BY_USER)}
        assert _answers(again) == {"c1": interrupted(INTERRUPTED)}

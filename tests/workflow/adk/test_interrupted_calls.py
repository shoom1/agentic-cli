"""On ADK each call is recorded, sealed and answered as the run's that made it.

ADK runs a sub-agent on the branch ``<agent>@<call id>``, and a provider that
numbers its call ids per agent gives the sub-agent's first call the id of
the delegation that started it. The plugin records a call that finished
under the run in progress (``MainAgentPlugin``'s: the call's own), answers a
stopped turn's seal from the record of the seal's own branch's run, and
marks each call running by its branch and id. ADK's seals pair a result
with a call by id alone, so ``unsealed`` answers, as the turn stops, a call
they missed, from the record of its branch's run; ``left_open`` pairs a
result with a call on its own branch only, and answers a call a stopped
turn left open as the user's.
"""

from __future__ import annotations

import asyncio
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
    unsealed,
)
from agentic_cli.workflow.agents import AgentRuns  # noqa: E402
from agentic_cli.tools.jobs.backends import JobState  # noqa: E402
from agentic_cli.workflow.interrupts import (  # noqa: E402
    INTERRUPTED,
    INTERRUPTED_BY_USER,
    NOT_RUN,
    OpenCalls,
    interrupted,
    job_started,
)
from agentic_cli.workflow.service_registry import (  # noqa: E402
    JOB_MANAGER,
    set_service_registry,
)
from agentic_cli.workflow.sessions import (  # noqa: E402
    SessionRef,
    ToolCallRef,
    get_active_tool_call,
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


def _context(
    call_id: str, *, branch: str | None, state_delta: dict[str, Any] | None = None
) -> Any:
    return SimpleNamespace(
        function_call_id=call_id, branch=branch,
        actions=SimpleNamespace(state_delta=state_delta or {}, artifact_delta={}),
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


class TestActiveToolCall:
    async def test_a_sub_agents_call_never_ends_its_parents_mark(self):
        """The researcher's call ``c1`` runs inside its coordinator's
        delegation ``c1``: each ends its own mark."""
        plugin = InterruptedCallsPlugin(OpenCalls())
        parents = _context("c1", branch=None)
        its_own = _context("c1", branch="researcher@c1")

        with turn_of(CONVERSATION):
            await plugin.before_tool_callback(
                tool=_tool("researcher"), tool_args={}, tool_context=parents
            )
            await plugin.before_tool_callback(
                tool=_tool("contract_lookup"), tool_args={}, tool_context=its_own
            )
            assert get_active_tool_call() == ToolCallRef("c1", "contract_lookup")
            await plugin.after_tool_callback(
                tool=_tool("contract_lookup"), tool_args={}, tool_context=its_own, result={}
            )
            assert get_active_tool_call() == ToolCallRef("c1", "researcher")
            await plugin.after_tool_callback(
                tool=_tool("researcher"), tool_args={}, tool_context=parents, result={}
            )

            assert get_active_tool_call() is None


class TestUnsealed:
    """ADK's seals pair a result with a call by id alone, so a call whose id a
    sub-agent's result used is left open as the turn stops: it is answered
    then, from the record of its own branch's run. Each session below ends
    with ADK's seal of the stopped invocation, as a stopped turn's does."""

    async def test_a_parents_calls_are_answered_from_its_own_record(self):
        calls, runs = OpenCalls(), AgentRuns()
        plugin = InterruptedCallsPlugin(calls, runs=runs)
        events = [
            _calls("coordinator", None, ("researcher", "c1"), ("contract_whoami", "c2")),
            _calls("researcher", "researcher@c1", ("contract_lookup", "c1")),
            _results("researcher", "researcher@c1", ("contract_lookup", "c1", {"value": 42})),
            _calls("researcher", "researcher@c1", ("contract_lookup", "c2")),
            _results("researcher", "researcher@c1", ("contract_lookup", "c2", {"value": 42})),
            _calls("researcher", "researcher@c1", ("contract_block", "c3")),
            # ADK seals only c3: the researcher's results hide c1 and c2.
            _seal("researcher", "researcher@c1", ("contract_block", "c3")),
        ]

        with turn_of(CONVERSATION):
            # The coordinator's own c2 finished beside its delegation.
            await plugin.after_tool_callback(
                tool=_tool("contract_whoami"), tool_args={},
                tool_context=_context("c2", branch=None, state_delta={"seen": True}),
                result={"success": True},
            )
            (answer,) = unsealed(events, calls, runs)

        assert (answer.author, answer.branch, answer.error_code) == ("coordinator", None, ABORTED)
        assert _answers(answer) == {
            "c1": interrupted(INTERRUPTED_BY_USER), "c2": {"success": True},
        }
        # Its state change is stored with its answer, as with its result.
        assert answer.actions.state_delta == {"seen": True}

    async def test_a_sub_agents_call_is_answered_from_its_runs_record(self):
        calls, runs = OpenCalls(), AgentRuns()
        events = [
            _calls("coordinator", None, ("contract_whoami", "c1")),
            _results("coordinator", None, ("contract_whoami", "c1", {"success": True})),
            _calls("coordinator", None, ("researcher", "c2")),
            _calls("researcher", "researcher@c2", ("write_file", "c1")),
            # ADK seals only the delegation: the coordinator's result hides c1.
            _seal("coordinator", None, ("researcher", "c2")),
        ]

        with turn_of(CONVERSATION):
            # The researcher's c1 waited at its prompt, in its run.
            token = set_active_run(branch_run(runs, CONVERSATION, "researcher@c2"))
            with pytest.raises(asyncio.CancelledError), calls.asking("c1"):
                raise asyncio.CancelledError
            reset_active_run(token)
            answers = unsealed(events, calls, runs)

        (answer,) = answers
        assert (answer.author, answer.branch) == ("researcher", "researcher@c2")
        assert _answers(answer) == {"c1": interrupted(NOT_RUN)}

    def test_a_stop_adk_has_not_sealed_is_left_to_the_next_turn(self):
        """Its seals would come after these answers, and answer a call
        twice: ``left_open`` answers what they leave."""
        events = [_calls("coordinator", None, ("researcher", "c1"))]

        with turn_of(CONVERSATION):
            assert unsealed(events, OpenCalls(), AgentRuns()) == []


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

    def test_a_sub_agents_call_is_never_answered_with_the_roots_job(self):
        """A provider that numbers its call ids repeats them across agents.
        The root's call keeps its job; a sub-agent's call of the same id is
        answered as interrupted, its run being numbered no more (a crash's,
        or one a stop ended), so no job is matched to it."""
        root_job = SimpleNamespace(
            job_id="j-root", call_id="call_0", session_id="s", user_id="u",
            state=JobState.RUNNING, started_by=None,
        )
        events = [
            _calls("coordinator", None, ("contract_start_job", "call_0"), ("researcher", "call_9")),
            _calls("researcher", "researcher@call_9", ("contract_lookup", "call_0")),
            _calls("leaf", "researcher@call_9.leaf@call_1", ("contract_lookup", "call_0")),
        ]

        registry = set_service_registry({JOB_MANAGER: SimpleNamespace(snapshot=lambda: [root_job])})
        try:
            with turn_of(CONVERSATION):
                answers = {answer.branch: _answers(answer) for answer in left_open(events, OpenCalls())}
        finally:
            registry.var.reset(registry)

        assert answers == {
            None: {"call_0": job_started(root_job), "call_9": interrupted(INTERRUPTED)},
            "researcher@call_9": {"call_0": interrupted(INTERRUPTED)},
            "researcher@call_9.leaf@call_1": {"call_0": interrupted(INTERRUPTED)},
        }

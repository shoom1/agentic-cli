"""On ADK each call is recorded, sealed and answered as the run's that made it.

A sub-agent run runs on its type's own runner, in a task of its own where
its run is the active one (``agents/delegate.py``), and a provider that
numbers its call ids per agent gives the sub-agent's first call the id of
the delegation that started it. The plugin records a call that finished
under the run in progress, answers a stopped invocation's seal from the
record of the run whose runner wrote it (the active run there), and marks
each call running by its invocation and id. ADK's seals pair a result with
a call by id alone, so ``unsealed`` answers, as the turn stops, a call of
the conversation's they missed; ``left_open`` pairs a result with a call on
its own branch only, and answers a call a stopped turn left open as the
user's. ``left_open_by`` answers the calls an instance's own agent left open
before its next run.
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
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_cli.workflow.adk.interrupted_calls_plugin import (  # noqa: E402
    ABORTED,
    InterruptedCallsPlugin,
    left_open,
    left_open_by,
    unsealed,
)
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
    RunRef,
    SessionRef,
    ToolCallRef,
    get_active_tool_call,
    reset_active_run,
    reset_active_turn,
    set_active_run,
    set_active_turn,
)

CONVERSATION = SessionRef("app", "u", "s")
RESEARCHER = RunRef(
    agent="researcher", instance="researcher-1", run="researcher-1#1", parent_run=None, depth=1
)


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
    call_id: str, *, invocation: str = "i1", state_delta: dict[str, Any] | None = None
) -> Any:
    return SimpleNamespace(
        function_call_id=call_id, invocation_id=invocation,
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
    """A stopped invocation's seals are written by the runner that ran its
    calls, in the task that iterates it: each is answered from the record of
    the run in progress there."""

    async def test_a_seal_is_answered_from_its_runs_record(self):
        calls = OpenCalls()
        plugin = InterruptedCallsPlugin(calls)

        with turn_of(CONVERSATION):
            # The researcher's call finishes in its run.
            token = set_active_run(RESEARCHER)
            await plugin.after_tool_callback(
                tool=_tool("contract_lookup"), tool_args={},
                tool_context=_context("c1", invocation="i-run"),
                result={"success": True, "value": 42},
            )
            its_own = await plugin.on_event_callback(
                invocation_context=None,
                event=_seal("researcher", None, ("contract_lookup", "c1")),
            )
            reset_active_run(token)
            parents = await plugin.on_event_callback(
                invocation_context=None, event=_seal("coordinator", None, ("task", "c1")),
            )

        assert _answers(parents) == {"c1": interrupted(INTERRUPTED_BY_USER)}
        assert _answers(its_own) == {"c1": {"success": True, "value": 42}}


    @pytest.mark.parametrize("cause, text", [
        ("its parent", "Not run: its parent stopped the run before this call ran."),
        (None, NOT_RUN),
    ])
    async def test_a_call_at_its_prompt_is_told_what_stopped_its_run(self, cause, text):
        calls = OpenCalls()
        plugin = InterruptedCallsPlugin(calls, stop_cause=lambda: cause)

        with turn_of(CONVERSATION):
            token = set_active_run(RESEARCHER)
            with pytest.raises(asyncio.CancelledError):
                with calls.asking("c1"):
                    raise asyncio.CancelledError
            sealed = await plugin.on_event_callback(
                invocation_context=None, event=_seal("researcher", None, ("write_file", "c1")),
            )
            reset_active_run(token)

        assert _answers(sealed) == {"c1": interrupted(text)}


class TestActiveToolCall:
    async def test_a_sub_agents_call_never_ends_its_parents_mark(self):
        """The researcher's call ``c1`` runs inside its coordinator's
        delegation ``c1``, on an invocation of its own: each ends its own mark."""
        plugin = InterruptedCallsPlugin(OpenCalls())
        parents = _context("c1", invocation="i-turn")
        its_own = _context("c1", invocation="i-run")

        with turn_of(CONVERSATION):
            await plugin.before_tool_callback(
                tool=_tool("task"), tool_args={}, tool_context=parents
            )
            await plugin.before_tool_callback(
                tool=_tool("contract_lookup"), tool_args={}, tool_context=its_own
            )
            assert get_active_tool_call() == ToolCallRef("c1", "contract_lookup")
            await plugin.after_tool_callback(
                tool=_tool("contract_lookup"), tool_args={}, tool_context=its_own, result={}
            )
            assert get_active_tool_call() == ToolCallRef("c1", "task")
            await plugin.after_tool_callback(
                tool=_tool("task"), tool_args={}, tool_context=parents, result={}
            )

            assert get_active_tool_call() is None


class TestUnsealed:
    """ADK's seals pair a result with a call by id alone, so a call whose id
    another branch's result used (a native ADK YAML app's agents share the
    conversation's session) is left open as the turn stops: it is answered
    then, from the conversation's record. Each session below ends with ADK's
    seal of the stopped invocation, as a stopped turn's does."""

    async def test_the_conversations_calls_are_answered_from_its_record(self):
        calls = OpenCalls()
        plugin = InterruptedCallsPlugin(calls)
        events = [
            _calls("coordinator", None, ("helper_tool", "c1"), ("contract_whoami", "c2")),
            _calls("helper", "helper", ("contract_lookup", "c1")),
            _results("helper", "helper", ("contract_lookup", "c1", {"value": 42})),
            _calls("helper", "helper", ("contract_lookup", "c2")),
            _results("helper", "helper", ("contract_lookup", "c2", {"value": 42})),
            _calls("helper", "helper", ("contract_block", "c3")),
            # ADK seals only c3: the helper's results hide c1 and c2.
            _seal("helper", "helper", ("contract_block", "c3")),
        ]

        with turn_of(CONVERSATION):
            # The coordinator's own c2 finished.
            await plugin.after_tool_callback(
                tool=_tool("contract_whoami"), tool_args={},
                tool_context=_context("c2", state_delta={"seen": True}),
                result={"success": True},
            )
            (answer,) = unsealed(events, calls)

        assert (answer.author, answer.branch, answer.error_code) == ("coordinator", None, ABORTED)
        assert _answers(answer) == {
            "c1": interrupted(INTERRUPTED_BY_USER), "c2": {"success": True},
        }
        # Its state change is stored with its answer, as with its result.
        assert answer.actions.state_delta == {"seen": True}

    def test_a_stop_adk_has_not_sealed_is_left_to_the_next_turn(self):
        """Its seals would come after these answers, and answer a call
        twice: ``left_open`` answers what they leave."""
        events = [_calls("coordinator", None, ("task", "c1"))]

        with turn_of(CONVERSATION):
            assert unsealed(events, OpenCalls()) == []


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

    async def test_after_a_failed_turn_a_call_that_returned_keeps_its_result(self):
        """A failed turn keeps the record of its calls that returned
        (``OpenCalls.failed``): the root's call left open that returned is
        answered with its result and the state it changed, one that did not
        as interrupted."""
        calls = OpenCalls()
        plugin = InterruptedCallsPlugin(calls)
        events = [_calls("coordinator", None, ("contract_whoami", "c1"), ("read_file", "c2"))]

        with turn_of(CONVERSATION):
            await plugin.after_tool_callback(
                tool=_tool("contract_whoami"), tool_args={},
                tool_context=_context("c1", state_delta={"seen": True}),
                result={"success": True},
            )
            calls.failed(CONVERSATION)
            (answer,) = left_open(events, calls)

        assert (answer.author, answer.branch, answer.error_code) == ("coordinator", None, ABORTED)
        assert _answers(answer) == {"c1": {"success": True}, "c2": interrupted(INTERRUPTED)}
        # Its state change is stored with its answer, as with its result.
        assert answer.actions.state_delta == {"seen": True}


SECOND = RunRef(
    agent="researcher", instance="researcher-1", run="researcher-1#2", parent_run=None, depth=1
)


class TestLeftOpenBy:
    """Before an instance's next run, its child session's calls an earlier
    run left open are answered as the instance's; the conversation's record
    and its stop are not its."""

    async def test_its_own_calls_are_answered_from_its_record(self):
        calls = OpenCalls()
        plugin = InterruptedCallsPlugin(calls)
        events = [
            _calls("coordinator", None, ("task", "c0")),  # an inheriting copy's: not its own
            _calls("researcher", None, ("contract_lookup", "r1"), ("contract_block", "r2")),
        ]
        token = set_active_run(RESEARCHER)
        try:
            with turn_of(CONVERSATION):
                await plugin.after_tool_callback(
                    tool=_tool("contract_lookup"), tool_args={},
                    tool_context=_context("r1", state_delta={"seen": True}),
                    result={"success": True, "value": 42},
                )
        finally:
            reset_active_run(token)
        calls.cancelled(CONVERSATION)

        with turn_of(CONVERSATION):
            (answer,) = left_open_by(events, calls, SECOND)
            stopped = calls.take_cancel(CONVERSATION)
            kept = calls.recorded("r1", run="researcher-1#2") is not None

        assert (answer.author, answer.branch, answer.error_code) == ("researcher", None, ABORTED)
        assert _answers(answer) == {
            "r1": {"success": True, "value": 42}, "r2": interrupted(INTERRUPTED),
        }
        assert answer.actions.state_delta == {"seen": True}
        # The conversation's stop and record are left as they were.
        assert stopped is True and kept is True

    def test_none_left_open_is_none(self):
        events = [
            _calls("researcher", None, ("contract_lookup", "r1")),
            _results("researcher", None, ("contract_lookup", "r1", {"success": True})),
        ]

        with turn_of(CONVERSATION):
            assert left_open_by(events, OpenCalls(), SECOND) == []


class TestPositionalPairing:
    def test_a_reused_id_left_open_is_answered_and_the_answered_one_is_not(self):
        """A conversation stored before reused ids were renamed may hold a
        provider's ``call_0`` twice: a result answers the latest call of its
        id before it, so the later call, left open, is answered, and the
        earlier one keeps its one result."""
        events = [
            _calls("coordinator", None, ("contract_lookup", "call_0")),
            _results("coordinator", None, ("contract_lookup", "call_0", {"success": True})),
            _calls("coordinator", None, ("read_file", "call_0")),
        ]

        with turn_of(CONVERSATION):
            (answer,) = left_open(events, OpenCalls())

        (response,) = answer.get_function_responses()
        assert (response.id, response.name) == ("call_0", "read_file")
        assert response.response == interrupted(INTERRUPTED)

    def test_an_earlier_call_left_open_is_answered_when_a_later_one_has_its_result(self):
        events = [
            _calls("coordinator", None, ("contract_block", "call_0")),
            _calls("coordinator", None, ("contract_lookup", "call_0")),
            _results("coordinator", None, ("contract_lookup", "call_0", {"success": True})),
        ]

        with turn_of(CONVERSATION):
            (answer,) = left_open(events, OpenCalls())

        (response,) = answer.get_function_responses()
        assert (response.id, response.name) == ("call_0", "contract_block")


def _callback(events: list[Event], branch: str | None = None) -> Any:
    """A model call's callback context: its session's events and its branch."""
    invocation = SimpleNamespace(branch=branch)
    return SimpleNamespace(
        session=SimpleNamespace(events=events), get_invocation_context=lambda: invocation,
    )


def _answer(*made: tuple[str, str], partial: bool | None = None) -> LlmResponse:
    """A model's answer that makes ``made`` calls, ``(name, id)`` each."""
    return LlmResponse(partial=partial, content=types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(id=call_id, name=name, args={}))
        for name, call_id in made
    ]))


def _ids(answer: LlmResponse) -> list[str | None]:
    return [part.function_call.id for part in answer.content.parts]


class TestAReusedCallId:
    """A provider that numbers its call ids per response (``call_0`` in every
    one) reuses an id the agent's conversation holds. ADK pairs a result
    with a call by id, so the reused one gets an id of its own as the answer
    arrives (``after_model_callback``), before ADK stores the answer and
    runs its calls: in place, since a plugin that returns an answer ends
    the other plugins' callbacks."""

    async def test_an_id_the_conversation_holds_is_renamed(self):
        events = [
            _calls("coordinator", None, ("contract_lookup", "call_0")),
            _results("coordinator", None, ("contract_lookup", "call_0", {"success": True})),
        ]
        answer = _answer(("read_file", "call_0"), ("contract_lookup", "call_1"))

        returned = await InterruptedCallsPlugin(OpenCalls()).after_model_callback(
            callback_context=_callback(events), llm_response=answer,
        )

        assert returned is None
        renamed, kept = _ids(answer)
        assert renamed.startswith("call_0-") and kept == "call_1"

    async def test_a_new_id_is_kept_as_the_provider_gave_it(self):
        answer = _answer(("contract_lookup", "call_0"))
        content = answer.content

        await InterruptedCallsPlugin(OpenCalls()).after_model_callback(
            callback_context=_callback([]), llm_response=answer,
        )

        assert answer.content is content
        assert _ids(answer) == ["call_0"]

    async def test_two_calls_of_one_id_in_one_answer_get_their_own(self):
        answer = _answer(("contract_lookup", "call_0"), ("contract_lookup", "call_0"))

        await InterruptedCallsPlugin(OpenCalls()).after_model_callback(
            callback_context=_callback([]), llm_response=answer,
        )

        first, second = _ids(answer)
        assert first == "call_0" and second.startswith("call_0-")

    async def test_only_the_agents_own_branch_and_those_above_it_count(self):
        """ADK shows an agent the events of its branch and its ancestors'
        (a native ADK YAML app's agents share a session): a sibling's call
        of the same id is none of its conversation."""
        events = [
            _calls("helper", "team.helper", ("contract_lookup", "call_0")),
            _calls("team", "team", ("contract_lookup", "call_1")),
        ]
        answer = _answer(("contract_lookup", "call_0"), ("contract_lookup", "call_1"))

        await InterruptedCallsPlugin(OpenCalls()).after_model_callback(
            callback_context=_callback(events, branch="team.other"), llm_response=answer,
        )

        kept, renamed = _ids(answer)
        assert kept == "call_0" and renamed.startswith("call_1-")

    async def test_a_partial_answer_is_left_to_the_whole_one(self):
        """ADK neither stores nor runs a streamed partial answer's calls."""
        events = [_calls("coordinator", None, ("contract_lookup", "call_0"))]
        answer = _answer(("contract_lookup", "call_0"), partial=True)

        await InterruptedCallsPlugin(OpenCalls()).after_model_callback(
            callback_context=_callback(events), llm_response=answer,
        )

        assert _ids(answer) == ["call_0"]

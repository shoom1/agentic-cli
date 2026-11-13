"""A delegation's result on ADK, shown as it is on Deep Agents.

ADK names a sub-agent's delegation tool after the sub-agent and hands its
parent the sub-agent's answer under ``result`` (``{"result": text}``), which
the generic rendering showed as "Result: 1 items". The event processor knows
the call is a delegation (it ends the run's ``AGENT_RUN`` there), so it shows
the answer's first line (``tool_summaries.delegation_summary``), and a
sub-agent that failed as a failed result; ``metadata["result"]`` stays ADK's.
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.adk")

from google.adk.events import Event  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_cli.workflow.adk.event_processor import ADKEventProcessor  # noqa: E402
from agentic_cli.workflow.adk.interrupted_calls_plugin import ABORTED  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.interrupts import INTERRUPTED_BY_USER  # noqa: E402


def _event(author: str, part: types.Part, *, error_code: str | None = None) -> Event:
    role = "user" if part.function_response is not None else "model"
    return Event(
        author=author, invocation_id="i1", error_code=error_code,
        content=types.Content(role=role, parts=[part]),
    )


def _call(name: str, call_id: str, args: dict) -> types.Part:
    return types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args))


def _response(name: str, call_id: str, response: dict) -> types.Part:
    return types.Part(
        function_response=types.FunctionResponse(id=call_id, name=name, response=response)
    )


async def _result_of(response: dict, *, error_code: str | None = None, name: str = "researcher"):
    """The ``TOOL_RESULT`` the coordinator's call ``c1`` of ``name`` comes back with."""
    processor = ADKEventProcessor(model="m", root="coordinator", delegates={"researcher"})
    call = _event("coordinator", _call(name, "c1", {"request": "Find X."}))
    [e async for e in processor.process_event(call, "s1")]
    answered = _event("coordinator", _response(name, "c1", response), error_code=error_code)
    events = [e async for e in processor.process_event(answered, "s1")]
    return next(e for e in events if e.type is EventType.TOOL_RESULT)


class TestADelegationsResult:
    async def test_shows_the_answers_first_line(self):
        result = await _result_of({"result": "X is 42.\nThe handbook says so."})

        assert result.content == "X is 42."
        assert result.metadata["success"] is True
        # What the parent read stays as ADK gave it.
        assert result.metadata["result"] == {"result": "X is 42.\nThe handbook says so."}

    async def test_an_empty_answer_is_no_answer(self):
        result = await _result_of({"result": ""})

        assert result.content == "No answer"

    async def test_a_sub_agent_that_failed_is_a_failed_result(self):
        result = await _result_of({"result": "Error running sub-agent: 529 overloaded"})

        assert result.content == "Failed: 529 overloaded"
        assert result.metadata["success"] is False
        assert result.metadata["error"] == "Error running sub-agent: 529 overloaded"

    async def test_a_stopped_turns_seal_keeps_its_rendering(self):
        result = await _result_of(
            {"success": False, "error": INTERRUPTED_BY_USER}, error_code=ABORTED
        )

        assert result.content == f"Failed: {INTERRUPTED_BY_USER}"

    async def test_another_tools_result_is_rendered_as_before(self):
        result = await _result_of({"result": "X is 42."}, name="lookup")

        assert result.content == "Result: 1 items"

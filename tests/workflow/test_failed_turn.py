"""A turn that fails yields one ``ERROR`` event and ends, on both backends.

The contract is the base ``process()``'s: the turn's one ``ERROR`` is not
recoverable and carries the error's text (``content``), whether a provider
rate-limited it (``metadata["rate_limited"]``, ``retry.is_rate_limit_error``)
and the exception's type (``metadata["error_type"]``); ``process()`` then
returns. A cancel still raises ``CancelledError``, and an ``ERROR`` the turn
already yielded (the overflow handler's ``CONTEXT_OVERFLOW``) is its one
``ERROR``. The same failures run on ADK and Deep Agents, through Anthropic's
API faked offline (``FakeAPI``), the provider clients retrying nothing.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import anthropic
import httpx
import pytest

pytest.importorskip("google.adk")
pytest.importorskip("deepagents")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.tools.registry import register_tool  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.context import OVERFLOW_ERROR_CODE, overflow_message  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.events import EventType, WorkflowEvent  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI  # noqa: E402
from tests.workflow.langchain_fakes import anthropic_reply, route  # noqa: E402
from tests.workflow.test_manager_models import route as route_adk  # noqa: E402

CLAUDE = "anthropic:claude-opus-4-7"
BACKENDS = ["adk", "deepagents"]

#: The type each backend's Claude adapter raises: ADK's is Anthropic's SDK
#: (its rate limit a subclass of ADK's own, worded with ADK's advice first),
#: Deep Agents' is langchain-anthropic's.
RAISED = {
    "adk": {400: "BadRequestError", 429: "_AnthropicRateLimitError", 529: "OverloadedError"},
    "deepagents": {
        400: "AnthropicInvalidRequestError",
        429: "AnthropicRateLimitError",
        529: "AnthropicOverloadedError",
    },
}


@register_tool(capabilities=EXEMPT, description="Answers at once (a test tool).")
async def ft_lookup(topic: str) -> dict:
    """Look a topic up."""
    return {"success": True, "fact": f"{topic} is 42"}


# --- the fake API -----------------------------------------------------------------------


class Script:
    """Anthropic's API answering each request with the next of ``answers``;
    the last one answers every request after it. ``"hold"`` keeps the
    request open until the turn is cancelled."""

    def __init__(self, *answers) -> None:
        self.answers = list(answers)
        self.held = asyncio.Event()
        self.api = FakeAPI(self._answer)

    async def _answer(self, request):
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if answer == "hold":
            self.held.set()
            await asyncio.Event().wait()
        return answer(request) if callable(answer) else answer


def say(text: str):
    return anthropic_reply({"type": "text", "text": text}, model="claude-opus-4-7")


def calls(name: str, args: dict, call_id: str):
    return anthropic_reply(
        {"type": "tool_use", "id": call_id, "name": name, "input": args},
        model="claude-opus-4-7",
    )


def refused(status: int, kind: str, message: str) -> httpx.Response:
    return httpx.Response(
        status, json={"type": "error", "error": {"type": kind, "message": message}}
    )


BAD_REQUEST = refused(400, "invalid_request_error", "max_tokens: 100000 > 64000")
RATE_LIMITED = refused(
    429, "rate_limit_error", "Number of request tokens has exceeded your rate limit."
)
OVERFLOW = refused(
    400, "invalid_request_error", "prompt is too long: 208310 tokens > 200000 maximum"
)
UNAVAILABLE = refused(529, "overloaded_error", "Overloaded")


# --- managers ---------------------------------------------------------------------------


@pytest.fixture
def hermetic(tmp_path, monkeypatch) -> Path:
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.chdir(tmp_path / "project")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
    return tmp_path


def serve(monkeypatch, script: Script, backend: str) -> Script:
    """ADK builds ``anthropic.AsyncAnthropic``; langchain-anthropic ``AsyncClient``."""
    if backend == "adk":
        route_adk(monkeypatch, anthropic, "AsyncAnthropic", script.api)
    else:
        route(monkeypatch, anthropic, "AsyncClient", script.api)
    return script


def solo(backend: str, *tools):
    """A main agent alone, on Claude, its client retrying nothing."""
    configs = [AgentConfig(
        name="solo", prompt="Answer.", tools=list(tools), include_state_tools=False,
    )]
    settings = BaseSettings(
        default_model=CLAUDE, session_store="memory", retry_max_attempts=0,
        prompt_caching_enabled=False,
    )
    if backend == "adk":
        return GoogleADKWorkflowManager(agent_configs=configs, settings=settings)
    return DeepAgentsWorkflowManager(agent_configs=configs, settings=settings)


async def turn_on(monkeypatch, backend: str, *answers, tools=()) -> list[WorkflowEvent]:
    """One turn of ``solo`` on ``answers``; what ``process()`` yields, raising nothing."""
    serve(monkeypatch, Script(*answers), backend)
    manager = solo(backend, *tools)
    try:
        return [event async for event in manager.process("Hello.", "u", session_id="s")]
    finally:
        await manager.cleanup()


def errors(events: list[WorkflowEvent]) -> list[WorkflowEvent]:
    return [event for event in events if event.type is EventType.ERROR]


def anthropic_error(response: httpx.Response) -> str:
    """What the Anthropic SDK says of ``response``, which ends the error's text."""
    return f"Error code: {response.status_code} - {json.loads(response.content)}"


def assert_failed(error: WorkflowEvent, backend: str, response: httpx.Response) -> None:
    """``error`` is the turn's one error for ``response``, as its backend raised it."""
    raised = RAISED[backend][response.status_code]
    assert error.content.endswith(anthropic_error(response))
    assert error.metadata["recoverable"] is False
    assert error.metadata["error_type"] == raised
    assert error.metadata["error_code"] == raised
    assert error.metadata["rate_limited"] is (response.status_code == 429)


# --- a turn that fails ------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_provider_error_is_one_error_and_the_turn_ends(hermetic, monkeypatch, backend):
    events = await turn_on(monkeypatch, backend, BAD_REQUEST)

    assert [event.type for event in events] == [EventType.ERROR]
    assert_failed(events[0], backend, BAD_REQUEST)


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_rate_limit_is_one_error_that_says_so(hermetic, monkeypatch, backend):
    events = await turn_on(monkeypatch, backend, RATE_LIMITED)

    (error,) = errors(events)
    assert_failed(error, backend, RATE_LIMITED)


@pytest.mark.parametrize("backend", BACKENDS)
async def test_an_overflow_nothing_recovers_is_one_clear_error(hermetic, monkeypatch, backend):
    events = await turn_on(monkeypatch, backend, OVERFLOW)

    (error,) = errors(events)
    assert error.content == overflow_message(CLAUDE)
    assert error.metadata["recoverable"] is False
    assert error.metadata["error_code"] == OVERFLOW_ERROR_CODE
    assert error.metadata.get("rate_limited", False) is False
    assert [event for event in events if event.type is EventType.TEXT] == []


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_model_error_after_a_tool_ran_is_one_error_last(hermetic, monkeypatch, backend):
    """The tool's call and result come first, then the turn's one error."""
    events = await turn_on(
        monkeypatch, backend, calls("ft_lookup", {"topic": "X"}, "toolu_1"), UNAVAILABLE,
        tools=[ft_lookup],
    )

    kinds = [event.type for event in events if event.type is not EventType.LLM_USAGE]
    assert kinds == [EventType.TOOL_CALL, EventType.TOOL_RESULT, EventType.ERROR]
    (error,) = errors(events)
    assert_failed(error, backend, UNAVAILABLE)


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_cancel_still_raises_and_says_nothing(hermetic, monkeypatch, backend):
    script = serve(monkeypatch, Script("hold"), backend)
    manager = solo(backend)
    seen: list[WorkflowEvent] = []

    async def consume() -> None:
        async for event in manager.process("Hello.", "u", session_id="s"):
            seen.append(event)

    try:
        running = asyncio.create_task(consume())
        await asyncio.wait_for(script.held.wait(), timeout=5)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    finally:
        await manager.cleanup()

    assert errors(seen) == []


@pytest.mark.parametrize("backend", BACKENDS)
async def test_the_next_turn_runs_after_a_failed_one(hermetic, monkeypatch, backend):
    serve(monkeypatch, Script(BAD_REQUEST, say("Hello back.")), backend)
    manager = solo(backend)
    try:
        first = [e async for e in manager.process("Hello.", "u", session_id="s")]
        second = [e async for e in manager.process("Again.", "u", session_id="s")]
    finally:
        await manager.cleanup()

    assert len(errors(first)) == 1
    assert errors(second) == []
    assert [e.content for e in second if e.type is EventType.TEXT] == ["Hello back."]


# --- ADK: an error the turn already reported is its one error ------------------------------


async def test_on_adk_an_error_already_yielded_is_not_followed_by_a_second(
    hermetic, monkeypatch
):
    """The overflow handler's answer is the turn's error: a failure after it
    (here the run raising) is logged, not reported again."""
    from google.adk.events import Event

    serve(monkeypatch, Script(say("Unused.")), "adk")
    manager = solo("adk")
    try:
        await manager.initialize_services()

        async def run_async(**kwargs):
            yield Event(
                author="solo", invocation_id="i1", error_code=OVERFLOW_ERROR_CODE,
                error_message=overflow_message(CLAUDE),
            )
            raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))

        monkeypatch.setattr(manager._runner, "run_async", run_async)
        events = [e async for e in manager.process("Hello.", "u", session_id="s")]
    finally:
        await manager.cleanup()

    (error,) = errors(events)
    assert error.metadata["error_code"] == OVERFLOW_ERROR_CODE

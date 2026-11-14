"""Every ADK model call timed (``workflow/adk/call_clock.py``).

``CallSentPlugin``, last in the chain, starts a call's clock as its request
goes out; ``CallLatencyPlugin``, at the head, marks the answer with the
milliseconds (``custom_metadata[LATENCY_KEY]``), which ADK copies onto the
answer's event and the event processor gives the call's ``LLM_USAGE``. A
call is told apart by its ``EventActions``; an error keeps its clock for the
answer a later plugin gives it; an answer already timed (the overflow
handler's retry) keeps its mark. No model is sent the mark.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import anthropic
import pytest

pytest.importorskip("google.adk")

from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.adk import call_clock  # noqa: E402
from agentic_cli.workflow.adk.call_clock import (  # noqa: E402
    LATENCY_KEY,
    CallClock,
    CallLatencyPlugin,
    CallSentPlugin,
    call_latency,
    mark_latency,
)
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message  # noqa: E402
from tests.workflow.test_manager_models import route  # noqa: E402


def _call(invocation: str = "inv-1") -> SimpleNamespace:
    """A model call's callback context: its own ``EventActions``."""
    return SimpleNamespace(invocation_id=invocation, agent_name="solo", actions=object())


def _answer(*, partial: bool = False) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part.from_text(text="Hi.")]),
        partial=partial,
    )


@pytest.fixture
def ticks(monkeypatch) -> list[float]:
    """``time.perf_counter`` as the clock reads it: the times given, in turn."""
    times: list[float] = []
    monkeypatch.setattr(call_clock, "time", SimpleNamespace(perf_counter=lambda: times.pop(0)))
    return times


@pytest.fixture
def plugins() -> tuple[CallSentPlugin, CallLatencyPlugin, CallClock]:
    clock = CallClock()
    return CallSentPlugin(clock), CallLatencyPlugin(clock), clock


class TestTheClock:
    async def test_an_answer_is_marked_with_how_long_its_call_took(self, plugins, ticks):
        sent, latency, clock = plugins
        call, answer = _call(), _answer()
        ticks += [10.0, 10.5]

        await sent.before_model_callback(callback_context=call, llm_request=None)
        returned = await latency.after_model_callback(callback_context=call, llm_response=answer)

        assert returned is None  # in place: every plugin after it still runs
        assert answer.custom_metadata == {LATENCY_KEY: 500.0}
        assert len(clock) == 0

    async def test_a_partial_answer_is_not_marked_and_keeps_its_clock(self, plugins, ticks):
        sent, latency, clock = plugins
        call, partial, final = _call(), _answer(partial=True), _answer()
        ticks += [10.0, 10.25]

        await sent.before_model_callback(callback_context=call, llm_request=None)
        await latency.after_model_callback(callback_context=call, llm_response=partial)
        await latency.after_model_callback(callback_context=call, llm_response=final)

        assert partial.custom_metadata is None
        assert final.custom_metadata == {LATENCY_KEY: 250.0}

    async def test_an_errors_clock_waits_for_the_answer_a_later_plugin_gives(
        self, plugins, ticks
    ):
        sent, latency, clock = plugins
        call, answer = _call(), _answer()
        ticks += [10.0, 12.0]

        await sent.before_model_callback(callback_context=call, llm_request=None)
        await latency.on_model_error_callback(
            callback_context=call, llm_request=None, error=RuntimeError("prompt is too long")
        )
        assert len(clock) == 1
        await latency.after_model_callback(callback_context=call, llm_response=answer)

        assert answer.custom_metadata == {LATENCY_KEY: 2_000.0}

    async def test_an_answer_already_timed_keeps_its_mark(self, plugins, ticks):
        # The overflow handler's retry, timed around its own request.
        sent, latency, clock = plugins
        call, answer = _call(), _answer()
        ticks += [10.0, 12.0]
        await sent.before_model_callback(callback_context=call, llm_request=None)
        mark_latency(answer, 300.0)

        await latency.after_model_callback(callback_context=call, llm_response=answer)

        assert answer.custom_metadata == {LATENCY_KEY: 300.0}
        assert len(clock) == 0

    async def test_parallel_calls_each_time_their_own(self, plugins, ticks):
        # Two delegations to one sub-agent share the invocation and the agent.
        sent, latency, _ = plugins
        first, second = _call(), _call()
        first_answer, second_answer = _answer(), _answer()
        ticks += [0.0, 1.0, 3.0, 4.0]

        await sent.before_model_callback(callback_context=first, llm_request=None)
        await sent.before_model_callback(callback_context=second, llm_request=None)
        await latency.after_model_callback(callback_context=second, llm_response=second_answer)
        await latency.after_model_callback(callback_context=first, llm_response=first_answer)

        assert second_answer.custom_metadata == {LATENCY_KEY: 2_000.0}
        assert first_answer.custom_metadata == {LATENCY_KEY: 4_000.0}

    async def test_a_run_lets_go_of_what_another_left(self, plugins, ticks):
        sent, latency, clock = plugins
        ticks += [10.0]
        await sent.before_model_callback(callback_context=_call("inv-1"), llm_request=None)

        await latency.before_run_callback(
            invocation_context=SimpleNamespace(invocation_id="inv-2")
        )

        assert len(clock) == 0

    def test_an_events_mark_is_read_as_milliseconds(self):
        assert call_latency(SimpleNamespace(custom_metadata={LATENCY_KEY: 12})) == 12.0
        assert call_latency(SimpleNamespace(custom_metadata={LATENCY_KEY: True})) is None
        assert call_latency(SimpleNamespace(custom_metadata={LATENCY_KEY: "12"})) is None
        assert call_latency(SimpleNamespace(custom_metadata=None)) is None


async def test_no_model_is_sent_the_mark(tmp_path, monkeypatch):
    """The mark stays on the session's event: no adapter sends
    ``custom_metadata``, so the next request holds the answer alone."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
    api = FakeAPI(anthropic_message({"type": "text", "text": "Hi."}))
    route(monkeypatch, anthropic, "AsyncAnthropic", api)
    manager = GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt="Answer.", include_state_tools=False)],
        settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6"),
    )
    try:
        for text in ("One.", "Two."):
            [e async for e in manager.process(text, "u", session_id="s")]
        rows = manager.usage_of(manager.session_ref("s", "u")).rows
    finally:
        await manager.cleanup()

    assert all(isinstance(row.latency_ms, float) for row in rows) and len(rows) == 2
    sent = json.dumps(api.bodies[-1])
    assert "Hi." in sent and "agentic_cli" not in sent and "latency" not in sent

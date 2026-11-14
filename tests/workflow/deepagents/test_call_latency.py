"""Every Deep Agents model call timed: its usage row says how long it took.

The turn's ``RunCallback`` sees every model call LangChain makes in the run,
with or without the raw log: it times each from its start to its end and
marks the answer (``response_metadata["agentic_cli"]["latency_ms"]``, beside
any other mark there, which it never replaces), so the event processor gives
the step's ``LLM_USAGE`` its ``latency_ms``; a summary's usage, which the
callback makes itself, carries it too. ``SubAgentOutcomeMiddleware``'s
outcome mark keeps it. No provider is sent the mark.
"""

from __future__ import annotations

import json
import uuid

import anthropic
import openai
import pytest

pytest.importorskip("deepagents")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, LLMResult  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents import event_processor  # noqa: E402
from agentic_cli.workflow.deepagents.event_processor import RunCallback  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.deepagents.middleware import (  # noqa: E402
    answer_latency,
    answer_outcome,
    marked_answer,
)
from agentic_cli.workflow.live_events import LiveEvents  # noqa: E402
from agentic_cli.workflow.usage import UsageLedgers  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message, openai_response  # noqa: E402
from tests.workflow.deepagents.test_event_processor import (  # noqa: E402
    MODELS,
    _model_update,
    _processor,
    _reported,
)
from tests.workflow.langchain_fakes import gemini_reply, route, route_google  # noqa: E402


@pytest.fixture
def ticks(monkeypatch) -> list[float]:
    """``time.perf_counter`` as the callback reads it: the times given, in turn."""
    times: list[float] = []

    class Clock:
        @staticmethod
        def perf_counter() -> float:
            return times.pop(0)

    monkeypatch.setattr(event_processor, "time", Clock)
    return times


async def _call(callback: RunCallback, answer: AIMessage, **metadata) -> None:
    """One model call through ``callback``, answered with ``answer``."""
    run = uuid.uuid4()
    await callback.on_chat_model_start(
        {}, [[HumanMessage("Hi.")]], run_id=run, metadata=metadata
    )
    await callback.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=answer)]]), run_id=run
    )


def _callback(live: LiveEvents | None = None, **kwargs) -> RunCallback:
    return RunCallback(
        root="coordinator", agent_models=MODELS, live=live or LiveEvents(), **kwargs
    )


class TestTheRunCallback:
    async def test_every_answer_is_marked_with_how_long_its_call_took(self, ticks):
        # No raw log, no summary: an agent's ordinary step.
        answer = AIMessage("Hi.", response_metadata={"stop_reason": "end_turn"})
        ticks += [10.0, 10.5]

        await _call(_callback(), answer, lc_agent_name="coordinator")

        assert answer_latency(answer) == 500.0
        assert answer.response_metadata["stop_reason"] == "end_turn"

    async def test_another_mark_stays(self, ticks):
        answer = AIMessage("Hi.", response_metadata={"agentic_cli": {"kind": "job_result"}})
        ticks += [10.0, 10.25]

        await _call(_callback(), answer)

        assert answer.response_metadata["agentic_cli"] == {
            "kind": "job_result", "latency_ms": 250.0,
        }

    async def test_a_summarys_usage_carries_it(self, ticks):
        ledgers = UsageLedgers()
        live = LiveEvents()
        summary = AIMessage("A summary.", usage_metadata={
            "input_tokens": 900, "output_tokens": 40, "total_tokens": 940,
        })
        ticks += [10.0, 11.0]

        reported = await _reported(live, lambda: _call(
            _callback(live, ledgers=ledgers), summary,
            lc_source="summarization", lc_agent_name="coordinator",
        ))

        (row,) = ledgers.ledger(None).rows
        assert (row.purpose, row.latency_ms) == ("summarization", 1_000.0)
        (usage,) = [e for e in reported if e.metadata.get("purpose") == "summarization"]
        assert usage.metadata["latency_ms"] == 1_000.0


class TestTheProcessor:
    def test_a_steps_row_has_its_calls_latency(self):
        ledgers = UsageLedgers()
        answer = AIMessage(
            "Hi.", usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
            response_metadata={"agentic_cli": {"latency_ms": 12.5}},
        )

        (usage,) = _processor(ledgers=ledgers).process((), "updates", _model_update(answer))[-1:]

        assert usage.metadata["latency_ms"] == 12.5
        assert ledgers.ledger(None).rows[0].latency_ms == 12.5


class TestTheOutcomeMark:
    def test_a_sub_agents_rewritten_answer_keeps_its_latency(self):
        answer = AIMessage("X is", response_metadata={
            "stop_reason": "max_tokens", "agentic_cli": {"latency_ms": 12.5},
        })

        marked = marked_answer(answer)

        assert answer_outcome(marked) == "cut"
        assert answer_latency(marked) == 12.5


@pytest.mark.parametrize("provider", ["claude", "gemini", "openai"])
async def test_no_provider_is_sent_the_mark(tmp_path, monkeypatch, provider):
    """The first answer, marked, goes back in the second request as the
    message it is: the mark stays in the checkpoint."""
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    if provider == "claude":
        api = FakeAPI(anthropic_message({"type": "text", "text": "Hi."},
                                        model="claude-opus-4-7"))
        route(monkeypatch, anthropic, "AsyncClient", api)
        model = "anthropic:claude-opus-4-7"
    elif provider == "gemini":
        api = FakeAPI(gemini_reply({"text": "Hi."}))
        route_google(monkeypatch, api)
        model = "google:gemini-2.5-flash"
    else:
        api = FakeAPI(openai_response("Hi."))
        route(monkeypatch, openai, "AsyncOpenAI", api)
        model = "openai:gpt-5.5"
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    for variable in ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(variable, "a-test-key")
    manager = DeepAgentsWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt="Answer.", include_state_tools=False)],
        settings=BaseSettings(default_model=model, session_store="memory"),
    )
    try:
        for text in ("One.", "Two."):
            [e async for e in manager.process(text, "u", session_id="s")]
        rows = manager.usage_of(manager.session_ref("s", "u")).rows
    finally:
        await manager.cleanup()

    assert len(rows) == 2 and all(isinstance(row.latency_ms, float) for row in rows)
    sent = json.dumps(api.bodies[-1])
    assert "Hi." in sent and "agentic_cli" not in sent and "latency" not in sent

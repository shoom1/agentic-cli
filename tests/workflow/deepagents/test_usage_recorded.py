"""Deep Agents records each call's usage in its conversation's ledger as it reports it.

The event processor records an agent's step as it makes its ``LLM_USAGE``,
before ``on_event`` (an app's hook that drops events cannot drop a row), and
a sub-agent run's end carries what the run used, by model; the run callback
records a summary's. A processor driven outside a manager files under no
conversation (None), as ``AgentRuns`` does.
"""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("deepagents")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, LLMResult  # noqa: E402

from agentic_cli.workflow.deepagents.event_processor import RunCallback  # noqa: E402
from agentic_cli.workflow.live_events import LiveEvents  # noqa: E402
from agentic_cli.workflow.usage import UsageLedgers  # noqa: E402
from tests.workflow.deepagents.test_event_processor import (  # noqa: E402
    MODELS,
    SUB,
    _call,
    _delegated,
    _model_update,
    _processor,
    _reported,
    _tools_update,
)


def _answer(text: str, tokens_in: int, tokens_out: int, **details) -> AIMessage:
    return AIMessage(text, usage_metadata={
        "input_tokens": tokens_in, "output_tokens": tokens_out,
        "total_tokens": tokens_in + tokens_out, **details,
    })


class TestTheProcessor:
    def test_an_agents_step_is_recorded(self):
        ledgers = UsageLedgers()

        _processor(ledgers=ledgers).process((), "updates", _model_update(
            _answer("Hi.", 100, 10, input_token_details={"cache_read": 60})
        ))

        (row,) = ledgers.ledger(None).rows
        assert (row.agent, row.run, row.purpose, row.input, row.output, row.cached) == (
            "coordinator", None, None, 100, 10, 60
        )

    def test_a_runs_end_carries_what_the_run_used(self):
        ledgers = UsageLedgers()
        processor = _processor(ledgers=ledgers)
        processor.process((), "updates", _model_update(AIMessage("", tool_calls=[
            _call("task", {"description": "Find X.", "subagent_type": "researcher"}, "c-" + SUB),
        ])))
        _delegated(processor, (), SUB, "researcher")
        processor.process((SUB,), "updates", _model_update(_answer("X is 42.", 50, 5)))

        _, ended = processor.process((), "updates", _tools_update(
            ToolMessage("X is 42.", tool_call_id="c-" + SUB, name="task")
        ))

        assert ended.metadata["usage"] == {"google:gemini-2.5-flash": {"input": 50, "output": 5}}
        assert ledgers.ledger(None).totals(run="researcher-1#1").calls == 1


class TestTheRunCallback:
    async def test_a_summarys_usage_is_recorded(self):
        ledgers = UsageLedgers()
        live = LiveEvents()
        callback = RunCallback(root="coordinator", agent_models=MODELS, live=live, ledgers=ledgers)
        run = uuid.uuid4()

        async def act():
            await callback.on_chat_model_start(
                {}, [[HumanMessage("Summarize.")]], run_id=run,
                metadata={"lc_source": "summarization", "lc_agent_name": "researcher"},
            )
            await callback.on_llm_end(
                LLMResult(generations=[[ChatGeneration(message=_answer("A summary.", 900, 40))]]),
                run_id=run,
            )

        await _reported(live, act)

        (row,) = ledgers.ledger(None).rows
        assert (row.purpose, row.agent, row.input, row.output) == ("summarization", "researcher", 900, 40)

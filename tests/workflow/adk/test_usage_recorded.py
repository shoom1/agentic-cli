"""ADK records each call's usage in its conversation's ledger as it reports it.

The event processor normalizes a call's counts by its model's provider kind
(the thinking counts in the output: apart from the candidates on Gemini and
ADK's Claude adapter, inside them on OpenAI's), records it before
``on_event`` (an app's hook that drops events cannot drop a row), and gives
a sub-agent run's end what the run used, by model. A processor driven outside
a manager files under no conversation (None), as ``AgentRuns`` does.
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.adk")

from google.adk.events import Event  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_cli.workflow.usage import UsageLedgers  # noqa: E402
from tests.workflow.adk.test_run_identity import (  # noqa: E402
    _delegated,
    _process,
    _processor,
    _results,
)


def _answer(author: str, prompt: int, candidates: int, thoughts: int | None = None,
            *, branch: str | None = None) -> Event:
    """A model's answer of ``author``'s that reports its usage."""
    return Event(
        author=author, invocation_id="i1", branch=branch,
        content=types.Content(role="model", parts=[types.Part.from_text(text="Done.")]),
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt, candidates_token_count=candidates,
            thoughts_token_count=thoughts,
        ),
    )


class TestTheProcessor:
    async def test_a_steps_usage_is_recorded_with_its_thinking_in_the_output(self):
        ledgers = UsageLedgers()
        processor = _processor(ledgers=ledgers, kinds={"coordinator": "google"})

        (usage, _) = await _process(processor, _answer("coordinator", 900, 5, 25))

        assert (usage.metadata["completion_tokens"], usage.metadata["thinking_tokens"]) == (30, 25)
        (row,) = ledgers.ledger(None).rows
        assert (row.agent, row.run, row.input, row.output, row.thinking) == (
            "coordinator", None, 900, 30, 25
        )

    async def test_an_openai_models_reasoning_is_counted_once(self):
        ledgers = UsageLedgers()
        processor = _processor(ledgers=ledgers, kinds={"coordinator": "openai"})

        await _process(processor, _answer("coordinator", 100, 70, 64))

        assert ledgers.ledger(None).rows[0].output == 70

    async def test_a_hook_that_drops_the_event_drops_no_row(self):
        ledgers = UsageLedgers()
        processor = _processor(ledgers=ledgers, on_event=lambda event: None)

        assert await _process(processor, _answer("coordinator", 100, 10)) == []
        assert ledgers.ledger(None).totals().calls == 1

    async def test_a_runs_end_carries_what_the_run_used(self):
        ledgers = UsageLedgers()
        processor = _processor(ledgers=ledgers)
        await _delegated(processor)
        await _process(processor, _answer("researcher", 50, 5, branch="researcher@toolu_c1"))

        (_, ended) = await _process(processor, _results(
            "coordinator", ("researcher", "toolu_c1", {"result": "X is 42."})
        ))

        assert ended.metadata["usage"] == {"m": {"input": 50, "output": 5}}


class TestTheManagersProcessor:
    async def test_an_agent_whose_model_is_unknown_counts_its_thinking_apart(
        self, tmp_path, monkeypatch
    ):
        # The default model is OpenAI's, whose adapters count the reasoning in
        # the completion. An agent the manager resolved no model for (a native
        # ADK YAML app's that ADK runs on its own model) may run on any: its
        # usage is read as an unknown kind's, the thinking apart.
        from agentic_cli.config import BaseSettings
        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager
        from agentic_cli.workflow.config import AgentConfig

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("OPENAI_API_KEY", "o-test-key")
        manager = GoogleADKWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt="Answer.", include_state_tools=False)],
            settings=BaseSettings(session_store="memory", default_model="openai:gpt-5"),
        )
        await manager.initialize_services()
        try:
            (usage, _) = await _process(
                manager._event_processor, _answer("stranger", 900, 5, 25)
            )
        finally:
            await manager.cleanup()

        assert (usage.metadata["completion_tokens"], usage.metadata["thinking_tokens"]) == (30, 25)

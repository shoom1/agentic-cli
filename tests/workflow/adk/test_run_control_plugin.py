"""ADK: a sub-agent run's limits at its model calls (``workflow/adk/run_control_plugin.py``).

What the plugin does to a request and an answer, alone. Whole turns, on both
backends, are in the contract suite (``TestTheStepLimit``,
``TestTheTokenLimit``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.adk")

from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_cli.workflow.adk.run_control_plugin import (  # noqa: E402
    RunControlPlugin,
    drop_calls,
    with_note,
)
from agentic_cli.workflow.agents.control import STEP_LIMIT_NOTE, RunControl  # noqa: E402
from agentic_cli.workflow.agents.envelope import TOKEN_LIMIT_ERROR  # noqa: E402
from agentic_cli.workflow.agents.policy import AgentLimits  # noqa: E402
from agentic_cli.workflow.sessions import RunRef  # noqa: E402

REF = RunRef(
    agent="researcher", instance="researcher-1", run="researcher-1#1", parent_run=None, depth=1
)


def user(*parts: types.Part) -> types.Content:
    return types.Content(role="user", parts=list(parts))


def model(*parts: types.Part) -> types.Content:
    return types.Content(role="model", parts=list(parts))


def text(value: str, **fields) -> types.Part:
    return types.Part(text=value, **fields)


def call(name: str) -> types.Part:
    return types.Part.from_function_call(name=name, args={})


def answered(name: str) -> types.Part:
    return types.Part.from_function_response(name=name, response={"success": True})


class TestTheNote:
    def test_it_is_a_part_of_a_last_user_content(self):
        result = user(answered("lookup"))
        contents = [user(text("# Task: Find X.")), model(call("lookup")), result]

        noted = with_note(contents, STEP_LIMIT_NOTE)

        assert noted[:2] == contents[:2]
        assert noted[2].parts[0] == answered("lookup")
        assert noted[2].parts[1].text == STEP_LIMIT_NOTE
        assert result.parts == [answered("lookup")]  # not changed in place

    def test_after_a_models_content_it_is_a_user_content_of_its_own(self):
        contents = [user(text("# Task: Find X.")), model(text("Hm."))]

        assert with_note(contents, STEP_LIMIT_NOTE)[-1] == user(text(STEP_LIMIT_NOTE))


class TestDroppedCalls:
    def test_the_text_and_thoughts_stay(self):
        answer = LlmResponse(content=model(text("why", thought=True), text("X is 42."),
                                           call("lookup")))

        drop_calls(answer)

        assert answer.content == model(text("why", thought=True), text("X is 42."))

    def test_calls_alone_leave_an_empty_text(self):
        answer = LlmResponse(content=model(text("why", thought=True), call("lookup")))

        drop_calls(answer)

        assert answer.content == model(text("why", thought=True), text(""))

    def test_an_answer_without_calls_is_left_as_it_is(self):
        content = model(text("X is 42."))
        answer = LlmResponse(content=content)

        drop_calls(answer)

        assert answer.content is content


class TestThePlugin:
    async def test_the_roots_calls_are_left_alone(self):
        plugin = RunControlPlugin(lambda: None, "anthropic")
        request = LlmRequest(contents=[user(text("Go."))])
        answer = LlmResponse(content=model(call("lookup")))

        assert await plugin.before_model_callback(callback_context=None, llm_request=request) is None
        assert await plugin.after_model_callback(callback_context=None, llm_response=answer) is None

        assert request.contents == [user(text("Go."))]
        assert answer.content == model(call("lookup"))

    async def test_a_call_past_the_token_limit_is_the_runs_last(self):
        control = RunControl(REF, AgentLimits(max_tokens=100))
        plugin = RunControlPlugin(lambda: control, "anthropic")
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=90, candidates_token_count=20
        )
        answer = LlmResponse(content=model(text("Found it."), call("lookup")), usage_metadata=usage)

        await plugin.before_model_callback(callback_context=None, llm_request=LlmRequest())
        await plugin.after_model_callback(callback_context=None, llm_response=answer)

        assert control.cancelled == ("limit", TOKEN_LIMIT_ERROR)
        assert answer.content == model(text("Found it."))

    async def test_a_partial_answer_counts_nothing(self):
        control = RunControl(REF, AgentLimits(max_tokens=1))
        plugin = RunControlPlugin(lambda: control, "anthropic")
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=90, candidates_token_count=20
        )

        await plugin.after_model_callback(
            callback_context=None,
            llm_response=LlmResponse(content=model(text("Fo")), usage_metadata=usage, partial=True),
        )

        assert control.cancelled is None

"""ADK: a sub-agent's refused or cut answer, said so (``workflow/adk/subagent_outcome_plugin.py``).

ADK's adapters report how an answer stopped as its ``finish_reason`` (Claude's
``refusal`` is ``SAFETY``, its ``max_tokens`` ``MAX_TOKENS``; a Gemini block
also sets ``error_code`` on an answer with no content), and the OpenAI
Responses adapter turns a refusal into text. The plugin rewrites a sub-agent's
final answer of either kind and marks it; the root's own answers are left as
they are. The whole turn, on both backends, is in the contract suite.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("google.adk")

import anthropic  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.adk.subagent_outcome_plugin import (  # noqa: E402
    OPENAI_REFUSAL_PREFIX,
    OUTCOME_KEY,
    SubAgentOutcomePlugin,
    mark_outcome,
)
from agentic_cli.workflow.agents.outcome import CUT, REFUSED, refusal_notice  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.sessions import RunRef, reset_active_run, set_active_run  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message  # noqa: E402
from tests.workflow.test_adk_delegation import (  # noqa: E402,F401
    Team,
    calls,
    manager_for,
    project,
    says,
    team,
    turn,
)
from tests.workflow.test_manager_models import route  # noqa: E402

RESEARCHER = RunRef("researcher", "researcher-1", "researcher-1#1", None, 1)
FINISH = types.FinishReason


def answer(*parts: types.Part, finish=None, error_code=None, partial=False) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=list(parts)) if parts else None,
        finish_reason=finish, error_code=error_code, partial=partial,
    )


def texts(response: LlmResponse) -> list[str]:
    return [part.text for part in response.content.parts if part.text and not part.thought]


class TestMarkOutcome:
    def test_claudes_refusal_becomes_the_notice(self):
        response = answer(finish=FINISH.SAFETY)  # stop_reason "refusal", no content

        assert mark_outcome(response) == REFUSED
        assert texts(response) == [refusal_notice()]
        assert response.custom_metadata == {OUTCOME_KEY: REFUSED}

    def test_a_gemini_block_without_content_becomes_the_notice_and_no_error(self):
        response = answer(finish=FINISH.PROHIBITED_CONTENT, error_code="PROHIBITED_CONTENT")
        response.error_message = "blocked"

        assert mark_outcome(response) == REFUSED
        assert texts(response) == [refusal_notice()]
        assert (response.error_code, response.error_message) == (None, None)

    def test_the_openai_adapters_refusal_text_is_the_explanation(self):
        response = answer(types.Part(text=f"{OPENAI_REFUSAL_PREFIX}I can't help with that."),
                          finish=FINISH.STOP)

        assert mark_outcome(response) == REFUSED
        assert texts(response) == [refusal_notice(explanation="I can't help with that.")]

    def test_the_responses_adapters_content_filter_is_a_refusal(self):
        """ADK's OpenAI Responses adapter gives an incomplete answer's
        ``content_filter`` as ``OTHER``, the reason in its error message."""
        response = answer(types.Part(text="Par"), finish=FINISH.OTHER, error_code=FINISH.OTHER)
        response.error_message = '{"reason": "content_filter"}'

        assert mark_outcome(response) == REFUSED
        assert texts(response) == [refusal_notice()]
        assert (response.error_code, response.error_message) == (None, None)

    @pytest.mark.parametrize("message", [
        None, "Not JSON.", '{"reason": "unknown"}', '["content_filter"]',
        '{"type": "response.failed"}',
    ])
    def test_an_other_stop_without_a_reason_it_names_is_left_as_it_is(self, message):
        response = answer(finish=FINISH.OTHER, error_code=FINISH.OTHER)
        response.error_message = message

        assert mark_outcome(response) is None
        assert (response.error_code, response.error_message) == (FINISH.OTHER, message)

    def test_a_reason_in_an_error_message_counts_only_for_an_other_stop(self):
        response = answer(types.Part(text="X is 42."), finish=FINISH.STOP)
        response.error_message = '{"reason": "content_filter"}'

        assert mark_outcome(response) is None

    def test_a_cut_keeps_its_text_and_its_thoughts(self):
        thought = types.Part(text="Thinking.", thought=True)
        response = answer(thought, types.Part(text="X "), types.Part(text="is"),
                          finish=FINISH.MAX_TOKENS)

        assert mark_outcome(response) == CUT
        assert response.content.parts[0] is thought
        assert texts(response) == ["X is\n\n[cut off at the output limit]"]
        assert response.custom_metadata == {OUTCOME_KEY: CUT}

    def test_an_answer_that_calls_a_tool_is_not_final(self):
        call = types.Part(function_call=types.FunctionCall(name="lookup", args={}, id="r1"))
        response = answer(call, finish=FINISH.MAX_TOKENS)

        assert mark_outcome(response) is None
        assert response.content.parts == [call] and response.custom_metadata is None

    @pytest.mark.parametrize("finish", [FINISH.STOP, None, FINISH.MALFORMED_FUNCTION_CALL])
    def test_any_other_answer_is_left_as_it_is(self, finish):
        response = answer(types.Part(text="X is 42."), finish=finish)

        assert mark_outcome(response) is None
        assert texts(response) == ["X is 42."] and response.custom_metadata is None

    def test_the_provider_key_another_plugin_wrote_is_kept(self):
        response = answer(finish=FINISH.SAFETY)
        response.custom_metadata = {"agentic_cli_provider": "anthropic"}

        mark_outcome(response)

        assert response.custom_metadata == {
            "agentic_cli_provider": "anthropic", OUTCOME_KEY: REFUSED,
        }


class TestThePlugin:
    async def test_only_a_sub_agent_runs_answer_is_rewritten(self):
        plugin = SubAgentOutcomePlugin()
        root = answer(types.Part(text="X is"), finish=FINISH.MAX_TOKENS)
        sub = answer(types.Part(text="X is"), finish=FINISH.MAX_TOKENS)

        assert await plugin.after_model_callback(callback_context=None, llm_response=root) is None
        token = set_active_run(RESEARCHER)
        try:
            assert (
                await plugin.after_model_callback(callback_context=None, llm_response=sub)
            ) is None
        finally:
            reset_active_run(token)

        assert texts(root) == ["X is"]
        assert texts(sub) == ["X is\n\n[cut off at the output limit]"]

    async def test_a_streamed_fragment_is_never_judged(self):
        plugin = SubAgentOutcomePlugin()
        fragment = answer(types.Part(text="X is"), finish=FINISH.MAX_TOKENS, partial=True)

        token = set_active_run(RESEARCHER)
        try:
            await plugin.after_model_callback(callback_context=None, llm_response=fragment)
        finally:
            reset_active_run(token)

        assert texts(fragment) == ["X is"]


class TestTheRootsOwnAnswer:
    async def test_the_roots_refusal_is_still_the_turns_failure(self, project, monkeypatch):
        refused = anthropic_message()
        refused.update(content=[], stop_reason="refusal")
        answers = Team(coordinator=[refused])
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team())
        try:
            events = await turn(manager, "Find X.")
        finally:
            await manager.cleanup()

        (error,) = [e for e in events if e.type is EventType.ERROR]
        assert error.content.endswith("returned no answer")
        assert (error.metadata["recoverable"], error.metadata["error_type"]) == (False, "SAFETY")


class TestTheRawLog:
    """With ``raw_llm_logging``, the log keeps a sub-agent's answer as the model
    gave it, and its parent reads the rewritten one: the plugin comes after
    the raw log in the chain."""

    @pytest.mark.parametrize("stop, blocks, given, read", [
        ("refusal", [], "LLM Response: ", refusal_notice()),
        ("max_tokens", [{"type": "text", "text": "X is"}], "LLM Response: X is",
         "X is\n\n[cut off at the output limit]"),
    ])
    async def test_the_log_has_the_answer_as_the_model_gave_it(
        self, project, monkeypatch, stop, blocks, given, read
    ):
        answer = anthropic_message(*blocks)
        answer.update(content=blocks, stop_reason=stop)
        answers = Team(
            coordinator=[calls("researcher", "toolu_c1", request="Find X."), says("Done.")],
            researcher=[answer],
        )
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = GoogleADKWorkflowManager(
            agent_configs=team(),
            settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6",
                                  raw_llm_logging=True),
        )
        try:
            events = await turn(manager, "Find X.")
        finally:
            await manager.cleanup()

        (logged,) = [e for e in events if e.type is EventType.LLM_RESPONSE
                     and e.metadata.get("author") == "researcher"]
        assert logged.content == given
        assert logged.metadata["finish_reason"].endswith(
            "SAFETY" if stop == "refusal" else "MAX_TOKENS"
        )
        (result,) = [e for e in events if e.type is EventType.TOOL_RESULT
                     and e.metadata.get("tool_call_id") == "toolu_c1"]
        assert result.metadata["result"] == {"result": read}


class TestTheADKItRestsOn:
    """Private ADK the plugin reads: if one of these moves, it would miss a refusal."""

    def test_claudes_stop_reasons_map_to_these_finish_reasons(self):
        from google.adk.models.anthropic_llm import to_google_genai_finish_reason

        assert to_google_genai_finish_reason("refusal") == FINISH.SAFETY
        assert to_google_genai_finish_reason("max_tokens") == FINISH.MAX_TOKENS

    def test_the_openai_responses_adapter_words_a_refusal_so(self):
        from google.adk.integrations.openai import _openai_responses_llm

        assert _openai_responses_llm._REFUSAL_PREFIX == OPENAI_REFUSAL_PREFIX

    def test_the_responses_adapter_gives_a_content_filter_as_other_and_its_reason(self):
        from google.adk.integrations.openai import _openai_responses_llm

        response = _openai_responses_llm._response_to_llm_response({
            "id": "resp_1", "status": "incomplete", "model": "gpt-5", "output": [],
            "incomplete_details": {"reason": "content_filter"},
        })

        assert (response.finish_reason, response.error_code) == (FINISH.OTHER, FINISH.OTHER)
        assert json.loads(response.error_message) == {"reason": "content_filter"}

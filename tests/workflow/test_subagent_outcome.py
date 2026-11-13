"""How a sub-agent's last answer ended (``workflow/agents/outcome.py``).

Both backends' hooks rewrite a refused or cut answer with these, and both
event processors warn of a cut with ``cut_off_warning``, so the texts are the
same on ADK and Deep Agents (the contract suite checks the whole turn).
"""

from __future__ import annotations

import pytest
from google.genai import types

from agentic_cli.workflow.agents import RUN_COMPLETED, RUN_REFUSED
from agentic_cli.workflow.agents.outcome import (
    CUT,
    CUT_OFF_NOTE,
    REFUSED,
    cut_off,
    cut_off_warning,
    outcome_of,
    refusal_notice,
)
from agentic_cli.workflow.events import EventType, WorkflowEvent
from agentic_cli.workflow.sessions import RunRef

RESEARCHER = RunRef("researcher", "researcher-2", "researcher-2#1", None, 1)


class TestOutcomeOf:
    @pytest.mark.parametrize("reason", [
        "refusal",                              # Claude
        "content_filter",                       # OpenAI
        "SAFETY", "PROHIBITED_CONTENT", "RECITATION", "BLOCKLIST", "SPII",  # Gemini
        types.FinishReason.SAFETY,              # ADK's enum
        # Gemini's prompt blocks: ADK's error_code, langchain's prompt_feedback.
        types.BlockedReason.SAFETY, types.BlockedReason.PROHIBITED_CONTENT,
        types.BlockedReason.JAILBREAK, types.BlockedReason.MODEL_ARMOR,
    ])
    def test_a_refusal(self, reason):
        assert outcome_of(reason) == REFUSED

    @pytest.mark.parametrize("reason", [
        "max_tokens", "length", "max_output_tokens", "MAX_TOKENS", types.FinishReason.MAX_TOKENS,
    ])
    def test_a_cut(self, reason):
        assert outcome_of(reason) == CUT

    @pytest.mark.parametrize("reason", [
        None, "end_turn", "stop", "tool_use", "STOP", types.FinishReason.STOP,
        types.FinishReason.MALFORMED_FUNCTION_CALL, "",
        # An unnamed stop says nothing of why: ADK's OpenAI adapters read on.
        types.FinishReason.OTHER, types.BlockedReason.OTHER,
    ])
    def test_anything_else_is_none(self, reason):
        assert outcome_of(reason) is None


class TestTheTexts:
    def test_the_refusal_notice_is_deep_agents_0_7_34s(self):
        assert refusal_notice() == (
            "The subagent stopped because a model refused the request. Retrying the same "
            "request is unlikely to succeed; rephrase it or continue without it."
        )

    def test_the_notice_names_a_category_and_an_explanation_when_given(self):
        assert refusal_notice(category="cyber", explanation="I can't help with that.") == (
            "The subagent stopped because a model refused the request. Category: cyber. "
            'Explanation: "I can\'t help with that." Retrying the same request is unlikely '
            "to succeed; rephrase it or continue without it."
        )

    def test_a_cut_answer_keeps_its_text_then_the_note(self):
        assert cut_off("X is  \n") == "X is\n\n[cut off at the output limit]"
        assert cut_off("") == cut_off("   ") == CUT_OFF_NOTE == "[cut off at the output limit]"


class TestTheWarning:
    def test_a_partial_end_is_warned_once_as_the_runs(self):
        end = WorkflowEvent.agent_run(
            RESEARCHER, RUN_COMPLETED, call_id="c1", result="X is", partial=True
        )

        warning = cut_off_warning(end)

        assert warning.type is EventType.ERROR
        assert warning.content == "researcher-2's answer was cut off at the output limit"
        assert warning.metadata == {
            "recoverable": True, "error_code": "MAX_TOKENS", "agent": "researcher",
        }
        assert warning.run == RESEARCHER

    @pytest.mark.parametrize("end", [
        WorkflowEvent.agent_run(RESEARCHER, RUN_COMPLETED, call_id="c1", result="X is 42."),
        WorkflowEvent.agent_run(RESEARCHER, RUN_REFUSED, call_id="c1", result="..."),
        WorkflowEvent.text("X is"),
    ])
    def test_no_warning_for_anything_else(self, end):
        assert cut_off_warning(end) is None

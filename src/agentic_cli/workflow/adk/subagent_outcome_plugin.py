"""A sub-agent's refused or cut answer, said so to its parent (``workflow/agents/outcome.py``).

ADK hands a sub-agent's last answer to its parent as the delegation's result,
whatever stopped it: a refusal reached the parent as an empty result, and an
answer cut at the output limit as a complete one. ``SubAgentOutcomePlugin``
looks at each final answer of a sub-agent run (``get_active_run()`` is set:
``MainAgentPlugin`` makes the run active as the agent starts) and rewrites one
that a model refused (a refusal finish reason: Claude's, a Gemini block,
OpenAI's ``content_filter``; or the text ADK's OpenAI Responses adapter makes
of a refusal) into the refusal notice, and one cut at the output limit into
its text followed by the cut-off note. It marks the answer with its outcome
(``custom_metadata[OUTCOME_KEY]``), which the event carries, so the event
processor ends the run ``refused``, or ``completed`` and ``partial`` with one
warning.

It changes the response in place and returns None, as
``ForeignReasoningPlugin`` does, so every plugin after it still runs; it sits
after the raw log, which records the answer as the model gave it. An answer
that calls tools is not final and is left alone, as is every answer of the
root and of an agent the root transfers to (no active run): their refusal or
cut ends the turn, as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from agentic_cli.workflow.agents.outcome import (
    REFUSED,
    cut_off,
    outcome_of,
    refusal_notice,
)
from agentic_cli.workflow.sessions import get_active_run

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_response import LlmResponse

#: The key a rewritten answer carries its outcome under, in its
#: ``custom_metadata`` (and so its event's), beside the provider key.
OUTCOME_KEY = "agentic_cli_outcome"

#: How ADK's OpenAI Responses adapter words a refusal: a text part starting
#: with this (``google.adk.integrations.openai._openai_responses_llm._REFUSAL_PREFIX``,
#: private ADK; a contract test pins it).
OPENAI_REFUSAL_PREFIX = "OpenAI refusal: "


class SubAgentOutcomePlugin(BasePlugin):
    """Rewrites and marks a sub-agent's refused or cut final answer (see the module docstring)."""

    def __init__(self) -> None:
        super().__init__(name="subagent_outcome")

    async def after_model_callback(
        self, *, callback_context: "CallbackContext", llm_response: "LlmResponse"
    ) -> Optional["LlmResponse"]:
        if not llm_response.partial and get_active_run() is not None:
            mark_outcome(llm_response)
        return None


def mark_outcome(llm_response: "LlmResponse") -> str | None:
    """Rewrite ``llm_response`` in place when a model refused or cut it; its outcome, else None.

    A refusal's text parts become the refusal notice (its explanation the
    text ADK's OpenAI adapter gave, if any); a cut answer's become its text
    followed by the cut-off note. Thoughts are kept; the error ADK records
    for an answer with no content (``error_code``) is cleared, since the
    answer now says what happened.
    """
    parts = list(llm_response.content.parts or []) if llm_response.content else []
    if any(part.function_call for part in parts):
        return None
    texts = [part.text for part in parts if part.text and not part.thought]
    refusal = next((text for text in texts if text.startswith(OPENAI_REFUSAL_PREFIX)), None)
    outcome = (
        REFUSED if refusal is not None
        else outcome_of(llm_response.finish_reason) or outcome_of(llm_response.error_code)
    )
    if outcome is None:
        return None
    if outcome == REFUSED:
        explanation = refusal.removeprefix(OPENAI_REFUSAL_PREFIX) if refusal else None
        text = refusal_notice(explanation=explanation)
    else:
        text = cut_off("".join(texts))
    thoughts = [part for part in parts if part.thought]
    llm_response.content = types.Content(role="model", parts=[*thoughts, types.Part(text=text)])
    llm_response.error_code = None
    llm_response.error_message = None
    llm_response.custom_metadata = {**(llm_response.custom_metadata or {}), OUTCOME_KEY: outcome}
    return outcome

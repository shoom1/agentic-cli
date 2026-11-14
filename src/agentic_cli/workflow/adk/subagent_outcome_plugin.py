"""A sub-agent's refused or cut answer, said so to its parent (``workflow/agents/outcome.py``).

ADK hands a sub-agent's last answer to its parent as the delegation's result,
whatever stopped it: a refusal reached the parent as an empty result, and an
answer cut at the output limit as a complete one. ``SubAgentOutcomePlugin``
looks at each final answer of a sub-agent run (``get_active_run()`` is set:
``MainAgentPlugin`` makes the run active as the agent starts) and reads how it
stopped from what ADK's adapters give it:

- its ``finish_reason``: Claude's ``refusal`` arrives as ``SAFETY`` and its
  ``max_tokens`` as ``MAX_TOKENS``, a Chat Completions ``content_filter`` as
  ``SAFETY`` and ``length`` as ``MAX_TOKENS``, Gemini's as themselves;
- its ``error_code``, which ADK sets on an answer with no content: a Gemini
  prompt's ``block_reason`` among them;
- the reason an ``OTHER`` stop names in its error message: ADK's OpenAI
  Responses adapter gives an incomplete answer's ``content_filter`` so
  (``{"reason": "content_filter"}``);
- a text part ``OpenAI refusal: <text>`` (``OPENAI_REFUSAL_PREFIX``): the
  Responses adapter's refusal, and ``OpenAICompatibleLlm``'s of a Chat
  Completions ``refusal`` (OpenRouter, OpenAI-compatible servers).

It rewrites one a model refused into the refusal notice (that text its
explanation), and one cut at the output limit into its text followed by the
cut-off note, its thoughts kept. It marks the answer with its outcome
(``custom_metadata[OUTCOME_KEY]``), which the event carries, so the event
processor ends the run ``refused``, or ``completed`` and ``partial`` with one
warning.

It changes the response in place and returns None, as
``ForeignReasoningPlugin`` does, so every plugin after it still runs; it sits
after the raw log, which records the answer as the model gave it. An answer
that calls tools is not final and is left alone, as is every answer of the
root and of an agent the root transfers to (no active run): they reach the
user as their model gave them (a Claude or Gemini refusal with no answer is the
turn's failure, "<model> returned no answer"; a root on OpenRouter or an
OpenAI-compatible server replies "OpenAI refusal: <text>", as
``OpenAICompatibleLlm`` words a Chat Completions refusal).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from agentic_cli.workflow.agents.outcome import (
    OPENAI_REFUSAL_PREFIX,
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
        else outcome_of(llm_response.finish_reason)
        or outcome_of(llm_response.error_code)
        or outcome_of(_named_reason(llm_response))
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


def _named_reason(llm_response: "LlmResponse") -> str | None:
    """The reason an ``OTHER`` stop names in its error message, else None.

    ADK's OpenAI Responses adapter gives every incomplete answer but a cut
    one (``content_filter`` among them) as ``OTHER``, and its
    ``incomplete_details`` as the error message, in JSON
    (``{"reason": "content_filter"}``).
    """
    stops = {
        getattr(code, "value", code)
        for code in (llm_response.finish_reason, llm_response.error_code)
    }
    if "OTHER" not in stops or not llm_response.error_message:
        return None
    try:
        details = json.loads(llm_response.error_message)
    except ValueError:
        return None
    reason = details.get("reason") if isinstance(details, dict) else None
    return reason if isinstance(reason, str) else None

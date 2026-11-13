"""How a sub-agent's last answer ended, when it did not end as an answer should.

A sub-agent's final answer can stop for two reasons that its parent must
know, since the parent reads only the text: a model refused the request
(Claude's ``refusal``, a ``content_filter``, a Gemini safety or recitation
block), or the answer was cut at the model's output limit (``max_tokens``,
``length``, ``max_output_tokens``). Each backend's hook on a sub-agent's model
calls (ADK's ``SubAgentOutcomePlugin``, Deep Agents'
``SubAgentOutcomeMiddleware``) rewrites such an answer the same way and marks
it with its outcome, and the event processors end the run by that mark:

- a refusal: the answer becomes Deep Agents' notice (``refusal_notice``, its
  wording from deepagents 0.7.34, so the result is the same whichever version
  is installed), and the run ends ``refused``;
- a cut: the text is kept and followed by ``CUT_OFF_NOTE`` (``cut_off``), the
  run ends ``completed`` with ``partial``, and one recoverable warning names
  it (``cut_off_warning``).

The root's own answers are not touched: its refusal or cut is the turn's
failure, as before.
"""

from __future__ import annotations

from agentic_cli.workflow.events import EventType, WorkflowEvent

#: A sub-agent's last answer was refused by its model.
REFUSED = "refused"
#: A sub-agent's last answer was cut at its model's output limit.
CUT = "cut"

#: Deep Agents 0.7.34's notice for a sub-agent whose model refused
#: (``deepagents/middleware/subagents.py::_refusal_notice``), in two parts.
REFUSAL_NOTICE = "The subagent stopped because a model refused the request."
REFUSAL_ADVICE = (
    "Retrying the same request is unlikely to succeed; rephrase it or continue without it."
)
#: A refused run's ``error``, one line.
REFUSAL_ERROR = "a model refused the request"

#: What follows the text of an answer cut at the output limit.
CUT_OFF_NOTE = "[cut off at the output limit]"
#: The recoverable warning's error code for a cut answer.
CUT_OFF_CODE = "MAX_TOKENS"

#: Stop or finish reasons that mean the model refused, lowercased: Claude's
#: ``refusal``; OpenAI's ``content_filter``; Gemini's blocks.
REFUSAL_REASONS = frozenset({
    "refusal",
    "content_filter",
    "safety",
    "recitation",
    "blocklist",
    "prohibited_content",
    "spii",
    "image_safety",
    "image_prohibited_content",
    "image_recitation",
})
#: Stop or finish reasons that mean the answer was cut at the output limit,
#: lowercased: Claude's ``max_tokens``, OpenAI's ``length`` (and the Responses
#: API's ``max_output_tokens``), Gemini's ``MAX_TOKENS``.
CUT_REASONS = frozenset({"max_tokens", "length", "max_output_tokens"})


def outcome_of(reason: object) -> str | None:
    """``REFUSED`` or ``CUT`` for a stop or finish reason, else None.

    ``reason`` is a provider's string, or an enum (ADK's ``FinishReason``),
    compared by its value, case ignored.
    """
    if reason is None:
        return None
    value = str(getattr(reason, "value", reason)).lower()
    if value in REFUSAL_REASONS:
        return REFUSED
    if value in CUT_REASONS:
        return CUT
    return None


def refusal_notice(*, category: str | None = None, explanation: str | None = None) -> str:
    """The result of a sub-agent whose model refused: Deep Agents 0.7.34's notice.

    Args:
        category: The refusal's category, when the provider gives one.
        explanation: The refusal's own text, when the provider gives one.
    """
    notice = REFUSAL_NOTICE
    if category:
        notice += f" Category: {category}."
    if explanation:
        notice += f' Explanation: "{explanation}"'
    return f"{notice} {REFUSAL_ADVICE}"


def cut_off(text: str) -> str:
    """The result of a sub-agent whose answer was cut: its text, then ``CUT_OFF_NOTE``."""
    kept = text.rstrip()
    return f"{kept}\n\n{CUT_OFF_NOTE}" if kept else CUT_OFF_NOTE


def cut_off_warning(end: WorkflowEvent) -> WorkflowEvent | None:
    """The one recoverable warning for a run ``end`` reports cut short, else None.

    ``end`` is an ``AGENT_RUN`` event; the warning names the run's instance
    and is the run's (``run``, ``metadata["agent"]``).
    """
    ref = end.run
    if end.type is not EventType.AGENT_RUN or ref is None or not end.metadata.get("partial"):
        return None
    warning = WorkflowEvent.error(
        f"{ref.instance}'s answer was cut off at the output limit",
        error_code=CUT_OFF_CODE,
        recoverable=True,
    )
    warning.metadata["agent"] = ref.agent
    warning.run = ref
    return warning

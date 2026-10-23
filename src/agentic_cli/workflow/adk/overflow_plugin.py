"""A provider's context overflow on ADK: recovered once, or one clear answer.

The summary and the fit guard keep a request within the model's window by
estimate, and an estimate can be wrong. A model call the provider refuses as
too long (``context.is_context_overflow``: each provider's status, code or
wording) reaches ``OverflowPlugin.on_model_error_callback``:

- **The main agent** summarizes its conversation now, as ADK does before a
  model call (a compaction event through ``compact_session``, whatever the
  saving), so later calls carry the summary; then the request is retried once
  as every later call will be: the summary, then the turns it kept
  (``WrittenSummary.kept_turns``, ``fit_guard_plugin.keep_newest_turns``). A
  summary that fails is one warning a turn, as one before a model call is.
- **A sub-agent** never summarizes the shared session; nor does the main
  agent when the summary covers nothing the request holds. The request is
  then retried with its oldest whole turns left out down to half the budget
  (``fit_guard_plugin.drop_turns``).
- The retry is built from the request as the fit guard saw it
  (``FitGuardPlugin.request_as_seen``, this call's own copy, never a parallel
  call's), without the model's context cache: a model's adapter may have
  rewritten the request in place (Gemini's cache moves the system
  instruction, the tools and the cached contents out of it). Without a copy,
  it is built from the request the error callback received.
  It runs on the agent's own model
  (``get_invocation_context().agent.canonical_model``), and its answer is the
  call's. Plugins' ``before_model_callback``\\ s do not run for it: the request
  was already built and checked. The turns it leaves out are reported as the
  fit guard reports its own, once a turn.

When the retry is refused as too long too, or nothing could be left out (the
newest turn alone is too long), the main agent answers with an error
response, ``error_code="CONTEXT_OVERFLOW"`` and the clear text
(``context.overflow_message``), never with model text, which ADK would store
as its reply; ``ADKEventProcessor`` turns it into the turn's one ``ERROR``. A
sub-agent answers its coordinator with text, "I could not do this task: …",
which is how a coordinator hears from it (its reply lives in its own branch
and reaches the coordinator as the tool's result), and the user is told
once, as a recoverable ``ERROR`` reported live; that answer carries no
``error_code``, which would show the user a second warning. Any other error
of the retry propagates.

An overflow an OpenAI Responses model reports in its answer (``status:
failed``) instead of raising reaches ``after_model_callback`` as an error
response: it gets the same answer, without a retry.
"""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING

from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from agentic_cli.logging import Loggers
from agentic_cli.workflow import live_events
from agentic_cli.workflow.adk.compaction import SUMMARY_FAILED, SUMMARY_FAILED_CODE
from agentic_cli.workflow.adk.fit_guard_plugin import (
    drop_turns,
    fold_summary,
    is_summary,
    keep_newest_turns,
    overhead_tokens,
    report_trim,
)
from agentic_cli.workflow.context import (
    OVERFLOW_ERROR_CODE,
    is_context_overflow,
    is_overflow_text,
    overflow_message,
)
from agentic_cli.workflow.events import WorkflowEvent

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest

    from agentic_cli.models import ProviderKind
    from agentic_cli.workflow.adk.compaction import WrittenSummary
    from agentic_cli.workflow.adk.fit_guard_plugin import FitGuardPlugin
    from agentic_cli.workflow.context import ContextBudget

logger = Loggers.workflow()


class OverflowPlugin(BasePlugin):
    """Answers a context overflow (see the module docstring).

    Args:
        root: The main agent's name.
        budgets: Each agent's context budget, by name. An agent not named
            here is left to ADK.
        kinds: Each agent's provider kind, by name.
        models: Each agent's model reference, as the user is told it, by name.
        summarize_now: Summarizes the main agent's conversation now, given
            the failed call's context; returns the summary written, or None
            when nothing was summarized; raises when the summary failed.
        fit_guard: The app's fit guard, whose copy of each request a retry
            is built from; without one, the request the callback is given.
    """

    def __init__(
        self,
        *,
        root: str,
        budgets: Mapping[str, "ContextBudget"],
        kinds: Mapping[str, "ProviderKind"],
        models: Mapping[str, str],
        summarize_now: Callable[["CallbackContext"], Awaitable["WrittenSummary | None"]],
        fit_guard: "FitGuardPlugin | None" = None,
    ) -> None:
        super().__init__(name="context_overflow")
        self._root = root
        self._budgets = dict(budgets)
        self._kinds = dict(kinds)
        self._models = dict(models)
        self._summarize_now = summarize_now
        self._fit_guard = fit_guard

    async def on_model_error_callback(
        self,
        *,
        callback_context: "CallbackContext",
        llm_request: "LlmRequest",
        error: Exception,
    ) -> LlmResponse | None:
        agent = callback_context.agent_name
        budget, kind = self._budgets.get(agent), self._kinds.get(agent)
        if budget is None or kind is None or not is_context_overflow(error):
            return None
        logger.info("adk_context_overflow", agent=agent, error_type=type(error).__name__)
        summary = await self._summary(callback_context) if agent == self._root else None
        request = self._as_seen(callback_context, llm_request)
        dropped = 0
        if summary is not None:
            # What every later call sends: the summary, then the turns it kept.
            contents, dropped = keep_newest_turns(request.contents, summary.kept_turns)
        if not dropped:
            contents, dropped = drop_turns(
                request.contents,
                target_tokens=budget.budget // 2 - overhead_tokens(request, kind),
                kind=kind,
            )
        if not dropped:
            return self._does_not_fit(agent)
        if summary is not None:
            contents = _with_summary(contents, summary.text)
        report_trim(len(request.contents), len(contents), agent, dropped)
        retry = request.model_copy(update={"contents": contents})
        model = callback_context.get_invocation_context().agent.canonical_model
        try:
            async with contextlib.aclosing(
                model.generate_content_async(retry, stream=False)
            ) as responses:
                async for response in responses:
                    logger.info("adk_context_overflow_retried", agent=agent,
                                turns_dropped=dropped, summarized=summary is not None)
                    return response
        except Exception as again:
            if not is_context_overflow(again):
                raise
        return self._does_not_fit(agent)

    async def after_model_callback(
        self, *, callback_context: "CallbackContext", llm_response: LlmResponse
    ) -> LlmResponse | None:
        agent = callback_context.agent_name
        code = llm_response.error_code
        if agent not in self._budgets or code is None or code == OVERFLOW_ERROR_CODE:
            return None
        if not is_overflow_text(llm_response.error_message or ""):
            return None
        logger.info("adk_context_overflow", agent=agent, error_type=str(code))
        return self._does_not_fit(agent)

    async def _summary(self, callback_context: "CallbackContext") -> "WrittenSummary | None":
        """The main agent's conversation summarized now, or None.

        A summary that failed is one warning a turn, as one before a model
        call is, and the retry goes on without it.
        """
        try:
            return await self._summarize_now(callback_context)
        except Exception as error:  # noqa: BLE001 - the retry goes on without it
            logger.debug("adk_overflow_summary_failed", error_type=type(error).__name__,
                         error=str(error))
            live = live_events.current()
            if live is not None and live.first(SUMMARY_FAILED_CODE):
                live.emit(WorkflowEvent.error(
                    SUMMARY_FAILED, error_code=SUMMARY_FAILED_CODE, recoverable=True
                ))
            return None

    def _as_seen(
        self, callback_context: "CallbackContext", llm_request: "LlmRequest"
    ) -> "LlmRequest":
        """The failed call's request as the fit guard saw it (or, without its
        copy, as this callback received it), without the context cache."""
        update: dict = {"cache_config": None, "cache_metadata": None}
        seen = self._fit_guard.request_as_seen(callback_context) if self._fit_guard else None
        if seen is not None:
            contents, config = seen
            update["contents"] = list(contents)
            if config is not None:
                update["config"] = config.model_copy()
        return llm_request.model_copy(update=update)

    def _does_not_fit(self, agent: str) -> LlmResponse:
        """The answer once nothing more can make the request fit."""
        model = self._models.get(agent, "the model")
        if agent == self._root:
            return LlmResponse(
                error_code=OVERFLOW_ERROR_CODE, error_message=overflow_message(model)
            )
        warning = WorkflowEvent.error(
            f"{agent}: the request no longer fits {model}'s context window.",
            error_code=OVERFLOW_ERROR_CODE,
            recoverable=True,
        )
        warning.metadata["agent"] = agent
        live_events.report(warning)
        return LlmResponse(content=types.Content(role="model", parts=[types.Part.from_text(
            text=f"I could not do this task: the request no longer fits {model}'s context window."
        )]))


def _with_summary(contents: list[types.Content], summary: str) -> list[types.Content]:
    """``contents`` with ``summary`` first, folded into the first kept user content.

    It takes the place of the summary they open with, which it stands for
    too; folded as the fit guard folds (``fold_summary``), so no provider sees
    one more user message in a row.
    """
    rest = list(contents)
    if rest and is_summary(rest[0]):
        remainder = list(rest[0].parts or [])[1:]
        rest = [rest[0].model_copy(update={"parts": remainder})] if remainder else []
        rest += contents[1:]
    return fold_summary([types.Content(role="user", parts=[types.Part.from_text(text=summary)]),
                         *rest])

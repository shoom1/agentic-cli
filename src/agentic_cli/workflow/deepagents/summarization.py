"""Summarization: Deep Agents' middleware, with our thresholds and no archive.

Every agent's stack has Deep Agents' ``SummarizationMiddleware``. It never
rewrites ``messages``: it records a summary event in private state and applies
it per request, so the checkpoint keeps the whole conversation for
``/sessions``, ``recent_messages`` and fact extraction.

**When.** Deep Agents checks each request against an input budget,
``B = 0.95 * window - max_tokens``. Its default trigger is 85% of the *whole*
window, which with a large output allowance sits above B, so the budget check
would end the turn before anything was summarized. Here each agent's
thresholds are its model's ``ContextBudget`` (``workflow/context.py``), as on
ADK: summarize at the smaller of ``context_summarize_at`` of B and the
``context_max_tokens`` ceiling, and keep the newest 12% of that word for
word. A model whose window is unknown counts as 128,000 tokens, not Deep
Agents' fixed 170,000 tokens or 6 messages, which never fire on a small
model.

**No archive.** Deep Agents writes the summarized messages to its virtual
filesystem and gives the model the file's path; with its file tools off the
model could never read it. Its backend here stores nothing: Deep Agents treats
a failed offload as non-fatal and writes a summary without a path. Its warning
that older messages "will not be recoverable" is wrong for us (the checkpoint
has them), so ``install_history_filters`` turns it into one debug log. The
summary prompt is LangChain's, as ADK's summaries use it
(``workflow/summary_prompt.py``): Deep Agents' adds a note on media it
archived, which here never happens.

**A summary the model refuses as too long** is retried with the middle of
the conversation cut out, as LangChain's helper does on its
``ContextOverflowError``. The helper's summary model is ``OverflowMapped``:
every refusal ``workflow/context.py`` recognizes (OpenRouter's, an
OpenAI-compatible server's, older Claude's) is raised as that error, inside
the helper's retry, so it is not sent again as it was.

**Reporting.** The summary call runs on the agent's own model, tagged
``lc_source="summarization"``; the run callback reports its start
(``CONTEXT_SUMMARIZING``) and its usage. Each summary applied is reported as
a custom run event (``CONTEXT_SUMMARIZED``), which the run callback turns
into ``CONTEXT_TRIMMED``, with the conversation's tokens before and after as
the middleware's own counter has them.

**On demand.** ``/compact`` (the manager's ``compact``) summarizes the main
agent's conversation now with the same middleware's pieces: the cut
(``compact_cutoff``, ADK's rule: the newest ``keep`` tokens kept, cut where a
user's message starts), a summary written by a copy of its LangChain helper
with the focus in its prompt (``write_summary``), and the event Deep Agents
would store for it (``summary_event``), marked with the newest answer it
was written after. While that answer is still the newest, its reported
count is the prompt the summary replaced, no evidence: the middleware's
``_should_summarize`` then decides on its own estimate of the request
alone, as ADK's engine stops at a summary. They rest on private Deep Agents
methods (``_apply_event_to_messages``, ``_compute_state_cutoff``,
``_build_new_messages_with_path``, ``_should_summarize``, ``_lc_helper``
and its summary model); contract tests pin them.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import warnings
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileUploadResponse,
    WriteResult,
)
from deepagents.middleware.summarization import (
    SUMMARIZATION_EVENT_KEY,
    SummarizationEvent,
    SummarizationMiddleware,
)
from langchain.agents.middleware._retry import default_retry_on
from langchain_core.callbacks import adispatch_custom_event, dispatch_custom_event
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import set_config_context

from agentic_cli.logging import Loggers
from agentic_cli.workflow.context import chars_per_token, context_budget, is_context_overflow
from agentic_cli.workflow.summary_prompt import SUMMARY_PROMPT, with_focus

if TYPE_CHECKING:
    from langchain.agents.middleware import ModelRequest, ModelResponse
    from langchain.agents.middleware.summarization import ContextSize
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AnyMessage
    from langchain_core.runnables import RunnableConfig

    from agentic_cli.config import BaseSettings
    from agentic_cli.models import ProviderKind, ResolvedModel

logger = Loggers.workflow()

#: The custom run event a summary reports itself with: ``{"agent",
#: "messages_before", "messages_after", "messages_summarized",
#: "tokens_before", "tokens_after"}``.
CONTEXT_SUMMARIZED = "agentic_cli.context_summarized"

#: Why the archive refuses every write.
_NO_ARCHIVE = "archive disabled"

#: Where a message the framework wrote is marked (``response_metadata``, which
#: no provider adapter sends), as job notifications are (``sessions.py``), and
#: the mark's ``kind`` on a summary ``/compact`` wrote.
_MARK = "agentic_cli"
COMPACTED = "compact"

#: Deep Agents' texts for a history it could not offload (an error log and a
#: warning), and the per-write warning before them. Matched at the start.
_NOT_RECOVERABLE = "Offloading conversation history to backend failed"
_OFFLOAD_FAILED = "Failed to offload conversation history"
_SUMMARIZATION_LOGGER = "deepagents.middleware.summarization"


def summarization_thresholds(
    resolved: "ResolvedModel", settings: "BaseSettings"
) -> tuple["ContextSize", "ContextSize"]:
    """The ``(trigger, keep)`` an agent's summarization runs on, in tokens.

    Args:
        resolved: The agent's model: its window and its ``max_tokens``.
        settings: ``context_summarize_at`` and ``context_max_tokens``.

    Raises:
        ContextBudgetError: The model's input budget is too small to work in.
    """
    budget = context_budget(resolved, settings)
    return ("tokens", budget.trigger), ("tokens", budget.keep)


class NoArchiveBackend(BackendProtocol):
    """A summarization backend that stores nothing.

    It implements what ``SummarizationMiddleware`` calls: writes and uploads
    fail with ``"archive disabled"`` and downloads find nothing, so every
    offload fails, which Deep Agents treats as non-fatal.
    """

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return [FileDownloadResponse(path=path, error="file_not_found") for path in paths]

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return self.download_files(paths)

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error=_NO_ARCHIVE)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return self.write(file_path, content)

    def edit(
        self, file_path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> EditResult:
        return EditResult(error=_NO_ARCHIVE)

    async def aedit(
        self, file_path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> EditResult:
        return self.edit(file_path, old_string, new_string, replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=path, error=_NO_ARCHIVE) for path, _ in files]

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return self.upload_files(files)


class OverflowMapped(Runnable):
    """A summary model whose context overflows are LangChain's ``ContextOverflowError``.

    The summary call goes to the agent's model directly, around the agent's
    middleware, so ``ContextOverflowMiddleware`` never sees it. LangChain's
    summary helper cuts the middle out of a summary request and sends it
    again only on ``ContextOverflowError``, which a partner package raises
    for its own provider's words but not for OpenRouter's, an
    OpenAI-compatible server's (vLLM, llama.cpp), older Claude's "exceed
    context limit" or Anthropic's 413. Every overflow ``workflow/context.py``
    recognizes is re-raised as that error, from the original; anything
    else, a ``ContextOverflowError`` included, passes unchanged.

    Args:
        model: The chat model that writes the summaries.
    """

    def __init__(self, model: Runnable) -> None:
        self.model = model

    def invoke(self, input: Any, config: "RunnableConfig | None" = None, **kwargs: Any) -> Any:  # noqa: A002 - Runnable's signature
        try:
            return self.model.invoke(input, config, **kwargs)
        except ContextOverflowError:
            raise
        except Exception as error:
            if is_context_overflow(error):
                raise ContextOverflowError(str(error)) from error
            raise

    async def ainvoke(
        self, input: Any, config: "RunnableConfig | None" = None, **kwargs: Any  # noqa: A002
    ) -> Any:
        try:
            return await self.model.ainvoke(input, config, **kwargs)
        except ContextOverflowError:
            raise
        except Exception as error:
            if is_context_overflow(error):
                raise ContextOverflowError(str(error)) from error
            raise


class ReportingSummarizationMiddleware(SummarizationMiddleware):
    """Deep Agents' summarization, reporting each summary it applies.

    The report is a custom run event, so it reaches the turn's run callback
    wherever the agent runs (a sub-agent included). Only the public model-call
    hooks are wrapped; what a summary did is read from Deep Agents' documented
    summary event (``SummarizationEvent.cutoff_index``) in the update it
    returns.

    Its helper's summary model is ``OverflowMapped``, inside LangChain's
    retry, so a summary refused as too long is cut and sent again, whatever
    the provider calls it. And a summary ``/compact`` wrote is not written
    again on the strength of a count reported before it (``_should_summarize``).

    Args:
        agent: The agent whose stack this is.
        **kwargs: ``SummarizationMiddleware``'s arguments.
    """

    def __init__(self, *, agent: str | None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        helper = self._lc_helper
        # Inside the retry: a mapped overflow is never retried as it was sent.
        helper._summary_model = OverflowMapped(helper._summary_chat_model).with_retry(
            retry_if_exception_type=default_retry_on
        )
        kind, value = kwargs.get("trigger") or (None, None)
        self._trigger_tokens: int | None = int(value) if kind == "tokens" else None

    def _should_summarize(self, messages: list["AnyMessage"], total_tokens: int) -> bool:
        """Whether to summarize the request, which holds ``messages``.

        Deep Agents summarizes when its estimate of the request
        (``total_tokens``) or the count the newest answer reported passes
        the trigger. Right after a summary ``/compact`` wrote, the newest
        answer is the one it was written after (its mark's ``after``), and
        its count is the prompt the summary replaced: then the estimate
        alone decides. Otherwise Deep Agents decides.
        """
        if self._trigger_tokens is not None and _counted_before_the_summary(messages):
            return total_tokens >= self._trigger_tokens
        return super()._should_summarize(messages, total_tokens)

    async def awrap_model_call(
        self,
        request: "ModelRequest",
        handler: Callable[["ModelRequest"], Awaitable["ModelResponse"]],
    ) -> Any:
        sent = False

        async def call(request: "ModelRequest") -> "ModelResponse":
            nonlocal sent
            sent = True
            return await handler(request)

        try:
            response = await super().awrap_model_call(request, call)
        except NoSummaryWritten as error:
            if sent:
                raise _refused_unsummarized(error) from error
            await self._anot_summarized()
            return await handler(request.override(messages=self._get_effective_messages(request)))
        report = self._report(request, response)
        if report is not None:
            try:
                await adispatch_custom_event(CONTEXT_SUMMARIZED, report)
            except RuntimeError:  # no run to report to (called outside a graph)
                logger.debug("deepagents_summary_not_reported", agent=self.agent)
        return response

    def wrap_model_call(
        self,
        request: "ModelRequest",
        handler: Callable[["ModelRequest"], "ModelResponse"],
    ) -> Any:
        sent = False

        def call(request: "ModelRequest") -> "ModelResponse":
            nonlocal sent
            sent = True
            return handler(request)

        try:
            response = super().wrap_model_call(request, call)
        except NoSummaryWritten as error:
            if sent:
                raise _refused_unsummarized(error) from error
            self._not_summarized()
            return handler(request.override(messages=self._get_effective_messages(request)))
        report = self._report(request, response)
        if report is not None:
            try:
                dispatch_custom_event(CONTEXT_SUMMARIZED, report)
            except RuntimeError:  # no run to report to (called outside a graph)
                logger.debug("deepagents_summary_not_reported", agent=self.agent)
        return response

    def _report(self, request: "ModelRequest", response: Any) -> dict[str, Any] | None:
        """What a summary did to the request, if this call applied a new one.

        Counts are of the conversation the model is sent: before, the state's
        messages (from the previous summary on, which stands for the rest);
        after, the new summary and the messages it kept. Tokens are the
        middleware's own count of those two conversations.
        """
        update = getattr(getattr(response, "command", None), "update", None)
        event = update.get(SUMMARIZATION_EVENT_KEY) if isinstance(update, dict) else None
        if not isinstance(event, dict):
            return None
        messages = list(request.messages)
        previous = (request.state or {}).get(SUMMARIZATION_EVENT_KEY)
        before = messages
        if isinstance(previous, dict):
            before = [previous["summary_message"], *messages[previous["cutoff_index"]:]]
        after = [event["summary_message"], *messages[event["cutoff_index"]:]]
        start = previous["cutoff_index"] if isinstance(previous, dict) else 0
        carried = 1 if isinstance(previous, dict) else 0  # the previous summary
        return {
            "agent": self.agent,
            "messages_before": len(before),
            "messages_after": len(after),
            "messages_summarized": event["cutoff_index"] - start + carried,
            "tokens_before": self.token_counter(before),
            "tokens_after": self.token_counter(after),
        }

    def _create_summary(self, messages_to_summarize: list[Any]) -> str:
        """Deep Agents' summary, refused when the model wrote no text.

        A summary call can end with no text: Claude ends one with
        ``stop_reason: "refusal"``. Deep Agents would store that empty
        summary in place of the messages it stands for, so the model would
        never see them again. Raising ``NoSummaryWritten`` instead stores
        nothing: the model call goes on with the conversation in full, the
        user is told once a turn (``SUMMARY_NOT_WRITTEN``), and the next call
        tries again. After the provider refused the request as too long,
        that is a context overflow.
        """
        return _written(super()._create_summary(messages_to_summarize))

    async def _acreate_summary(self, messages_to_summarize: list[Any]) -> str:
        """``_create_summary``, async."""
        return _written(await super()._acreate_summary(messages_to_summarize))

    async def _anot_summarized(self) -> None:
        logger.debug("deepagents_summary_not_written", agent=self.agent)
        try:
            await adispatch_custom_event(SUMMARY_NOT_WRITTEN, {"agent": self.agent})
        except RuntimeError:  # no run to report to (called outside a graph)
            pass

    def _not_summarized(self) -> None:
        logger.debug("deepagents_summary_not_written", agent=self.agent)
        try:
            dispatch_custom_event(SUMMARY_NOT_WRITTEN, {"agent": self.agent})
        except RuntimeError:  # no run to report to (called outside a graph)
            pass


#: The custom run event a summary that could not be written reports: ``{"agent"}``.
SUMMARY_NOT_WRITTEN = "agentic_cli.summary_not_written"
#: What the user is told, once a turn, when a summary could not be written.
SUMMARY_FAILED = "Could not summarize the earlier conversation; it is sent in full instead."
SUMMARY_FAILED_CODE = "summary_failed"


class NoSummaryWritten(Exception):
    """A summary call came back with no text, so no summary is stored."""


def _written(summary: str) -> str:
    if not summary.strip():
        raise NoSummaryWritten("the model wrote no summary")
    return summary


def _refused_unsummarized(error: NoSummaryWritten) -> Exception:
    """The overflow a request is left with when no summary could shrink it."""
    from langchain_core.exceptions import ContextOverflowError

    return ContextOverflowError(f"The request is too long, and {error}.")


def build_summarization(
    model: "BaseChatModel", resolved: "ResolvedModel", settings: "BaseSettings"
) -> SummarizationMiddleware:
    """An agent's summarization middleware, on its own model.

    Args:
        model: The agent's chat model, which also writes the summaries.
        resolved: The agent's resolved model (thresholds; ``agent`` names it).
        settings: The manager's settings (``context_summarize_at``,
            ``context_max_tokens``).

    Raises:
        ContextBudgetError: The model's input budget is too small to work in.
    """
    install_history_filters()
    trigger, keep = summarization_thresholds(resolved, settings)
    return ReportingSummarizationMiddleware(
        agent=resolved.agent,
        model=model,
        backend=NoArchiveBackend(),
        trigger=trigger,
        keep=keep,
        # Deep Agents' prompt adds how to read media it archived: none here.
        summary_prompt=SUMMARY_PROMPT,
        # The older history is summarized whole, as create_deep_agent does;
        # the constructor's default would summarize only its last 4,000 tokens.
        trim_tokens_to_summarize=None,
        truncate_args_settings=None,
    )


# --- on demand (/compact) -------------------------------------------------------------------


def compact_cutoff(
    messages: Sequence["AnyMessage"], *, keep_tokens: int, kind: "ProviderKind"
) -> int | None:
    """Where ``/compact`` cuts the conversation the model is sent: ``messages[:i]``
    is summarized, the rest kept.

    ADK's rule (``workflow/adk/compaction.py::summary_range``), on messages:
    walking back from the newest message until about ``keep_tokens`` are kept
    (each counted as LangChain's approximate counter counts it, at the
    provider's characters per token), the cut goes back to where that user
    turn started (a ``HumanMessage``; the summary so far is none), and further
    back while a tool call before the cut has no result before it. The newest
    turn is never summarized.

    Args:
        messages: The conversation as the model is sent it: the summary so
            far first, if there is one (``_apply_event_to_messages``).
        keep_tokens: About how much to keep word for word (the agent's
            ``ContextBudget.keep``).
        kind: The agent's provider, whose characters per token count.

    Returns:
        The cut; None when nothing would be summarized but the summary so
        far, or nothing at all.
    """
    ratio = chars_per_token(kind)
    cut, kept = len(messages), 0
    while cut > 0 and kept < keep_tokens:
        cut -= 1
        kept += count_tokens_approximately([messages[cut]], chars_per_token=ratio)
    starts = [
        index for index, message in enumerate(messages)
        if isinstance(message, HumanMessage) and not _is_summary(message)
    ]
    for start in reversed(starts):
        if start <= cut and _closed(messages[:start]):
            break
    else:
        return None
    if start == 0 or (start == 1 and _is_summary(messages[0])):
        return None
    return start


def _is_summary(message: "AnyMessage") -> bool:
    """A summary message, as Deep Agents marks it (``lc_source``)."""
    return (
        isinstance(message, HumanMessage)
        and message.additional_kwargs.get("lc_source") == "summarization"
    )


def _closed(messages: Sequence["AnyMessage"]) -> bool:
    """Whether every tool call in ``messages`` has its result among them.

    A call a cancelled turn left open is answered before the next run, right
    after it (``PatchToolCallsMiddleware``): summarized, its answer would
    open the next request.
    """
    called = {
        call["id"]
        for message in messages if isinstance(message, AIMessage)
        for call in message.tool_calls if call.get("id")
    }
    answered = {message.tool_call_id for message in messages if isinstance(message, ToolMessage)}
    return called <= answered


async def write_summary(
    middleware: SummarizationMiddleware,
    messages: list["AnyMessage"],
    *,
    focus: str | None = None,
    callbacks: Sequence["BaseCallbackHandler"] = (),
) -> str:
    """A summary of ``messages``, as ``middleware`` writes one, on ``focus``.

    A copy of the middleware's LangChain helper, with ``focus`` spliced into
    its prompt (``with_focus``), writes it on the agent's model, tagged
    ``lc_source="summarization"``, as a summary in a turn is written. It runs
    in a context whose run config holds ``callbacks`` (the turn's run
    callback), so they see the call as a summary in a turn.

    Returns:
        The summary, stripped (empty when the model wrote none).

    Raises:
        Exception: The model call failed, after LangChain's retries.
    """
    helper = copy.copy(middleware._lc_helper)
    helper.summary_prompt = with_focus(helper.summary_prompt, focus)
    with set_config_context({"callbacks": list(callbacks)}) as context:
        summary = await asyncio.create_task(
            helper._acreate_summary(messages), context=context
        )
    return summary.strip()


def summary_event(
    middleware: SummarizationMiddleware,
    previous: SummarizationEvent | None,
    cut: int,
    summary: str,
    *,
    after: str | None = None,
    written_on: int | None = None,
) -> SummarizationEvent:
    """The event Deep Agents stores for ``summary``, of the sent messages before ``cut``.

    Its cut is moved from the messages the model is sent (the summary so far
    at 0) to the state's, and its message is the one Deep Agents writes when
    there is no archive: "Here is a summary of the conversation to date:"
    and the summary. With ``after`` (the id of the newest answer in the
    messages sent), the message is marked ``{"kind": "compact", "after":
    after}`` in its ``response_metadata``, which ``_should_summarize`` reads.
    With ``written_on`` (how many messages the state holds), the event says
    so as Deep Agents' own do since 0.7.30 (``created_at_index``): the
    messages it kept from before it then lose their reasoning, which Claude
    binds to the conversation before them. An older Deep Agents ignores it.
    """
    (message,) = middleware._build_new_messages_with_path(summary, None)
    if after is not None:
        message.response_metadata[_MARK] = {"kind": COMPACTED, "after": after}
    event: SummarizationEvent = {
        "cutoff_index": middleware._compute_state_cutoff(previous, cut),
        "summary_message": message,
        "file_path": None,
    }
    if written_on is not None:
        event["created_at_index"] = written_on  # type: ignore[typeddict-unknown-key]
    return event


def newest_answer(messages: Sequence["AnyMessage"]) -> AIMessage | None:
    """The newest ``AIMessage`` in ``messages``, if any."""
    return next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)


def _counted_before_the_summary(messages: Sequence["AnyMessage"]) -> bool:
    """Whether the newest answer is the one a ``/compact`` summary was written after.

    ``messages`` are the request's, the summary first; the summary's mark
    names that answer (``summary_event``).
    """
    if not messages:
        return False
    mark = messages[0].response_metadata.get(_MARK)
    if not isinstance(mark, dict) or mark.get("kind") != COMPACTED:
        return False
    newest = newest_answer(messages)
    return newest is not None and newest.id is not None and newest.id == mark.get("after")


class _HistoryNotArchived(logging.Filter):
    """Turns Deep Agents' offload-failure logs into one debug log.

    The per-write warning is dropped; the "not recoverable" error becomes
    ``deepagents_history_not_archived``. Every other record passes.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = str(record.msg)
        if message.startswith(_NOT_RECOVERABLE):
            logger.debug("deepagents_history_not_archived")
            return False
        return not message.startswith(_OFFLOAD_FAILED)


def install_history_filters() -> None:
    """Quiet Deep Agents' "older messages will not be recoverable" notices.

    A logging filter on Deep Agents' summarization logger and a warnings
    filter, both anchored to the start of those messages; nothing else from
    Deep Agents is silenced. Idempotent.
    """
    source = logging.getLogger(_SUMMARIZATION_LOGGER)
    if not any(isinstance(f, _HistoryNotArchived) for f in source.filters):
        source.addFilter(_HistoryNotArchived())
    # warnings.filterwarnings replaces an identical entry rather than adding one.
    warnings.filterwarnings("ignore", message=_NOT_RECOVERABLE, category=UserWarning)

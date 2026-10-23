"""Each model request fits its agent's input budget, whatever the history holds.

The summary is triggered by the provider's count of the agent's previous
prompt, so it lags one call; a sub-agent that inherits the conversation may
run on a smaller model, and never compacts the shared session; a model whose
window is unknown counts as 128,000 tokens, which may be more than it has.
``FitGuardPlugin`` covers all three, in ``before_model_callback``, last of the
plugins, after ADK has built the request (the summary included):

- It estimates the request: the system prompt, the tool declarations and the
  contents, as JSON characters, at the provider's characters per token
  (``context.chars_per_token``), plus 10%.
- Over the agent's budget B, it drops the oldest whole turns from this
  request (``drop_turns``) until it is under the agent's trigger: a turn
  starts at a user's text, never at a tool result or right after a call, so
  a call keeps its result; the newest turn stays, and so does a summary that
  opens the conversation. The first kept user content starts with a note,
  ``FIT_NOTE``, as a part of its own: no new message, so no provider sees one
  more user message in a row. The session is untouched: the next request
  starts from the whole history again.
- It reports ``CONTEXT_TRIMMED(source="fit_guard", turns_dropped=n)`` once a
  turn.
- When the newest turn alone is over the budget, nothing it could drop would
  help: the request goes as it is, and the overflow handler answers.
- As its last step, a summary that opens the conversation (a user content of
  its own) is folded into the next content when that is a user's text, as a
  part before its own (``fold_summary``): a strict chat template refuses two
  user messages in a row.

It keeps each request as it came in, until the call ends
(``request_as_seen``): the overflow handler retries from that copy, since a
model's adapter may rewrite the request in place (Gemini's context cache moves
the system instruction, the tools and the cached contents out of it).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from agentic_cli.logging import Loggers
from agentic_cli.workflow import live_events
from agentic_cli.workflow.context import ESTIMATE_MARGIN, chars_per_token
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.summary_prompt import SUMMARY_PREFIX

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse

    from agentic_cli.models import ProviderKind
    from agentic_cli.workflow.context import ContextBudget

logger = Loggers.workflow()

#: What the model is told where older turns were left out.
FIT_NOTE = "[Earlier turns were left out to fit the context window.]"
#: The note's own size, as a part of a content's JSON (and its comma).
_NOTE_CHARS = len(types.Part.from_text(text=FIT_NOTE).model_dump_json(exclude_none=True)) + 1

#: A request as it came in: its contents, and a copy of its config.
Seen = tuple[list[types.Content], "types.GenerateContentConfig | None"]


class FitGuardPlugin(BasePlugin):
    """Keeps each request within its agent's budget (see the module docstring).

    Args:
        budgets: Each agent's context budget, by name. An agent not named
            here (a native ADK YAML app's) is left alone.
        kinds: Each agent's provider kind, by name, for its characters per token.
    """

    def __init__(
        self, budgets: Mapping[str, "ContextBudget"], kinds: Mapping[str, "ProviderKind"]
    ) -> None:
        super().__init__(name="fit_guard")
        self._budgets = dict(budgets)
        self._kinds = dict(kinds)
        # (invocation, agent) → its model call's request as it came in.
        self._seen: dict[tuple[str, str], Seen] = {}

    async def before_model_callback(
        self, *, callback_context: "CallbackContext", llm_request: "LlmRequest"
    ) -> None:
        agent = callback_context.agent_name
        budget, kind = self._budgets.get(agent), self._kinds.get(agent)
        if budget is None or kind is None:
            return None
        config = llm_request.config
        self._seen[_key(callback_context)] = (
            list(llm_request.contents), config.model_copy() if config is not None else None
        )
        self._fit(llm_request, agent, budget, kind)
        llm_request.contents = fold_summary(llm_request.contents)
        return None

    async def after_model_callback(
        self, *, callback_context: "CallbackContext", llm_response: "LlmResponse"
    ) -> None:
        if not llm_response.partial:
            self._seen.pop(_key(callback_context), None)
        return None

    async def on_model_error_callback(
        self, *, callback_context: "CallbackContext", llm_request: "LlmRequest",
        error: Exception,
    ) -> None:
        self._seen.pop(_key(callback_context), None)
        return None

    async def after_run_callback(self, *, invocation_context: "InvocationContext") -> None:
        invocation = invocation_context.invocation_id
        self._seen = {key: seen for key, seen in self._seen.items() if key[0] != invocation}

    def request_as_seen(self, callback_context: "CallbackContext") -> Seen | None:
        """The request of the model call in flight as it came in, before any
        plugin after this one, or the model's adapter, changed it; None when
        this agent has no budget or no call is in flight."""
        return self._seen.get(_key(callback_context))

    def _fit(
        self, llm_request: "LlmRequest", agent: str, budget: "ContextBudget",
        kind: "ProviderKind",
    ) -> None:
        """Leave the oldest turns out of ``llm_request`` when it is over the budget."""
        estimate = estimate_request_tokens(llm_request, kind)
        if estimate <= budget.budget:
            return
        overhead = overhead_tokens(llm_request, kind)
        contents, dropped = drop_turns(
            llm_request.contents, target_tokens=budget.trigger - overhead, kind=kind
        )
        if not dropped or overhead + _tokens(_contents_chars(contents), kind) > budget.budget:
            # The newest turn alone is over: the overflow handler answers.
            logger.debug("adk_request_over_budget", agent=agent, estimate=estimate,
                         budget=budget.budget)
            return
        before = len(llm_request.contents)
        llm_request.contents = contents
        logger.debug("adk_request_fitted", agent=agent, estimate=estimate,
                     budget=budget.budget, turns_dropped=dropped)
        report_trim(before, len(contents), agent, dropped)


def report_trim(before: int, after: int, agent: str, dropped: int) -> None:
    """Report turns left out of a request, at most once a turn (the fit guard's key)."""
    live = live_events.current()
    if live is not None and live.first("fit_guard"):
        live.emit(WorkflowEvent.context_trimmed(
            before, after, "fit_guard", agent, turns_dropped=dropped
        ))


def estimate_request_tokens(llm_request: "LlmRequest", kind: "ProviderKind") -> int:
    """About how many tokens ``llm_request`` is for a ``kind`` model.

    The characters of the system instruction, the tool declarations (JSON)
    and the contents (JSON), at the provider's characters per token, plus 10%.
    """
    return _tokens(_config_chars(llm_request) + _contents_chars(llm_request.contents), kind)


def overhead_tokens(llm_request: "LlmRequest", kind: "ProviderKind") -> int:
    """The part of ``estimate_request_tokens`` no turn can lower: the system
    instruction and the tool declarations."""
    return _tokens(_config_chars(llm_request), kind)


def fold_summary(contents: list[types.Content]) -> list[types.Content]:
    """``contents`` with a summary that opens them folded into the next one.

    When the summary (a user content of its own) is followed by a user's
    text, that content gets the summary's parts before its own, so no
    provider sees two user messages in a row. Otherwise (a tool result, a
    model's turn, nothing) ``contents`` as they are; they are never changed
    in place.
    """
    if len(contents) < 2 or not _is_summary(contents[0]) or not _users_text(contents[1]):
        return contents
    summary, first = contents[0], contents[1]
    merged = first.model_copy(update={"parts": [*(summary.parts or []), *(first.parts or [])]})
    return [merged, *contents[2:]]


def drop_turns(
    contents: list[types.Content], *, target_tokens: float, kind: "ProviderKind"
) -> tuple[list[types.Content], int]:
    """``contents`` without their oldest whole turns, under ``target_tokens`` if it can be.

    A turn starts at a user content with a text and no tool result, after a
    content that made no call: so a call and its result stay together. Turns
    go oldest first until the rest, with the note, is under the target; the
    newest turn always stays, and so does a summary that opens the
    conversation. The first kept user content gets ``FIT_NOTE`` as a part
    before its own; ``contents`` themselves are left as they are.

    Each content is measured once; what goes is subtracted from a running
    sum.

    Returns:
        The contents to send, and how many turns were left out (0, with
        ``contents`` themselves, when there is only one turn).
    """
    head = contents[:1] if contents and _is_summary(contents[0]) else []
    body = contents[len(head):]
    starts = [i for i in range(len(body)) if _starts_turn(body, i)]
    if not starts or starts[-1] == 0:
        return contents, 0
    sizes = [len(content.model_dump_json(exclude_none=True)) for content in contents]
    remaining = sum(sizes) + _NOTE_CHARS
    sizes = sizes[len(head):]
    # A partial turn before the first start goes with the first turn.
    cuts = [cut for cut in starts if cut > 0]
    previous = 0
    for dropped, cut in enumerate(cuts, start=1):
        remaining -= sum(sizes[previous:cut])
        previous = cut
        if _tokens(remaining, kind) < target_tokens:
            break
    return [*head, _noted(body[cut]), *body[cut + 1:]], dropped


def _noted(content: types.Content) -> types.Content:
    """``content`` with ``FIT_NOTE`` as a part before its own."""
    return content.model_copy(update={"parts": [types.Part.from_text(text=FIT_NOTE),
                                                *(content.parts or [])]})


def _starts_turn(contents: list[types.Content], i: int) -> bool:
    if not _users_text(contents[i]):
        return False
    previous = contents[i - 1] if i > 0 else None
    return previous is None or not any(part.function_call for part in previous.parts or [])


def _users_text(content: types.Content) -> bool:
    """A user content holding a text, and no tool result."""
    if content.role != "user" or not content.parts:
        return False
    if any(part.function_response for part in content.parts):
        return False
    return any(part.text and not part.thought for part in content.parts)


def _is_summary(content: types.Content) -> bool:
    """A user content whose first part is the summary ADK sends first once the
    conversation was summarized (folded or not)."""
    first = content.parts[0] if content.parts else None
    return content.role == "user" and bool(first and first.text) and first.text.startswith(
        SUMMARY_PREFIX
    )


def _config_chars(llm_request: "LlmRequest") -> int:
    config = llm_request.config
    if config is None:
        return 0
    chars = 0
    instruction = config.system_instruction
    if isinstance(instruction, str):
        chars += len(instruction)
    elif isinstance(instruction, types.Content):
        chars += len(instruction.model_dump_json(exclude_none=True))
    elif instruction is not None:
        chars += len(str(instruction))
    for tool in config.tools or []:
        dump = getattr(tool, "model_dump_json", None)
        chars += len(dump(exclude_none=True)) if dump is not None else len(str(tool))
    return chars


def _contents_chars(contents: list[types.Content]) -> int:
    return sum(len(content.model_dump_json(exclude_none=True)) for content in contents)


def _tokens(chars: int, kind: "ProviderKind") -> int:
    return round(chars / chars_per_token(kind) * ESTIMATE_MARGIN)


def _key(callback_context: "CallbackContext") -> tuple[str, str]:
    return callback_context.invocation_id, callback_context.agent_name

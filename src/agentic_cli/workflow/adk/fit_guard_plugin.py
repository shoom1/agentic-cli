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
  a call keeps its result; the newest turn stays, and so do the memory index,
  the roster of the main agent's sub-agents and a summary that open the
  conversation (``protected_head``). The first
  kept user content starts with a note, ``FIT_NOTE``, as a part of its own:
  no new message, so no provider sees one more user message in a row. The
  session is untouched: the next request starts from the whole history again.
- In a sub-agent run's request (``keep_task``: a sub-agent type's chain) the
  run's task message, its invocation's user content, and its instance's
  brief (the event of its child session whose state delta set ``BRIEF_KEY``:
  ``brief_of``; a follow-up's message is not the brief) are never left out
  either: a message steered to the run after a tool result starts a turn,
  and the turns between the task message and the first kept one go; that
  turn's opening content joins the task message, after the note, as one
  user content.
- It reports ``CONTEXT_TRIMMED(source="fit_guard", turns_dropped=n)`` once a
  turn.
- When the newest turn alone is over the budget, nothing it could drop would
  help: the request goes as it is, and the overflow handler answers.
- As its last step, what opens the conversation (the memory index, the roster, a
  summary: each a user content of its own) is folded into the next content when that
  is a user's text, as parts before its own (``fold_summary``): a strict chat
  template refuses two user messages in a row.

It keeps each request as it came in, until the call ends
(``request_as_seen``): the overflow handler retries from that copy, since a
model's adapter may rewrite the request in place (Gemini's context cache moves
the system instruction, the tools and the cached contents out of it). A copy
belongs to one model call: it is kept under the identity of the call's
``EventActions`` (``callback_context.actions``), which ADK makes for each
call and hands to all three of its callbacks. It is released when the
call ends (``after_model_callback``, ``on_model_error_callback``). Before
google-adk 2.14 a call cancelled mid-request fires neither (2.14 runs the
error callback with the ``CancelledError``), so as each run ends
(``after_run_callback``, also for a stopped one; ``on_run_error_callback``
for one that fails) the copies it left are dropped, its own only: two runs
of one sub-agent type share their runner, and this plugin, at once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple

from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from agentic_cli.logging import Loggers
from agentic_cli.memory import INDEX_OPEN
from agentic_cli.workflow import live_events
from agentic_cli.workflow.context import ESTIMATE_MARGIN, chars_per_token
from agentic_cli.workflow.adk.sessions import BRIEF_KEY
from agentic_cli.workflow.agents.roster import ROSTER_OPEN
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.summary_prompt import SUMMARY_PREFIX

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events import Event
    from google.adk.events.event_actions import EventActions
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


class _Held(NamedTuple):
    """One model call's copy, under the id of the call's ``EventActions``:
    holding the object keeps that id from going to another while it is here."""

    actions: "EventActions"
    invocation: str
    seen: Seen


class FitGuardPlugin(BasePlugin):
    """Keeps each request within its agent's budget (see the module docstring).

    Args:
        budgets: Each agent's context budget, by name. An agent not named
            here (a native ADK YAML app's whose model did not resolve) is left
            alone.
        kinds: Each agent's provider kind, by name, for its characters per token.
        keep_task: Whether each request's task message (its invocation's
            user content) is kept whatever is left out: a sub-agent type's
            chain, whose runs each start from one.
    """

    def __init__(
        self, budgets: Mapping[str, "ContextBudget"], kinds: Mapping[str, "ProviderKind"],
        *, keep_task: bool = False,
    ) -> None:
        super().__init__(name="fit_guard")
        self._budgets = dict(budgets)
        self._kinds = dict(kinds)
        self._keep_task = keep_task
        # id(a model call's EventActions) → its request as it came in.
        self._seen: dict[int, _Held] = {}

    async def before_model_callback(
        self, *, callback_context: "CallbackContext", llm_request: "LlmRequest"
    ) -> None:
        agent = callback_context.agent_name
        budget, kind = self._budgets.get(agent), self._kinds.get(agent)
        if budget is None or kind is None:
            return None
        config = llm_request.config
        actions = callback_context.actions
        self._seen[id(actions)] = _Held(actions, callback_context.invocation_id, (
            list(llm_request.contents), config.model_copy() if config is not None else None
        ))
        task = callback_context.user_content if self._keep_task else None
        brief = brief_of(callback_context.session.events) if self._keep_task else None
        self._fit(llm_request, agent, budget, kind, task, brief)
        llm_request.contents = fold_summary(llm_request.contents)
        return None

    async def after_model_callback(
        self, *, callback_context: "CallbackContext", llm_response: "LlmResponse"
    ) -> None:
        if not llm_response.partial:
            self._release(callback_context)
        return None

    async def on_model_error_callback(
        self, *, callback_context: "CallbackContext", llm_request: "LlmRequest",
        error: Exception,
    ) -> None:
        self._release(callback_context)
        return None

    async def after_run_callback(self, *, invocation_context: "InvocationContext") -> None:
        # What a cancelled call left: before google-adk 2.14 it fired neither release.
        self._forget(invocation_context.invocation_id)
        return None

    async def on_run_error_callback(
        self, *, invocation_context: "InvocationContext", error: Exception
    ) -> None:
        self._forget(invocation_context.invocation_id)
        return None

    def _forget(self, invocation: str) -> None:
        """Drop the copies ``invocation`` left; another run's stay."""
        self._seen = {
            key: held for key, held in self._seen.items() if held.invocation != invocation
        }

    def request_as_seen(self, callback_context: "CallbackContext") -> Seen | None:
        """The request of this model call as it came in, before any plugin
        after this one, or the model's adapter, changed it; None when this
        agent has no budget or the call has ended. Never another call's: a
        parallel call of the same agent has its own."""
        held = self._seen.get(id(callback_context.actions))
        return held.seen if held is not None else None

    def _release(self, callback_context: "CallbackContext") -> None:
        self._seen.pop(id(callback_context.actions), None)

    def _fit(
        self, llm_request: "LlmRequest", agent: str, budget: "ContextBudget",
        kind: "ProviderKind", task: types.Content | None = None,
        brief: types.Content | None = None,
    ) -> None:
        """Leave the oldest turns out of ``llm_request`` when it is over the budget.

        ``task``, a sub-agent run's task message, and ``brief``, its
        instance's, are never left out.
        """
        estimate = estimate_request_tokens(llm_request, kind)
        if estimate <= budget.budget:
            return
        overhead = overhead_tokens(llm_request, kind)
        contents, dropped = drop_turns(
            llm_request.contents, target_tokens=budget.trigger - overhead, kind=kind, task=task,
            brief=brief,
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
    """``contents`` with what opens them (the index, the roster, a summary) folded into the next one.

    The index, the roster and a summary are user contents of their own. Their parts go
    before those of the user's text that follows, the index's first, so no
    provider sees two user messages in a row. When no user text follows (a
    tool result, a model's turn), they become one content.
    A lone index, roster or summary before no user text is left as it is. ``contents``
    are never changed in place.
    """
    head = protected_head(contents)
    if not head:
        return contents
    rest = contents[len(head):]
    parts = [part for content in head for part in (content.parts or [])]
    if rest and _users_text(rest[0]):
        merged = rest[0].model_copy(update={"parts": [*parts, *(rest[0].parts or [])]})
        return [merged, *rest[1:]]
    if len(head) == 1:
        return contents
    return [head[0].model_copy(update={"parts": parts}), *rest]


def drop_turns(
    contents: list[types.Content], *, target_tokens: float, kind: "ProviderKind",
    task: types.Content | None = None, brief: types.Content | None = None,
) -> tuple[list[types.Content], int]:
    """``contents`` without their oldest whole turns, under ``target_tokens`` if it can be.

    A turn starts at a user content with a text and no tool result, after a
    content that made no call: so a call and its result stay together. Turns
    go oldest first, the oldest whatever the target (the caller knows the
    request is too big), until the rest, with the note, is under the target;
    the newest turn always stays, and so do the index, the roster and a
    summary that open the conversation. The first kept user content gets ``FIT_NOTE`` as a part
    before its own; ``contents`` themselves are left as they are.

    ``task`` is a sub-agent run's task message (its invocation's user
    content), which stays too: a message steered to the run after a tool
    result starts a turn, and once the turns go past the task message, the
    first kept one's opening content joins it, after the note, as one user
    content. An inheriting run's history before it goes first.

    ``brief`` is the run's instance's brief, which stays as the task message
    does (a follow-up's task message is its own message, and the brief the
    turn before it): what is kept opens with the first of them, the note,
    the other when the cut went past it too, then the first kept turn's
    opening content. One content when they are the same message (a first
    run's).

    Each content is measured once; what goes is subtracted from a running
    sum.

    Returns:
        The contents to send, and how many turns were left out (0, with
        ``contents`` themselves, when there is only one turn).
    """
    head = protected_head(contents)
    body = contents[len(head):]
    starts = [i for i in range(len(body)) if _starts_turn(body, i)]
    if not starts or starts[-1] == 0:
        return contents, 0
    kept = sorted({at for at in (_brief_at(body, brief), _task_at(body, task)) if at is not None})
    sizes = [len(content.model_dump_json(exclude_none=True)) for content in contents]
    remaining = sum(sizes) + _NOTE_CHARS
    sizes = sizes[len(head):]
    # A partial turn before the first start goes with the first turn.
    cuts = [cut for cut in starts if cut > 0]
    previous = 0
    for dropped, cut in enumerate(cuts, start=1):
        remaining -= sum(sizes[previous:cut])
        # The brief and the task message stay.
        remaining += sum(sizes[at] for at in kept if previous <= at < cut)
        previous = cut
        if _tokens(remaining, kind) < target_tokens:
            break
    before = [body[at] for at in kept if at < cut]
    if before:
        return [*head, _joined(before, body[cut]), *body[cut + 1:]], dropped
    return [*head, _noted(body[cut]), *body[cut + 1:]], dropped


def keep_newest_turns(
    contents: list[types.Content], turns: int
) -> tuple[list[types.Content], int]:
    """``contents`` from their ``turns``-th newest turn on, cut as ``drop_turns`` cuts.

    The index, the roster and a summary that open them stay, and the first kept user
    content gets ``FIT_NOTE``. For a retry after a summary: the turns the
    summary kept, what every later call sends besides it.

    Returns:
        The contents, and how many turns were left out (0, with ``contents``
        themselves, when they hold no more than ``turns``).
    """
    head = protected_head(contents)
    body = contents[len(head):]
    starts = [i for i in range(len(body)) if _starts_turn(body, i)]
    if turns <= 0 or len(starts) <= turns:
        return contents, 0
    cut = starts[-turns]
    return [*head, _noted(body[cut]), *body[cut + 1:]], len(starts) - turns


def _noted(content: types.Content) -> types.Content:
    """``content`` with ``FIT_NOTE`` as a part before its own."""
    return content.model_copy(update={"parts": [types.Part.from_text(text=FIT_NOTE),
                                                *(content.parts or [])]})


def _joined(kept: Sequence[types.Content], content: types.Content) -> types.Content:
    """``content`` after the first of ``kept``, ``FIT_NOTE`` and the rest of
    ``kept``, as one user content."""
    first, *rest = kept
    return content.model_copy(update={"parts": [
        *(first.parts or []),
        types.Part.from_text(text=FIT_NOTE),
        *(part for other in rest for part in other.parts or []),
        *(content.parts or []),
    ]})


def _user_contents_like(contents: list[types.Content], like: types.Content | None) -> list[int]:
    """Where the user contents with ``like``'s parts are in ``contents``."""
    if like is None:
        return []
    return [
        i for i, content in enumerate(contents)
        if content.role == "user" and content.parts == like.parts
    ]


def _task_at(contents: list[types.Content], task: types.Content | None) -> int | None:
    """Where ``task`` (a run's task message) is in ``contents``: the last user
    content with its parts; None for none."""
    found = _user_contents_like(contents, task)
    return found[-1] if found else None


def _brief_at(contents: list[types.Content], brief: types.Content | None) -> int | None:
    """Where ``brief`` (an instance's first message) is in ``contents``: the
    first user content with its parts; None for none."""
    found = _user_contents_like(contents, brief)
    return found[0] if found else None


def brief_of(events: Sequence["Event"]) -> types.Content | None:
    """The brief of the instance whose child session holds ``events``: the
    content of the first event whose state delta set ``BRIEF_KEY`` (its first
    run's message), or None.

    A copied conversation (an inheriting instance's) carries no state delta,
    so a parent's brief is never taken for the child's.
    """
    for event in events:
        delta = event.actions.state_delta if event.actions is not None else None
        if delta and BRIEF_KEY in delta:
            return event.content
    return None


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


def is_summary(content: types.Content) -> bool:
    """Whether ``content`` opens with the summary ADK sends once the
    conversation was summarized: a user content whose first part it is
    (folded into a user's text or not)."""
    first = content.parts[0] if content.parts else None
    return content.role == "user" and bool(first and first.text) and first.text.startswith(
        SUMMARY_PREFIX
    )


def is_memory_index(content: types.Content) -> bool:
    """Whether ``content`` is the memory index the main agent's requests open with."""
    first = content.parts[0] if content.parts else None
    return content.role == "user" and bool(first and first.text) and first.text.startswith(INDEX_OPEN)


def is_roster(content: types.Content) -> bool:
    """Whether ``content`` is the roster of sub-agents the main agent's requests carry."""
    first = content.parts[0] if content.parts else None
    return content.role == "user" and bool(first and first.text) and first.text.startswith(ROSTER_OPEN)


def protected_head(contents: list[types.Content]) -> list[types.Content]:
    """What opens ``contents`` and is never dropped: the index, the roster, then a summary."""
    count = 0
    for opens in (is_memory_index, is_roster, is_summary):
        if count < len(contents) and opens(contents[count]):
            count += 1
    return contents[:count]


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

"""How long each model call takes, on ADK: a clock per call.

ADK makes an ``EventActions`` for each model call and hands that object to
all three of its model callbacks (before, after, on error), so a call is
timed under its identity, as the fit guard keeps its copy of a request:
parallel delegations to one sub-agent share their invocation and their
agent's name, and a clock kept by either timed one call from another's
request. The clock holds the object, so its id goes to no other call while
this one is out. A failed call keeps its clock: a plugin after the one that
reads it may answer the error (the overflow handler), and that answer reaches
``after_model_callback`` like any response. Each run lets go of what another
invocation left: an error nobody answered, and a call cancelled mid-request,
which before google-adk 2.14 fires no callback.

``CallClock`` is that clock; the raw log (``LLMLoggingPlugin``) keeps one.
Every app times every call with two plugins sharing another (the manager's
``_plugin_chain``): ``CallSentPlugin``, last, starts it as the request goes
out, after every plugin that edits the request; ``CallLatencyPlugin``, at
the head, stops it as the answer comes in, before any plugin could replace
the answer, and marks the answer with the milliseconds
(``custom_metadata[LATENCY_KEY]``), which ADK copies onto the answer's event,
so the event processor gives the call's ``LLM_USAGE`` its ``latency_ms``. An
answer already marked keeps its mark: the overflow handler's retry is timed
around its own request, as Deep Agents times every model call. No adapter
sends ``custom_metadata`` to a model.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

from google.adk.plugins.base_plugin import BasePlugin

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.models import LlmRequest, LlmResponse

#: The key an answer carries its call's latency under, in its
#: ``custom_metadata`` (and so its event's), in milliseconds.
LATENCY_KEY = "agentic_cli_latency_ms"


class _Clock(NamedTuple):
    """When one model call's request went out, under the id of the call's
    ``EventActions``: holding the object keeps that id from going to another
    call while this one is out."""

    actions: Any
    invocation: str | None
    started: float


class CallClock:
    """One clock per model call, kept under the identity of its ``EventActions``."""

    def __init__(self) -> None:
        # id(a model call's EventActions) -> when its request went out.
        self._clocks: dict[int, _Clock] = {}

    def start(self, callback_context: "CallbackContext") -> None:
        """Start the clock of the call ``callback_context`` is for."""
        actions = getattr(callback_context, "actions", None)
        if actions is not None:
            self._clocks[id(actions)] = _Clock(
                actions, getattr(callback_context, "invocation_id", None), time.perf_counter()
            )

    def elapsed(self, callback_context: "CallbackContext", *, done: bool) -> float | None:
        """Milliseconds since the call's request went out, None if it was not
        timed; its clock is let go once the call is ``done``."""
        actions = getattr(callback_context, "actions", None)
        if actions is None:
            return None
        clocks = self._clocks
        clock = clocks.pop(id(actions), None) if done else clocks.get(id(actions))
        return None if clock is None else (time.perf_counter() - clock.started) * 1000

    def keep_only(self, invocation: str | None) -> None:
        """Let go of every clock another invocation left (turns run one at a time)."""
        self._clocks = {
            key: clock for key, clock in self._clocks.items() if clock.invocation == invocation
        }

    def clear(self) -> None:
        """Let go of every clock."""
        self._clocks.clear()

    def __len__(self) -> int:
        """How many calls are out: timed, and not yet answered."""
        return len(self._clocks)


def mark_latency(llm_response: "LlmResponse", latency_ms: float) -> None:
    """Mark ``llm_response`` with its call's latency, unless it carries one already."""
    metadata = dict(llm_response.custom_metadata or {})
    metadata.setdefault(LATENCY_KEY, latency_ms)
    llm_response.custom_metadata = metadata


def call_latency(adk_event: Any) -> float | None:
    """The latency an answer's event carries (``LATENCY_KEY``), in milliseconds; None without."""
    latency = (getattr(adk_event, "custom_metadata", None) or {}).get(LATENCY_KEY)
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        return None
    return float(latency)


class CallLatencyPlugin(BasePlugin):
    """Marks each model call's answer with how long the call took (see the module docstring).

    At the head of the chain, so its ``after_model_callback`` runs before
    any plugin that could replace the answer. It changes the answer in place
    and returns None, so every plugin after it still runs.
    """

    def __init__(self, clock: CallClock) -> None:
        super().__init__(name="call_latency")
        self.clock = clock

    async def after_model_callback(
        self, *, callback_context: "CallbackContext", llm_response: "LlmResponse"
    ) -> Optional["LlmResponse"]:
        if llm_response.partial is True:
            return None
        latency = self.clock.elapsed(callback_context, done=True)
        if latency is not None:
            mark_latency(llm_response, latency)
        return None

    async def on_model_error_callback(
        self, *, callback_context: "CallbackContext", llm_request: "LlmRequest",
        error: Exception,
    ) -> Optional["LlmResponse"]:
        # The clock stays: an answer to the error (the overflow handler's)
        # reaches after_model_callback, timed, and lets it go.
        return None

    async def before_run_callback(
        self, *, invocation_context: "InvocationContext"
    ) -> None:
        self.clock.keep_only(invocation_context.invocation_id)
        return None


class CallSentPlugin(BasePlugin):
    """Starts each model call's clock as its request goes out: last in the chain."""

    def __init__(self, clock: CallClock) -> None:
        super().__init__(name="call_sent")
        self.clock = clock

    async def before_model_callback(
        self, *, callback_context: "CallbackContext", llm_request: "LlmRequest"
    ) -> None:
        self.clock.start(callback_context)
        return None

"""ADK plugin: a sub-agent run's control at each of its model calls (``workflow/agents/control.py``).

In each sub-agent type's plugin chain (``GoogleADKWorkflowManager._agent_plugin_chain``),
for the run in progress (``get_active_run``: the run's own task,
``agents/delegate.py``), whose control the manager's ``Delegations`` holds:

- before each model call the call is counted (``RunControl.model_call``),
  and the run's last allowed one gets ``STEP_LIMIT_NOTE`` after its last
  message, in the request only: as a part of the last content when that is
  the user's (a tool result, the task message), so no provider sees two
  user messages in a row. Its tools stay declared: Claude refuses a request
  whose messages hold tool calls or results when it defines no tools, and no
  call that answer makes is run (below);
- after each, what it used (``workflow/usage.py::adk_tokens``, by its
  model's provider kind) goes to ``RunControl.used``;
- on the run's last call (``RunControl.final_call``: its step limit's, or
  one past its token limit) the answer's function calls are dropped, so ADK
  runs none and the run ends with that answer; an answer that was calls
  alone keeps an empty text, which ADK takes as the run's last answer
  (``MODEL_RETURNED_NO_CONTENT``, no failure for a run).

It changes the request and the answer in place and returns None, as
``ForeignReasoningPlugin`` does, so every plugin after it still runs. It sits
before ``SubAgentOutcomePlugin``, which then reads an answer whose calls were
dropped as the run's final one. The root's calls (no active run) and a run
without a control are left alone.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from agentic_cli.workflow.agents.control import STEP_LIMIT_NOTE
from agentic_cli.workflow.usage import adk_tokens

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse

    from agentic_cli.workflow.agents.control import RunControl


class RunControlPlugin(BasePlugin):
    """A sub-agent run's limits at its model calls (see the module docstring).

    Args:
        control: The control of the run in progress (None: no run, or none
            to control), asked at each callback.
        kind: The provider kind of the type's model (``ProviderKind``'s
            value), which its usage is read by; None when unknown.
    """

    def __init__(self, control: Callable[[], RunControl | None], kind: str | None) -> None:
        super().__init__(name="run_control")
        self._control = control
        self._kind = kind

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        control = self._control()
        if control is None:
            return None
        if control.model_call():
            llm_request.contents = with_note(list(llm_request.contents or []), STEP_LIMIT_NOTE)
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        if llm_response.partial:
            return None
        control = self._control()
        if control is None:
            return None
        if llm_response.usage_metadata is not None:
            tokens = adk_tokens(llm_response.usage_metadata, self._kind)
            control.used(tokens.input, tokens.output)
        if control.final_call:
            drop_calls(llm_response)
        return None


def with_note(contents: list[types.Content], note: str) -> list[types.Content]:
    """``contents`` with ``note`` after their last message, as the user's.

    A part of the last content when that is the user's (a tool result, the
    task message, a steered message); else a user content of its own.
    ``contents`` are not changed in place.
    """
    text = types.Part.from_text(text=note)
    if contents and contents[-1].role == "user":
        last = contents[-1]
        return [*contents[:-1], last.model_copy(update={"parts": [*(last.parts or []), text]})]
    return [*contents, types.Content(role="user", parts=[text])]


def drop_calls(llm_response: Any) -> None:
    """Drop ``llm_response``'s function calls, in place: it is the run's last answer.

    Its text and thoughts are kept; an answer that was calls alone keeps an
    empty text, so ADK takes it as an answer (none is not one: ADK would
    call the model again).
    """
    content = llm_response.content
    if content is None or not any(part.function_call for part in content.parts or ()):
        return
    kept = [part for part in content.parts or () if not part.function_call]
    if not any(part.text is not None and not part.thought for part in kept):
        kept.append(types.Part.from_text(text=""))
    llm_response.content = types.Content(role=content.role or "model", parts=kept)

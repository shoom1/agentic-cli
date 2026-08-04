"""Correct the model-visible description of ADK's ``transfer_to_agent`` tool.

ADK builds ``TransferToAgentTool``'s declaration from the ``transfer_to_agent``
function's docstring, which (through 1.37.0, the newest 1.x) contains:

    Note:
      For most use cases, you should use TransferToAgentTool instead of this
      function directly.

That paragraph is written for *Python callers*, but it ships to the model as
the tool's description — so the model is told, in the tool it is supposed to
call, to call something else. Gemini 3.1 followed the instruction and emitted
``TransferToAgentTool``, which ADK rejects (``Tool 'TransferToAgentTool' not
found``), and delegation failed outright.

Upstream fixed the docstring in ADK 2.x, which is a major release outside this
project's ``google-adk>=1.34,<2`` pin. So the description is corrected here, on
the prepared request, under conditions narrow enough that the correction simply
stops applying once the installed ADK no longer needs it:

- only when the tool object is **exactly** ``TransferToAgentTool`` — an
  application tool that happens to be named ``transfer_to_agent`` is left alone;
- only when the misleading sentence is actually present — so it is idempotent,
  and an ADK release that fixes the text makes this a no-op;
- only ``description`` is written. The declaration name, parameter schema,
  required fields and ADK's enum of valid agent names are untouched, and the
  upstream function's ``__doc__`` is never mutated (ADK rebuilds the
  declaration per request, so the edit is scoped to one ``LlmRequest``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from google.adk.plugins.base_plugin import BasePlugin

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types

logger = Loggers.workflow()

#: The declaration name ADK uses. Also the name the model must call.
TRANSFER_TOOL_NAME = "transfer_to_agent"

#: The fragment that makes ADK's generated description actively harmful. Its
#: presence is the trigger; its absence means there is nothing to correct.
_MISLEADING_FRAGMENT = "you should use TransferToAgentTool"

#: Replacement: says plainly what to call, and defers to the parameter schema
#: for *which* agents are valid rather than restating them (ADK's enum is the
#: authority and stays intact).
ROUTING_DESCRIPTION = (
    "Route the current request to another agent. "
    f'Call this function by its exact name: {TRANSFER_TOOL_NAME}'
    '(agent_name="<allowed agent>"). '
    "Choose agent_name from the enum of valid agent names in this tool's "
    "parameter schema — those are the only accepted values. "
    "Transfer when another agent's description fits the user's request better "
    "than your own."
)


def _native_transfer_tool_type() -> type | None:
    """ADK's exact ``TransferToAgentTool`` class, or None if unavailable."""
    try:
        from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool
    except ImportError:  # pragma: no cover - ADK always ships it today
        return None
    return TransferToAgentTool


def _function_declarations(llm_request: "LlmRequest") -> Iterator["types.FunctionDeclaration"]:
    """Every function declaration attached to the prepared request."""
    config = getattr(llm_request, "config", None)
    for tool in getattr(config, "tools", None) or []:
        for declaration in getattr(tool, "function_declarations", None) or []:
            yield declaration


def correct_transfer_declaration(llm_request: "LlmRequest") -> bool:
    """Rewrite the transfer tool's description on this request, if warranted.

    Returns:
        True if a declaration was corrected (useful for tests and logging).
    """
    transfer_type = _native_transfer_tool_type()
    if transfer_type is None:
        return False

    tool = (getattr(llm_request, "tools_dict", None) or {}).get(TRANSFER_TOOL_NAME)
    # Exact type, never isinstance: a subclass may declare anything it likes,
    # and an application tool sharing the name is not ADK's routing primitive.
    if type(tool) is not transfer_type:
        return False

    corrected = False
    for declaration in _function_declarations(llm_request):
        if declaration.name != TRANSFER_TOOL_NAME:
            continue
        if _MISLEADING_FRAGMENT not in (declaration.description or ""):
            continue  # already correct (or fixed upstream) — leave it alone
        declaration.description = ROUTING_DESCRIPTION
        corrected = True

    return corrected


class TransferToolDescriptionPlugin(BasePlugin):
    """Applies :func:`correct_transfer_declaration` to every model request."""

    def __init__(self) -> None:
        super().__init__(name="transfer_tool_description")
        self._corrections = 0

    @property
    def corrections(self) -> int:
        """How many requests have been corrected (diagnostics/tests)."""
        return self._corrections

    async def before_model_callback(
        self,
        *,
        callback_context: "CallbackContext",
        llm_request: "LlmRequest",
    ) -> None:
        """Fix the transfer description in place; never blocks the request."""
        try:
            if correct_transfer_declaration(llm_request):
                self._corrections += 1
        except Exception as exc:  # noqa: BLE001 - never break a turn over this
            logger.warning("transfer_description_correction_failed", error=str(exc))
        return None

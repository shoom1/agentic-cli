"""ADK's built-in agent-transfer tool: permission identity and description.

Two verified defects blocked multi-agent delegation, and both are covered here.

1. **Permission identity.** ADK auto-injects an exact
   ``TransferToAgentTool`` into any agent with ``sub_agents``. It is a
   ``FunctionTool`` *subclass*, and the plugin only unwrapped exact
   ``FunctionTool``/``LongRunningFunctionTool``, so the routing primitive
   resolved to ``_UNVERIFIED`` and was denied as unregistered — even with
   permissions disabled. Delegation could not work at all.

2. **Model-visible description.** Through 1.37, ADK built the declaration from
   a ``transfer_to_agent`` docstring that advised callers to "use
   TransferToAgentTool instead of this function directly". The model followed
   that advice and emitted ``TransferToAgentTool``, which ADK rejects, so a
   plugin rewrote the description. ADK 2 fixed the docstring and the plugin
   went; the guard at the end fails if a later ADK brings the advice back.

The identity fix must not become a hole: adding a type to the trusted list is
exactly the kind of change that can quietly re-admit forged tools, so the
negative cases are asserted alongside the positive one.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("google.adk")

from google.adk.tools import FunctionTool, LongRunningFunctionTool  # noqa: E402
from google.adk.tools.transfer_to_agent_tool import (  # noqa: E402
    TransferToAgentTool,
    transfer_to_agent,
)

from agentic_cli.workflow.adk.permission_plugin import (  # noqa: E402
    _UNVERIFIED,
    PermissionPlugin,
)
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402

#: The declaration name ADK gives its routing tool, and the name the model calls.
TRANSFER_TOOL_NAME = "transfer_to_agent"

AGENT_NAMES = ["arxiv_specialist", "data_analyst", "report_writer"]


def _tool() -> TransferToAgentTool:
    return TransferToAgentTool(agent_names=list(AGENT_NAMES))


def _description_is_safe(description: str) -> bool:
    """No advice to call the class instead of the function."""
    return (
        "TransferToAgentTool" not in description
        and "instead of this function" not in description
    )


# ---------------------------------------------------------------------------
# 1. Permission identity
# ---------------------------------------------------------------------------


class TestTransferToolResolvesToItsRegisteredDefinition:
    def test_exact_transfer_tool_resolves_to_the_exempt_definition(self):
        from agentic_cli.tools.registry import identify_tool

        defn = PermissionPlugin._resolve_definition(_tool())

        assert defn is not _UNVERIFIED, (
            "ADK's built-in transfer tool was denied as unregistered"
        )
        assert defn is identify_tool(transfer_to_agent), (
            "resolved to something other than the registered transfer_to_agent"
        )
        assert defn.name == TRANSFER_TOOL_NAME
        assert defn.capabilities is EXEMPT, (
            "the routing primitive must stay EXEMPT, not acquire capabilities"
        )

    async def test_before_tool_callback_allows_it(self):
        result = await PermissionPlugin().before_tool_callback(
            tool=_tool(),
            tool_args={"agent_name": "arxiv_specialist"},
            tool_context=None,
        )
        assert result is None, f"the transfer tool was gated: {result}"


class TestIdentityGuaranteesSurvive:
    """Widening the trusted-type list must not admit anything else."""

    async def _denied(self, tool: Any) -> None:
        assert PermissionPlugin._resolve_definition(tool) is _UNVERIFIED
        result = await PermissionPlugin().before_tool_callback(
            tool=tool, tool_args={}, tool_context=None
        )
        assert result is not None and result.get("success") is False
        assert "not registered" in result.get("error", "")

    async def test_arbitrary_function_tool_subclass_is_denied(self):
        class SneakyTool(FunctionTool):
            """A subclass may override run_async and do anything."""

        await self._denied(SneakyTool(func=transfer_to_agent))

    async def test_transfer_tool_subclass_is_denied(self):
        class SubclassedTransfer(TransferToAgentTool):
            pass

        await self._denied(SubclassedTransfer(agent_names=list(AGENT_NAMES)))

    async def test_forged_object_named_transfer_to_agent_is_denied(self):
        forged = SimpleNamespace(name=TRANSFER_TOOL_NAME)
        await self._denied(forged)

    async def test_forged_object_named_after_the_class_is_denied(self):
        class TransferToAgentTool:  # shadows the real name deliberately
            name = TRANSFER_TOOL_NAME

        await self._denied(TransferToAgentTool())

    async def test_forged_object_carrying_the_real_func_is_denied(self):
        """A copied ``.func`` is an attribute, not evidence of behaviour."""
        forged = SimpleNamespace(name=TRANSFER_TOOL_NAME, func=transfer_to_agent)
        await self._denied(forged)

    async def test_long_running_function_tool_still_resolves(self):
        """The pre-existing trusted types keep working."""
        from agentic_cli.tools.registry import identify_tool

        defn = PermissionPlugin._resolve_definition(
            LongRunningFunctionTool(func=transfer_to_agent)
        )
        assert defn is identify_tool(transfer_to_agent)


# ---------------------------------------------------------------------------
# 2. Model-visible description
# ---------------------------------------------------------------------------


class TestTheInstalledAdkTransferTool:
    """ADK's own declaration is safe for the model, with no correction applied."""

    def test_installed_adk_transfer_declaration_is_safe(self):
        declaration = _tool()._get_declaration()

        assert declaration.name == TRANSFER_TOOL_NAME
        assert _description_is_safe(declaration.description or ""), (
            f"the model would be told to call the class: {declaration.description!r}"
        )
        schema = declaration.parameters_json_schema
        assert schema["properties"]["agent_name"]["enum"] == AGENT_NAMES
        assert schema["required"] == ["agent_name"]

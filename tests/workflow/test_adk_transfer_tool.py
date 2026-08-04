"""ADK's built-in agent-transfer tool: permission identity and description.

Two verified defects blocked multi-agent delegation, and both are covered here.

1. **Permission identity.** ADK auto-injects an exact
   ``TransferToAgentTool`` into any agent with ``sub_agents``. It is a
   ``FunctionTool`` *subclass*, and the plugin only unwrapped exact
   ``FunctionTool``/``LongRunningFunctionTool``, so the routing primitive
   resolved to ``_UNVERIFIED`` and was denied as unregistered — even with
   permissions disabled. Delegation could not work at all.

2. **Model-visible description.** ADK builds the declaration from the
   ``transfer_to_agent`` docstring, which advises callers to "use
   TransferToAgentTool instead of this function directly". The model followed
   that advice and emitted ``TransferToAgentTool``, which ADK rejects.

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
from google.genai import types  # noqa: E402

from agentic_cli.workflow.adk.permission_plugin import (  # noqa: E402
    _UNVERIFIED,
    PermissionPlugin,
)
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from agentic_cli.workflow.adk.transfer_tool_description import (  # noqa: E402
    ROUTING_DESCRIPTION,
    TRANSFER_TOOL_NAME,
    TransferToolDescriptionPlugin,
    correct_transfer_declaration,
)

AGENT_NAMES = ["arxiv_specialist", "data_analyst", "report_writer"]


def _tool() -> TransferToAgentTool:
    return TransferToAgentTool(agent_names=list(AGENT_NAMES))


def _request(tool: Any, declaration: types.FunctionDeclaration | None = None):
    """A minimal LlmRequest carrying one tool and its declaration."""
    from google.adk.models.llm_request import LlmRequest

    if declaration is None:
        declaration = tool._get_declaration()
    return LlmRequest(
        config=types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=[declaration])]
        ),
        tools_dict={declaration.name: tool},
    )


#: A description in the shape ADK 1.x generates. The correction-path tests use
#: *this*, not whatever the installed ADK happens to produce: the production
#: workaround is deliberately a no-op once upstream fixes its docstring (2.x
#: already has), so tests that required the installed declaration to be broken
#: would start failing the day the pinned range picks up the fix. What must keep
#: working is the correction itself, given input that needs correcting.
MISLEADING_DESCRIPTION = (
    "Transfer the question to another agent.\n\n"
    "This tool hands off control to another agent when it's more suitable to\n"
    "answer the user's question according to the agent's description.\n\n"
    "Note:\n"
    "  For most use cases, you should use TransferToAgentTool instead of this\n"
    "  function directly. TransferToAgentTool provides additional enum "
    "constraints\n"
    "  that prevent LLMs from hallucinating invalid agent names.\n\n"
    "Args:\n"
    "  agent_name: the agent name to transfer to."
)

SAFE_DESCRIPTION = "Transfer the query to another agent."


def _declaration_with(description: str) -> types.FunctionDeclaration:
    """The native tool's real declaration, with the description swapped.

    Name, parameters, required fields and ADK's agent-name enum all come from
    the genuine tool, so schema assertions stay meaningful.
    """
    declaration = _tool()._get_declaration()
    declaration.description = description
    return declaration


def _misleading_request():
    """An exact native tool whose declaration needs correcting."""
    return _request(_tool(), _declaration_with(MISLEADING_DESCRIPTION))


def _description_is_safe(description: str) -> bool:
    """No advice to call the class, under either ADK's wording or ours."""
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


class TestTransferDeclarationIsCorrected:
    """The correction path, driven by a synthetic misleading declaration.

    These never depend on the installed ADK still being broken — only on the
    correction doing the right thing when handed input that needs it.
    """

    def test_exactly_one_transfer_declaration_with_the_right_name(self):
        request = _misleading_request()
        correct_transfer_declaration(request)

        decls = [
            d
            for tool in request.config.tools
            for d in tool.function_declarations
            if d.name == TRANSFER_TOOL_NAME
        ]
        assert len(decls) == 1
        assert decls[0].name == TRANSFER_TOOL_NAME

    def test_description_no_longer_advises_calling_the_class(self):
        request = _misleading_request()
        assert correct_transfer_declaration(request) is True

        description = request.config.tools[0].function_declarations[0].description
        assert _description_is_safe(description)

    def test_description_states_the_callable_name(self):
        request = _misleading_request()
        correct_transfer_declaration(request)

        description = request.config.tools[0].function_declarations[0].description
        assert f'{TRANSFER_TOOL_NAME}(agent_name="' in description

    def test_agent_name_enum_and_schema_are_preserved(self):
        before = _tool()._get_declaration().parameters.model_dump(exclude_none=True)

        request = _misleading_request()
        correct_transfer_declaration(request)
        after = request.config.tools[0].function_declarations[0].parameters

        assert after.properties["agent_name"].enum == AGENT_NAMES
        assert after.required == ["agent_name"]
        assert after.model_dump(exclude_none=True) == before

    def test_correction_is_idempotent(self):
        request = _misleading_request()
        assert correct_transfer_declaration(request) is True
        first = request.config.tools[0].function_declarations[0].description

        # A second pass has nothing left to do.
        assert correct_transfer_declaration(request) is False
        assert request.config.tools[0].function_declarations[0].description == first

    def test_already_correct_description_is_left_alone(self):
        """An ADK release that ships a sane docstring needs no correction."""
        request = _request(_tool(), _declaration_with(SAFE_DESCRIPTION))

        assert correct_transfer_declaration(request) is False
        assert (
            request.config.tools[0].function_declarations[0].description
            == SAFE_DESCRIPTION
        )

    def test_the_upstream_function_doc_is_never_mutated(self):
        before = transfer_to_agent.__doc__
        correct_transfer_declaration(_misleading_request())
        assert transfer_to_agent.__doc__ == before


class TestAgainstTheInstalledAdk:
    """One integration check that works whichever docstring ADK ships.

    Either the installed declaration is misleading and the correction fixes it,
    or it is already safe and no correction is needed. Both are acceptable; a
    declaration that is still misleading *after* the plugin has run is not.
    """

    def test_installed_declaration_ends_up_safe(self):
        request = _request(_tool())
        initial = request.config.tools[0].function_declarations[0].description or ""

        corrected = correct_transfer_declaration(request)
        final = request.config.tools[0].function_declarations[0].description or ""

        if _description_is_safe(initial):
            assert corrected is False, (
                "an already-safe ADK description was rewritten anyway"
            )
        else:
            assert corrected is True, (
                "the installed ADK description advises calling the class and "
                "was not corrected"
            )

        assert _description_is_safe(final), (
            f"the model would still be told to call the class: {final!r}"
        )
        # Whichever branch ran, the contract the model needs is intact.
        declaration = request.config.tools[0].function_declarations[0]
        assert declaration.name == TRANSFER_TOOL_NAME
        assert declaration.parameters.properties["agent_name"].enum == AGENT_NAMES
        assert declaration.parameters.required == ["agent_name"]


class TestOnlyTheNativeToolIsRewritten:
    def test_application_tool_with_the_same_name_is_untouched(self):
        """A same-named application tool is not ADK's routing primitive."""

        def transfer_to_agent(agent_name: str) -> dict:  # noqa: A001 - deliberate
            """You should use TransferToAgentTool instead of this function."""
            return {"success": True}

        app_tool = FunctionTool(func=transfer_to_agent)
        declaration = app_tool._get_declaration()
        original = declaration.description

        request = _request(app_tool, declaration)
        assert correct_transfer_declaration(request) is False
        assert request.config.tools[0].function_declarations[0].description == original

    def test_impostor_base_tool_named_transfer_to_agent_is_untouched(self):
        """A real BaseTool that merely takes the name is not ADK's tool.

        ``LlmRequest.tools_dict`` is typed to ``BaseTool``, so the realistic
        impostor is a genuine tool object with the right name — which is the
        case the exact-type check has to reject.
        """
        from google.adk.tools import BaseTool

        class ImpostorTool(BaseTool):
            def __init__(self) -> None:
                super().__init__(name=TRANSFER_TOOL_NAME, description="impostor")

        declaration = types.FunctionDeclaration(
            name=TRANSFER_TOOL_NAME,
            description="you should use TransferToAgentTool instead",
        )

        request = _request(ImpostorTool(), declaration)
        assert correct_transfer_declaration(request) is False
        assert (
            request.config.tools[0].function_declarations[0].description
            == "you should use TransferToAgentTool instead"
        )

    def test_subclass_of_the_native_tool_is_untouched(self):
        class SubclassedTransfer(TransferToAgentTool):
            pass

        tool = SubclassedTransfer(agent_names=list(AGENT_NAMES))
        request = _request(tool)
        assert correct_transfer_declaration(request) is False


class TestPluginWiring:
    async def test_plugin_corrects_the_request_and_counts_it(self):
        """Deterministic: driven by a synthetic misleading declaration."""
        plugin = TransferToolDescriptionPlugin()
        request = _misleading_request()

        await plugin.before_model_callback(
            callback_context=SimpleNamespace(invocation_id="i1"),
            llm_request=request,
        )

        assert plugin.corrections == 1
        description = request.config.tools[0].function_declarations[0].description
        assert description == ROUTING_DESCRIPTION

    async def test_plugin_does_not_count_an_already_safe_request(self):
        plugin = TransferToolDescriptionPlugin()
        request = _request(_tool(), _declaration_with(SAFE_DESCRIPTION))

        await plugin.before_model_callback(
            callback_context=SimpleNamespace(invocation_id="i1"),
            llm_request=request,
        )

        assert plugin.corrections == 0

    async def test_plugin_never_blocks_a_request(self):
        """A malformed request must not fail the turn."""
        plugin = TransferToolDescriptionPlugin()
        broken = SimpleNamespace(tools_dict={TRANSFER_TOOL_NAME: _tool()}, config=None)

        result = await plugin.before_model_callback(
            callback_context=SimpleNamespace(invocation_id="i1"), llm_request=broken
        )
        assert result is None

    def test_manager_registers_the_plugin(self):
        """The correction must actually be wired into the runner."""
        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

        manager = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
        manager._settings = SimpleNamespace(
            raw_llm_logging=False, app_name="t", verbose_thinking=False
        )

        plugins = manager._init_plugins()

        assert any(isinstance(p, TransferToolDescriptionPlugin) for p in plugins)

"""Backend-native tool variants are explicit, and bare names are deterministic.

The ADK and LangGraph state tools (``save_plan``/``get_plan``/``save_tasks``/
``get_tasks``) are the same *tool* with two native implementations: different
signatures (``ToolContext`` vs ``InjectedState``/``Command``) and different
docstrings, hence different model-visible schemas.

They used to contest the registry name, and whichever module imported second
won — silently, because their permission metadata happened to match. So
``ToolDefinition.func`` (what a bare ``"save_plan"`` in an ``AgentConfig``
resolves to) depended on import order, and could hand an ADK agent a LangGraph
tool. They now register as declared *variants* of a backend-neutral contract,
and a bare name that has no neutral implementation fails as ambiguous instead
of guessing.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest


def _probe(*imports: str) -> dict:
    """Import the given modules, in order, in a fresh interpreter."""
    script = textwrap.dedent(
        """
        import json
        {imports}
        from agentic_cli.tools.registry import get_registry, identify_tool
        from agentic_cli.tools.tool_resolver import resolve_tool

        registry = get_registry()
        definition = registry.get("save_plan")
        out = {{
            "declared": definition is not None,
            "has_neutral_implementation": bool(
                definition is not None and definition.func is not None
            ),
            "variants": len(definition.variants) if definition else 0,
            "capabilities_exempt": definition is not None
            and not isinstance(definition.capabilities, list),
        }}
        try:
            resolve_tool("save_plan")
            out["bare_name"] = "resolved"
        except ValueError as exc:
            out["bare_name"] = "ambiguous" if "ambiguous" in str(exc) else "error"

        bound = []
        for module_name, attr in (
            ("agentic_cli.tools.adk.state_tools", "save_plan"),
            ("agentic_cli.tools.langgraph.state_tools", "save_plan"),
        ):
            import importlib
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            bound.append(identify_tool(getattr(module, attr)) is definition)
        out["all_variants_bound"] = bool(bound) and all(bound)
        print(json.dumps(out))
        """
    ).format(imports="\n".join(imports))
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


ADK_FIRST = (
    "import agentic_cli.tools.adk.state_tools",
    "import agentic_cli.tools.langgraph.state_tools",
)
LANGGRAPH_FIRST = (
    "import agentic_cli.tools.langgraph.state_tools",
    "import agentic_cli.tools.adk.state_tools",
)


@pytest.fixture(scope="module", autouse=True)
def _require_langgraph():
    pytest.importorskip("langgraph")


class TestImportOrderIsIrrelevant:
    """The registry must look the same whichever backend module loads first."""

    def test_both_orders_agree(self):
        assert _probe(*ADK_FIRST) == _probe(*LANGGRAPH_FIRST)

    def test_both_orders_register_two_variants(self):
        for order in (ADK_FIRST, LANGGRAPH_FIRST):
            out = _probe(*order)
            assert out["declared"] is True
            assert out["variants"] == 2, out
            assert out["all_variants_bound"] is True

    def test_bare_name_is_ambiguous_in_both_orders(self):
        for order in (ADK_FIRST, LANGGRAPH_FIRST):
            out = _probe(*order)
            assert out["has_neutral_implementation"] is False
            assert out["bare_name"] == "ambiguous", out

    def test_a_single_backend_still_registers_its_variant(self):
        out = _probe("import agentic_cli.tools.adk.state_tools")
        assert out["declared"] is True
        assert out["variants"] == 1
        assert out["bare_name"] == "ambiguous"


class TestVariantsShareOneContract:
    """In-process: both variants gate under the same declared capabilities."""

    def test_both_variants_resolve_to_the_declared_tool(self):
        from agentic_cli.tools.adk import state_tools as adk_state_tools
        from agentic_cli.tools.langgraph import state_tools as lg_state_tools
        from agentic_cli.tools.registry import get_registry, identify_tool

        definition = get_registry().get("save_plan")
        assert definition is not None
        assert identify_tool(adk_state_tools.save_plan) is definition
        assert identify_tool(lg_state_tools.save_plan) is definition

    def test_the_declaration_has_no_neutral_implementation(self):
        from agentic_cli.tools.registry import get_registry

        definition = get_registry().get("save_plan")
        assert definition.func is None
        assert len(definition.variants) == 2

    def test_bare_name_resolution_names_the_variants(self):
        from agentic_cli.tools.tool_resolver import resolve_tool

        with pytest.raises(ValueError, match="ambiguous") as exc:
            resolve_tool("save_plan")

        message = str(exc.value)
        assert "save_plan" in message
        assert "include_state_tools" in message

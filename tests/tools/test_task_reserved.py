"""``task`` is a framework tool name: the delegation tool (and the parent's other tools).

A coordinator on the Deep Agents backend delegates through a ``task`` tool
that Deep Agents builds. It has no backend-neutral implementation, so the
framework declares the name and its permission contract
(``tools/_core/delegation.py``), and the backend binds the object it builds to
that declaration. The declaration exists as soon as ``agentic_cli.tools`` is
imported, so no application tool can take the name: one name, one tool.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from agentic_cli.tools import register_tool
from agentic_cli.tools._core.delegation import (
    CHECK_AGENTS_TOOL,
    FOLLOW_UP_TOOLS,
    MESSAGE_AGENT_TOOL,
    STOP_AGENT_TOOL,
    TASK_TOOL,
    declare_delegation_tool,
    declare_follow_up_tools,
)
from agentic_cli.tools.registry import (
    ToolCategory,
    bind_tool_identity,
    get_registry,
    identify_tool,
)
from agentic_cli.tools.tool_resolver import resolve_tools
from agentic_cli.workflow.permissions import EXEMPT


def test_task_is_declared_by_the_framework():
    definition = get_registry().get("task")

    assert TASK_TOOL == "task"
    assert definition is not None
    assert definition.func is None  # no backend-neutral implementation
    assert definition.description == "Delegate a task to a sub-agent."
    assert definition.capabilities is EXEMPT
    assert definition.category is ToolCategory.PLANNING


def test_declaring_it_again_is_idempotent():
    before = get_registry().get("task")

    assert declare_delegation_tool() is before
    assert get_registry().get("task") is before


class TestTheNameIsReserved:
    def test_an_application_tool_named_task_raises(self):
        def run_task(description: str) -> dict:
            """An application's own delegation."""
            return {"success": True}

        with pytest.raises(ValueError, match="'task' is already registered"):
            register_tool(name="task", capabilities=EXEMPT)(run_task)

    def test_a_function_named_task_raises(self):
        with pytest.raises(ValueError, match="'task' is already registered"):

            @register_tool(capabilities=EXEMPT)
            def task(description: str) -> dict:
                """Shadows the delegation tool."""
                return {"success": True}

    def test_it_is_reserved_before_any_tool_registers(self, tmp_path):
        """A fresh interpreter: the first thing an application can import
        (the registry) already has ``task`` declared."""
        code = textwrap.dedent(
            """
            from agentic_cli.tools.registry import register_tool
            from agentic_cli.workflow.permissions import EXEMPT

            def task(description: str) -> dict:
                \"\"\"An application tool.\"\"\"
                return {}

            try:
                register_tool(capabilities=EXEMPT)(task)
            except ValueError as error:
                print("refused:", error)
            else:
                print("registered")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            timeout=120,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("refused: Tool 'task' is already registered")

    def test_a_bare_name_reference_resolves_to_nothing(self):
        with pytest.raises(ValueError, match="'task' is ambiguous"):
            resolve_tools(["task"])


def test_identify_tool_resolves_a_bound_object_to_the_declaration():
    class DelegationTool:
        """Stands in for the object Deep Agents builds."""

    tool = DelegationTool()
    definition = get_registry().get("task")

    assert identify_tool(tool) is None
    bind_tool_identity(tool, definition)

    assert identify_tool(tool) is definition
    assert identify_tool(DelegationTool()) is None


# -- the parent's tools: message_agent, stop_agent, check_agents ----------------------------


class TestTheParentsToolsAreReserved:
    """``message_agent``, ``stop_agent`` and ``check_agents`` are declared
    beside ``task``, exempt, with no backend-neutral callable: each backend
    builds its own and binds it to its declaration."""

    def test_their_names(self):
        assert (MESSAGE_AGENT_TOOL, STOP_AGENT_TOOL, CHECK_AGENTS_TOOL) == (
            "message_agent", "stop_agent", "check_agents",
        )
        assert FOLLOW_UP_TOOLS == ("message_agent", "stop_agent", "check_agents")

    @pytest.mark.parametrize("name, description", [
        ("message_agent", "Send a message to one of your sub-agents."),
        ("stop_agent", "Stop one of your sub-agents."),
        ("check_agents", "Check on your sub-agents."),
    ])
    def test_each_is_declared_by_the_framework(self, name, description):
        definition = get_registry().get(name)

        assert definition is not None
        assert definition.func is None
        assert definition.description == description
        assert definition.capabilities is EXEMPT
        assert definition.category is ToolCategory.PLANNING

    def test_declaring_them_again_is_idempotent(self):
        before = tuple(get_registry().get(name) for name in FOLLOW_UP_TOOLS)

        assert declare_follow_up_tools() == before

    @pytest.mark.parametrize("name", ["message_agent", "stop_agent", "check_agents"])
    def test_an_application_tool_of_one_of_their_names_raises(self, name):
        def run(agent: str) -> dict:
            """An application's own."""
            return {"success": True}

        with pytest.raises(ValueError, match=f"'{name}' is already registered"):
            register_tool(name=name, capabilities=EXEMPT)(run)

        with pytest.raises(ValueError, match=f"'{name}' is ambiguous"):
            resolve_tools([name])

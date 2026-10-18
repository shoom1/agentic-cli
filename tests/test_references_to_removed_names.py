"""References to names 0.7.0 removed fail clearly, and nothing the model reads names them.

A config, a tool declaration or a prompt written for 0.6 can still name a
removed tool or service key. Each must fail with a message that points at the
replacement, never resolve to something else or silently do nothing.
"""

from __future__ import annotations

import pytest

from agentic_cli.tools.tool_resolver import resolve_tool


def test_a_config_naming_kb_ingest_url_fails_and_names_the_replacement():
    with pytest.raises(ValueError, match="Unknown tool name 'kb_ingest_url'") as exc:
        resolve_tool("kb_ingest_url")
    assert "kb_ingest_file" in str(exc.value)


def test_no_tool_description_or_demo_prompt_names_a_removed_tool(mock_context):
    import agentic_cli.tools  # noqa: F401  (registers the framework tools)
    from agentic_cli.tools.registry import get_registry
    from examples.research_demo.agents import AGENT_CONFIGS

    texts = {
        d.name: f"{d.description}\n{getattr(d.func, '__doc__', '') or ''}"
        for d in get_registry().list_tools()
    }
    texts.update({f"agent {c.name}": c.get_prompt(mock_context.settings) for c in AGENT_CONFIGS})
    for removed in ("kb_ingest_url", "google_search_tool"):
        assert [name for name, text in texts.items() if removed in text] == [], removed


def test_kb_manager_among_other_keys_is_refused_and_names_knowledge_base():
    from agentic_cli.tools.registry import ToolRegistry
    from agentic_cli.workflow.permissions import EXEMPT

    registry = ToolRegistry()
    with pytest.raises(ValueError, match="kb_manager") as exc:

        @registry.register(capabilities=EXEMPT, requires=("knowledge_base", "kb_manager"))
        def _tool() -> dict:
            """Tool."""
            return {"success": True}

    assert "requires='knowledge_base'" in str(exc.value)

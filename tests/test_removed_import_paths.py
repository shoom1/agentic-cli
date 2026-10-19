"""The import paths 0.7.0 removed fail instead of forwarding."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "agentic_cli.knowledge_base",
        "agentic_cli.knowledge_base.manager",
        "agentic_cli.tools.kb_convert",
        "agentic_cli.workflow.adk.transfer_tool_description",
        "agentic_cli.workflow.models",
    ],
)
def test_a_removed_module_cannot_be_imported(module, monkeypatch):
    monkeypatch.delitem(sys.modules, module, raising=False)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_memory_tools_no_longer_exports_a_memory_store():
    with pytest.raises(ImportError):
        from agentic_cli.tools.memory_tools import MemoryStore  # noqa: F401


def test_kb_ingest_url_is_gone():
    import agentic_cli.tools as tools
    from agentic_cli.tools.registry import get_registry

    with pytest.raises(ImportError):
        from agentic_cli.tools import kb_ingest_url  # noqa: F401
    assert "kb_ingest_url" not in tools.__all__
    assert get_registry().get("kb_ingest_url") is None


def test_tools_google_search_tool_is_gone_and_loads_no_adk():
    """Asked for in a fresh interpreter: the lookup must not import ADK first."""
    body = textwrap.dedent("""
        import json, sys
        import agentic_cli.tools as tools
        try:
            tools.google_search_tool
            raised = False
        except AttributeError:
            raised = True
        print(json.dumps({
            "raised": raised,
            "adk": sorted(m for m in sys.modules if m.startswith("google.adk")),
        }))
    """)
    proc = subprocess.run([sys.executable, "-c", body], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])

    assert result == {"raised": True, "adk": []}


@pytest.mark.parametrize("name", ["ModelRegistry", "ModelFamily", "ModelInfo", "DiscoveryState"])
def test_the_model_registry_is_not_exported_from_the_workflow(name):
    import agentic_cli.workflow as workflow

    assert name not in workflow.__all__
    with pytest.raises(AttributeError):
        getattr(workflow, name)


@pytest.mark.parametrize(
    "name",
    [
        "get_model",
        "get_available_models",
        "supports_thinking_effort",
        "check_model",
        "set_model_registry",
        "has_google_key",
        "has_anthropic_key",
        "has_any_api_key",
    ],
)
def test_the_registry_settings_helpers_are_gone(name):
    from agentic_cli.config import BaseSettings

    assert not hasattr(BaseSettings, name)


def test_thinking_effort_levels_constant_is_gone():
    import agentic_cli.workflow.settings as settings_module

    assert not hasattr(settings_module, "THINKING_EFFORT_LEVELS")


def test_the_manager_exposes_its_catalog_not_a_registry():
    from agentic_cli.workflow.base_manager import BaseWorkflowManager

    assert not hasattr(BaseWorkflowManager, "model_registry")
    assert isinstance(BaseWorkflowManager.catalog, property)


@pytest.mark.parametrize(
    "field, value",
    [
        ("effort", "high"),
        ("extra_params", {"service_tier": "auto"}),
        ("request_timeout", 900.0),
        ("max_retries", 5),
    ],
)
def test_direct_anthropic_llm_refuses_the_fields_it_lost(field, value):
    """An ignored field would silently drop the effort or timeout it carried."""
    from pydantic import ValidationError

    from agentic_cli.workflow.adk.anthropic_llm import DirectAnthropicLlm

    with pytest.raises(ValidationError, match=f"{field}\n  Extra inputs are not permitted"):
        DirectAnthropicLlm(model="claude-sonnet-4-5", **{field: value})


def test_the_anthropic_extra_param_client_is_gone():
    from agentic_cli.workflow.adk import anthropic_llm

    assert not hasattr(anthropic_llm, "_ExtraParamClient")
    assert not hasattr(anthropic_llm, "_ExtraParamMessages")


@pytest.mark.parametrize(
    "name",
    [
        "_get_planner",
        "_gemini3_thinking_config",
        "_gemini25_thinking_config",
        "_anthropic_thinking_budget",
        "_anthropic_effort",
        "_anthropic_max_tokens",
        "_build_model_arg",
        "_get_generate_content_config",
        "_generate_config_kwargs_from_settings",
        "_resolve_thinking",
        "_resolve_model_for_config",
    ],
)
def test_the_adk_manager_s_model_helpers_are_gone(name):
    """The ADK model factory builds every agent's model."""
    from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

    assert not hasattr(GoogleADKWorkflowManager, name)


@pytest.mark.parametrize(
    "module, name",
    [
        ("agentic_cli.workflow.adk.manager", "_is_anthropic_model"),
        ("agentic_cli.workflow.adk.manager", "_anthropic_uses_adaptive"),
        ("agentic_cli.workflow.adk.manager", "_claude_version"),
        ("agentic_cli.workflow.factory", "_is_claude_model"),
        ("agentic_cli.cli.workflow_controller", "_is_claude_model"),
    ],
)
def test_the_model_name_guesses_are_gone(module, name):
    assert not hasattr(importlib.import_module(module), name)


@pytest.mark.parametrize("name", ["get_llm", "get_thinking_config", "_THINKING_BUDGETS"])
def test_the_langgraph_builder_s_model_helpers_are_gone(name):
    """The LangChain adapter builds every LangGraph agent's model."""
    pytest.importorskip("langgraph")
    from agentic_cli.workflow.langgraph.graph_builder import LangGraphBuilder

    assert not hasattr(LangGraphBuilder, name)


def test_the_langgraph_model_name_guess_is_gone():
    pytest.importorskip("langgraph")
    from agentic_cli.workflow.langgraph import graph_builder

    assert not hasattr(graph_builder, "_has_thinking_knob")


@pytest.mark.parametrize(
    "name",
    [
        "_resolve_model_for_config",
        "_BUILDABLE_PROVIDER_KINDS",
        "_UNDERSTOOD_THINKING_MODES",
        "_BACKEND_NAME",
    ],
)
def test_the_langgraph_manager_s_model_helpers_are_gone(name):
    pytest.importorskip("langgraph")
    from agentic_cli.workflow.langgraph.manager import LangGraphWorkflowManager

    assert not hasattr(LangGraphWorkflowManager, name)


@pytest.mark.parametrize(
    "name",
    [
        "_backend_model_problem",
        "_resolved",
        "_BUILDABLE_PROVIDER_KINDS",
        "_UNDERSTOOD_THINKING_MODES",
        "_BACKEND_NAME",
    ],
)
def test_the_backend_model_gate_is_gone(name):
    """Every backend builds every provider kind and thinking mode its record
    allows, from the models resolved at startup."""
    from agentic_cli.workflow import base_manager

    assert not hasattr(base_manager.BaseWorkflowManager, name)
    assert not hasattr(base_manager, "_PROVIDER_LABELS")

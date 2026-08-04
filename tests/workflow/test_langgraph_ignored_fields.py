"""LangGraph warns that ADK-only AgentConfig fields are ignored."""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.langgraph import manager as mgr_mod  # noqa: E402
from agentic_cli.workflow.langgraph.manager import (  # noqa: E402
    LangGraphWorkflowManager,
    _ignored_adk_only_fields,
)
from agentic_cli.workflow.mcp import MCPServerConfig  # noqa: E402
from agentic_cli.workflow.model_settings import ModelSettings  # noqa: E402


class TestIgnoredFieldsHelper:
    def test_detects_model_settings(self):
        cfg = AgentConfig(name="a", prompt="p", model_settings=ModelSettings(temperature=0.5))
        assert _ignored_adk_only_fields([cfg]) == [("a", ["model_settings"])]

    def test_detects_multiple_fields_in_canonical_order(self):
        cfg = AgentConfig(
            name="a", prompt="p",
            skills=["x"],
            mcp_servers=[MCPServerConfig(name="s", command="echo")],
        )
        # Order follows _ADK_ONLY_FIELDS: model_settings, mcp_servers, skills.
        assert _ignored_adk_only_fields([cfg]) == [("a", ["mcp_servers", "skills"])]

    def test_only_affected_agents_listed(self):
        a = AgentConfig(name="a", prompt="p", skills=["x"])
        b = AgentConfig(name="b", prompt="p")
        assert _ignored_adk_only_fields([a, b]) == [("a", ["skills"])]

    def test_empty_when_no_adk_only_fields(self):
        assert _ignored_adk_only_fields([AgentConfig(name="a", prompt="p")]) == []


class TestManagerWarns:
    def test_warns_on_construction(self, mock_context, monkeypatch):
        calls = []
        monkeypatch.setattr(
            mgr_mod.logger, "warning", lambda *a, **k: calls.append((a, k))
        )
        cfg = AgentConfig(name="a", prompt="p", model_settings=ModelSettings(temperature=0.5))
        LangGraphWorkflowManager(
            agent_configs=[cfg], settings=mock_context.settings, model="gemini-2.5-flash"
        )
        assert any(
            a and a[0] == "langgraph_ignoring_adk_only_fields" for a, _ in calls
        )

    def test_no_warning_without_adk_only_fields(self, mock_context, monkeypatch):
        calls = []
        monkeypatch.setattr(
            mgr_mod.logger, "warning", lambda *a, **k: calls.append((a, k))
        )
        cfg = AgentConfig(name="a", prompt="p")
        LangGraphWorkflowManager(
            agent_configs=[cfg], settings=mock_context.settings, model="gemini-2.5-flash"
        )
        assert not any(
            a and a[0] == "langgraph_ignoring_adk_only_fields" for a, _ in calls
        )

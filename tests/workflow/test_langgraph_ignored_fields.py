"""LangGraph warns that ADK-only AgentConfig fields are ignored.

An agent's ``model_settings`` is not one of them: LangGraph builds each agent's
model from its resolved model, which carries them.
"""

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
    def test_model_settings_are_not_ignored(self):
        cfg = AgentConfig(name="a", prompt="p", model_settings=ModelSettings(temperature=0.5))
        assert _ignored_adk_only_fields([cfg]) == []

    def test_detects_multiple_fields_in_canonical_order(self):
        cfg = AgentConfig(
            name="a", prompt="p",
            skills=["x"],
            mcp_servers=[MCPServerConfig(name="s", command="echo")],
        )
        # Order follows _ADK_ONLY_FIELDS: mcp_servers, skills.
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
        cfg = AgentConfig(name="a", prompt="p", skills=["x"])
        LangGraphWorkflowManager(
            agent_configs=[cfg], settings=mock_context.settings, model="gemini-2.5-flash"
        )
        assert any(
            a and a[0] == "langgraph_ignoring_adk_only_fields" for a, _ in calls
        )

    def test_the_warning_names_the_backends_that_use_them(self, mock_context, monkeypatch):
        calls = []
        monkeypatch.setattr(
            mgr_mod.logger, "warning", lambda *a, **k: calls.append((a, k))
        )
        cfg = AgentConfig(name="a", prompt="p", skills=["x"])
        LangGraphWorkflowManager(
            agent_configs=[cfg], settings=mock_context.settings, model="gemini-2.5-flash"
        )
        (note,) = [
            k["note"] for a, k in calls if a and a[0] == "langgraph_ignoring_adk_only_fields"
        ]
        assert note == (
            "mcp_servers/skills are used by the ADK and Deep Agents backends; "
            "LangGraph ignores them"
        )

    @pytest.mark.parametrize(
        "fields", [{}, {"model_settings": ModelSettings(temperature=0.5)}], ids=["none", "model"]
    )
    def test_no_warning_without_adk_only_fields(self, mock_context, monkeypatch, fields):
        calls = []
        monkeypatch.setattr(
            mgr_mod.logger, "warning", lambda *a, **k: calls.append((a, k))
        )
        cfg = AgentConfig(name="a", prompt="p", **fields)
        LangGraphWorkflowManager(
            agent_configs=[cfg], settings=mock_context.settings, model="gemini-2.5-flash"
        )
        assert not any(
            a and a[0] == "langgraph_ignoring_adk_only_fields" for a, _ in calls
        )

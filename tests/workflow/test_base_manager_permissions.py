"""Test that BaseWorkflowManager constructs and publishes PermissionEngine."""

from pathlib import Path

import pytest

from agentic_cli.workflow.permissions import PermissionEngine
from agentic_cli.workflow.service_registry import PERMISSION_ENGINE


class TestEngineInServiceRegistry:
    def test_base_manager_registers_permission_engine(self, tmp_path, monkeypatch):
        """After _ensure_managers_initialized, the engine is published to the registry dict."""
        from agentic_cli.config import BaseSettings
        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager
        from agentic_cli.workflow.config import AgentConfig

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))

        settings = BaseSettings(google_api_key="test")
        configs = [AgentConfig(name="root", prompt="hi", tools=[])]
        manager = GoogleADKWorkflowManager(
            agent_configs=configs, settings=settings, model="gemini-2.5-flash",
        )
        manager._ensure_managers_initialized()
        assert isinstance(manager.services[PERMISSION_ENGINE], PermissionEngine)

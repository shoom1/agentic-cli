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

    async def test_the_engine_knows_which_code_runs_on_the_host(self, tmp_path, monkeypatch):
        """The manager tells its engine, from its own settings and at each
        check, which code would run without a sandbox: that code is approved
        per call, and a settings change applies at once."""
        from agentic_cli.config import BaseSettings
        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager
        from agentic_cli.workflow.config import AgentConfig
        from agentic_cli.workflow.permissions import Capability
        from agentic_cli.workflow.permissions.prompt import ALLOW_ONCE_CHOICE, DENY_CHOICE

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        settings = BaseSettings(google_api_key="test")
        manager = GoogleADKWorkflowManager(
            agent_configs=[AgentConfig(name="root", prompt="hi", tools=[])],
            settings=settings,
            model="gemini-2.5-flash",
        )
        manager._ensure_managers_initialized()
        engine = manager.services[PERMISSION_ENGINE]
        requests = []

        async def ask(request):
            requests.append(request)
            return DENY_CHOICE

        monkeypatch.setattr(manager, "request_user_input", ask)
        python = [Capability("python.exec")]

        await engine.check("execute_python", python, {"code": "1"})
        settings.update_setting("os_sandbox_enabled", False)
        await engine.check("execute_python", python, {"code": "2"})

        assert len(requests[0].choices) == 4
        assert requests[1].choices == [ALLOW_ONCE_CHOICE, DENY_CHOICE]

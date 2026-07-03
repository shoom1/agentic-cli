"""Trust-model regression tests (P0-2, P0-3).

The project ``./.{app}/settings.json`` is untrusted — a cloned repo can ship it.
It must not be able to (a) disable the permission engine, or (b) grant
allow-rules; and the agent must not be able to write the app's own config to
self-escalate. See docs/reviews/2026-07-03-*.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_cli.workflow.permissions.capabilities import Capability
from agentic_cli.workflow.permissions.engine import PermissionEngine
from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource
from agentic_cli.workflow.permissions.store import BUILTIN_RULES, PermissionContext


def _stub_settings(*, enabled: bool = True, app_name: str = "myapp") -> MagicMock:
    s = MagicMock()
    s.permissions_enabled = enabled
    s.app_name = app_name
    return s


def _stub_workflow(response: str = "Deny") -> MagicMock:
    w = MagicMock()
    w.request_user_input = AsyncMock(return_value=response)
    return w


def _engine(workdir: Path, home: Path, app: str = "myapp", response: str = "Deny") -> PermissionEngine:
    ctx = PermissionContext(workdir=workdir, home=home, app_name=app)
    return PermissionEngine(_stub_settings(app_name=app), _stub_workflow(response), ctx)


# ---------------------------------------------------------------------------
# P0-2: project settings can tighten (deny) but never loosen (allow / disable)
# ---------------------------------------------------------------------------

class TestProjectRuleTrust:
    def _dirs(self, tmp_path, monkeypatch):
        proj, home = tmp_path / "proj", tmp_path / "home"
        proj.mkdir()
        home.mkdir()
        monkeypatch.chdir(proj)
        monkeypatch.setenv("HOME", str(home))
        return proj, home

    def test_project_allow_rules_are_ignored(self, tmp_path, monkeypatch):
        proj, home = self._dirs(tmp_path, monkeypatch)
        (proj / ".myapp").mkdir()
        (proj / ".myapp/settings.json").write_text(json.dumps({
            "permissions": {"allow": [{"capability": "http.read", "target": "*"}]},
        }))
        engine = _engine(proj, home)
        project_allows = [
            r for r in engine.rules
            if r.source is RuleSource.PROJECT and r.effect is Effect.ALLOW
        ]
        assert project_allows == []

    def test_project_deny_rules_are_honored(self, tmp_path, monkeypatch):
        proj, home = self._dirs(tmp_path, monkeypatch)
        (proj / ".myapp").mkdir()
        (proj / ".myapp/settings.json").write_text(json.dumps({
            "permissions": {"deny": [{"capability": "http.read", "target": "https://evil.test/**"}]},
        }))
        engine = _engine(proj, home)
        project_denies = [
            r for r in engine.rules
            if r.source is RuleSource.PROJECT and r.effect is Effect.DENY
        ]
        assert len(project_denies) == 1

    def test_user_allow_rules_still_honored(self, tmp_path, monkeypatch):
        """User config is trusted — it CAN grant allow-rules."""
        proj, home = self._dirs(tmp_path, monkeypatch)
        (home / ".myapp").mkdir()
        (home / ".myapp/settings.json").write_text(json.dumps({
            "permissions": {"allow": [{"capability": "http.read", "target": "*"}]},
        }))
        engine = _engine(proj, home)
        user_allows = [
            r for r in engine.rules
            if r.source is RuleSource.USER and r.effect is Effect.ALLOW
        ]
        assert len(user_allows) == 1


class TestProjectCannotDisablePermissions:
    def test_project_settings_json_cannot_set_permissions_enabled(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("AGENTIC_PERMISSIONS_ENABLED", raising=False)
        (tmp_path / ".agentic_cli").mkdir()
        (tmp_path / ".agentic_cli/settings.json").write_text(
            json.dumps({"permissions_enabled": False})
        )
        from agentic_cli.config import BaseSettings

        settings = BaseSettings()
        assert settings.permissions_enabled is True


# ---------------------------------------------------------------------------
# P0-3: the agent cannot write the app's own config to self-escalate
# ---------------------------------------------------------------------------

class TestConfigWriteDeny:
    def test_context_substitutes_app_name(self, tmp_path):
        ctx = PermissionContext(workdir=tmp_path, home=Path("/home"), app_name="myapp")
        assert ctx.substitute("${workdir}/.${app_name}/**") == f"{tmp_path}/.myapp/**"

    def test_builtin_rules_deny_app_config_writes(self):
        config_denies = [
            r for r in BUILTIN_RULES
            if r.effect is Effect.DENY
            and r.capability == "filesystem.write"
            and "${app_name}" in r.target
        ]
        # Both ${workdir} and ${home} config dirs must be covered.
        assert any("${workdir}" in r.target for r in config_denies)
        assert any("${home}" in r.target for r in config_denies)

    @pytest.mark.asyncio
    async def test_write_to_project_config_denied(self, tmp_path):
        engine = _engine(tmp_path, tmp_path / "home", response="Allow always")
        target = str(tmp_path / ".myapp" / "settings.json")
        result = await engine.check(
            "write_file",
            [Capability("filesystem.write", target_arg="path")],
            {"path": target},
        )
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_write_to_user_config_denied(self, tmp_path):
        home = tmp_path / "home"
        engine = _engine(tmp_path, home, response="Allow always")
        target = str(home / ".myapp" / "settings.json")
        result = await engine.check(
            "write_file",
            [Capability("filesystem.write", target_arg="path")],
            {"path": target},
        )
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_config_denied_even_with_broadened_workdir_grant(self, tmp_path):
        """A prior 'Allow always' that broadened to <workdir>/** must not reach
        the config path — deny-wins."""
        engine = _engine(tmp_path, tmp_path / "home")
        engine._session_rules.append(
            Rule("filesystem.write", f"{tmp_path}/**", Effect.ALLOW, RuleSource.SESSION)
        )
        target = str(tmp_path / ".myapp" / "settings.json")
        result = await engine.check(
            "write_file",
            [Capability("filesystem.write", target_arg="path")],
            {"path": target},
        )
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_ordinary_workdir_write_not_overblocked(self, tmp_path):
        engine = _engine(tmp_path, tmp_path / "home", response="Allow once")
        target = str(tmp_path / "output.txt")
        result = await engine.check(
            "write_file",
            [Capability("filesystem.write", target_arg="path")],
            {"path": target},
        )
        assert result.allowed is True

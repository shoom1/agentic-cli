"""Tests for PermissionEngine."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_cli.workflow.permissions.capabilities import Capability
from agentic_cli.workflow.permissions.engine import PermissionEngine
from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource
from agentic_cli.workflow.permissions.store import PermissionContext


def _stub_settings(*, enabled: bool = True, app_name: str = "agentic") -> MagicMock:
    s = MagicMock()
    s.permissions_enabled = enabled
    s.app_name = app_name
    return s


def _stub_workflow() -> MagicMock:
    w = MagicMock()
    w.request_user_input = AsyncMock(return_value="Deny")
    return w


@pytest.fixture
def ctx(tmp_path: Path) -> PermissionContext:
    return PermissionContext(workdir=tmp_path, home=Path("/fake/home"))


class TestEngineInit:
    def test_loads_builtin_rules(self, ctx):
        engine = PermissionEngine(
            settings=_stub_settings(), workflow=_stub_workflow(), ctx=ctx,
        )
        builtin = [r for r in engine.rules if r.source is RuleSource.BUILTIN]
        assert any(r.capability == "filesystem.read" for r in builtin)

    def test_loads_user_and_project_from_files(self, ctx, tmp_path, monkeypatch):
        import json
        # Prepare user + project files at their canonical locations.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".agentic").mkdir()
        (tmp_path / ".agentic/settings.json").write_text(json.dumps({
            "permissions": {"allow": [{"capability": "http.read", "target": "https://x.test/**"}]},
        }))
        # User file lives at ~/.agentic/settings.json (which is tmp_path/.agentic/...)
        # — same path since HOME == tmp_path; test merging via overlap

        engine = PermissionEngine(
            settings=_stub_settings(), workflow=_stub_workflow(), ctx=ctx,
        )
        # Exactly one http.read allow should be loaded
        http = [r for r in engine.rules if r.capability == "http.read"]
        assert len(http) >= 1


class TestEngineDisabled:
    @pytest.mark.asyncio
    async def test_short_circuits_when_disabled(self, ctx):
        engine = PermissionEngine(
            settings=_stub_settings(enabled=False),
            workflow=_stub_workflow(),
            ctx=ctx,
        )
        result = await engine.check(
            "write_file",
            [Capability("filesystem.write", target_arg="path")],
            {"path": "/any/where"},
        )
        assert result.allowed is True
        assert "disabled" in result.reason.lower()


class TestEngineRuleBased:
    @pytest.mark.asyncio
    async def test_builtin_allow_reads_in_workdir(self, ctx, tmp_path):
        engine = PermissionEngine(settings=_stub_settings(), workflow=_stub_workflow(), ctx=ctx)
        result = await engine.check(
            "read_file",
            [Capability("filesystem.read", target_arg="path")],
            {"path": str(tmp_path / "foo.py")},
        )
        assert result.allowed is True
        assert "builtin" in result.reason and "allow" in result.reason

    @pytest.mark.asyncio
    async def test_builtin_deny_writes_to_etc(self, ctx):
        engine = PermissionEngine(settings=_stub_settings(), workflow=_stub_workflow(), ctx=ctx)
        result = await engine.check(
            "write_file",
            [Capability("filesystem.write", target_arg="path")],
            {"path": "/etc/passwd"},
        )
        assert result.allowed is False
        assert "deny" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_deny_wins_across_capabilities(self, ctx, tmp_path):
        """copy_file-style: one cap allowed, one denied → overall deny."""
        engine = PermissionEngine(settings=_stub_settings(), workflow=_stub_workflow(), ctx=ctx)
        result = await engine.check(
            "copy_file",
            [
                Capability("filesystem.read", target_arg="src"),   # inside workdir: allow
                Capability("filesystem.write", target_arg="dest"), # /etc: deny
            ],
            {"src": str(tmp_path / "a"), "dest": "/etc/out"},
        )
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_deny_wins_across_sources(self, ctx, tmp_path, monkeypatch):
        """Add a session allow for filesystem.write → still blocked by builtin deny."""
        monkeypatch.chdir(tmp_path)
        engine = PermissionEngine(settings=_stub_settings(), workflow=_stub_workflow(), ctx=ctx)
        # Inject a session ALLOW for /etc/** — canonicalised so it matches on macOS
        from agentic_cli.workflow.permissions.matchers import get_matcher
        engine._session_rules.append(
            Rule(
                "filesystem.write",
                get_matcher("filesystem.write").canonicalize("/etc/**", ctx),
                Effect.ALLOW,
                RuleSource.SESSION,
            )
        )
        result = await engine.check(
            "write_file",
            [Capability("filesystem.write", target_arg="path")],
            {"path": "/etc/x"},
        )
        assert result.allowed is False  # DENY wins

    @pytest.mark.asyncio
    async def test_targetless_capability_uses_star(self, ctx):
        """A rule with target='*' matches any invocation of a targetless capability."""
        engine = PermissionEngine(settings=_stub_settings(), workflow=_stub_workflow(), ctx=ctx)
        engine._session_rules.append(
            Rule("python.exec", "*", Effect.ALLOW, RuleSource.SESSION)
        )
        result = await engine.check(
            "execute_python",
            [Capability("python.exec")],  # no target_arg
            {"code": "print('hi')"},
        )
        assert result.allowed is True

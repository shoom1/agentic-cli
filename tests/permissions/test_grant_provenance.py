"""P0-1 grant-provenance tests.

Interactive 'Allow always' grants live in the USER-side
~/.{app}/project_grants.json keyed by resolved project path — so a cloned repo
(a different path) carries no grants, and a repo-shipped permissions.local.json
is no longer trusted.
"""

import json
from pathlib import Path

import pytest

from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource
from agentic_cli.workflow.permissions.store import (
    PermissionContext,
    append_project_rule,
    load_project_grants,
)


def _ctx(workdir: Path) -> PermissionContext:
    return PermissionContext(workdir=workdir, home=Path("/fake/home"), app_name="agentic")


class TestLoadProjectGrants:
    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        assert load_project_grants("agentic", _ctx(tmp_path / "proj")) == []

    def test_roundtrip_current_project(self, tmp_path, monkeypatch):
        home = tmp_path / "home"; proj = tmp_path / "proj"; proj.mkdir()
        monkeypatch.setenv("HOME", str(home))
        append_project_rule(
            "agentic",
            Rule("http.read", "https://ok.test/**", Effect.ALLOW, RuleSource.PROJECT),
            proj,
        )
        rules = load_project_grants("agentic", _ctx(proj))
        assert len(rules) == 1
        assert rules[0].effect is Effect.ALLOW
        assert rules[0].source is RuleSource.PROJECT

    def test_grant_under_different_path_not_loaded(self, tmp_path, monkeypatch):
        """Simulated clone: a grant recorded for project A is invisible from B."""
        home = tmp_path / "home"; a = tmp_path / "a"; b = tmp_path / "b"
        a.mkdir(); b.mkdir()
        monkeypatch.setenv("HOME", str(home))
        append_project_rule("agentic", Rule("http.read", "*", Effect.ALLOW, RuleSource.PROJECT), a)
        assert load_project_grants("agentic", _ctx(b)) == []

    def test_malformed_json_raises(self, tmp_path, monkeypatch):
        from agentic_cli.settings_persistence import get_user_project_grants_path
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        p = get_user_project_grants_path("agentic")
        p.parent.mkdir(parents=True)
        p.write_text("{not json")
        with pytest.raises(ValueError):
            load_project_grants("agentic", _ctx(tmp_path / "proj"))

    def test_non_dict_grants_file_raises(self, tmp_path, monkeypatch):
        """A project_grants.json containing a JSON array (not an object) must raise ValueError."""
        from agentic_cli.settings_persistence import get_user_project_grants_path
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        p = get_user_project_grants_path("agentic")
        p.parent.mkdir(parents=True)
        p.write_text("[]")
        with pytest.raises(ValueError, match="Expected a JSON object"):
            load_project_grants("agentic", _ctx(tmp_path / "proj"))


class TestLegacyLocalFileNotTrusted:
    def test_repo_permissions_local_json_is_ignored(self, tmp_path, monkeypatch):
        """A repo-shipped ./.{app}/permissions.local.json allow-rule is no longer
        loaded as trusted (P0-1 drops the repo-local trusted load)."""
        from unittest.mock import AsyncMock, MagicMock
        from agentic_cli.workflow.permissions.engine import PermissionEngine

        home = tmp_path / "home"; proj = tmp_path / "proj"; proj.mkdir()
        monkeypatch.chdir(proj)
        monkeypatch.setenv("HOME", str(home))
        (proj / ".agentic").mkdir()
        (proj / ".agentic" / "permissions.local.json").write_text(json.dumps({
            "permissions": {"allow": [{"capability": "http.read", "target": "*"}]}
        }))
        s = MagicMock(); s.permissions_enabled = True; s.app_name = "agentic"
        w = MagicMock(); w.request_user_input = AsyncMock(return_value="Deny")
        engine = PermissionEngine(settings=s, workflow=w, ctx=_ctx(proj))
        project_allows = [
            r for r in engine.rules
            if r.source is RuleSource.PROJECT and r.effect is Effect.ALLOW
        ]
        assert project_allows == []


class TestChainBlocked:
    @pytest.mark.asyncio
    async def test_project_cannot_preauthorize_via_settings_or_local_file(
        self, tmp_path, monkeypatch
    ):
        """The reproduced chain: a cloned repo ships settings.json (sensitive
        keys) + a permissions.local.json allow-rule. Neither pre-authorizes:
        sensitive settings are dropped AND the repo allow-rule is untrusted, so a
        gated tool call still reaches the approval prompt (here: user denies)."""
        from unittest.mock import AsyncMock, MagicMock
        from agentic_cli.config import BaseSettings
        from agentic_cli.workflow.permissions.capabilities import Capability
        from agentic_cli.workflow.permissions.engine import PermissionEngine

        home = tmp_path / "home"; proj = tmp_path / "proj"; proj.mkdir()
        monkeypatch.chdir(proj)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("AGENTIC_STATEFUL_EXECUTOR_BACKEND", raising=False)
        (proj / ".agentic_cli").mkdir()
        (proj / ".agentic_cli" / "settings.json").write_text(json.dumps({
            "stateful_executor_backend": "local",
            "raw_llm_logging": True,
        }))
        (proj / ".agentic_cli" / "permissions.local.json").write_text(json.dumps({
            "permissions": {"allow": [{"capability": "http.read", "target": "*"}]}
        }))
        settings = BaseSettings()  # app_name default "agentic_cli"
        assert settings.stateful_executor_backend == "none"   # sensitive dropped
        assert settings.raw_llm_logging is False

        w = MagicMock(); w.request_user_input = AsyncMock(return_value="Deny")
        ctx = PermissionContext(workdir=proj, home=home, app_name="agentic_cli")
        engine = PermissionEngine(settings=settings, workflow=w, ctx=ctx)
        result = await engine.check(
            "web_fetch",
            [Capability("http.read", target_arg="url")],
            {"url": "https://evil.test/x"},
        )
        assert result.allowed is False           # repo allow-rule NOT trusted → prompt → denied
        w.request_user_input.assert_awaited_once()

"""Tests for PermissionContext + JSON store helpers."""

from pathlib import Path

import pytest

from agentic_cli.workflow.permissions.store import PermissionContext


class TestPermissionContext:
    def test_fields(self, tmp_path: Path):
        ctx = PermissionContext(workdir=tmp_path, home=Path("/fake/home"))
        assert ctx.workdir == tmp_path
        assert ctx.home == Path("/fake/home")

    def test_substitute_workdir(self, tmp_path: Path):
        ctx = PermissionContext(workdir=tmp_path, home=Path("/fake/home"))
        out = ctx.substitute("${workdir}/src/**")
        assert out == f"{tmp_path}/src/**"

    def test_substitute_home(self, tmp_path: Path):
        ctx = PermissionContext(workdir=tmp_path, home=Path("/fake/home"))
        assert ctx.substitute("${home}/.cache") == "/fake/home/.cache"

    def test_substitute_multiple(self, tmp_path: Path):
        ctx = PermissionContext(workdir=tmp_path, home=Path("/fake/home"))
        out = ctx.substitute("${workdir}:${home}")
        assert out == f"{tmp_path}:/fake/home"

    def test_substitute_unknown_variable_passes_through(self, tmp_path: Path):
        ctx = PermissionContext(workdir=tmp_path, home=Path("/fake/home"))
        assert ctx.substitute("${unknown}/x") == "${unknown}/x"


class TestBuiltinRules:
    def test_has_expected_entries(self):
        from agentic_cli.workflow.permissions.rules import Effect, RuleSource
        from agentic_cli.workflow.permissions.store import BUILTIN_RULES

        allows = [r for r in BUILTIN_RULES if r.effect is Effect.ALLOW]
        denies = [r for r in BUILTIN_RULES if r.effect is Effect.DENY]

        # Reads within workdir allowed:
        assert any(r.capability == "filesystem.read" and "${workdir}" in r.target for r in allows)
        # System dirs denied:
        system_targets = {r.target for r in denies if r.capability == "filesystem.write"}
        for path in ("/etc/**", "/usr/**", "/bin/**", "/sbin/**", "/boot/**", "/System/**"):
            assert path in system_targets
        # Credential dirs denied:
        for path in ("${home}/.ssh/**", "${home}/.aws/**", "${home}/.gnupg/**"):
            assert path in system_targets

    def test_all_builtin_have_builtin_source(self):
        from agentic_cli.workflow.permissions.rules import RuleSource
        from agentic_cli.workflow.permissions.store import BUILTIN_RULES
        for rule in BUILTIN_RULES:
            assert rule.source is RuleSource.BUILTIN


class TestLoadRules:
    def test_missing_file_returns_empty(self, tmp_path: Path):
        from agentic_cli.workflow.permissions.rules import RuleSource
        from agentic_cli.workflow.permissions.store import PermissionContext, load_rules

        ctx = PermissionContext(workdir=tmp_path, home=Path("/fake/home"))
        rules = load_rules(tmp_path / "missing.json", RuleSource.USER, ctx)
        assert rules == []

    def test_missing_permissions_section_returns_empty(self, tmp_path: Path):
        import json
        from agentic_cli.workflow.permissions.rules import RuleSource
        from agentic_cli.workflow.permissions.store import PermissionContext, load_rules

        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"default_model": "gpt-4"}))
        ctx = PermissionContext(workdir=tmp_path, home=Path("/fake/home"))
        assert load_rules(path, RuleSource.USER, ctx) == []

    def test_parses_allow_and_deny(self, tmp_path: Path):
        import json
        from agentic_cli.workflow.permissions.rules import Effect, RuleSource
        from agentic_cli.workflow.permissions.store import PermissionContext, load_rules

        path = tmp_path / "settings.json"
        path.write_text(json.dumps({
            "permissions": {
                "allow": [{"capability": "filesystem.read", "target": "${workdir}/**"}],
                "deny":  [{"capability": "filesystem.write", "target": "/etc/**"}],
            }
        }))
        ctx = PermissionContext(workdir=tmp_path, home=Path("/fake/home"))
        rules = load_rules(path, RuleSource.PROJECT, ctx)

        assert len(rules) == 2
        allow = next(r for r in rules if r.effect is Effect.ALLOW)
        deny = next(r for r in rules if r.effect is Effect.DENY)
        assert allow.capability == "filesystem.read"
        # PathMatcher canonicalises ${workdir} to the absolute path + /**
        assert str(tmp_path) in allow.target and allow.target.endswith("/**")
        assert allow.source is RuleSource.PROJECT
        # PathMatcher resolves the path (resolves symlinks on macOS /etc -> /private/etc)
        assert deny.target == str(Path("/etc/**").resolve(strict=False))

    def test_malformed_json_raises(self, tmp_path: Path):
        from agentic_cli.workflow.permissions.rules import RuleSource
        from agentic_cli.workflow.permissions.store import PermissionContext, load_rules

        path = tmp_path / "settings.json"
        path.write_text("{not json")
        ctx = PermissionContext(workdir=tmp_path, home=Path("/fake/home"))
        with pytest.raises(ValueError):
            load_rules(path, RuleSource.USER, ctx)


class TestAppendProjectRule:
    def test_creates_file_when_absent(self, tmp_path, monkeypatch):
        from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource
        from agentic_cli.workflow.permissions.store import append_project_rule

        monkeypatch.chdir(tmp_path)
        rule = Rule("filesystem.write", "/abs/foo", Effect.ALLOW, RuleSource.PROJECT)
        append_project_rule("agentic", rule)

        import json
        data = json.loads((tmp_path / ".agentic/settings.json").read_text())
        assert data["permissions"]["allow"] == [
            {"capability": "filesystem.write", "target": "/abs/foo"}
        ]

    def test_preserves_other_settings_keys(self, tmp_path, monkeypatch):
        import json
        from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource
        from agentic_cli.workflow.permissions.store import append_project_rule

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".agentic").mkdir()
        (tmp_path / ".agentic/settings.json").write_text(json.dumps({
            "default_model": "claude-sonnet-4",
            "thinking_effort": "medium",
        }))

        rule = Rule("filesystem.write", "/abs/foo", Effect.ALLOW, RuleSource.PROJECT)
        append_project_rule("agentic", rule)

        data = json.loads((tmp_path / ".agentic/settings.json").read_text())
        assert data["default_model"] == "claude-sonnet-4"
        assert data["thinking_effort"] == "medium"
        assert data["permissions"]["allow"][0]["capability"] == "filesystem.write"

    def test_deduplicates_identical_rules(self, tmp_path, monkeypatch):
        import json
        from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource
        from agentic_cli.workflow.permissions.store import append_project_rule

        monkeypatch.chdir(tmp_path)
        rule = Rule("filesystem.write", "/abs/foo", Effect.ALLOW, RuleSource.PROJECT)
        append_project_rule("agentic", rule)
        append_project_rule("agentic", rule)

        data = json.loads((tmp_path / ".agentic/settings.json").read_text())
        assert len(data["permissions"]["allow"]) == 1

    def test_writes_deny_section_for_deny_effect(self, tmp_path, monkeypatch):
        import json
        from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource
        from agentic_cli.workflow.permissions.store import append_project_rule

        monkeypatch.chdir(tmp_path)
        rule = Rule("filesystem.write", "/etc/foo", Effect.DENY, RuleSource.PROJECT)
        append_project_rule("agentic", rule)

        data = json.loads((tmp_path / ".agentic/settings.json").read_text())
        assert data["permissions"]["deny"] == [
            {"capability": "filesystem.write", "target": "/etc/foo"}
        ]
        assert "allow" not in data["permissions"] or data["permissions"]["allow"] == []

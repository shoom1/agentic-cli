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
        # Memory + KB: agent-internal stores allowed by default.
        assert any(r.capability == "memory.*" and r.target == "*" for r in allows)
        assert any(r.capability == "kb.*" and r.target == "*" for r in allows)
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
    """Interactive 'Allow always' grants persist to the USER-side, path-keyed
    ~/.{app}/project_grants.json (P0-1) — never into the repo, so a clone
    carries no grants."""

    def _grants_path(self, home: Path, app: str = "agentic") -> Path:
        return home / f".{app}" / "project_grants.json"

    def test_creates_user_grants_file_keyed_by_project(self, tmp_path, monkeypatch):
        import json
        from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource
        from agentic_cli.workflow.permissions.store import append_project_rule

        home = tmp_path / "home"; proj = tmp_path / "proj"; proj.mkdir()
        monkeypatch.setenv("HOME", str(home))
        rule = Rule("filesystem.write", "/abs/foo", Effect.ALLOW, RuleSource.PROJECT)
        append_project_rule("agentic", rule, proj)

        data = json.loads(self._grants_path(home).read_text())
        assert data[str(proj.resolve())]["permissions"]["allow"] == [
            {"capability": "filesystem.write", "target": "/abs/foo"}
        ]

    def test_writes_nothing_into_project_dir(self, tmp_path, monkeypatch):
        from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource
        from agentic_cli.workflow.permissions.store import append_project_rule

        home = tmp_path / "home"; proj = tmp_path / "proj"; proj.mkdir()
        monkeypatch.setenv("HOME", str(home))
        append_project_rule(
            "agentic",
            Rule("filesystem.write", "/abs/foo", Effect.ALLOW, RuleSource.PROJECT),
            proj,
        )
        assert not (proj / ".agentic").exists()  # nothing dropped inside the repo

    def test_deduplicates_identical_rules(self, tmp_path, monkeypatch):
        import json
        from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource
        from agentic_cli.workflow.permissions.store import append_project_rule

        home = tmp_path / "home"; proj = tmp_path / "proj"; proj.mkdir()
        monkeypatch.setenv("HOME", str(home))
        rule = Rule("filesystem.write", "/abs/foo", Effect.ALLOW, RuleSource.PROJECT)
        append_project_rule("agentic", rule, proj)
        append_project_rule("agentic", rule, proj)

        data = json.loads(self._grants_path(home).read_text())
        assert len(data[str(proj.resolve())]["permissions"]["allow"]) == 1

    def test_two_projects_kept_separate(self, tmp_path, monkeypatch):
        import json
        from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource
        from agentic_cli.workflow.permissions.store import append_project_rule

        home = tmp_path / "home"; a = tmp_path / "a"; b = tmp_path / "b"
        a.mkdir(); b.mkdir()
        monkeypatch.setenv("HOME", str(home))
        append_project_rule("agentic", Rule("http.read", "*", Effect.ALLOW, RuleSource.PROJECT), a)
        append_project_rule("agentic", Rule("filesystem.write", "/x", Effect.ALLOW, RuleSource.PROJECT), b)

        data = json.loads(self._grants_path(home).read_text())
        assert set(data.keys()) == {str(a.resolve()), str(b.resolve())}

    def test_writes_deny_section_for_deny_effect(self, tmp_path, monkeypatch):
        import json
        from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource
        from agentic_cli.workflow.permissions.store import append_project_rule

        home = tmp_path / "home"; proj = tmp_path / "proj"; proj.mkdir()
        monkeypatch.setenv("HOME", str(home))
        append_project_rule(
            "agentic",
            Rule("filesystem.write", "/etc/foo", Effect.DENY, RuleSource.PROJECT),
            proj,
        )
        data = json.loads(self._grants_path(home).read_text())
        assert data[str(proj.resolve())]["permissions"]["deny"] == [
            {"capability": "filesystem.write", "target": "/etc/foo"}
        ]

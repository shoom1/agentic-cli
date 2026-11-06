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

    def test_the_framework_s_saved_files_are_readable_wherever_it_starts(self):
        """The two allows name the folders the framework saves into for
        ``read_file``, so started from home (where the project read allow is
        left out) they are still read without a prompt."""
        from agentic_cli.tools.webfetch import saved
        from agentic_cli.workflow import tool_output
        from agentic_cli.workflow.permissions.rules import Effect
        from agentic_cli.workflow.permissions.store import BUILTIN_RULES

        allows = {r.target for r in BUILTIN_RULES if r.effect is Effect.ALLOW}
        for folder in (tool_output.FOLDER_NAME, saved.FOLDER_NAME):
            assert f"${{workdir}}/.${{app_name}}/{folder}/**" in allows

    def test_all_builtin_have_builtin_source(self):
        from agentic_cli.workflow.permissions.rules import RuleSource
        from agentic_cli.workflow.permissions.store import BUILTIN_RULES
        for rule in BUILTIN_RULES:
            assert rule.source is RuleSource.BUILTIN


#: A directory where the project file should be.
_DIRECTORY = object()


class TestSettingsRules:
    """The trusted rules are ``settings.permissions``, whatever source set it."""

    def test_allow_and_deny_are_user_rules_with_their_patterns_expanded(self, tmp_path: Path):
        from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource
        from agentic_cli.workflow.permissions.store import settings_rules
        from agentic_cli.workflow.settings import PermissionsConfig

        class Settings:
            permissions = PermissionsConfig.model_validate({
                "allow": [{"capability": "filesystem.write", "target": "${workdir}/build/**"}],
                "deny": [{"capability": "http.read", "target": "https://Tracker.example.net./**"}],
            })

        ctx = PermissionContext(workdir=tmp_path, home=Path("/fake/home"))
        assert settings_rules(Settings(), ctx) == [
            Rule("filesystem.write", f"{tmp_path}/build/**", Effect.ALLOW, RuleSource.USER),
            Rule("http.read", "https://tracker.example.net/**", Effect.DENY, RuleSource.USER),
        ]

    def test_settings_without_the_field_give_none(self, tmp_path: Path):
        from agentic_cli.workflow.permissions.store import settings_rules

        ctx = PermissionContext(workdir=tmp_path, home=Path("/fake/home"))
        assert settings_rules(object(), ctx) == []

    def test_a_rule_that_names_no_url_is_skipped_with_a_warning(self, tmp_path: Path):
        import structlog

        from agentic_cli.workflow.permissions.store import settings_rules
        from agentic_cli.workflow.settings import PermissionsConfig

        class Settings:
            permissions = PermissionsConfig.model_validate({"deny": [
                {"capability": "http.read", "target": "https://i\u2764.ws/**"},
                {"capability": "http.read", "target": "https://tracker.example.net/**"},
            ]})

        ctx = PermissionContext(workdir=tmp_path, home=Path("/fake/home"))
        with structlog.testing.capture_logs() as logs:
            rules = settings_rules(Settings(), ctx)

        assert [r.target for r in rules] == ["https://tracker.example.net/**"]
        [skipped] = [e for e in logs if e["event"] == "permission_rule_skipped"]
        assert skipped["setting"] == "permissions"
        assert skipped["target"] == "https://i\u2764.ws/**"


class TestLoadProjectDenies:
    """The untrusted project file's deny rules, each entry checked on its own."""

    def _load(self, tmp_path: Path, body: object):
        """Load ``body`` (JSON, or the file's bytes) as a project file."""
        import json

        import structlog

        from agentic_cli.workflow.permissions.store import load_project_denies

        path = tmp_path / "settings.json"
        if isinstance(body, bytes):
            path.write_bytes(body)
        elif body is not _DIRECTORY:
            path.write_text(json.dumps(body))
        else:
            path.mkdir()
        ctx = PermissionContext(workdir=tmp_path, home=Path("/fake/home"))
        with structlog.testing.capture_logs() as logs:
            rules = load_project_denies(path, ctx)
        return path, rules, [e for e in logs if e["event"].startswith("permission_")]

    def test_missing_file_returns_empty(self, tmp_path: Path):
        from agentic_cli.workflow.permissions.store import load_project_denies

        ctx = PermissionContext(workdir=tmp_path, home=Path("/fake/home"))
        assert load_project_denies(tmp_path / "missing.json", ctx) == []

    def test_missing_permissions_section_returns_empty(self, tmp_path: Path):
        _, rules, logs = self._load(tmp_path, {"default_model": "gpt-4"})
        assert rules == []
        assert logs == []

    def test_reads_deny_rules_and_never_allow_rules(self, tmp_path: Path):
        from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource

        _, rules, logs = self._load(tmp_path, {"permissions": {
            "allow": [{"capability": "filesystem.read", "target": "${workdir}/**"}],
            "deny":  [{"capability": "filesystem.write", "target": "/etc/**"}],
        }})

        # PathMatcher resolves the path (on macOS /etc is /private/etc).
        etc = str(Path("/etc/**").resolve(strict=False))
        assert rules == [Rule("filesystem.write", etc, Effect.DENY, RuleSource.PROJECT)]
        assert logs == []

    def test_malformed_entries_are_reported(self, tmp_path: Path):
        """Each malformed entry is skipped with one warning naming the file and
        the entry; the valid ones still load, and nothing raises."""
        path, rules, logs = self._load(tmp_path, {"permissions": {"deny": [
            {"capability": "filesystem.read"},
            "filesystem.read ./secrets",
            {"capability": "filesystem.read", "target": ["a", "b"]},
            None,
            {"capability": "filesystem.read", "target": "${workdir}/secrets/**"},
        ]}})

        assert [r.target for r in rules] == [f"{tmp_path}/secrets/**"]
        assert [(e["event"], e["path"], e["entry"]) for e in logs] == [
            ("permission_project_rule_invalid", str(path), f"permissions.deny[{i}]")
            for i in range(4)
        ]
        assert logs[0]["error"] == "target: Field required"
        assert all(e["log_level"] == "warning" for e in logs)

    @pytest.mark.parametrize(
        "section, entry",
        [
            (["oops"], "permissions"),
            ("deny", "permissions"),
            (7, "permissions"),
            ({"deny": "filesystem.read /x"}, "permissions.deny"),
            ({"deny": {"capability": "filesystem.read", "target": "/x"}}, "permissions.deny"),
        ],
    )
    def test_a_section_of_the_wrong_shape_is_reported(self, tmp_path: Path, section, entry):
        path, rules, logs = self._load(tmp_path, {"permissions": section})

        assert rules == []
        assert [(e["event"], e["path"], e["entry"]) for e in logs] == [
            ("permission_project_rule_invalid", str(path), entry),
        ]

    @pytest.mark.parametrize(
        "body",
        [b"{not json", b"[]", b"null", b'"text"', b'{"permissions": "\xff"}', _DIRECTORY],
        ids=["not-json", "list", "null", "string", "not-utf8", "directory"],
    )
    def test_a_file_that_is_not_a_json_object_is_reported(self, tmp_path: Path, body):
        path, rules, logs = self._load(tmp_path, body)

        assert rules == []
        assert [(e["event"], e["path"]) for e in logs] == [
            ("permission_project_file_invalid", str(path)),
        ]


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

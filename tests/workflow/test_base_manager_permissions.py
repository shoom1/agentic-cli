"""The permission engine a workflow manager builds, and the rules it enforces.

The engine's trusted rules are ``settings.permissions``, from every source the
settings read (the constructor, the environment, the user's file, the app's
class default). The project file's section is untrusted: its deny rules
apply, each entry checked on its own, and its allow rules do not.
"""

import json
from pathlib import Path

import pytest
import structlog
from pydantic import Field

from agentic_cli.config import BaseSettings
from agentic_cli.workflow.permissions import (
    Capability,
    CheckResult,
    Effect,
    PermissionEngine,
    RuleSource,
)
from agentic_cli.workflow.service_registry import PERMISSION_ENGINE
from agentic_cli.workflow.settings import PermissionRuleConfig, PermissionsConfig

APP = "permapp"
_SECRET_DENY = {"capability": "filesystem.read", "target": "${workdir}/secret/**"}


class TestEngineInServiceRegistry:
    def test_base_manager_registers_permission_engine(self, tmp_path, monkeypatch):
        """After _ensure_managers_initialized, the engine is published to the registry dict."""
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

@pytest.fixture
def layout(tmp_path, monkeypatch) -> dict[str, Path]:
    """A fake home and a project beside it, with a file under ``secret/``.

    Reading inside the project is allowed by a builtin rule, so a deny on
    ``secret/`` is what makes that read fail.
    """
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    (project / "secret").mkdir(parents=True)
    (project / "secret" / "notes.txt").write_text("x")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    monkeypatch.delenv("AGENTIC_PERMISSIONS", raising=False)
    return {"home": home, "project": project, "secret": project / "secret" / "notes.txt"}


def _write(path: Path, body: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body if isinstance(body, str) else json.dumps(body))
    return path


def _user_file(layout, body: object) -> Path:
    return _write(layout["home"] / f".{APP}" / "settings.json", body)


def _project_file(layout, body: object) -> Path:
    return _write(layout["project"] / f".{APP}" / "settings.json", body)


def _engine(settings: BaseSettings) -> PermissionEngine:
    from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager
    from agentic_cli.workflow.config import AgentConfig

    manager = GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="root", prompt="hi", tools=[])],
        settings=settings,
        model="gemini-2.5-flash",
    )
    manager._ensure_managers_initialized()
    return manager.services[PERMISSION_ENGINE]


def _settings(**kwargs) -> BaseSettings:
    return BaseSettings(app_name=APP, google_api_key="test", **kwargs)


async def _read(engine: PermissionEngine, path: Path) -> CheckResult:
    return await engine.check(
        "read_file", [Capability("filesystem.read", target_arg="path")], {"path": str(path)}
    )


async def _fetch(engine: PermissionEngine, url: str) -> CheckResult:
    return await engine.check(
        "web_fetch", [Capability("http.read", target_arg="url")], {"url": url}
    )


class TestSettingsRulesAreEnforced:
    """Every source of ``settings.permissions`` reaches the engine."""

    async def test_the_read_is_allowed_without_a_rule(self, layout):
        """The control: the deny rules below are what refuses the read."""
        assert (await _read(_engine(_settings()), layout["secret"])).allowed

    async def test_a_deny_given_to_the_constructor_is_enforced(self, layout):
        engine = _engine(_settings(permissions={"deny": [_SECRET_DENY]}))

        result = await _read(engine, layout["secret"])

        assert not result.allowed
        assert "deny" in result.reason

    async def test_a_deny_shipped_as_the_settings_class_default_is_enforced(self, layout):
        class AppSettings(BaseSettings):
            permissions: PermissionsConfig = Field(
                default_factory=lambda: PermissionsConfig(
                    deny=[PermissionRuleConfig(**_SECRET_DENY)]
                )
            )

        engine = _engine(AppSettings(app_name=APP, google_api_key="test"))

        assert not (await _read(engine, layout["secret"])).allowed

    async def test_a_deny_from_the_environment_is_enforced(self, layout, monkeypatch):
        monkeypatch.setenv("AGENTIC_PERMISSIONS", json.dumps({"deny": [_SECRET_DENY]}))

        engine = _engine(_settings())

        assert not (await _read(engine, layout["secret"])).allowed

    async def test_an_allow_given_to_the_constructor_is_honored(self, layout):
        url = "https://docs.example.org/page"
        engine = _engine(_settings(permissions={
            "allow": [{"capability": "http.read", "target": "https://docs.example.org/**"}],
        }))

        result = await _fetch(engine, url)

        assert result.allowed
        assert "user/allow" in result.reason

    async def test_the_user_file_rules_apply_once(self, layout):
        _user_file(layout, {"permissions": {
            "allow": [{"capability": "http.read", "target": "https://docs.example.org/**"}],
            "deny": [_SECRET_DENY],
        }})

        engine = _engine(_settings())

        assert not (await _read(engine, layout["secret"])).allowed
        assert (await _fetch(engine, "https://docs.example.org/page")).allowed
        trusted = [(r.capability, r.effect) for r in engine.rules if r.source is RuleSource.USER]
        assert sorted(trusted) == [("filesystem.read", Effect.DENY), ("http.read", Effect.ALLOW)]

    async def test_saved_grants_still_load(self, layout):
        grant = {"capability": "http.read", "target": "https://docs.example.org/**"}
        _write(layout["home"] / f".{APP}" / "project_grants.json", {
            str(layout["project"].resolve()): {"permissions": {"allow": [grant]}},
        })

        engine = _engine(_settings())

        assert (await _fetch(engine, "https://docs.example.org/page")).allowed
        project = [(r.effect, r.target) for r in engine.rules if r.source is RuleSource.PROJECT]
        assert project == [(Effect.ALLOW, grant["target"])]

    async def test_started_from_home_the_user_file_is_read_once(self, layout, monkeypatch):
        """There the project file is the user's own file: its rules are the
        settings' (trusted), and not read a second time as project rules."""
        monkeypatch.chdir(layout["home"])
        _user_file(layout, {"permissions": {"deny": [
            {"capability": "http.read", "target": "https://tracker.example.net/**"},
        ]}})

        engine = _engine(_settings())

        named = [r for r in engine.rules if "tracker.example.net" in r.target]
        assert [(r.source, r.effect) for r in named] == [(RuleSource.USER, Effect.DENY)]


def _app_settings(**default) -> BaseSettings:
    class AppSettings(BaseSettings):
        permissions: PermissionsConfig = Field(
            default_factory=lambda: PermissionsConfig.model_validate(default)
        )

    return AppSettings(app_name=APP, google_api_key="test")


class TestEverySourceApplies:
    """The trusted rules are the union of every source's: none replaces another's."""

    _DOCS = {"capability": "http.read", "target": "https://docs.example.org/**"}
    _TRACKER = {"capability": "http.read", "target": "https://tracker.example.net/**"}

    async def test_an_app_default_deny_and_a_user_file_allow_both_apply(self, layout):
        _user_file(layout, {"permissions": {"allow": [self._DOCS]}})

        engine = _engine(_app_settings(deny=[_SECRET_DENY]))

        assert not (await _read(engine, layout["secret"])).allowed
        assert (await _fetch(engine, "https://docs.example.org/page")).allowed

    async def test_an_environment_deny_and_a_user_file_deny_both_deny(self, layout, monkeypatch):
        _user_file(layout, {"permissions": {"deny": [_SECRET_DENY]}})
        monkeypatch.setenv("AGENTIC_PERMISSIONS", json.dumps({"deny": [self._TRACKER]}))

        engine = _engine(_settings())

        assert not (await _read(engine, layout["secret"])).allowed
        result = await _fetch(engine, "https://tracker.example.net/p")
        assert (result.allowed, "user/deny" in result.reason) == (False, True)

    async def test_a_constructor_deny_and_the_class_default_deny_both_deny(self, layout):
        class AppSettings(BaseSettings):
            permissions: PermissionsConfig = Field(
                default_factory=lambda: PermissionsConfig(
                    deny=[PermissionRuleConfig(**_SECRET_DENY)]
                )
            )

        engine = _engine(AppSettings(
            app_name=APP, google_api_key="test", permissions={"deny": [self._TRACKER]},
        ))

        assert not (await _read(engine, layout["secret"])).allowed
        result = await _fetch(engine, "https://tracker.example.net/p")
        assert (result.allowed, "user/deny" in result.reason) == (False, True)

    async def test_a_rule_given_in_two_sources_appears_once(self, layout, monkeypatch):
        _user_file(layout, {"permissions": {"deny": [_SECRET_DENY]}})
        monkeypatch.setenv("AGENTIC_PERMISSIONS", json.dumps({"deny": [_SECRET_DENY]}))

        engine = _engine(_app_settings(deny=[_SECRET_DENY]))

        trusted = [r for r in engine.rules if r.source is RuleSource.USER]
        assert [(r.capability, r.effect) for r in trusted] == [("filesystem.read", Effect.DENY)]


class TestProjectFileRules:
    """The project file can only deny, and a malformed entry costs only itself."""

    async def test_its_denies_apply_and_its_allows_do_not(self, layout):
        _project_file(layout, {"permissions": {
            "allow": [{"capability": "http.read", "target": "*"}],
            "deny": [_SECRET_DENY],
        }})

        engine = _engine(_settings())

        assert not (await _read(engine, layout["secret"])).allowed
        project = [(r.capability, r.effect) for r in engine.rules if r.source is RuleSource.PROJECT]
        assert project == [("filesystem.read", Effect.DENY)]

    async def test_malformed_entries_warn_and_a_valid_sibling_still_denies(self, layout):
        path = _project_file(layout, {"permissions": {"deny": [
            {"capability": "filesystem.read"},
            "oops",
            42,
            _SECRET_DENY,
        ]}})
        settings = _settings()

        with structlog.testing.capture_logs() as logs:
            engine = _engine(settings)

        assert not (await _read(engine, layout["secret"])).allowed
        invalid = [e for e in logs if e["event"] == "permission_project_rule_invalid"]
        assert [(e["path"], e["entry"]) for e in invalid] == [
            (str(path), "permissions.deny[0]"),
            (str(path), "permissions.deny[1]"),
            (str(path), "permissions.deny[2]"),
        ]
        assert all(e["log_level"] == "warning" for e in invalid)

    @pytest.mark.parametrize(
        "section, entry",
        [
            (["oops"], "permissions"),
            ("deny everything", "permissions"),
            ({"deny": {"capability": "filesystem.read", "target": "/x"}}, "permissions.deny"),
        ],
    )
    async def test_a_section_that_is_not_the_right_shape_warns(self, layout, section, entry):
        path = _project_file(layout, {"permissions": section})
        settings = _settings()

        with structlog.testing.capture_logs() as logs:
            engine = _engine(settings)

        assert [r for r in engine.rules if r.source is RuleSource.PROJECT] == []
        invalid = [e for e in logs if e["event"] == "permission_project_rule_invalid"]
        assert [(e["path"], e["entry"]) for e in invalid] == [(str(path), entry)]

    @pytest.mark.parametrize("body", ["{not json", "[1, 2]", "\"text\""])
    async def test_a_file_that_is_not_a_json_object_warns(self, layout, body):
        settings = _settings()  # the settings read the file first: written after
        path = _project_file(layout, body)

        with structlog.testing.capture_logs() as logs:
            engine = _engine(settings)

        assert [r for r in engine.rules if r.source is RuleSource.PROJECT] == []
        invalid = [e for e in logs if e["event"] == "permission_project_file_invalid"]
        assert [e["path"] for e in invalid] == [str(path)]

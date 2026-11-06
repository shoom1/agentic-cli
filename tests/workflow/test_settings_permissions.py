"""Tests for permissions settings."""

import json

import pytest
from pydantic import Field, ValidationError

from agentic_cli.config import BaseSettings
from agentic_cli.workflow.settings import PermissionsConfig


class TestPermissionsSettings:
    def test_permissions_enabled_default_true(self):
        s = BaseSettings()
        assert s.permissions_enabled is True

    def test_permissions_default_empty(self):
        s = BaseSettings()
        assert s.permissions.allow == []
        assert s.permissions.deny == []

    def test_hitl_enabled_removed(self):
        """hitl_enabled should no longer exist."""
        s = BaseSettings()
        assert not hasattr(s, "hitl_enabled")

    def test_permissions_accepts_allow_and_deny(self):
        from agentic_cli.workflow.settings import (
            PermissionRuleConfig,
            PermissionsConfig,
        )

        cfg = PermissionsConfig(
            allow=[PermissionRuleConfig(capability="filesystem.read", target="/x/**")],
            deny=[PermissionRuleConfig(capability="filesystem.write", target="/etc/**")],
        )
        assert cfg.allow[0].capability == "filesystem.read"
        assert cfg.deny[0].target == "/etc/**"

    def test_base_settings_has_permissions_field(self):
        s = BaseSettings()
        from agentic_cli.workflow.settings import PermissionsConfig
        assert isinstance(s.permissions, PermissionsConfig)

    def test_an_app_ships_rules_as_its_class_default_without_a_warning(self):
        """The engine enforces an app's default rules, so overriding the
        field must not print pydantic's "shadows an attribute in parent"
        warning over the UI at every start."""
        import warnings

        from pydantic import Field

        from agentic_cli.workflow.settings import (
            PermissionRuleConfig,
            PermissionsConfig,
        )

        deny = PermissionRuleConfig(capability="filesystem.read", target="/srv/keys/**")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            class AppSettings(BaseSettings):
                permissions: PermissionsConfig = Field(
                    default_factory=lambda: PermissionsConfig(deny=[deny])
                )

            settings = AppSettings()

        assert [str(w.message) for w in caught] == []
        assert settings.permissions.deny == [deny]


def _rule(host: str) -> dict:
    return {"capability": "http.read", "target": f"https://{host}/**"}


class TestPermissionsAreTheUnionOfTheSources:
    """``settings.permissions`` holds every trusted source's rules: the app's
    class default, the user file, the environment and the constructor, each
    list concatenated in that order with exact duplicates dropped. No source's
    section replaces another's (pydantic-settings replaces the lists inside a
    dict it merges), and the project file is none of them."""

    APP = "unionapp"

    @pytest.fixture(autouse=True)
    def _dirs(self, tmp_path, monkeypatch):
        self.home, self.project = tmp_path / "home", tmp_path / "project"
        self.home.mkdir()
        self.project.mkdir()
        monkeypatch.setenv("HOME", str(self.home))
        monkeypatch.chdir(self.project)
        monkeypatch.delenv("AGENTIC_PERMISSIONS", raising=False)

    def _file(self, root, permissions: dict) -> None:
        path = root / f".{self.APP}" / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"permissions": permissions}))

    def _app_class(self, **default):
        class AppSettings(BaseSettings):
            permissions: PermissionsConfig = Field(
                default_factory=lambda: PermissionsConfig.model_validate(default)
            )

        return AppSettings

    @staticmethod
    def _hosts(rules) -> list[str]:
        return [r.target.removeprefix("https://").removesuffix("/**") for r in rules]

    def test_a_user_file_allow_keeps_the_app_s_default_deny(self):
        self._file(self.home, {"allow": [_rule("user-allow")]})

        settings = self._app_class(deny=[_rule("app-deny")])(app_name=self.APP)

        assert self._hosts(settings.permissions.allow) == ["user-allow"]
        assert self._hosts(settings.permissions.deny) == ["app-deny"]

    def test_an_environment_deny_list_adds_to_the_user_file_s(self, monkeypatch):
        self._file(self.home, {"deny": [_rule("user-deny")]})
        monkeypatch.setenv("AGENTIC_PERMISSIONS", json.dumps({"deny": [_rule("env-deny")]}))

        settings = BaseSettings(app_name=self.APP)

        assert self._hosts(settings.permissions.deny) == ["user-deny", "env-deny"]

    def test_a_constructor_deny_adds_to_the_class_default(self):
        settings = self._app_class(deny=[_rule("app-deny")])(
            app_name=self.APP, permissions={"deny": [_rule("ctor-deny")]}
        )

        assert self._hosts(settings.permissions.deny) == ["app-deny", "ctor-deny"]

    def test_every_source_in_order_and_each_rule_once(self, monkeypatch):
        self._file(self.home, {"allow": [_rule("shared")], "deny": [_rule("user-deny")]})
        monkeypatch.setenv("AGENTIC_PERMISSIONS__DENY", json.dumps([_rule("env-deny")]))

        settings = self._app_class(allow=[_rule("shared")], deny=[_rule("app-deny")])(
            app_name=self.APP,
            permissions=PermissionsConfig.model_validate(
                {"allow": [_rule("shared")], "deny": [_rule("ctor-deny"), _rule("user-deny")]}
            ),
        )

        assert self._hosts(settings.permissions.allow) == ["shared"]
        assert self._hosts(settings.permissions.deny) == [
            "app-deny", "user-deny", "env-deny", "ctor-deny",
        ]

    def test_the_project_file_is_no_source(self):
        self._file(self.project, {"allow": [_rule("project-allow")], "deny": [_rule("project-deny")]})
        self._file(self.home, {"deny": [_rule("user-deny")]})

        settings = BaseSettings(app_name=self.APP)

        assert settings.permissions.allow == []
        assert self._hosts(settings.permissions.deny) == ["user-deny"]

    def test_a_malformed_section_in_a_trusted_source_is_still_refused(self, monkeypatch):
        """A union cannot hide a bad value under a good one: it is validated."""
        self._file(self.home, {"deny": "everything"})
        monkeypatch.setenv("AGENTIC_PERMISSIONS", json.dumps({"deny": [_rule("env-deny")]}))

        with pytest.raises(ValidationError):
            BaseSettings(app_name=self.APP)


class TestTheUnionUsesThePublicSourcesHook:
    """Each source's rules are filed by the framework's own sources hook
    (``SettingsSourcesMixin.settings_customise_sources``), never by
    overriding pydantic-settings' private methods, so a pydantic-settings
    release cannot silently change which rules apply."""

    def test_no_settings_class_of_the_framework_overrides_a_private_method(self):
        from pydantic_settings import BaseSettings as PydanticBaseSettings

        private = {
            name for name in vars(PydanticBaseSettings)
            if name.startswith("_settings_") or name.startswith("_PydanticBaseSettings")
        }
        overridden = {
            (cls.__qualname__, name)
            for cls in BaseSettings.__mro__
            if cls.__module__.startswith("agentic_cli")
            for name in vars(cls)
            if name in private
        }
        assert overridden == set()

    def test_a_class_composed_from_the_mixins_gets_the_union(self, tmp_path, monkeypatch):
        from pydantic_settings import BaseSettings as PydanticBaseSettings
        from pydantic_settings import SettingsConfigDict

        from agentic_cli.settings_mixins import AppSettingsMixin, CLISettingsMixin
        from agentic_cli.workflow.settings import WorkflowSettingsMixin

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.chdir(tmp_path)
        user_file = tmp_path / "home" / ".composedapp" / "settings.json"
        user_file.parent.mkdir(parents=True)
        user_file.write_text(json.dumps({"permissions": {"allow": [_rule("user-allow")]}}))
        monkeypatch.setenv("APP_PERMISSIONS", json.dumps({"deny": [_rule("env-deny")]}))

        class Composed(
            WorkflowSettingsMixin, AppSettingsMixin, CLISettingsMixin, PydanticBaseSettings
        ):
            model_config = SettingsConfigDict(env_prefix="APP_", extra="ignore")
            permissions: PermissionsConfig = Field(
                default_factory=lambda: PermissionsConfig.model_validate(
                    {"deny": [_rule("app-deny")]}
                )
            )

        settings = Composed(app_name="composedapp", permissions={"deny": [_rule("ctor-deny")]})

        hosts = TestPermissionsAreTheUnionOfTheSources._hosts
        assert hosts(settings.permissions.allow) == ["user-allow"]
        assert hosts(settings.permissions.deny) == ["app-deny", "env-deny", "ctor-deny"]

    def test_the_sources_are_filed_under_their_own_names(self, tmp_path, monkeypatch):
        """A source that reads the others' values (``settings_sources_data``)
        finds them under the names it would without the filing."""
        from pydantic_settings import PydanticBaseSettingsSource

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        seen: dict = {}

        class Peek(PydanticBaseSettingsSource):
            def get_field_value(self, field, field_name):
                return None, field_name, False

            def __call__(self):
                seen.update(self.settings_sources_data)
                return {}

        class PeekingSettings(BaseSettings):
            @classmethod
            def settings_customise_sources(cls, settings_cls, **sources):
                return (*super().settings_customise_sources(settings_cls, **sources),
                        Peek(settings_cls))

        PeekingSettings(thinking_effort="high", permissions={"deny": [_rule("ctor-deny")]})

        assert seen["InitSettingsSource"]["thinking_effort"] == "high"

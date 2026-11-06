"""Tests for permissions settings."""

from agentic_cli.config import BaseSettings


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

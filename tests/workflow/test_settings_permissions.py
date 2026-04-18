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

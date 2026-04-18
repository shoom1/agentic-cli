"""Tests for ToolDefinition.capabilities field + register_tool capabilities= kwarg.

This task adds `capabilities` as an optional kwarg (default EXEMPT) alongside
the existing `permission_level` kwarg so tools can be migrated incrementally.
Both fields coexist on `ToolDefinition` throughout Phase 2–5; the old field
is removed in the final cleanup task.
"""

import pytest

from agentic_cli.tools.registry import (
    PermissionLevel,
    ToolCategory,
    ToolRegistry,
)
from agentic_cli.workflow.permissions import Capability, EXEMPT
from agentic_cli.workflow.permissions.capabilities import _CapabilityExempt


class TestCapabilitiesKwarg:
    def test_default_is_exempt(self):
        reg = ToolRegistry()

        @reg.register(name="no_caps")
        def no_caps() -> dict:
            return {}

        defn = reg.get("no_caps")
        assert isinstance(defn.capabilities, _CapabilityExempt)

    def test_exempt_stored(self):
        reg = ToolRegistry()

        @reg.register(name="exempt", capabilities=EXEMPT)
        def exempt() -> dict:
            return {}

        defn = reg.get("exempt")
        assert defn.capabilities is EXEMPT

    def test_list_stored(self):
        reg = ToolRegistry()
        caps = [Capability("filesystem.read", target_arg="path")]

        @reg.register(name="reader", capabilities=caps)
        def reader(path: str) -> dict:
            return {}

        defn = reg.get("reader")
        assert defn.capabilities == caps

    def test_empty_list_rejected(self):
        reg = ToolRegistry()
        with pytest.raises(ValueError, match="EXEMPT"):
            @reg.register(name="bad", capabilities=[])
            def bad() -> dict:
                return {}

    def test_non_list_non_exempt_rejected(self):
        reg = ToolRegistry()
        with pytest.raises(TypeError):
            @reg.register(name="bad2", capabilities="invalid")  # type: ignore[arg-type]
            def bad2() -> dict:
                return {}

    def test_list_with_non_capability_rejected(self):
        reg = ToolRegistry()
        with pytest.raises(TypeError):
            @reg.register(name="bad3", capabilities=["not a Capability"])  # type: ignore[list-item]
            def bad3() -> dict:
                return {}

    def test_permission_level_still_works(self):
        """Old kwarg continues to be accepted during migration."""
        reg = ToolRegistry()

        @reg.register(
            name="legacy",
            permission_level=PermissionLevel.SAFE,
            category=ToolCategory.READ,
        )
        def legacy() -> dict:
            return {}

        defn = reg.get("legacy")
        assert defn.permission_level == PermissionLevel.SAFE
        # capabilities defaults to EXEMPT when not specified:
        assert isinstance(defn.capabilities, _CapabilityExempt)

    def test_both_kwargs_can_coexist(self):
        """Tools in the middle of migration have both fields set."""
        reg = ToolRegistry()
        caps = [Capability("filesystem.read", target_arg="path")]

        @reg.register(
            name="migrating",
            permission_level=PermissionLevel.SAFE,
            capabilities=caps,
            category=ToolCategory.READ,
        )
        def migrating(path: str) -> dict:
            return {}

        defn = reg.get("migrating")
        assert defn.permission_level == PermissionLevel.SAFE
        assert defn.capabilities == caps

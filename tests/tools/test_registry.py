"""Tests for ToolDefinition.capabilities field + register_tool capabilities= kwarg.

capabilities= is a required kwarg on @register_tool / ToolRegistry.register.
Omitting it raises TypeError.
"""

import pytest

from agentic_cli.tools.registry import (
    ToolCategory,
    ToolRegistry,
)
from agentic_cli.workflow.permissions import Capability, EXEMPT
from agentic_cli.workflow.permissions.capabilities import _CapabilityExempt


class TestCapabilitiesKwarg:
    def test_capabilities_kwarg_is_required(self):
        """Omitting capabilities= should raise TypeError."""
        reg = ToolRegistry()
        with pytest.raises(TypeError):
            @reg.register(name="no_caps")
            def no_caps() -> dict:
                return {}

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

"""Tests for capability declarations."""

import pytest

from agentic_cli.workflow.permissions.capabilities import (
    EXEMPT,
    Capability,
    ResolvedCapability,
    _CapabilityExempt,
)


class TestCapability:
    def test_targeted_capability(self):
        cap = Capability("filesystem.read", target_arg="path")
        assert cap.name == "filesystem.read"
        assert cap.target_arg == "path"

    def test_targetless_capability(self):
        cap = Capability("python.exec")
        assert cap.name == "python.exec"
        assert cap.target_arg is None

    def test_capability_is_hashable(self):
        a = Capability("filesystem.read", target_arg="path")
        b = Capability("filesystem.read", target_arg="path")
        assert a == b
        assert hash(a) == hash(b)


class TestResolvedCapability:
    def test_carries_name_and_target(self):
        r = ResolvedCapability(name="filesystem.read", target="/abs/path")
        assert r.name == "filesystem.read"
        assert r.target == "/abs/path"


class TestExempt:
    def test_exempt_is_singleton_of_sentinel_type(self):
        assert isinstance(EXEMPT, _CapabilityExempt)
        # Same instance referenced everywhere
        from agentic_cli.workflow.permissions.capabilities import EXEMPT as also_exempt
        assert EXEMPT is also_exempt

    def test_exempt_is_truthy(self):
        # So `not caps` in adapters doesn't misclassify EXEMPT as "missing".
        assert bool(EXEMPT)

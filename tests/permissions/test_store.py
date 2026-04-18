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

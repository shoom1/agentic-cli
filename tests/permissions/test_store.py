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

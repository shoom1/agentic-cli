# tests/permissions/test_matchers.py
"""Tests for permission matchers + cap-name glob."""

from pathlib import Path

import pytest

from agentic_cli.workflow.permissions.matchers import (
    Matcher,
    StringGlobMatcher,
)
from agentic_cli.workflow.permissions.store import PermissionContext


@pytest.fixture
def ctx(tmp_path: Path) -> PermissionContext:
    return PermissionContext(workdir=tmp_path, home=Path("/fake/home"))


class TestStringGlobMatcher:
    def test_canonicalize_strips_and_substitutes(self, ctx):
        m = StringGlobMatcher()
        assert m.canonicalize("  ${workdir}/foo  ", ctx) == f"{ctx.workdir}/foo"

    def test_matches_exact(self, ctx):
        m = StringGlobMatcher()
        assert m.matches("python.exec", "python.exec")

    def test_matches_glob_star(self, ctx):
        m = StringGlobMatcher()
        assert m.matches("*", "anything")
        assert m.matches("foo*", "foobar")
        assert not m.matches("foo*", "barbaz")


class TestMatcherProtocol:
    def test_string_glob_matcher_satisfies_protocol(self):
        assert isinstance(StringGlobMatcher(), Matcher)

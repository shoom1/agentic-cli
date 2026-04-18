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


class TestPathMatcher:
    def test_canonicalize_substitutes_workdir(self, ctx):
        from agentic_cli.workflow.permissions.matchers import PathMatcher
        m = PathMatcher()
        assert m.canonicalize("${workdir}/src/**", ctx) == f"{ctx.workdir}/src/**"

    def test_canonicalize_expands_home(self, ctx):
        from agentic_cli.workflow.permissions.matchers import PathMatcher
        m = PathMatcher()
        # ctx.home is a fake absolute path; expanduser() won't change it,
        # but an input like "~/x" should be expanded via the OS user's home
        # only when there isn't a ${home} substitution — we use ${home}.
        assert m.canonicalize("${home}/.cache", ctx) == "/fake/home/.cache"

    def test_canonicalize_anchors_relative_to_workdir(self, ctx):
        from agentic_cli.workflow.permissions.matchers import PathMatcher
        m = PathMatcher()
        assert m.canonicalize("foo/bar", ctx) == f"{ctx.workdir}/foo/bar"

    def test_canonicalize_resolves_dotdot(self, ctx):
        from agentic_cli.workflow.permissions.matchers import PathMatcher
        m = PathMatcher()
        # ${workdir} is tmp_path; ../x resolves out of it.
        out = m.canonicalize("${workdir}/a/../b", ctx)
        assert out == f"{ctx.workdir}/b"

    def test_match_single_star_is_single_segment(self, ctx):
        from agentic_cli.workflow.permissions.matchers import PathMatcher
        m = PathMatcher()
        assert m.matches("/a/*/c", "/a/b/c")
        assert not m.matches("/a/*/c", "/a/b/x/c")

    def test_match_double_star_crosses_segments(self, ctx):
        from agentic_cli.workflow.permissions.matchers import PathMatcher
        m = PathMatcher()
        assert m.matches("/a/**/z", "/a/z")
        assert m.matches("/a/**/z", "/a/b/z")
        assert m.matches("/a/**/z", "/a/b/c/z")
        assert not m.matches("/a/**/z", "/x/b/z")

    def test_match_question_mark_single_char(self, ctx):
        from agentic_cli.workflow.permissions.matchers import PathMatcher
        m = PathMatcher()
        assert m.matches("/a/?/c", "/a/b/c")
        assert not m.matches("/a/?/c", "/a/bb/c")

    def test_match_case_sensitive(self, ctx):
        from agentic_cli.workflow.permissions.matchers import PathMatcher
        m = PathMatcher()
        assert not m.matches("/A/b", "/a/b")


class TestURLMatcher:
    def test_canonicalize_lowercases_scheme_and_host(self, ctx):
        from agentic_cli.workflow.permissions.matchers import URLMatcher
        m = URLMatcher()
        assert m.canonicalize("HTTPS://API.GitHub.com/repos", ctx) == "https://api.github.com/repos"

    def test_canonicalize_strips_default_port(self, ctx):
        from agentic_cli.workflow.permissions.matchers import URLMatcher
        m = URLMatcher()
        assert m.canonicalize("https://example.com:443/x", ctx) == "https://example.com/x"
        assert m.canonicalize("http://example.com:80/x", ctx) == "http://example.com/x"

    def test_canonicalize_defaults_to_https(self, ctx):
        from agentic_cli.workflow.permissions.matchers import URLMatcher
        m = URLMatcher()
        assert m.canonicalize("api.github.com/x", ctx) == "https://api.github.com/x"

    def test_canonicalize_drops_fragment(self, ctx):
        from agentic_cli.workflow.permissions.matchers import URLMatcher
        m = URLMatcher()
        assert m.canonicalize("https://example.com/x#frag", ctx) == "https://example.com/x"

    def test_matches_host_exact(self, ctx):
        from agentic_cli.workflow.permissions.matchers import URLMatcher
        m = URLMatcher()
        assert m.matches("https://api.github.com/**", "https://api.github.com/repos/foo")
        assert not m.matches("https://api.github.com/**", "https://pypi.org/x")

    def test_matches_path_glob(self, ctx):
        from agentic_cli.workflow.permissions.matchers import URLMatcher
        m = URLMatcher()
        assert m.matches("https://api.github.com/repos/**", "https://api.github.com/repos/owner/repo")
        assert not m.matches("https://api.github.com/repos/**", "https://api.github.com/gists/1")


class TestShellMatcher:
    def test_canonicalize_trims_and_substitutes(self, ctx):
        from agentic_cli.workflow.permissions.matchers import ShellMatcher
        m = ShellMatcher()
        assert m.canonicalize("  ls ${workdir}  ", ctx) == f"ls {ctx.workdir}"

    def test_matches_whole_command_glob(self, ctx):
        from agentic_cli.workflow.permissions.matchers import ShellMatcher
        m = ShellMatcher()
        assert m.matches("git *", "git status")
        assert m.matches("git commit*", "git commit -m 'x'")
        assert not m.matches("git *", "svn up")

    def test_matches_does_not_tokenize(self, ctx):
        """Shell matcher is intentionally naive; risk assessor handles semantics."""
        from agentic_cli.workflow.permissions.matchers import ShellMatcher
        m = ShellMatcher()
        assert m.matches("git *", "git rm -rf /")  # permission matcher allows


class TestMatcherRegistry:
    def test_get_matcher_dispatches_by_namespace(self):
        from agentic_cli.workflow.permissions.matchers import (
            PathMatcher, URLMatcher, ShellMatcher, StringGlobMatcher, get_matcher,
        )
        assert isinstance(get_matcher("filesystem.read"), PathMatcher)
        assert isinstance(get_matcher("http.read"), URLMatcher)
        assert isinstance(get_matcher("shell.exec"), ShellMatcher)
        assert isinstance(get_matcher("python.exec"), StringGlobMatcher)  # fallback
        assert isinstance(get_matcher("kb.read"), StringGlobMatcher)


class TestCapMatches:
    def test_exact(self):
        from agentic_cli.workflow.permissions.matchers import _cap_matches
        assert _cap_matches("filesystem.read", "filesystem.read")
        assert not _cap_matches("filesystem.read", "filesystem.write")

    def test_namespace_glob(self):
        from agentic_cli.workflow.permissions.matchers import _cap_matches
        assert _cap_matches("filesystem.*", "filesystem.read")
        assert _cap_matches("filesystem.*", "filesystem.write")
        assert not _cap_matches("filesystem.*", "http.read")

    def test_star_matches_everything(self):
        from agentic_cli.workflow.permissions.matchers import _cap_matches
        assert _cap_matches("*", "filesystem.read")
        assert _cap_matches("*", "anything")

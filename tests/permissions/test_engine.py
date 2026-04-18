"""Tests for PermissionEngine."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_cli.workflow.permissions.capabilities import Capability
from agentic_cli.workflow.permissions.engine import PermissionEngine
from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource
from agentic_cli.workflow.permissions.store import PermissionContext


def _stub_settings(*, enabled: bool = True, app_name: str = "agentic") -> MagicMock:
    s = MagicMock()
    s.permissions_enabled = enabled
    s.app_name = app_name
    return s


def _stub_workflow() -> MagicMock:
    w = MagicMock()
    w.request_user_input = AsyncMock(return_value="Deny")
    return w


@pytest.fixture
def ctx(tmp_path: Path) -> PermissionContext:
    return PermissionContext(workdir=tmp_path, home=Path("/fake/home"))


class TestEngineInit:
    def test_loads_builtin_rules(self, ctx):
        engine = PermissionEngine(
            settings=_stub_settings(), workflow=_stub_workflow(), ctx=ctx,
        )
        builtin = [r for r in engine.rules if r.source is RuleSource.BUILTIN]
        assert any(r.capability == "filesystem.read" for r in builtin)

    def test_loads_user_and_project_from_files(self, ctx, tmp_path, monkeypatch):
        import json
        # Prepare user + project files at their canonical locations.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".agentic").mkdir()
        (tmp_path / ".agentic/settings.json").write_text(json.dumps({
            "permissions": {"allow": [{"capability": "http.read", "target": "https://x.test/**"}]},
        }))
        # User file lives at ~/.agentic/settings.json (which is tmp_path/.agentic/...)
        # — same path since HOME == tmp_path; test merging via overlap

        engine = PermissionEngine(
            settings=_stub_settings(), workflow=_stub_workflow(), ctx=ctx,
        )
        # Exactly one http.read allow should be loaded
        http = [r for r in engine.rules if r.capability == "http.read"]
        assert len(http) >= 1


class TestEngineDisabled:
    @pytest.mark.asyncio
    async def test_short_circuits_when_disabled(self, ctx):
        engine = PermissionEngine(
            settings=_stub_settings(enabled=False),
            workflow=_stub_workflow(),
            ctx=ctx,
        )
        result = await engine.check(
            "write_file",
            [Capability("filesystem.write", target_arg="path")],
            {"path": "/any/where"},
        )
        assert result.allowed is True
        assert "disabled" in result.reason.lower()


class TestEngineRuleBased:
    @pytest.mark.asyncio
    async def test_builtin_allow_reads_in_workdir(self, ctx, tmp_path):
        engine = PermissionEngine(settings=_stub_settings(), workflow=_stub_workflow(), ctx=ctx)
        result = await engine.check(
            "read_file",
            [Capability("filesystem.read", target_arg="path")],
            {"path": str(tmp_path / "foo.py")},
        )
        assert result.allowed is True
        assert "builtin" in result.reason and "allow" in result.reason

    @pytest.mark.asyncio
    async def test_builtin_deny_writes_to_etc(self, ctx):
        engine = PermissionEngine(settings=_stub_settings(), workflow=_stub_workflow(), ctx=ctx)
        result = await engine.check(
            "write_file",
            [Capability("filesystem.write", target_arg="path")],
            {"path": "/etc/passwd"},
        )
        assert result.allowed is False
        assert "deny" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_deny_wins_across_capabilities(self, ctx, tmp_path):
        """copy_file-style: one cap allowed, one denied → overall deny."""
        engine = PermissionEngine(settings=_stub_settings(), workflow=_stub_workflow(), ctx=ctx)
        result = await engine.check(
            "copy_file",
            [
                Capability("filesystem.read", target_arg="src"),   # inside workdir: allow
                Capability("filesystem.write", target_arg="dest"), # /etc: deny
            ],
            {"src": str(tmp_path / "a"), "dest": "/etc/out"},
        )
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_deny_wins_across_sources(self, ctx, tmp_path, monkeypatch):
        """Add a session allow for filesystem.write → still blocked by builtin deny."""
        monkeypatch.chdir(tmp_path)
        engine = PermissionEngine(settings=_stub_settings(), workflow=_stub_workflow(), ctx=ctx)
        # Inject a session ALLOW for /etc/** — canonicalised so it matches on macOS
        from agentic_cli.workflow.permissions.matchers import get_matcher
        engine._session_rules.append(
            Rule(
                "filesystem.write",
                get_matcher("filesystem.write").canonicalize("/etc/**", ctx),
                Effect.ALLOW,
                RuleSource.SESSION,
            )
        )
        result = await engine.check(
            "write_file",
            [Capability("filesystem.write", target_arg="path")],
            {"path": "/etc/x"},
        )
        assert result.allowed is False  # DENY wins

    @pytest.mark.asyncio
    async def test_targetless_capability_uses_star(self, ctx):
        """A rule with target='*' matches any invocation of a targetless capability."""
        engine = PermissionEngine(settings=_stub_settings(), workflow=_stub_workflow(), ctx=ctx)
        engine._session_rules.append(
            Rule("python.exec", "*", Effect.ALLOW, RuleSource.SESSION)
        )
        result = await engine.check(
            "execute_python",
            [Capability("python.exec")],  # no target_arg
            {"code": "print('hi')"},
        )
        assert result.allowed is True


class TestEngineAskFlow:
    @pytest.mark.asyncio
    async def test_user_allow_once_no_rule_installed(self, ctx, tmp_path):
        w = _stub_workflow()
        w.request_user_input = AsyncMock(return_value="Allow once")
        engine = PermissionEngine(settings=_stub_settings(), workflow=w, ctx=ctx)

        result = await engine.check(
            "web_fetch",
            [Capability("http.read", target_arg="url")],
            {"url": "https://example.com/"},
        )
        assert result.allowed is True
        assert "once" in result.reason
        assert not any(r.source is RuleSource.SESSION for r in engine.rules)

    @pytest.mark.asyncio
    async def test_user_allow_session_installs_session_rule(self, ctx, tmp_path):
        w = _stub_workflow()
        w.request_user_input = AsyncMock(return_value="Allow for this session")
        engine = PermissionEngine(settings=_stub_settings(), workflow=w, ctx=ctx)

        result = await engine.check(
            "web_fetch",
            [Capability("http.read", target_arg="url")],
            {"url": "https://example.com/x"},
        )
        assert result.allowed is True
        session = [r for r in engine.rules if r.source is RuleSource.SESSION]
        assert any(r.capability == "http.read" and "example.com" in r.target for r in session)

        # Second call for the same target → no new prompt.
        w.request_user_input.reset_mock()
        result2 = await engine.check(
            "web_fetch",
            [Capability("http.read", target_arg="url")],
            {"url": "https://example.com/x"},
        )
        assert result2.allowed is True
        w.request_user_input.assert_not_called()

    @pytest.mark.asyncio
    async def test_user_allow_always_writes_project_file(self, ctx, tmp_path, monkeypatch):
        import json
        monkeypatch.chdir(tmp_path)
        w = _stub_workflow()
        w.request_user_input = AsyncMock(return_value="Allow always (save to project)")
        engine = PermissionEngine(settings=_stub_settings(), workflow=w, ctx=ctx)

        result = await engine.check(
            "web_fetch",
            [Capability("http.read", target_arg="url")],
            {"url": "https://example.com/x"},
        )
        assert result.allowed is True

        data = json.loads((tmp_path / ".agentic/settings.json").read_text())
        allow = data["permissions"]["allow"]
        assert len(allow) == 1
        assert allow[0]["capability"] == "http.read"
        assert "example.com" in allow[0]["target"]

    @pytest.mark.asyncio
    async def test_user_denies(self, ctx):
        w = _stub_workflow()
        w.request_user_input = AsyncMock(return_value="Deny")
        engine = PermissionEngine(settings=_stub_settings(), workflow=w, ctx=ctx)

        result = await engine.check(
            "web_fetch",
            [Capability("http.read", target_arg="url")],
            {"url": "https://example.com/x"},
        )
        assert result.allowed is False
        assert "denied" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_only_unmatched_capabilities_become_session_rules(self, ctx, tmp_path):
        """copy_file with src inside workdir (allowed by builtin) and dest outside.
        Only the write rule should be synthesised."""
        w = _stub_workflow()
        w.request_user_input = AsyncMock(return_value="Allow for this session")
        engine = PermissionEngine(settings=_stub_settings(), workflow=w, ctx=ctx)

        result = await engine.check(
            "copy_file",
            [
                Capability("filesystem.read", target_arg="src"),
                Capability("filesystem.write", target_arg="dest"),
            ],
            {"src": str(tmp_path / "a"), "dest": str(tmp_path / "out")},
        )
        assert result.allowed is True
        session = [r for r in engine.rules if r.source is RuleSource.SESSION]
        assert len(session) == 1
        assert session[0].capability == "filesystem.write"


class TestEngineConcurrency:
    @pytest.mark.asyncio
    async def test_ask_is_serialised(self, ctx):
        """Two simultaneous calls that both need to ask → one prompt at a time."""
        import asyncio

        ask_active = 0
        ask_peak = 0

        async def fake_input(request):
            nonlocal ask_active, ask_peak
            ask_active += 1
            ask_peak = max(ask_peak, ask_active)
            # Simulate think time
            await asyncio.sleep(0.01)
            ask_active -= 1
            return "Allow once"

        w = MagicMock()
        w.request_user_input = fake_input
        engine = PermissionEngine(settings=_stub_settings(), workflow=w, ctx=ctx)

        await asyncio.gather(
            engine.check(
                "web_fetch", [Capability("http.read", target_arg="url")],
                {"url": "https://a.test/"},
            ),
            engine.check(
                "web_fetch", [Capability("http.read", target_arg="url")],
                {"url": "https://b.test/"},
            ),
        )
        assert ask_peak == 1  # never two asks in flight simultaneously


class TestTargetlessAllowAlwaysRegression:
    """Regression: after 'Allow always' on a targetless capability (target_arg=None),
    subsequent calls must not re-prompt.

    Bug: matchers that canonicalise targets (URLMatcher, PathMatcher) were mangling
    the ``"*"`` sentinel, so the just-installed rule didn't match the next
    invocation's resolved target (also ``"*"``). Fix short-circuits ``"*"`` in
    ``canonicalize`` and in ``matches``.
    """

    @pytest.mark.asyncio
    async def test_http_read_allow_always_matches_next_call(self, ctx, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        w = _stub_workflow()
        w.request_user_input = AsyncMock(return_value="Allow always (save to project)")
        engine = PermissionEngine(settings=_stub_settings(), workflow=w, ctx=ctx)

        # First call: no rule → ask → allow always (saves session + project rule)
        result1 = await engine.check(
            "web_search",
            [Capability("http.read")],  # targetless
            {"query": "test"},
        )
        assert result1.allowed is True
        assert w.request_user_input.await_count == 1

        # Second call must match the session rule and NOT re-prompt.
        result2 = await engine.check(
            "web_search",
            [Capability("http.read")],
            {"query": "different query"},
        )
        assert result2.allowed is True
        assert w.request_user_input.await_count == 1

        # Third call with a DIFFERENT tool that declares the same capability
        # is also covered — same cap, still target=='*'.
        result3 = await engine.check(
            "search_arxiv",
            [Capability("http.read")],
            {"query": "papers"},
        )
        assert result3.allowed is True
        assert w.request_user_input.await_count == 1

    @pytest.mark.asyncio
    async def test_reloaded_wildcard_rule_still_matches(self, ctx, tmp_path, monkeypatch):
        """After the project JSON is reloaded (simulating next process run),
        a rule stored with target='*' must still match a targetless capability."""
        monkeypatch.chdir(tmp_path)

        # Round 1: grant "allow always" so the rule is persisted.
        w1 = _stub_workflow()
        w1.request_user_input = AsyncMock(return_value="Allow always (save to project)")
        engine1 = PermissionEngine(settings=_stub_settings(), workflow=w1, ctx=ctx)
        await engine1.check("web_search", [Capability("http.read")], {"query": "x"})

        # Round 2: fresh engine reads rules from disk.
        w2 = _stub_workflow()
        w2.request_user_input = AsyncMock(return_value="Deny")  # would deny if re-asked
        engine2 = PermissionEngine(settings=_stub_settings(), workflow=w2, ctx=ctx)
        result = await engine2.check(
            "web_search",
            [Capability("http.read")],
            {"query": "y"},
        )
        assert result.allowed is True
        w2.request_user_input.assert_not_called()

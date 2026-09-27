"""The engine judges exactly the location the tool will act on.

A call argument is a *target*: it is resolved with the shared resolver
(``agentic_cli.paths.resolve_path``) and never has ``${workdir}``-style
placeholders expanded. Rule patterns keep their placeholders. These tests use a
temporary project directory with an ordinary file beside it.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_cli.workflow.permissions.capabilities import Capability
from agentic_cli.workflow.permissions.engine import PermissionEngine
from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource
from agentic_cli.workflow.permissions.store import PermissionContext, load_project_grants

APP = "resolvetest"


def _settings() -> MagicMock:
    s = MagicMock()
    s.permissions_enabled = True
    s.app_name = APP
    return s


def _workflow(answer: str = "Deny") -> MagicMock:
    w = MagicMock()
    w.request_user_input = AsyncMock(return_value=answer)
    return w


@pytest.fixture
def layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """``root/project`` is the workdir and cwd; ``root/outside/data.txt`` is not
    part of the project. HOME is isolated so no real config is read."""
    root = tmp_path.resolve()
    project = root / "project"
    outside = root / "outside"
    home = root / "home"
    for d in (project, outside, home):
        d.mkdir()
    (outside / "data.txt").write_text("outside\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    return {"root": root, "project": project, "outside": outside, "home": home}


def _engine(layout, workflow=None) -> PermissionEngine:
    ctx = PermissionContext(workdir=layout["project"], home=layout["home"], app_name=APP)
    return PermissionEngine(settings=_settings(), workflow=workflow or _workflow(), ctx=ctx)


PLACEHOLDER_ARG = "d/${workdir}/../../../outside/data.txt"


class TestPlaceholdersInArgumentsAreLiteral:
    def test_target_resolves_to_the_real_location(self, layout):
        engine = _engine(layout)
        [resolved] = engine._resolve(
            [Capability("filesystem.write", target_arg="path")], {"path": PLACEHOLDER_ARG}
        )
        assert resolved.target == str(layout["outside"] / "data.txt")

    @pytest.mark.asyncio
    async def test_project_grant_does_not_cover_a_path_outside_the_project(self, layout):
        workflow = _workflow("Deny")
        engine = _engine(layout, workflow)
        engine._session_rules.append(
            Rule("filesystem.write", f"{layout['project']}/**", Effect.ALLOW, RuleSource.SESSION)
        )
        result = await engine.check(
            "write_file",
            [Capability("filesystem.write", target_arg="path")],
            {"path": PLACEHOLDER_ARG},
        )
        assert result.allowed is False
        workflow.request_user_input.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_builtin_workdir_read_allow_does_not_cover_it_either(self, layout):
        workflow = _workflow("Deny")
        engine = _engine(layout, workflow)
        result = await engine.check(
            "read_file",
            [Capability("filesystem.read", target_arg="path")],
            {"path": PLACEHOLDER_ARG},
        )
        assert result.allowed is False
        workflow.request_user_input.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_real_project_file_is_still_allowed_without_asking(self, layout):
        workflow = _workflow("Deny")
        engine = _engine(layout, workflow)
        result = await engine.check(
            "read_file",
            [Capability("filesystem.read", target_arg="path")],
            {"path": "src/main.py"},
        )
        assert result.allowed is True
        workflow.request_user_input.assert_not_awaited()


class TestRelativeTargetsFollowTheToolsDirectory:
    def test_relative_target_is_anchored_at_cwd_even_if_workdir_differs(self, layout, monkeypatch):
        monkeypatch.chdir(layout["outside"])
        engine = _engine(layout)  # ctx.workdir is still the project
        [resolved] = engine._resolve(
            [Capability("filesystem.read", target_arg="path")], {"path": "data.txt"}
        )
        assert resolved.target == str(layout["outside"] / "data.txt")


class TestStarArgumentIsNotTheWildcard:
    def test_star_path_argument_resolves_to_a_file_named_star(self, layout):
        engine = _engine(layout)
        [resolved] = engine._resolve(
            [Capability("filesystem.read", target_arg="path")], {"path": "*"}
        )
        assert resolved.target == str(layout["project"] / "*")

    @pytest.mark.asyncio
    async def test_session_grant_on_a_star_path_does_not_cover_every_read(
        self, layout, monkeypatch
    ):
        monkeypatch.chdir(layout["outside"])  # outside the project: this asks
        engine = _engine(layout, _workflow("Allow for this session"))
        await engine.check(
            "read_file",
            [Capability("filesystem.read", target_arg="path")],
            {"path": "*"},
        )
        session = [r for r in engine.rules if r.source is RuleSource.SESSION]
        assert [r.target for r in session] == [f"{layout['outside']}/**"]

    def test_targetless_capability_still_resolves_to_the_wildcard(self, layout):
        engine = _engine(layout)
        [resolved] = engine._resolve([Capability("python.exec")], {"code": "1"})
        assert resolved.target == "*"


class TestUnresolvableTargetsAreDenied:
    @pytest.mark.asyncio
    async def test_null_byte_target_is_denied_not_raised(self, layout):
        workflow = _workflow("Allow once")
        engine = _engine(layout, workflow)
        result = await engine.check(
            "read_file",
            [Capability("filesystem.read", target_arg="path")],
            {"path": "bad\x00name"},
        )
        assert result.allowed is False
        assert "invalid target" in result.reason
        workflow.request_user_input.assert_not_awaited()


class TestRulePatternsStillExpand:
    def test_user_settings_rule_expands_workdir(self, layout):
        cfg = layout["home"] / f".{APP}"
        cfg.mkdir()
        (cfg / "settings.json").write_text(json.dumps({
            "permissions": {"allow": [
                {"capability": "filesystem.write", "target": "${workdir}/build/**"},
            ]},
        }))
        engine = _engine(layout)
        user = [r for r in engine.rules if r.source is RuleSource.USER]
        assert [r.target for r in user] == [f"{layout['project']}/build/**"]


class TestSavedGrantsLoadLiterally:
    """Grants are written by the engine as concrete, already-resolved targets.
    Re-expanding placeholder text on load would turn a directory name into a
    different location."""

    def test_placeholder_text_in_a_saved_grant_is_not_expanded(self, layout):
        literal = f"{layout['project']}/${{workdir}}/**"
        grants = layout["home"] / f".{APP}" / "project_grants.json"
        grants.parent.mkdir(parents=True)
        grants.write_text(json.dumps({
            str(layout["project"]): {"permissions": {"allow": [
                {"capability": "filesystem.write", "target": literal},
            ]}},
        }))
        ctx = PermissionContext(workdir=layout["project"], home=layout["home"], app_name=APP)
        [rule] = load_project_grants(APP, ctx)
        assert rule.target == literal

    def test_saved_url_grant_round_trips(self, layout):
        grants = layout["home"] / f".{APP}" / "project_grants.json"
        grants.parent.mkdir(parents=True)
        grants.write_text(json.dumps({
            str(layout["project"]): {"permissions": {"allow": [
                {"capability": "http.read", "target": "https://example.com/docs"},
            ]}},
        }))
        ctx = PermissionContext(workdir=layout["project"], home=layout["home"], app_name=APP)
        [rule] = load_project_grants(APP, ctx)
        assert rule.target == "https://example.com/docs"

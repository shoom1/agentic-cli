"""Seam 2: a resource capability names its target, and grants never widen to
"everything".

Filesystem, network and shell capabilities act on a resource named by a
target. A declaration without one resolves to the wildcard, so approving one
tool approved that capability for every tool and every resource.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_cli.workflow.permissions.capabilities import (
    Capability,
    ResolvedCapability,
    is_resource_capability,
)
from agentic_cli.workflow.permissions.engine import (
    PermissionEngine,
    broaden_target_for_grant,
)
from agentic_cli.workflow.permissions.prompt import (
    ALLOW_ALWAYS_CHOICE,
    ALLOW_ONCE_CHOICE,
    ALLOW_SESSION_CHOICE,
    DENY_CHOICE,
    build_request,
)
from agentic_cli.workflow.permissions.rules import RuleSource
from agentic_cli.workflow.permissions.store import (
    PermissionContext,
    load_project_grants,
    load_rules,
)

APP = "captargets"


def _settings() -> MagicMock:
    s = MagicMock()
    s.permissions_enabled = True
    s.app_name = APP
    return s


def _workflow(answer: str) -> MagicMock:
    w = MagicMock()
    w.request_user_input = AsyncMock(return_value=answer)
    return w


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    root = tmp_path.resolve()
    home, work = root / "home", root / "work"
    home.mkdir()
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    return {"root": root, "home": home, "work": work}


def _engine(env, answer: str = "Deny") -> PermissionEngine:
    ctx = PermissionContext(workdir=env["work"], home=env["home"], app_name=APP)
    return PermissionEngine(settings=_settings(), workflow=_workflow(answer), ctx=ctx)


class TestFixedTargets:
    def test_fixed_target_is_resolved_as_a_target(self, env):
        engine = _engine(env)
        [rc] = engine._resolve(
            [Capability("http.read", target="HTTPS://Example.com:443/api")], {}
        )
        assert rc == ResolvedCapability("http.read", "https://example.com/api")

    def test_fixed_target_expands_no_placeholders(self, env):
        engine = _engine(env)
        [rc] = engine._resolve(
            [Capability("http.read", target="https://example.com/${home}")], {}
        )
        assert rc.target == "https://example.com/${home}"

    def test_target_and_target_arg_are_mutually_exclusive(self):
        with pytest.raises(ValueError):
            Capability("http.read", target_arg="url", target="https://example.com")


class TestResourceCapabilities:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("filesystem.read", True),
            ("filesystem.write", True),
            ("http.read", True),
            ("shell.exec", True),
            ("search.web", False),
            ("python.exec", False),
            ("kb.write", False),
        ],
    )
    def test_is_resource_capability(self, name, expected):
        assert is_resource_capability(name) is expected

    def test_registering_a_targetless_resource_capability_warns(self):
        from agentic_cli.tools.registry import ToolRegistry

        def my_fetch(q: str) -> dict:
            return {"success": True}

        with pytest.warns(UserWarning, match="my_fetch.*http.read.*target"):
            ToolRegistry().register(my_fetch, capabilities=[Capability("http.read")])

    def test_targeted_or_non_resource_capabilities_do_not_warn(self):
        from agentic_cli.tools.registry import ToolRegistry

        def a(url: str) -> dict:
            return {"success": True}

        def b(code: str) -> dict:
            return {"success": True}

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reg = ToolRegistry()
            reg.register(a, capabilities=[Capability("http.read", target_arg="url")])
            reg.register(b, capabilities=[Capability("python.exec")])

    def test_every_framework_resource_capability_names_a_target(self):
        import agentic_cli.tools  # noqa: F401
        import agentic_cli.tools.sandbox  # noqa: F401
        from agentic_cli.tools.registry import get_registry

        offenders = [
            (d.name, c.name)
            for d in get_registry().list_tools()
            if d.func is not None
            and getattr(d.func, "__module__", "").startswith("agentic_cli.")
            and isinstance(d.capabilities, list)
            for c in d.capabilities
            if is_resource_capability(c.name) and c.target_arg is None and c.target is None
        ]
        assert offenders == []


class TestNoWildcardGrantsForResources:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("answer", [ALLOW_SESSION_CHOICE, ALLOW_ALWAYS_CHOICE])
    async def test_targetless_resource_approval_is_not_stored(self, env, answer):
        engine = _engine(env, answer)
        result = await engine.check("third_party", [Capability("http.read")], {})
        assert result.allowed is True
        assert [r for r in engine.rules if r.source is RuleSource.SESSION] == []
        assert not (env["home"] / f".{APP}" / "project_grants.json").exists()

    def test_prompt_offers_only_once_or_deny_when_nothing_can_be_remembered(self):
        req = build_request("third_party", [ResolvedCapability("http.read", "*")])
        assert req.choices == [ALLOW_ONCE_CHOICE, DENY_CHOICE]

    def test_prompt_offers_all_choices_otherwise(self):
        req = build_request(
            "web_fetch", [ResolvedCapability("http.read", "https://example.com/a")]
        )
        assert req.choices == [
            ALLOW_ONCE_CHOICE, ALLOW_SESSION_CHOICE, ALLOW_ALWAYS_CHOICE, DENY_CHOICE,
        ]


class TestSavedWildcardGrantsAreDropped:
    def _write_grants(self, env, entries):
        path = env["home"] / f".{APP}" / "project_grants.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({str(env["work"]): {"permissions": {"allow": entries}}}))

    def test_resource_wildcard_grant_is_dropped_on_load(self, env):
        self._write_grants(env, [
            {"capability": "http.read", "target": "*"},
            {"capability": "python.exec", "target": "*"},
            {"capability": "http.read", "target": "https://example.com/a"},
        ])
        ctx = PermissionContext(workdir=env["work"], home=env["home"], app_name=APP)
        loaded = {(r.capability, r.target) for r in load_project_grants(APP, ctx)}
        assert loaded == {("python.exec", "*"), ("http.read", "https://example.com/a")}

    def test_explicit_user_rule_with_a_wildcard_is_kept(self, env):
        cfg = env["home"] / f".{APP}" / "settings.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps({
            "permissions": {"allow": [{"capability": "http.read", "target": "*"}]},
        }))
        ctx = PermissionContext(workdir=env["work"], home=env["home"], app_name=APP)
        rules = load_rules(cfg, RuleSource.USER, ctx)
        assert [(r.capability, r.target) for r in rules] == [("http.read", "*")]


class TestFilesystemGrantScope:
    def _grant(self, target: str, home: Path) -> str:
        return broaden_target_for_grant(ResolvedCapability("filesystem.read", target), home=home)

    def test_file_widens_to_its_directory(self, env):
        f = env["work"] / "a.txt"
        assert self._grant(str(f), env["home"]) == f"{env['work']}/**"

    def test_directory_covers_itself_not_its_parent(self, env):
        d = env["work"] / "data"
        d.mkdir()
        assert self._grant(str(d), env["home"]) == f"{d}/**"

    def test_home_directory_covers_home(self, env):
        assert self._grant(str(env["home"]), env["home"]) == f"{env['home']}/**"

    def test_a_file_directly_in_an_ancestor_of_home_is_not_widened(self, env):
        f = env["root"] / "f.txt"  # root is the parent of home
        assert self._grant(str(f), env["home"]) == str(f)

    def test_an_ancestor_of_home_is_not_widened(self, env):
        assert self._grant(str(env["root"]), env["home"]) == str(env["root"])

    def test_the_filesystem_root_is_never_widened(self, env):
        assert self._grant("/", env["home"]) == "/"
        assert self._grant("/f.txt", env["home"]) == "/f.txt"

    @pytest.mark.asyncio
    async def test_session_grant_on_a_directory_does_not_cover_its_siblings(self, env):
        (env["home"] / "docs").mkdir()
        (env["home"] / "other").mkdir()
        engine = _engine(env, ALLOW_SESSION_CHOICE)
        caps = [Capability("filesystem.read", target_arg="path")]
        await engine.check("glob", caps, {"path": str(env["home"] / "docs")})
        engine._workflow.request_user_input = AsyncMock(return_value="Deny")
        sibling = await engine.check("glob", caps, {"path": str(env["home"] / "other")})
        assert sibling.allowed is False

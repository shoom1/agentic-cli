"""The memory folders are written only through the memory tools (spec §14)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.tools.exec_policy import python_sandbox_policy
from agentic_cli.tools.os_sandbox import OSSandboxPolicy
from agentic_cli.workflow.permissions import Capability, PermissionContext, PermissionEngine
from agentic_cli.workflow.permissions.store import memory_folder_denies


def _settings(tmp_path) -> BaseSettings:
    # A workspace outside ~/.{app_name}: the case the app-config deny misses.
    return BaseSettings(app_name="app", workspace_dir=tmp_path / "data")


class _Workflow:
    async def request_user_input(self, request):
        raise AssertionError("a protected write must be denied, not asked")


@pytest.mark.parametrize("target", ["memory/prefers-tabs-a1.md", "projects/app-0123/memory/x.md",
                                    "projects/app-0123/project-path.txt"])
async def test_the_file_tools_cannot_write_there(tmp_path, target):
    settings = _settings(tmp_path)
    ctx = PermissionContext(workdir=tmp_path / "proj", home=tmp_path / "home", app_name="app")
    engine = PermissionEngine(settings=settings, workflow=_Workflow(), ctx=ctx)
    path = str(settings.workspace_dir / target)
    result = await engine.check("write_file", [Capability("filesystem.write", target_arg="path")], {"path": path})
    assert not result.allowed
    # By the deny rule, not by a prompt that failed (which also denies).
    assert result.reason.startswith("rule: builtin/deny filesystem.write")


def test_the_rules_are_builtin_denies_of_both_folders(tmp_path):
    settings = _settings(tmp_path)
    ctx = PermissionContext(workdir=tmp_path, home=tmp_path / "home", app_name="app")
    targets = {rule.target for rule in memory_folder_denies(settings, ctx)}
    root = (tmp_path / "data").resolve()
    assert targets == {f"{root}/memory/**", f"{root}/projects/**"}


def test_executed_code_cannot_see_them(tmp_path):
    settings = _settings(tmp_path)
    policy = python_sandbox_policy(settings)
    if policy is None:
        pytest.skip("the OS sandbox is off in these settings")
    hidden = {Path(p) for p in policy.deny_read_paths}
    root = (tmp_path / "data").resolve()
    assert {root / "memory", root / "projects"} <= hidden


def test_executed_code_cannot_write_there_either(tmp_path):
    """Hidden from reads is not enough where the folders sit in a writable
    place: code could plant a file the next conversation reads as a memory."""
    settings = _settings(tmp_path)
    root = (tmp_path / "data").resolve()
    (root / "memory").mkdir(parents=True)
    (root / "projects").mkdir()
    policy = python_sandbox_policy(settings)
    if policy is None:
        pytest.skip("the OS sandbox is off in these settings")
    denied = {Path(p) for p in policy.deny_write_paths}
    assert {root / "memory", root / "projects"} <= denied
    # What every policy denies stays denied.
    assert set(OSSandboxPolicy().deny_write_paths) <= set(policy.deny_write_paths)


def test_a_folder_that_does_not_exist_is_not_made_a_mount_point(tmp_path):
    """The sandbox backends bind over a missing deny-write path to block its
    creation, which can leave a stray file where the store makes its folder."""
    policy = python_sandbox_policy(_settings(tmp_path))
    if policy is None:
        pytest.skip("the OS sandbox is off in these settings")
    assert policy.deny_write_paths == OSSandboxPolicy().deny_write_paths

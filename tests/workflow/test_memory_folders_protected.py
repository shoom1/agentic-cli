"""The memory folders are written only through the memory tools (spec §14)."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.tools.exec_policy import python_sandbox_policy
from agentic_cli.tools.os_sandbox import OSSandboxPolicy, get_os_sandbox
from agentic_cli.tools.os_sandbox.bubblewrap import BubblewrapSandbox
from agentic_cli.tools.os_sandbox.seatbelt import SeatbeltSandbox
from agentic_cli.workflow.permissions import Capability, PermissionContext, PermissionEngine
from agentic_cli.workflow.memory_services import open_memory_store, project_memory_dir
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


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """A project (the cwd, where the sandbox may write) under a hermetic home."""
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "proj").mkdir()
    monkeypatch.chdir(tmp_path / "proj")
    return (tmp_path / "proj").resolve()


def _inside(project: Path, **values) -> BaseSettings:
    """Settings whose workspace_dir sits in the project: a folder the sandbox may write."""
    return BaseSettings(app_name="app", workspace_dir=project / "data", **values)


def _policy(settings: BaseSettings) -> OSSandboxPolicy:
    policy = python_sandbox_policy(settings)
    if policy is None:
        pytest.skip("the OS sandbox is off in these settings")
    return policy


def _paths(names: list[str]) -> set[Path]:
    return {Path(name) for name in names}


def test_building_the_policy_makes_the_folders_private(project):
    """Code that finds a folder missing could create it, and the files it
    plants there would be read as memories: the folders exist before it runs."""
    settings = _inside(project)
    _policy(settings)
    key_folder = project_memory_dir(settings)
    assert key_folder is not None
    for folder in (project / "data" / "memory", project / "data" / "projects", key_folder):
        assert folder.is_dir()
        assert stat.S_IMODE(folder.stat().st_mode) == 0o700


def test_a_project_scope_is_made_only_where_the_store_has_one(project, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path / "home")  # started from home: no project scope
    settings = _inside(project)
    _policy(settings)
    assert (project / "data" / "memory").is_dir()
    assert list((project / "data" / "projects").iterdir()) == []


def test_the_store_opens_on_the_folders_the_policy_made(project):
    settings = _inside(project)
    _policy(settings)
    store = open_memory_store(settings)
    assert store is not None
    assert store.memories() == []
    memory = store.save("Prefers tabs", "Indent with tabs.", "user", scope="user")
    assert [m.id for m in store.memories()] == [memory.id]


def test_executed_code_cannot_write_there_either(project):
    """Hidden from reads is not enough where the folders sit in a writable
    place: code could plant a file the next conversation reads as a memory.
    The folders are denied as subpaths, so every project's folder is."""
    settings = _inside(project)
    root = project / "data"
    policy = _policy(settings)
    assert {root / "memory", root / "projects"} <= _paths(policy.deny_write_paths)
    assert {root / "memory", root / "projects"} <= _paths(policy.deny_read_paths)
    # What every policy denies stays denied.
    assert set(OSSandboxPolicy().deny_write_paths) <= set(policy.deny_write_paths)


def test_a_folder_that_does_not_exist_is_not_made_a_mount_point(project, tmp_path):
    """A folder that cannot be made is still denied (seatbelt names a path, not
    a mount) but is left out of the list bubblewrap binds, which would bind
    /dev/null over it and can leave a stray file where the store makes its folder."""
    (tmp_path / "blocker").write_text("a file, so nothing can be made below it")
    settings = BaseSettings(app_name="app", workspace_dir=tmp_path / "blocker" / "data")
    policy = _policy(settings)
    root = (tmp_path / "blocker" / "data").resolve(strict=False)
    assert {root / "memory", root / "projects"} <= _paths(policy.deny_write_missing_paths)
    assert {root / "memory", root / "projects"} <= _paths(policy.deny_read_paths)
    assert policy.deny_write_paths == OSSandboxPolicy().deny_write_paths


def test_the_workspace_and_what_is_above_it_cannot_be_renamed(project, tmp_path):
    """Code that may write the workspace's parent could move the workspace
    aside, write into the moved folder, and move it back: seatbelt names a
    path, so the path itself is denied (what is below stays writable)."""
    policy = _policy(_inside(project))
    assert _paths(policy.deny_rename_paths) == {project / "data", project}
    policy = _policy(_inside(project, os_sandbox_writable_paths=[str(tmp_path)]))
    assert _paths(policy.deny_rename_paths) == {project / "data", project, tmp_path.resolve()}


def test_a_workspace_the_code_cannot_write_needs_no_rename_deny(project, tmp_path):
    outside = BaseSettings(app_name="app", workspace_dir=tmp_path / "home" / "ws")
    assert _policy(outside).deny_rename_paths == []


class TestEachBackend:
    def test_seatbelt_denies_the_missing_folders_and_the_renames(self, project):
        settings = BaseSettings(app_name="app", workspace_dir=project.parent / "blocker" / "data")
        (project.parent / "blocker").write_text("a file")
        policy = _policy(settings)
        policy.deny_rename_paths.append(str(project / "data"))
        profile = SeatbeltSandbox()._build_profile(project, policy)
        root = (project.parent / "blocker" / "data").resolve(strict=False)
        assert f'(deny file-write* (subpath "{root}/memory"))' in profile
        assert f'(deny file-write-unlink (literal "{project / "data"}"))' in profile

    def test_the_rename_rule_follows_the_write_allows(self):
        from agentic_cli.tools.os_sandbox.seatbelt_profile import generate_seatbelt_profile

        profile = generate_seatbelt_profile(
            writable_paths=[Path("/p")], deny_write_paths=[], readable_paths=[],
            deny_read_paths=[], deny_rename_paths=[Path("/p/data")],
        )
        assert profile.index('(allow file-write* (subpath "/p"))') < profile.index(
            '(deny file-write-unlink (literal "/p/data"))'
        )

    def test_bubblewrap_binds_what_exists_and_needs_no_rename_rule(self, project):
        """A bind mount follows its directory when an ancestor is renamed, so a
        write through the new name still meets the read-only mount."""
        settings = _inside(project)
        policy = _policy(settings)
        args = BubblewrapSandbox()._build_bwrap_args(project, policy)
        root = (project / "data").resolve()
        binds = {(args[i + 1], args[i + 2]) for i in range(len(args)) if args[i] == "--ro-bind"}
        assert (str(root / "memory"), str(root / "memory")) in binds
        assert (str(root / "projects"), str(root / "projects")) in binds
        policy.deny_rename_paths.append(str(project / "data"))
        policy.deny_write_missing_paths.append(str(root / "never-made"))
        assert BubblewrapSandbox()._build_bwrap_args(project, policy) == args


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
    reason="runs the real macOS sandbox",
)
class TestRealSeatbelt:
    """A sandboxed shell tries what a model's code could."""

    @staticmethod
    def _run(project: Path, settings: BaseSettings, command: str) -> subprocess.CompletedProcess:
        wrapped = get_os_sandbox().wrap_shell_command(command, project, _policy(settings))
        assert wrapped.success, wrapped.error
        return subprocess.run(wrapped.command, shell=True, cwd=project, capture_output=True, text=True)

    def test_code_cannot_create_the_folder_and_plant_a_memory(self, project):
        settings = _inside(project)
        plant = "mkdir -p data/memory && echo bad > data/memory/planted-abcd1234.md"
        assert self._run(project, settings, plant).returncode != 0
        assert not (project / "data" / "memory" / "planted-abcd1234.md").exists()
        store = open_memory_store(settings)
        assert store is not None and store.memories() == []

    def test_code_cannot_write_a_folder_that_exists(self, project):
        settings = _inside(project)
        assert self._run(project, settings, "echo bad > data/memory/planted.md").returncode != 0
        assert self._run(project, settings, "mkdir data/projects/other-0123").returncode != 0
        assert not (project / "data" / "memory" / "planted.md").exists()

    def test_code_cannot_move_the_workspace_aside_to_write_into_it(self, project):
        settings = _inside(project)
        store = open_memory_store(settings)
        assert store is not None
        store.save("Real memory", "Kept.", "user", scope="user")
        script = (
            "mv data data_old && echo bad > data_old/memory/planted-abcd1234.md; "
            "mv data_old data 2>/dev/null; true"
        )
        self._run(project, settings, script)
        assert not list((project / "data" / "memory").glob("planted-*"))
        assert not (project / "data_old").exists()
        assert [m.description for m in store.memories()] == ["Real memory"]

    def test_everything_else_in_the_project_stays_writable(self, project):
        settings = _inside(project)
        result = self._run(
            project, settings, "echo ok > notes.md && mkdir data/other && echo ok > data/other/a.md"
        )
        assert result.returncode == 0, result.stderr
        assert (project / "data" / "other" / "a.md").read_text().strip() == "ok"

"""Tests for bubblewrap sandbox argument generation.

Tests cover:
- Basic bwrap argument structure (flags, ro-bind root)
- Network isolation (--unshare-net toggle)
- Writable path bind mounts
- Deny-write path handling (existing and nonexistent targets)
- Deny-read path handling (directories vs files)
- Device and proc mounts
- wrap_shell_command full output format
- wrap_python_command full output format
- Error handling when _build_bwrap_args fails
- Working directory always writable
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agentic_cli.tools.shell.os_sandbox.bubblewrap import BubblewrapSandbox
from agentic_cli.tools.shell.os_sandbox.policy import OSSandboxPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_policy(**overrides) -> OSSandboxPolicy:
    """Create a policy with no deny lists for cleaner assertions."""
    defaults = {
        "enabled": True,
        "writable_paths": [],
        "deny_write_paths": [],
        "deny_read_paths": [],
        "allow_network": False,
    }
    defaults.update(overrides)
    return OSSandboxPolicy(**defaults)


# ---------------------------------------------------------------------------
# Basic bwrap args
# ---------------------------------------------------------------------------

class TestBasicBwrapArgs:
    def test_starts_with_bwrap_and_isolation_flags(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        args = sandbox._build_bwrap_args(tmp_path, policy)

        assert args[0] == "bwrap"
        assert "--new-session" in args
        assert "--die-with-parent" in args
        assert "--unshare-pid" in args

    def test_ro_bind_root(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        args = sandbox._build_bwrap_args(tmp_path, policy)

        # Find the --ro-bind / / triple
        idx = args.index("--ro-bind")
        assert args[idx + 1] == "/"
        assert args[idx + 2] == "/"


# ---------------------------------------------------------------------------
# Network isolation
# ---------------------------------------------------------------------------

class TestNetworkIsolation:
    def test_unshare_net_when_network_disallowed(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(allow_network=False)
        args = sandbox._build_bwrap_args(tmp_path, policy)

        assert "--unshare-net" in args

    def test_no_unshare_net_when_network_allowed(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(allow_network=True)
        args = sandbox._build_bwrap_args(tmp_path, policy)

        assert "--unshare-net" not in args


# ---------------------------------------------------------------------------
# Writable paths
# ---------------------------------------------------------------------------

class TestWritablePaths:
    def test_existing_writable_path_gets_bind_mount(self, tmp_path: Path):
        writable_dir = tmp_path / "workspace"
        writable_dir.mkdir()

        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(writable_paths=[str(writable_dir)])
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved = str(writable_dir.resolve())
        # Find --bind <path> <path> for the writable dir
        bind_pairs = [
            (args[i + 1], args[i + 2])
            for i in range(len(args))
            if args[i] == "--bind" and i + 2 < len(args)
        ]
        assert (resolved, resolved) in bind_pairs

    def test_nonexistent_writable_path_gets_tmpfs(self, tmp_path: Path):
        missing_dir = tmp_path / "does_not_exist"

        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(writable_paths=[str(missing_dir)])
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved = str(missing_dir.resolve())
        # Find --tmpfs <path> for the nonexistent writable path
        tmpfs_targets = [
            args[i + 1]
            for i in range(len(args))
            if args[i] == "--tmpfs" and i + 1 < len(args)
        ]
        assert resolved in tmpfs_targets


# ---------------------------------------------------------------------------
# Deny write paths
# ---------------------------------------------------------------------------

class TestDenyWritePaths:
    def test_existing_deny_write_path_gets_ro_bind(self, tmp_path: Path):
        protected_dir = tmp_path / "protected"
        protected_dir.mkdir()

        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(deny_write_paths=[str(protected_dir)])
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved = str(protected_dir.resolve())
        # Find --ro-bind <path> <path> for the deny-write dir
        ro_bind_pairs = [
            (args[i + 1], args[i + 2])
            for i in range(len(args))
            if args[i] == "--ro-bind" and i + 2 < len(args)
        ]
        assert (resolved, resolved) in ro_bind_pairs

    def test_nonexistent_deny_write_path_with_existing_parent_gets_devnull(
        self, tmp_path: Path
    ):
        # Parent exists but target does not
        missing_file = tmp_path / "secret.txt"

        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(deny_write_paths=[str(missing_file)])
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved = str(missing_file.resolve())
        # Find --ro-bind /dev/null <path>
        devnull_targets = [
            args[i + 2]
            for i in range(len(args))
            if args[i] == "--ro-bind"
            and i + 2 < len(args)
            and args[i + 1] == "/dev/null"
        ]
        assert resolved in devnull_targets

    def test_nonexistent_deny_write_path_with_missing_parent_ignored(
        self, tmp_path: Path
    ):
        # Both parent and target do not exist
        deep_missing = tmp_path / "no_parent" / "no_file.txt"

        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(deny_write_paths=[str(deep_missing)])
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved = str(deep_missing.resolve())
        # Should not appear anywhere in args (neither ro-bind nor devnull)
        all_args_str = " ".join(args)
        assert resolved not in all_args_str


# ---------------------------------------------------------------------------
# Deny read paths
# ---------------------------------------------------------------------------

class TestDenyReadPaths:
    def test_directory_deny_read_gets_tmpfs(self, tmp_path: Path):
        secret_dir = tmp_path / "secrets"
        secret_dir.mkdir()

        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(deny_read_paths=[str(secret_dir)])
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved = str(secret_dir.resolve())
        tmpfs_targets = [
            args[i + 1]
            for i in range(len(args))
            if args[i] == "--tmpfs" and i + 1 < len(args)
        ]
        assert resolved in tmpfs_targets

    def test_file_deny_read_gets_devnull(self, tmp_path: Path):
        secret_file = tmp_path / "credentials.json"
        secret_file.write_text("{}")

        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(deny_read_paths=[str(secret_file)])
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved = str(secret_file.resolve())
        devnull_targets = [
            args[i + 2]
            for i in range(len(args))
            if args[i] == "--ro-bind"
            and i + 2 < len(args)
            and args[i + 1] == "/dev/null"
        ]
        assert resolved in devnull_targets

    def test_nonexistent_deny_read_path_ignored(self, tmp_path: Path):
        missing = tmp_path / "ghost"

        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(deny_read_paths=[str(missing)])
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved = str(missing.resolve())
        # Neither is_dir() nor is_file() returns True, so it should be skipped
        # Check it is not in any --tmpfs or --ro-bind /dev/null target
        all_args_str = " ".join(args)
        # The resolved path should not be an argument target for deny-read
        tmpfs_targets = [
            args[i + 1]
            for i in range(len(args))
            if args[i] == "--tmpfs" and i + 1 < len(args)
        ]
        devnull_targets = [
            args[i + 2]
            for i in range(len(args))
            if args[i] == "--ro-bind"
            and i + 2 < len(args)
            and args[i + 1] == "/dev/null"
        ]
        assert resolved not in tmpfs_targets
        assert resolved not in devnull_targets


# ---------------------------------------------------------------------------
# Device and proc
# ---------------------------------------------------------------------------

class TestDeviceAndProc:
    def test_dev_and_proc_present(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        args = sandbox._build_bwrap_args(tmp_path, policy)

        # --dev /dev
        dev_idx = [
            i for i in range(len(args)) if args[i] == "--dev" and i + 1 < len(args)
        ]
        assert len(dev_idx) >= 1
        assert args[dev_idx[0] + 1] == "/dev"

        # --proc /proc
        proc_idx = [
            i for i in range(len(args)) if args[i] == "--proc" and i + 1 < len(args)
        ]
        assert len(proc_idx) >= 1
        assert args[proc_idx[0] + 1] == "/proc"


# ---------------------------------------------------------------------------
# wrap_shell_command
# ---------------------------------------------------------------------------

class TestWrapShellCommand:
    def test_format(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        result = sandbox.wrap_shell_command("echo hello", tmp_path, policy)

        assert result.success is True
        assert result.sandbox_type == "bubblewrap"
        assert result.command.startswith("bwrap ")
        assert "-- /bin/bash -c 'echo hello'" in result.command

    def test_single_quotes_escaped(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        result = sandbox.wrap_shell_command("echo 'it'\\''s'", tmp_path, policy)

        assert result.success is True
        # The command should contain the escaped single quote
        assert "-- /bin/bash -c " in result.command

    def test_command_with_special_chars(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        cmd = "ls -la /tmp && echo 'done'"
        result = sandbox.wrap_shell_command(cmd, tmp_path, policy)

        assert result.success is True
        assert "-- /bin/bash -c " in result.command


# ---------------------------------------------------------------------------
# wrap_python_command
# ---------------------------------------------------------------------------

class TestWrapPythonCommand:
    def test_format(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        python_args = ["/usr/bin/python3", "-c", "print('hi')"]
        result = sandbox.wrap_python_command(python_args, tmp_path, policy)

        assert result.success is True
        assert result.sandbox_type == "bubblewrap"
        assert result.command.startswith("bwrap ")
        assert "-- /usr/bin/python3 -c" in result.command

    def test_python_args_joined(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        python_args = ["/usr/bin/python3", "-u", "-c", "import sys; print(sys.path)"]
        result = sandbox.wrap_python_command(python_args, tmp_path, policy)

        assert result.success is True
        # All python args should appear after --
        cmd = result.command
        separator_idx = cmd.index("-- ")
        after_separator = cmd[separator_idx + 3:]
        assert "/usr/bin/python3" in after_separator
        assert "-u" in after_separator
        assert "-c" in after_separator


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_wrap_shell_command_returns_failure_on_build_error(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()

        with patch.object(
            sandbox, "_build_bwrap_args", side_effect=RuntimeError("mock failure")
        ):
            result = sandbox.wrap_shell_command("echo hi", tmp_path, policy)

        assert result.success is False
        assert "mock failure" in result.error
        assert result.sandbox_type == "bubblewrap"
        # Original command is returned as-is on failure
        assert result.command == "echo hi"

    def test_wrap_python_command_returns_failure_on_build_error(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        python_args = ["/usr/bin/python3", "-c", "pass"]

        with patch.object(
            sandbox, "_build_bwrap_args", side_effect=RuntimeError("mock failure")
        ):
            result = sandbox.wrap_python_command(python_args, tmp_path, policy)

        assert result.success is False
        assert "mock failure" in result.error
        assert result.sandbox_type == "bubblewrap"


# ---------------------------------------------------------------------------
# Working dir always writable
# ---------------------------------------------------------------------------

class TestWorkingDirAlwaysWritable:
    def test_working_dir_gets_bind_mount(self, tmp_path: Path):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy()
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved_wd = str(tmp_path.resolve())
        bind_pairs = [
            (args[i + 1], args[i + 2])
            for i in range(len(args))
            if args[i] == "--bind" and i + 2 < len(args)
        ]
        assert (resolved_wd, resolved_wd) in bind_pairs

    def test_working_dir_writable_even_without_explicit_writable_paths(
        self, tmp_path: Path
    ):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(writable_paths=[])
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved_wd = str(tmp_path.resolve())
        bind_pairs = [
            (args[i + 1], args[i + 2])
            for i in range(len(args))
            if args[i] == "--bind" and i + 2 < len(args)
        ]
        assert (resolved_wd, resolved_wd) in bind_pairs

    def test_working_dir_not_duplicated_if_also_in_writable_paths(
        self, tmp_path: Path
    ):
        sandbox = BubblewrapSandbox()
        policy = _minimal_policy(writable_paths=[str(tmp_path)])
        args = sandbox._build_bwrap_args(tmp_path, policy)

        resolved_wd = str(tmp_path.resolve())
        bind_pairs = [
            (args[i + 1], args[i + 2])
            for i in range(len(args))
            if args[i] == "--bind" and i + 2 < len(args)
        ]
        # Should appear exactly once, not twice
        count = bind_pairs.count((resolved_wd, resolved_wd))
        assert count == 1

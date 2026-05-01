"""Tests for OS sandbox core components.

Tests cover:
- OSSandboxPolicy: defaults, path resolution, deduplication
- SandboxCapabilities: platform detection, cache clearing
- get_os_sandbox factory: force_noop, platform dispatch, caching
- NoOpSandbox: pass-through behavior
- ShellSecurityConfig: serialization of os_sandbox_policy
- ExecutionSandbox integration: command wrapping with mocked subprocess
- SafePythonExecutor integration: Python wrapping with mocked subprocess
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentic_cli.tools.shell.os_sandbox import (
    NoOpSandbox,
    OSSandboxPolicy,
    SandboxCapabilities,
    detect_sandbox_capabilities,
    get_os_sandbox,
    reset_cached_sandbox,
)
from agentic_cli.tools.shell.os_sandbox.base import OSSandboxResult
from agentic_cli.tools.shell.os_sandbox.detect import clear_detection_cache
from agentic_cli.tools.shell.os_sandbox.policy import (
    DEFAULT_READABLE,
    MANDATORY_DENY_WRITE,
)
from agentic_cli.tools.shell.config import ShellSecurityConfig
from agentic_cli.tools.shell.sandbox import ExecutionSandbox
from agentic_cli.tools.executor import SafePythonExecutor


# ---------------------------------------------------------------------------
# Fixtures for cache isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_detection_cache():
    """Clear the detection cache before and after every test."""
    clear_detection_cache()
    yield
    clear_detection_cache()


@pytest.fixture(autouse=True)
def _reset_sandbox_cache():
    """Reset the factory cache before and after every test."""
    reset_cached_sandbox()
    yield
    reset_cached_sandbox()


# ---------------------------------------------------------------------------
# TestOSSandboxPolicy
# ---------------------------------------------------------------------------


class TestOSSandboxPolicy:
    def test_default_policy_has_mandatory_deny_write(self):
        policy = OSSandboxPolicy()
        for entry in MANDATORY_DENY_WRITE:
            assert entry in policy.deny_write_paths

    def test_resolved_writable_paths_includes_working_dir(self, tmp_path: Path):
        policy = OSSandboxPolicy()
        paths = policy.resolved_writable_paths(tmp_path)
        assert tmp_path.resolve() in paths

    def test_resolved_writable_paths_includes_extra_paths(self, tmp_path: Path):
        extra = tmp_path / "extra"
        extra.mkdir()
        policy = OSSandboxPolicy(writable_paths=[str(extra)])
        paths = policy.resolved_writable_paths(tmp_path)
        assert extra.resolve() in paths
        assert tmp_path.resolve() in paths

    def test_resolved_writable_paths_deduplicates(self, tmp_path: Path):
        # Pass the working_dir itself as an extra writable path
        policy = OSSandboxPolicy(writable_paths=[str(tmp_path)])
        paths = policy.resolved_writable_paths(tmp_path)
        resolved = tmp_path.resolve()
        assert paths.count(resolved) == 1

    def test_resolved_deny_write_paths_expands_and_resolves(self):
        policy = OSSandboxPolicy(deny_write_paths=["~/.bashrc", "/etc/"])
        resolved = policy.resolved_deny_write_paths()
        assert len(resolved) == 2
        home_bashrc = Path("~/.bashrc").expanduser().resolve()
        assert home_bashrc in resolved
        assert all(isinstance(p, Path) for p in resolved)

    def test_resolved_deny_read_paths(self):
        policy = OSSandboxPolicy(deny_read_paths=["~/.ssh/", "/secret"])
        resolved = policy.resolved_deny_read_paths()
        assert len(resolved) == 2
        home_ssh = Path("~/.ssh/").expanduser().resolve()
        assert home_ssh in resolved

    def test_resolved_readable_paths_returns_defaults(self):
        policy = OSSandboxPolicy()
        readable = policy.resolved_readable_paths()
        assert len(readable) == len(DEFAULT_READABLE)
        # Check a known entry
        assert Path("/usr/") in readable


# ---------------------------------------------------------------------------
# TestSandboxCapabilities
# ---------------------------------------------------------------------------


class TestSandboxCapabilities:
    def test_darwin_with_sandbox_exec(self):
        with patch.object(sys, "platform", "darwin"), \
             patch("agentic_cli.tools.shell.os_sandbox.detect.sys") as mock_sys, \
             patch("agentic_cli.tools.shell.os_sandbox.detect.shutil") as mock_shutil:
            mock_sys.platform = "darwin"
            mock_shutil.which.return_value = "/usr/bin/sandbox-exec"
            caps = detect_sandbox_capabilities()
            assert caps.seatbelt_available is True
            assert caps.bubblewrap_available is False
            assert caps.platform == "darwin"

    def test_linux_with_bwrap(self):
        with patch("agentic_cli.tools.shell.os_sandbox.detect.sys") as mock_sys, \
             patch("agentic_cli.tools.shell.os_sandbox.detect.shutil") as mock_shutil:
            mock_sys.platform = "linux"
            mock_shutil.which.return_value = "/usr/bin/bwrap"
            caps = detect_sandbox_capabilities()
            assert caps.bubblewrap_available is True
            assert caps.seatbelt_available is False
            assert caps.platform == "linux"

    def test_darwin_no_sandbox_exec(self):
        with patch("agentic_cli.tools.shell.os_sandbox.detect.sys") as mock_sys, \
             patch("agentic_cli.tools.shell.os_sandbox.detect.shutil") as mock_shutil:
            mock_sys.platform = "darwin"
            mock_shutil.which.return_value = None
            caps = detect_sandbox_capabilities()
            assert caps.seatbelt_available is False

    def test_win32_both_false(self):
        with patch("agentic_cli.tools.shell.os_sandbox.detect.sys") as mock_sys, \
             patch("agentic_cli.tools.shell.os_sandbox.detect.shutil") as mock_shutil:
            mock_sys.platform = "win32"
            caps = detect_sandbox_capabilities()
            assert caps.seatbelt_available is False
            assert caps.bubblewrap_available is False

    def test_clear_detection_cache_resets(self):
        """After calling detect once, clearing the cache allows a new detection."""
        with patch("agentic_cli.tools.shell.os_sandbox.detect.sys") as mock_sys, \
             patch("agentic_cli.tools.shell.os_sandbox.detect.shutil") as mock_shutil:
            mock_sys.platform = "win32"
            caps1 = detect_sandbox_capabilities()
            assert caps1.platform == "win32"

        clear_detection_cache()

        with patch("agentic_cli.tools.shell.os_sandbox.detect.sys") as mock_sys, \
             patch("agentic_cli.tools.shell.os_sandbox.detect.shutil") as mock_shutil:
            mock_sys.platform = "darwin"
            mock_shutil.which.return_value = "/usr/bin/sandbox-exec"
            caps2 = detect_sandbox_capabilities()
            assert caps2.platform == "darwin"
            assert caps2.seatbelt_available is True


# ---------------------------------------------------------------------------
# TestGetOSSandbox
# ---------------------------------------------------------------------------


class TestGetOSSandbox:
    def test_force_noop_returns_noop(self):
        sandbox = get_os_sandbox(force_noop=True)
        assert isinstance(sandbox, NoOpSandbox)
        assert sandbox.sandbox_type == "none"

    def test_darwin_with_sandbox_exec_returns_seatbelt(self):
        with patch(
            "agentic_cli.tools.shell.os_sandbox.detect_sandbox_capabilities"
        ) as mock_detect:
            mock_detect.return_value = SandboxCapabilities(
                platform="darwin",
                seatbelt_available=True,
                bubblewrap_available=False,
            )
            sandbox = get_os_sandbox()
            from agentic_cli.tools.shell.os_sandbox.seatbelt import SeatbeltSandbox

            assert isinstance(sandbox, SeatbeltSandbox)

    def test_linux_with_bwrap_returns_bubblewrap(self):
        with patch(
            "agentic_cli.tools.shell.os_sandbox.detect_sandbox_capabilities"
        ) as mock_detect:
            mock_detect.return_value = SandboxCapabilities(
                platform="linux",
                seatbelt_available=False,
                bubblewrap_available=True,
            )
            sandbox = get_os_sandbox()
            from agentic_cli.tools.shell.os_sandbox.bubblewrap import (
                BubblewrapSandbox,
            )

            assert isinstance(sandbox, BubblewrapSandbox)

    def test_no_tools_returns_noop(self):
        with patch(
            "agentic_cli.tools.shell.os_sandbox.detect_sandbox_capabilities"
        ) as mock_detect:
            mock_detect.return_value = SandboxCapabilities(
                platform="win32",
                seatbelt_available=False,
                bubblewrap_available=False,
            )
            sandbox = get_os_sandbox()
            assert isinstance(sandbox, NoOpSandbox)

    def test_reset_cached_sandbox_clears_cache(self):
        with patch(
            "agentic_cli.tools.shell.os_sandbox.detect_sandbox_capabilities"
        ) as mock_detect:
            mock_detect.return_value = SandboxCapabilities(
                platform="win32",
                seatbelt_available=False,
                bubblewrap_available=False,
            )
            first = get_os_sandbox()
            assert isinstance(first, NoOpSandbox)

        reset_cached_sandbox()

        with patch(
            "agentic_cli.tools.shell.os_sandbox.detect_sandbox_capabilities"
        ) as mock_detect:
            mock_detect.return_value = SandboxCapabilities(
                platform="darwin",
                seatbelt_available=True,
                bubblewrap_available=False,
            )
            second = get_os_sandbox()
            from agentic_cli.tools.shell.os_sandbox.seatbelt import SeatbeltSandbox

            assert isinstance(second, SeatbeltSandbox)


# ---------------------------------------------------------------------------
# TestNoOpSandbox
# ---------------------------------------------------------------------------


class TestNoOpSandbox:
    def test_wrap_shell_command_returns_unchanged(self, tmp_path: Path):
        sandbox = NoOpSandbox()
        policy = OSSandboxPolicy()
        result = sandbox.wrap_shell_command("echo hello", tmp_path, policy)
        assert result.command == "echo hello"
        assert result.success is True
        assert result.sandbox_type == "none"

    def test_wrap_python_command_returns_joined(self, tmp_path: Path):
        sandbox = NoOpSandbox()
        policy = OSSandboxPolicy()
        args = [sys.executable, "-c", "print('hi')"]
        result = sandbox.wrap_python_command(args, tmp_path, policy)
        assert result.command == shlex.join(args)
        assert result.success is True
        assert result.sandbox_type == "none"

    def test_is_available(self):
        sandbox = NoOpSandbox()
        assert sandbox.is_available() is True

    def test_sandbox_type(self):
        sandbox = NoOpSandbox()
        assert sandbox.sandbox_type == "none"


# ---------------------------------------------------------------------------
# TestShellSecurityConfigOSSandbox
# ---------------------------------------------------------------------------


class TestShellSecurityConfigOSSandbox:
    def test_from_dict_with_os_sandbox_policy(self):
        data = {
            "os_sandbox_policy": {
                "enabled": True,
                "writable_paths": ["/tmp/work"],
                "deny_write_paths": ["~/.bashrc"],
                "deny_read_paths": ["/secret"],
                "allow_network": True,
            }
        }
        config = ShellSecurityConfig.from_dict(data)
        assert config.os_sandbox_policy is not None
        assert config.os_sandbox_policy.enabled is True
        assert "/tmp/work" in config.os_sandbox_policy.writable_paths
        assert "~/.bashrc" in config.os_sandbox_policy.deny_write_paths
        assert "/secret" in config.os_sandbox_policy.deny_read_paths
        assert config.os_sandbox_policy.allow_network is True

    def test_from_dict_without_os_sandbox_policy(self):
        config = ShellSecurityConfig.from_dict({})
        assert config.os_sandbox_policy is None

    def test_to_dict_serializes_os_sandbox_policy(self):
        policy = OSSandboxPolicy(
            enabled=True,
            writable_paths=["/extra"],
            allow_network=True,
        )
        config = ShellSecurityConfig(os_sandbox_policy=policy)
        d = config.to_dict()
        assert d["os_sandbox_policy"] is not None
        assert d["os_sandbox_policy"]["enabled"] is True
        assert "/extra" in d["os_sandbox_policy"]["writable_paths"]
        assert d["os_sandbox_policy"]["allow_network"] is True

    def test_to_dict_with_none_os_sandbox_policy(self):
        config = ShellSecurityConfig(os_sandbox_policy=None)
        d = config.to_dict()
        assert d["os_sandbox_policy"] is None

    def test_merge_with_preserves_os_sandbox_policy(self):
        policy = OSSandboxPolicy(writable_paths=["/merge"])
        base = ShellSecurityConfig(os_sandbox_policy=None)
        other = ShellSecurityConfig(os_sandbox_policy=policy)
        merged = base.merge_with(other)
        assert merged.os_sandbox_policy is not None
        assert "/merge" in merged.os_sandbox_policy.writable_paths

    def test_merge_with_keeps_base_when_other_is_none(self):
        policy = OSSandboxPolicy(writable_paths=["/base"])
        base = ShellSecurityConfig(os_sandbox_policy=policy)
        other = ShellSecurityConfig(os_sandbox_policy=None)
        merged = base.merge_with(other)
        assert merged.os_sandbox_policy is not None
        assert "/base" in merged.os_sandbox_policy.writable_paths


# ---------------------------------------------------------------------------
# TestExecutionSandboxIntegration
# ---------------------------------------------------------------------------


class TestExecutionSandboxIntegration:
    def _make_mock_popen(self, stdout: bytes = b"ok\n", stderr: bytes = b"", returncode: int = 0):
        """Create a mock Popen that returns predictable output."""
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (stdout, stderr)
        mock_proc.returncode = returncode
        mock_popen = MagicMock(return_value=mock_proc)
        return mock_popen

    def test_no_policy_executes_as_before(self, tmp_path: Path):
        sandbox = ExecutionSandbox(os_sandbox_policy=None)
        mock_popen = self._make_mock_popen()

        with patch("agentic_cli.tools.shell.sandbox.subprocess.Popen", mock_popen):
            result = sandbox.execute("echo hello", working_dir=tmp_path)

        assert result.success is True
        assert result.stdout == "ok\n"
        # Popen was called with the original command (no wrapping)
        call_args = mock_popen.call_args
        assert "echo hello" in call_args[0][0]

    def test_disabled_policy_executes_as_before(self, tmp_path: Path):
        policy = OSSandboxPolicy(enabled=False)
        sandbox = ExecutionSandbox(os_sandbox_policy=policy)
        mock_popen = self._make_mock_popen()

        with patch("agentic_cli.tools.shell.sandbox.subprocess.Popen", mock_popen):
            result = sandbox.execute("echo hello", working_dir=tmp_path)

        assert result.success is True
        # The command was passed through without wrapping
        call_args = mock_popen.call_args
        assert "echo hello" in call_args[0][0]

    def test_enabled_policy_wraps_command(self, tmp_path: Path):
        policy = OSSandboxPolicy(enabled=True)
        sandbox = ExecutionSandbox(os_sandbox_policy=policy)

        mock_os_sandbox = MagicMock()
        mock_os_sandbox.wrap_shell_command.return_value = OSSandboxResult(
            command="sandbox-exec -p '...' /bin/bash -c 'echo hello'",
            sandbox_type="seatbelt",
            success=True,
        )
        mock_popen = self._make_mock_popen()

        with patch(
            "agentic_cli.tools.shell.sandbox.subprocess.Popen", mock_popen
        ), patch(
            "agentic_cli.tools.shell.os_sandbox.get_os_sandbox",
            return_value=mock_os_sandbox,
        ):
            result = sandbox.execute("echo hello", working_dir=tmp_path)

        assert result.success is True
        # wrap_shell_command was called
        mock_os_sandbox.wrap_shell_command.assert_called_once()
        call_args = mock_os_sandbox.wrap_shell_command.call_args
        assert call_args[0][0] == "echo hello"
        # Popen received the wrapped command
        popen_cmd = mock_popen.call_args[0][0]
        assert "sandbox-exec" in popen_cmd

    def test_enabled_policy_wrap_failure_fails_closed(self, tmp_path: Path):
        policy = OSSandboxPolicy(enabled=True)
        sandbox = ExecutionSandbox(os_sandbox_policy=policy)

        mock_os_sandbox = MagicMock()
        mock_os_sandbox.sandbox_type = "seatbelt"
        mock_os_sandbox.wrap_shell_command.return_value = OSSandboxResult(
            command="echo hello",
            sandbox_type="seatbelt",
            success=False,
            error="sandbox-exec not found",
        )
        mock_popen = self._make_mock_popen()

        with patch(
            "agentic_cli.tools.shell.sandbox.subprocess.Popen", mock_popen
        ), patch(
            "agentic_cli.tools.shell.os_sandbox.get_os_sandbox",
            return_value=mock_os_sandbox,
        ):
            result = sandbox.execute("echo hello", working_dir=tmp_path)

        # Wrap failure must fail closed instead of dropping back to plain subprocess.
        assert result.success is False
        assert "sandbox" in (result.error or "").lower()
        mock_popen.assert_not_called()

    def test_enabled_policy_no_real_sandbox_fails_closed(self, tmp_path: Path):
        policy = OSSandboxPolicy(enabled=True)
        sandbox = ExecutionSandbox(os_sandbox_policy=policy)

        mock_os_sandbox = MagicMock()
        mock_os_sandbox.sandbox_type = "none"
        mock_popen = self._make_mock_popen()

        with patch(
            "agentic_cli.tools.shell.sandbox.subprocess.Popen", mock_popen
        ), patch(
            "agentic_cli.tools.shell.os_sandbox.get_os_sandbox",
            return_value=mock_os_sandbox,
        ):
            result = sandbox.execute("echo hello", working_dir=tmp_path)

        assert result.success is False
        assert "sandbox" in (result.error or "").lower()
        # We refused to run, so wrap and Popen must not have been called.
        mock_os_sandbox.wrap_shell_command.assert_not_called()
        mock_popen.assert_not_called()


# ---------------------------------------------------------------------------
# TestSafePythonExecutorIntegration
# ---------------------------------------------------------------------------


class TestSafePythonExecutorIntegration:
    def test_no_policy_executes_as_before(self):
        executor = SafePythonExecutor(os_sandbox_policy=None)

        mock_proc = MagicMock()
        mock_proc.stdout = (
            "hello\n"
            "\n__AGENTIC_EXECUTOR_RESULT_SENTINEL__\n"
            '{"success": true, "output": "", "result": null, '
            '"error": "", "execution_time_ms": 0}'
        )
        mock_proc.stderr = ""
        mock_proc.returncode = 0

        with patch("agentic_cli.tools.executor.subprocess.run", return_value=mock_proc):
            result = executor.execute("print('hello')")

        assert result["success"] is True

    def test_enabled_policy_wraps_python_command(self):
        policy = OSSandboxPolicy(enabled=True)
        executor = SafePythonExecutor(os_sandbox_policy=policy)

        mock_os_sandbox = MagicMock()
        mock_os_sandbox.wrap_python_command.return_value = OSSandboxResult(
            command=f"sandbox-exec -p '...' {sys.executable} -c 'script'",
            sandbox_type="seatbelt",
            success=True,
        )

        mock_proc = MagicMock()
        mock_proc.stdout = (
            "\n__AGENTIC_EXECUTOR_RESULT_SENTINEL__\n"
            '{"success": true, "output": "", "result": "42", '
            '"error": "", "execution_time_ms": 0}'
        )
        mock_proc.stderr = ""
        mock_proc.returncode = 0

        with patch(
            "agentic_cli.tools.executor.subprocess.run", return_value=mock_proc
        ) as mock_run, patch(
            "agentic_cli.tools.shell.os_sandbox.get_os_sandbox",
            return_value=mock_os_sandbox,
        ):
            result = executor.execute("1 + 1")

        assert result["success"] is True
        # wrap_python_command was called
        mock_os_sandbox.wrap_python_command.assert_called_once()
        # subprocess.run was called with shell=True (wrapped command)
        assert mock_run.call_args.kwargs.get("shell") is True

    def test_enabled_policy_wrap_failure_fails_closed(self):
        policy = OSSandboxPolicy(enabled=True)
        executor = SafePythonExecutor(os_sandbox_policy=policy)

        mock_os_sandbox = MagicMock()
        mock_os_sandbox.sandbox_type = "seatbelt"
        mock_os_sandbox.wrap_python_command.return_value = OSSandboxResult(
            command="",
            sandbox_type="seatbelt",
            success=False,
            error="sandbox-exec not found",
        )

        with patch(
            "agentic_cli.tools.executor.subprocess.run"
        ) as mock_run, patch(
            "agentic_cli.tools.shell.os_sandbox.get_os_sandbox",
            return_value=mock_os_sandbox,
        ):
            result = executor.execute("1 + 1")

        # Wrap failure must fail closed rather than running unwrapped.
        assert result["success"] is False
        assert "sandbox" in (result.get("error") or "").lower()
        mock_run.assert_not_called()

    def test_enabled_policy_no_real_sandbox_fails_closed(self):
        policy = OSSandboxPolicy(enabled=True)
        executor = SafePythonExecutor(os_sandbox_policy=policy)

        mock_os_sandbox = MagicMock()
        mock_os_sandbox.sandbox_type = "none"

        with patch(
            "agentic_cli.tools.executor.subprocess.run"
        ) as mock_run, patch(
            "agentic_cli.tools.shell.os_sandbox.get_os_sandbox",
            return_value=mock_os_sandbox,
        ):
            result = executor.execute("1 + 1")

        assert result["success"] is False
        assert "sandbox" in (result.get("error") or "").lower()
        mock_os_sandbox.wrap_python_command.assert_not_called()
        mock_run.assert_not_called()

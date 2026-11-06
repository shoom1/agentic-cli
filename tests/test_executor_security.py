"""Security tests for SafePythonExecutor module gating.

File/network/pickle-capable libraries (numpy, pandas, ...) re-expose the very
capabilities the restricted builtins remove (e.g. numpy.DataSource().open(),
pandas.read_pickle(url)). Outside the OS sandbox they hand the code the host in
one call, so a run gets them only when it runs inside the sandbox: the set is
chosen from where the call actually runs, when it runs.
"""

import shlex
import subprocess
from types import SimpleNamespace

import pytest

from agentic_cli.tools.executor import SafePythonExecutor
from agentic_cli.tools.shell.os_sandbox.policy import OSSandboxPolicy
from tests.tools.test_executor_security import _runner_modules


class _FakeSandbox:
    """Stand-in OS sandbox. ``sandbox_type='none'`` = unavailable; otherwise the
    wrap just runs the plain command (no real isolation — we only test wiring)."""

    def __init__(self, sandbox_type: str = "seatbelt") -> None:
        self._type = sandbox_type
        self.wrapped: list[list[str]] = []

    @property
    def sandbox_type(self) -> str:
        return self._type

    def wrap_python_command(self, argv, working_dir, policy):
        from agentic_cli.tools.shell.os_sandbox.base import OSSandboxResult

        self.wrapped.append(list(argv))
        return OSSandboxResult(command=shlex.join(argv), sandbox_type=self._type)


def _patch_sandbox(monkeypatch, sandbox_type: str) -> _FakeSandbox:
    sandbox = _FakeSandbox(sandbox_type)
    monkeypatch.setattr(
        "agentic_cli.tools.shell.os_sandbox.get_os_sandbox",
        lambda *a, **k: sandbox,
    )
    return sandbox


def _scripts_run(monkeypatch) -> list[str]:
    """The runner script of each unwrapped subprocess the executor starts."""
    scripts: list[str] = []
    real_run = subprocess.run

    def _run(*args, **kwargs):
        if not kwargs.get("shell"):
            scripts.append(args[0][2])
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _run)
    return scripts


class TestModuleGatingWithoutSandbox:
    """No OS sandbox -> only the pure-computation core is importable."""

    def setup_method(self):
        self.executor = SafePythonExecutor()  # os_sandbox_policy=None

    def test_the_run_allows_core_modules_only(self, monkeypatch):
        scripts = _scripts_run(monkeypatch)
        assert self.executor.execute("1 + 1")["success"] is True
        assert _runner_modules(scripts[0]) == self.executor.CORE_MODULES

    def test_numpy_import_rejected(self):
        ok, err = self.executor.validate_code("import numpy")
        assert ok is False
        assert "numpy" in err

    def test_pandas_from_import_rejected(self):
        ok, _ = self.executor.validate_code("from pandas import read_pickle")
        assert ok is False

    def test_datasource_escape_blocked_at_validation(self):
        escape = "import numpy as np\nnp.DataSource().open('/etc/hosts').read()"
        ok, _ = self.executor.validate_code(escape)
        assert ok is False

    def test_numpy_not_prebound_in_namespace(self):
        # Even if numpy is installed in the host env, `np`/`numpy` must not be
        # reachable without an (allowed) import — otherwise the import gate is moot.
        result = self.executor.execute("np")
        assert result["success"] is False
        result2 = self.executor.execute("numpy")
        assert result2["success"] is False

    def test_core_module_still_works(self):
        ok, _ = self.executor.validate_code("import math")
        assert ok is True
        result = self.executor.execute("import math\nmath.sqrt(16)")
        assert result["success"] is True
        assert result["result"] == "4.0"


class TestModuleGatingWithSandbox:
    """OS sandbox enabled AND available -> the heavy libraries are permitted."""

    def test_a_sandboxed_run_allows_every_module(self, monkeypatch):
        sandbox = _patch_sandbox(monkeypatch, "seatbelt")
        executor = SafePythonExecutor(os_sandbox_policy=OSSandboxPolicy(enabled=True))
        assert executor.execute("1 + 1")["success"] is True
        assert _runner_modules(sandbox.wrapped[0][2]) == executor.ALLOWED_MODULES

    def test_numpy_import_allowed_when_sandbox_available(self, monkeypatch):
        _patch_sandbox(monkeypatch, "seatbelt")
        executor = SafePythonExecutor(os_sandbox_policy=OSSandboxPolicy(enabled=True))
        result = executor.execute("import numpy\n'ok'")
        assert "not allowed" not in (result["error"] or ""), result

    def test_disabled_policy_is_treated_as_no_sandbox(self, monkeypatch):
        sandbox = _patch_sandbox(monkeypatch, "seatbelt")
        executor = SafePythonExecutor(os_sandbox_policy=SimpleNamespace(enabled=False))
        result = executor.execute("import numpy")
        assert result["success"] is False and "not allowed" in result["error"]
        assert sandbox.wrapped == []


class TestOsSandboxAvailabilityAware:
    """Enabling the sandbox but having no real backend must NOT re-enable the
    heavy modules — availability, not the flag, gates SANDBOXED_MODULES (5b)."""

    def test_enabled_but_unavailable_falls_back_core_only(self, monkeypatch):
        _patch_sandbox(monkeypatch, "none")
        scripts = _scripts_run(monkeypatch)
        ex = SafePythonExecutor(
            os_sandbox_policy=OSSandboxPolicy(enabled=True, strict=False)
        )
        assert ex.execute("1 + 1")["success"] is True
        assert _runner_modules(scripts[0]) == ex.CORE_MODULES

    def test_unavailable_non_strict_falls_back_and_runs(self, monkeypatch):
        """Default (non-strict): no backend -> hardened in-process, not refusal."""
        _patch_sandbox(monkeypatch, "none")
        ex = SafePythonExecutor(
            os_sandbox_policy=OSSandboxPolicy(enabled=True, strict=False)
        )
        # numpy still rejected (CORE only) ...
        assert ex.execute("import numpy")["success"] is False
        # ... but ordinary compute runs rather than being refused.
        result = ex.execute("import math\nprint(math.sqrt(16))")
        assert result["success"] is True, result
        assert "4.0" in result["output"]

    def test_unavailable_strict_refuses(self, monkeypatch):
        _patch_sandbox(monkeypatch, "none")
        ex = SafePythonExecutor(
            os_sandbox_policy=OSSandboxPolicy(enabled=True, strict=True)
        )
        result = ex.execute("1 + 1")
        assert result["success"] is False
        assert "sandbox" in result["error"].lower()

    def test_available_runs_under_sandbox(self, monkeypatch):
        _patch_sandbox(monkeypatch, "seatbelt")
        ex = SafePythonExecutor(os_sandbox_policy=OSSandboxPolicy(enabled=True))
        result = ex.execute("import math\nprint(math.sqrt(16))")
        assert result["success"] is True, result


class TestOsSandboxSettingsDefaults:
    def test_os_sandbox_enabled_defaults_true(self):
        from agentic_cli.config import BaseSettings

        settings = BaseSettings()
        assert settings.os_sandbox_enabled is True

    def test_os_sandbox_strict_defaults_true(self):
        """With no sandbox backend, or a wrap that fails, code is refused rather
        than run as the user: falling back is an opt-in."""
        from agentic_cli.config import BaseSettings

        settings = BaseSettings()
        assert settings.os_sandbox_strict is True

    def test_a_policy_is_strict_unless_told_otherwise(self, monkeypatch):
        """``OSSandboxPolicy`` has the settings' default: an executor given an
        enabled policy never falls back without one, unless it says so."""
        assert OSSandboxPolicy().strict is True
        _patch_sandbox(monkeypatch, "none")
        ex = SafePythonExecutor(os_sandbox_policy=OSSandboxPolicy(enabled=True))
        result = ex.execute("1 + 1")
        assert result["success"] is False
        assert "sandbox" in result["error"].lower()


class TestExecPolicy:
    """Which code-execution capabilities may run on the host, by settings."""

    @pytest.mark.parametrize(
        "settings, uncontained",
        [
            ({}, set()),
            ({"os_sandbox_enabled": False}, {"python.exec"}),
            ({"os_sandbox_strict": False}, {"python.exec"}),
            ({"stateful_executor_backend": "local"}, {"python.exec.stateful"}),
            ({"stateful_executor_backend": "docker"}, set()),
            (
                {"os_sandbox_enabled": False, "stateful_executor_backend": "local"},
                {"python.exec", "python.exec.stateful"},
            ),
        ],
    )
    def test_uncontained_exec_capabilities(self, settings, uncontained):
        from agentic_cli.config import BaseSettings
        from agentic_cli.tools.exec_policy import uncontained_exec_capabilities

        assert uncontained_exec_capabilities(BaseSettings(**settings)) == uncontained

    def test_the_tool_s_policy_follows_the_same_settings(self):
        from agentic_cli.config import BaseSettings
        from agentic_cli.tools.exec_policy import python_sandbox_policy

        assert python_sandbox_policy(BaseSettings(os_sandbox_enabled=False)) is None
        assert python_sandbox_policy(BaseSettings()).strict is True
        assert python_sandbox_policy(BaseSettings(os_sandbox_strict=False)).strict is False

"""Security tests for SafePythonExecutor module gating.

File/network/pickle-capable libraries (numpy, pandas, ...) re-expose the very
capabilities the restricted builtins remove (e.g. numpy.DataSource().open(),
pandas.read_pickle(url)). Without a real OS sandbox they are an escape from the
in-process restrictions, so they must be unavailable unless os_sandbox is on.
"""

import shlex
from types import SimpleNamespace

import pytest

from agentic_cli.tools.executor import SafePythonExecutor
from agentic_cli.tools.shell.os_sandbox.policy import OSSandboxPolicy


class _FakeSandbox:
    """Stand-in OS sandbox. ``sandbox_type='none'`` = unavailable; otherwise the
    wrap just runs the plain command (no real isolation — we only test wiring)."""

    def __init__(self, sandbox_type: str = "seatbelt") -> None:
        self._type = sandbox_type

    @property
    def sandbox_type(self) -> str:
        return self._type

    def wrap_python_command(self, argv, working_dir, policy):
        from agentic_cli.tools.shell.os_sandbox.base import OSSandboxResult

        return OSSandboxResult(command=shlex.join(argv), sandbox_type=self._type)


def _patch_sandbox(monkeypatch, sandbox_type: str) -> None:
    monkeypatch.setattr(
        "agentic_cli.tools.shell.os_sandbox.get_os_sandbox",
        lambda *a, **k: _FakeSandbox(sandbox_type),
    )


class TestModuleGatingWithoutSandbox:
    """No OS sandbox -> only the pure-computation core is importable."""

    def setup_method(self):
        self.executor = SafePythonExecutor()  # os_sandbox_policy=None

    def test_effective_set_is_core_only(self):
        assert self.executor.effective_allowed_modules == self.executor.CORE_MODULES
        assert "numpy" not in self.executor.effective_allowed_modules
        assert "pandas" not in self.executor.effective_allowed_modules

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

    def test_effective_set_is_full_when_sandbox_available(self, monkeypatch):
        _patch_sandbox(monkeypatch, "seatbelt")
        executor = SafePythonExecutor(os_sandbox_policy=OSSandboxPolicy(enabled=True))
        assert executor.effective_allowed_modules == executor.ALLOWED_MODULES
        assert "numpy" in executor.effective_allowed_modules

    def test_numpy_import_allowed_when_sandbox_available(self, monkeypatch):
        _patch_sandbox(monkeypatch, "seatbelt")
        executor = SafePythonExecutor(os_sandbox_policy=OSSandboxPolicy(enabled=True))
        ok, _ = executor.validate_code("import numpy")
        assert ok is True

    def test_disabled_policy_is_treated_as_no_sandbox(self):
        policy = SimpleNamespace(enabled=False)
        executor = SafePythonExecutor(os_sandbox_policy=policy)
        assert executor.effective_allowed_modules == executor.CORE_MODULES


class TestOsSandboxAvailabilityAware:
    """Enabling the sandbox but having no real backend must NOT re-enable the
    heavy modules — availability, not the flag, gates SANDBOXED_MODULES (5b)."""

    def test_enabled_but_unavailable_is_core_only(self, monkeypatch):
        _patch_sandbox(monkeypatch, "none")
        ex = SafePythonExecutor(os_sandbox_policy=OSSandboxPolicy(enabled=True))
        assert ex.effective_allowed_modules == ex.CORE_MODULES
        assert "numpy" not in ex.effective_allowed_modules

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

    def test_os_sandbox_strict_defaults_false(self):
        from agentic_cli.config import BaseSettings

        settings = BaseSettings()
        assert settings.os_sandbox_strict is False

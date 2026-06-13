"""Security tests for SafePythonExecutor module gating.

File/network/pickle-capable libraries (numpy, pandas, ...) re-expose the very
capabilities the restricted builtins remove (e.g. numpy.DataSource().open(),
pandas.read_pickle(url)). Without a real OS sandbox they are an escape from the
in-process restrictions, so they must be unavailable unless os_sandbox is on.
"""

from types import SimpleNamespace

from agentic_cli.tools.executor import SafePythonExecutor


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
    """OS sandbox enabled -> the heavy libraries are permitted again."""

    def test_effective_set_is_full_when_sandbox_enabled(self):
        policy = SimpleNamespace(enabled=True)
        executor = SafePythonExecutor(os_sandbox_policy=policy)
        assert executor.effective_allowed_modules == executor.ALLOWED_MODULES
        assert "numpy" in executor.effective_allowed_modules

    def test_numpy_import_allowed_when_sandbox_enabled(self):
        policy = SimpleNamespace(enabled=True)
        executor = SafePythonExecutor(os_sandbox_policy=policy)
        ok, _ = executor.validate_code("import numpy")
        assert ok is True

    def test_disabled_policy_is_treated_as_no_sandbox(self):
        policy = SimpleNamespace(enabled=False)
        executor = SafePythonExecutor(os_sandbox_policy=policy)
        assert executor.effective_allowed_modules == executor.CORE_MODULES

"""Safe Python code execution for exploration tasks.

Provides a sandboxed environment for executing Python code
with restricted capabilities.
"""

from __future__ import annotations

import ast
import contextlib
import io
import signal
import sys
import time
import traceback
from typing import Any


class TimeoutError(Exception):
    """Raised when code execution times out."""

    pass


class SafePythonExecutor:
    """Executes Python code in a restricted environment.

    Provides safety through:
    - AST validation (blocks dangerous patterns)
    - Restricted namespace (limited builtins)
    - Execution timeout
    - Output capture
    """

    # Modules allowed for import
    ALLOWED_MODULES = {
        # Core math/science
        "numpy",
        "pandas",
        "scipy",
        "sympy",
        "math",
        "statistics",
        "cmath",
        # Collections and utilities
        "collections",
        "itertools",
        "functools",
        "operator",
        # Data handling
        "json",
        "re",
        "datetime",
        "decimal",
        "fractions",
        # Random (for simulations)
        "random",
    }

    # Builtins that are blocked
    BLOCKED_BUILTINS = {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "__import__",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "type",
        "isinstance",
        "issubclass",
        "object",
        "super",
        "classmethod",
        "staticmethod",
        "property",
        "breakpoint",
        "help",
        "license",
        "credits",
        "copyright",
        "exit",
        "quit",
    }

    # Dangerous AST nodes
    BLOCKED_AST_NODES = {
        ast.Import,  # We handle imports specially
        ast.ImportFrom,
    }

    def __init__(self, default_timeout: int = 30) -> None:
        """Initialize the executor.

        Args:
            default_timeout: Default execution timeout in seconds.
        """
        self.default_timeout = default_timeout
        self._namespace: dict[str, Any] = {}
        self._setup_namespace()

    def _setup_namespace(self) -> None:
        """Set up the restricted execution namespace."""
        # Safe builtins
        safe_builtins = {
            k: v
            for k, v in __builtins__.items()  # type: ignore[attr-defined]
            if k not in self.BLOCKED_BUILTINS
        }

        # Add safe import function
        safe_builtins["__import__"] = self._safe_import

        self._namespace = {
            "__builtins__": safe_builtins,
            "__name__": "__main__",
            "__doc__": None,
        }

        # Pre-import common modules
        self._preload_modules()

    def _preload_modules(self) -> None:
        """Pre-load commonly used modules into namespace."""
        preload = ["math", "json", "re", "datetime", "collections"]

        for module_name in preload:
            try:
                module = __import__(module_name)
                self._namespace[module_name] = module
            except ImportError:
                pass

        # Try to import numpy and pandas (may not be available)
        try:
            import numpy as np

            self._namespace["np"] = np
            self._namespace["numpy"] = np
        except ImportError:
            pass

        try:
            import pandas as pd

            self._namespace["pd"] = pd
            self._namespace["pandas"] = pd
        except ImportError:
            pass

    def _safe_import(
        self,
        name: str,
        globals_: dict | None = None,
        locals_: dict | None = None,
        fromlist: tuple = (),
        level: int = 0,
    ) -> Any:
        """Safe import function that only allows whitelisted modules."""
        if name not in self.ALLOWED_MODULES:
            raise ImportError(
                f"Module '{name}' is not allowed. "
                f"Allowed modules: {', '.join(sorted(self.ALLOWED_MODULES))}"
            )
        return __import__(name, globals_, locals_, fromlist, level)

    def validate_code(self, code: str) -> tuple[bool, str]:
        """Validate code for safety.

        Args:
            code: Python code to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Check for dangerous patterns
        for node in ast.walk(tree):
            # Check for blocked node types
            if type(node) in self.BLOCKED_AST_NODES:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Check if importing allowed modules
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.split(".")[0] not in self.ALLOWED_MODULES:
                                return (
                                    False,
                                    f"Import of '{alias.name}' is not allowed",
                                )
                    elif isinstance(node, ast.ImportFrom):
                        if (
                            node.module
                            and node.module.split(".")[0] not in self.ALLOWED_MODULES
                        ):
                            return (
                                False,
                                f"Import from '{node.module}' is not allowed",
                            )

            # Check for dangerous attribute access
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("_"):
                    return (
                        False,
                        f"Access to private attribute '{node.attr}' is not allowed",
                    )

            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BLOCKED_BUILTINS:
                        return False, f"Call to '{node.func.id}' is not allowed"

        return True, ""

    def execute(
        self,
        code: str,
        context: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Execute Python code safely.

        Args:
            code: Python code to execute.
            context: Optional variables to inject into namespace.
            timeout_seconds: Maximum execution time.

        Returns:
            Dict with execution results.
        """
        start_time = time.time()
        timeout = timeout_seconds or self.default_timeout

        # Validate code
        is_valid, error = self.validate_code(code)
        if not is_valid:
            return {
                "success": False,
                "output": "",
                "result": None,
                "error": f"Validation error: {error}",
                "execution_time_ms": 0,
            }

        # Create execution namespace
        exec_namespace = dict(self._namespace)
        if context:
            exec_namespace.update(context)

        # Capture stdout
        stdout_capture = io.StringIO()

        try:
            # Parse code to get the last expression
            tree = ast.parse(code)
            last_expr = None

            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = tree.body.pop()

            # Execute with timeout
            with self._timeout_context(timeout):
                with contextlib.redirect_stdout(stdout_capture):
                    # Execute statements
                    if tree.body:
                        exec(compile(tree, "<string>", "exec"), exec_namespace)

                    # Evaluate last expression for result
                    result = None
                    if last_expr:
                        result = eval(
                            compile(
                                ast.Expression(body=last_expr.value),
                                "<string>",
                                "eval",
                            ),
                            exec_namespace,
                        )

            execution_time = (time.time() - start_time) * 1000

            # Format result
            result_str = None
            if result is not None:
                try:
                    result_str = repr(result)
                    # Truncate long results
                    if len(result_str) > 10000:
                        result_str = result_str[:10000] + "... (truncated)"
                except Exception:
                    result_str = "<unable to represent result>"

            return {
                "success": True,
                "output": stdout_capture.getvalue(),
                "result": result_str,
                "error": "",
                "execution_time_ms": round(execution_time, 2),
            }

        except TimeoutError:
            return {
                "success": False,
                "output": stdout_capture.getvalue(),
                "result": None,
                "error": f"Execution timed out after {timeout} seconds",
                "execution_time_ms": timeout * 1000,
            }
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = "".join(traceback.format_exception_only(type(e), e))
            return {
                "success": False,
                "output": stdout_capture.getvalue(),
                "result": None,
                "error": error_msg.strip(),
                "execution_time_ms": round(execution_time, 2),
            }

    @contextlib.contextmanager
    def _timeout_context(self, seconds: int):
        """Context manager for execution timeout.

        Note: Uses SIGALRM on Unix systems. On Windows,
        timeout is not enforced.
        """
        if sys.platform == "win32":
            # Windows doesn't support SIGALRM
            yield
            return

        def timeout_handler(signum: int, frame: Any) -> None:
            raise TimeoutError(f"Execution timed out after {seconds} seconds")

        # Set the signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)

        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


class MockPythonExecutor:
    """Mock Python executor for testing."""

    def __init__(self, default_timeout: int = 30) -> None:
        """Initialize mock executor."""
        self.default_timeout = default_timeout

    def validate_code(self, code: str) -> tuple[bool, str]:
        """Always returns valid for mock."""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

    def execute(
        self,
        code: str,
        context: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Return mock execution result."""
        is_valid, error = self.validate_code(code)
        if not is_valid:
            return {
                "success": False,
                "output": "",
                "result": None,
                "error": error,
                "execution_time_ms": 0,
            }

        return {
            "success": True,
            "output": "Mock execution output",
            "result": "42",
            "error": "",
            "execution_time_ms": 10.0,
        }

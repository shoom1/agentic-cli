"""Safe Python code execution via subprocess isolation.

Provides a sandboxed environment for executing Python code
with restricted capabilities, memory limits, and cross-platform timeout.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Sentinel used to separate user stdout from the JSON result envelope.
# The parent uses rsplit(SENTINEL, 1) so user code printing the sentinel
# does not break parsing.
_RESULT_SENTINEL = "__AGENTIC_EXECUTOR_RESULT_SENTINEL__"


class ExecutionTimeoutError(Exception):
    """Raised when code execution times out."""

    pass


# Backward-compatible alias
TimeoutError = ExecutionTimeoutError


class SafePythonExecutor:
    """Executes Python code in an isolated subprocess.

    Provides safety through:
    - AST validation (blocks dangerous patterns, runs in-process for fast feedback)
    - Subprocess isolation (code never runs in the agent process)
    - Execution timeout (cross-platform via subprocess.run)
    - Memory limits (resource.setrlimit on Unix)
    - Output capture with sentinel protocol
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
        # type, isinstance, hasattr are safe introspection builtins — allowed
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

    def __init__(
        self, default_timeout: int = 30, max_memory_mb: int = 512
    ) -> None:
        """Initialize the executor.

        Args:
            default_timeout: Default execution timeout in seconds.
            max_memory_mb: Maximum memory for subprocess (MB, Unix only).
        """
        self.default_timeout = default_timeout
        self.max_memory_mb = max_memory_mb

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
        """Execute Python code safely in an isolated subprocess.

        Args:
            code: Python code to execute.
            context: Optional variables to inject into namespace.
            timeout_seconds: Maximum execution time.

        Returns:
            Dict with execution results.
        """
        timeout = timeout_seconds or self.default_timeout

        # Validate code in-process for fast feedback
        logger.debug("python_executor.validation", code_length=len(code))
        is_valid, error = self.validate_code(code)
        if not is_valid:
            return {
                "success": False,
                "output": "",
                "result": None,
                "error": f"Validation error: {error}",
                "execution_time_ms": 0,
            }

        return self._execute_in_subprocess(code, context, timeout)

    def _execute_in_subprocess(
        self,
        code: str,
        context: dict[str, Any] | None,
        timeout: int,
    ) -> dict[str, Any]:
        """Run validated code in a subprocess.

        Args:
            code: Pre-validated Python code.
            context: Optional variables to inject.
            timeout: Execution timeout in seconds.

        Returns:
            Dict with execution results.
        """
        start_time = time.time()
        script = self._build_runner_script(context)

        logger.debug(
            "python_executor.subprocess_start",
            timeout=timeout,
            max_memory_mb=self.max_memory_mb,
        )

        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                input=code,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.time() - start_time) * 1000
            logger.warning("python_executor.timeout", timeout=timeout)
            return {
                "success": False,
                "output": "",
                "result": None,
                "error": f"Execution timed out after {timeout} seconds",
                "execution_time_ms": round(elapsed, 2),
            }

        elapsed = (time.time() - start_time) * 1000

        # Parse result using sentinel protocol
        stdout = proc.stdout or ""

        if _RESULT_SENTINEL in stdout:
            user_output, result_json = stdout.rsplit(_RESULT_SENTINEL, 1)
            user_output = user_output.rstrip("\n")
            try:
                result = json.loads(result_json.strip())
                result["output"] = user_output
                result["execution_time_ms"] = round(elapsed, 2)
                logger.info(
                    "python_executor.complete",
                    success=result["success"],
                    execution_time_ms=result["execution_time_ms"],
                )
                return result
            except json.JSONDecodeError:
                pass

        # No sentinel found — process crashed or was killed
        if proc.returncode in (-9, 137):
            logger.warning(
                "python_executor.subprocess_crash",
                returncode=proc.returncode,
                reason="oom",
            )
            return {
                "success": False,
                "output": stdout,
                "result": None,
                "error": "Process killed (likely out of memory)",
                "execution_time_ms": round(elapsed, 2),
            }

        # Other unexpected failure
        stderr = (proc.stderr or "").strip()
        logger.warning(
            "python_executor.subprocess_crash",
            returncode=proc.returncode,
            stderr=stderr[:500],
        )
        return {
            "success": False,
            "output": stdout,
            "result": None,
            "error": stderr or f"Subprocess exited with code {proc.returncode}",
            "execution_time_ms": round(elapsed, 2),
        }

    def _build_runner_script(self, context: dict[str, Any] | None) -> str:
        """Build the self-contained runner script for the subprocess.

        The script:
        1. Sets resource limits (Unix only)
        2. Reads code from stdin
        3. Sets up restricted namespace
        4. Executes code, capturing stdout
        5. Prints user output, sentinel, then JSON result

        Args:
            context: Optional variables to inject into the execution namespace.

        Returns:
            Python source code string for the runner script.
        """
        context_repr = repr(context) if context else "None"
        allowed_modules_repr = repr(self.ALLOWED_MODULES)
        blocked_builtins_repr = repr(self.BLOCKED_BUILTINS)
        max_memory_bytes = self.max_memory_mb * 1024 * 1024

        return textwrap.dedent(f"""\
            import sys
            import ast
            import io
            import json
            import traceback
            import time
            import contextlib

            # --- Memory limit (Unix only) ---
            try:
                import resource
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    ({max_memory_bytes}, {max_memory_bytes}),
                )
            except (ImportError, ValueError, OSError):
                pass  # Windows or unsupported

            SENTINEL = {repr(_RESULT_SENTINEL)}
            ALLOWED_MODULES = {allowed_modules_repr}
            BLOCKED_BUILTINS = {blocked_builtins_repr}

            def safe_import(name, globals_=None, locals_=None, fromlist=(), level=0):
                if name not in ALLOWED_MODULES:
                    raise ImportError(
                        f"Module '{{name}}' is not allowed. "
                        f"Allowed modules: {{', '.join(sorted(ALLOWED_MODULES))}}"
                    )
                return __builtins__.__import__(name, globals_, locals_, fromlist, level) \\
                    if hasattr(__builtins__, '__import__') \\
                    else __import__(name, globals_, locals_, fromlist, level)

            def build_namespace(context):
                raw = __builtins__
                if isinstance(raw, dict):
                    builtins_dict = dict(raw)
                else:
                    builtins_dict = {{k: getattr(raw, k) for k in dir(raw) if not k.startswith('_') or k == '__build_class__'}}
                for b in BLOCKED_BUILTINS:
                    builtins_dict.pop(b, None)
                builtins_dict["__import__"] = safe_import
                ns = {{"__builtins__": builtins_dict, "__name__": "__main__", "__doc__": None}}
                # Pre-load common modules
                for mod_name in ["math", "json", "re", "datetime", "collections"]:
                    try:
                        ns[mod_name] = __import__(mod_name)
                    except ImportError:
                        pass
                # Try numpy/pandas
                for mod_name, alias in [("numpy", "np"), ("pandas", "pd")]:
                    try:
                        m = __import__(mod_name)
                        ns[mod_name] = m
                        ns[alias] = m
                    except ImportError:
                        pass
                if context:
                    ns.update(context)
                return ns

            def main():
                code = sys.stdin.read()
                context = {context_repr}
                ns = build_namespace(context)

                stdout_capture = io.StringIO()
                start = time.time()

                try:
                    tree = ast.parse(code)
                    last_expr = None
                    if tree.body and isinstance(tree.body[-1], ast.Expr):
                        last_expr = tree.body.pop()

                    with contextlib.redirect_stdout(stdout_capture):
                        if tree.body:
                            exec(compile(tree, "<string>", "exec"), ns)
                        result = None
                        if last_expr:
                            result = eval(
                                compile(ast.Expression(body=last_expr.value), "<string>", "eval"),
                                ns,
                            )

                    elapsed = (time.time() - start) * 1000
                    result_str = None
                    if result is not None:
                        try:
                            result_str = repr(result)
                            if len(result_str) > 10000:
                                result_str = result_str[:10000] + "... (truncated)"
                        except Exception:
                            result_str = "<unable to represent result>"

                    user_output = stdout_capture.getvalue()
                    sys.stdout.write(user_output)
                    sys.stdout.write("\\n" + SENTINEL + "\\n")
                    json.dump({{
                        "success": True,
                        "output": "",
                        "result": result_str,
                        "error": "",
                        "execution_time_ms": 0,
                    }}, sys.stdout)

                except Exception as e:
                    elapsed = (time.time() - start) * 1000
                    error_msg = "".join(traceback.format_exception_only(type(e), e)).strip()
                    user_output = stdout_capture.getvalue()
                    sys.stdout.write(user_output)
                    sys.stdout.write("\\n" + SENTINEL + "\\n")
                    json.dump({{
                        "success": False,
                        "output": "",
                        "result": None,
                        "error": error_msg,
                        "execution_time_ms": 0,
                    }}, sys.stdout)

            main()
        """)


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

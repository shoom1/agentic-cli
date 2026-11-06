"""Python code execution in a subprocess, inside the OS sandbox when asked.

Where each call runs is decided when it runs: inside the OS sandbox
(sandbox-exec on macOS, bwrap on Linux) when the policy asks for it and a
backend wraps the command; refused when the policy is strict (the default)
and that fails; otherwise as the user. Outside the sandbox the executor's
checks are not a security boundary: CPython has no in-process sandbox.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from agentic_cli.tools.shell.os_sandbox.policy import OSSandboxPolicy

logger = structlog.get_logger(__name__)

# Sentinel used to separate user stdout from the JSON result envelope.
# The parent uses rsplit(SENTINEL, 1) so user code printing the sentinel
# does not break parsing.
_RESULT_SENTINEL = "__AGENTIC_EXECUTOR_RESULT_SENTINEL__"

# The runner's first line on stderr: the interpreter started. A sandboxed run
# without it never got that far, so the sandbox tool itself failed (bwrap
# denied user namespaces, a seatbelt profile the kernel rejected).
_STARTED_MARKER = "__AGENTIC_EXECUTOR_STARTED__"


class ExecutionTimeoutError(Exception):
    """Raised when code execution times out."""

    pass


# Backward-compatible alias
TimeoutError = ExecutionTimeoutError

def default_opt_out(app_name: str | None) -> str:
    """Where to turn the OS sandbox off, when the caller names no better
    place: the user's settings file (the setting is the user's: the project
    file and ``/settings`` cannot change it)."""
    where = f"~/.{app_name}/settings.json" if app_name else "your user settings file"
    return f'set "os_sandbox_enabled": false in {where}'


def how_to_run(opt_out: str) -> str:
    """The end of every refusal: how to get a working sandbox, how to opt out
    (``opt_out``, e.g. ``default_opt_out``), and what opting out means."""
    return (
        "To run it, install a working OS sandbox: sandbox-exec ships with macOS; "
        "on Linux install bubblewrap (`bwrap`), and on Ubuntu 24.04 or later also "
        "allow it unprivileged user namespaces (an AppArmor profile for bwrap, or "
        "kernel.apparmor_restrict_unprivileged_userns=0); on Windows there is none. "
        f"Or turn the sandbox off ({opt_out}) to run code without one, with your "
        "privileges."
    )


@dataclass(frozen=True)
class _Run:
    """Where one call's code runs, decided before it is validated.

    ``command`` is what ``subprocess.run`` gets (a shell string when the
    sandbox wrapped it), ``modules`` what the code may import there, and
    ``refusal`` the error when it runs nowhere.
    """

    command: str | list[str] = ""
    shell: bool = False
    modules: frozenset[str] = frozenset()
    sandboxed: bool = False
    sandbox_type: str = ""
    refusal: str | None = None


class SafePythonExecutor:
    """Runs Python code in a fresh subprocess, inside the OS sandbox when asked.

    Where the code runs is decided at each call, before it is validated:

    * The policy is enabled and a backend (sandbox-exec on macOS, bwrap on
      Linux) wraps the command: the code runs inside the OS sandbox, which is
      the security boundary (it writes only to the working directory and the
      configured paths, reads no credential store, and has no network unless
      the policy allows it). ``CORE_MODULES`` and ``SANDBOXED_MODULES`` are
      importable.
    * The policy is enabled but there is no backend, the wrap fails, or the
      sandbox tool fails as it starts (the runner never reported in): the
      code is refused when the policy is strict (the default), with the cause
      and ``how_to_run``, and otherwise runs as below.
    * No policy (or a disabled one), or a non-strict fallback: the code runs
      with the user's privileges, and only ``CORE_MODULES`` are importable.
      The AST checks and the restricted builtins still apply, but they are not
      a security boundary: CPython has no in-process sandbox, and the allowed
      standard modules reach the host themselves
      (``re.enum.sys.modules["os"]``, ``json.codecs.open``). Code that runs
      this way must be approved call by call (``tools/exec_policy.py``).

    Every run also has a timeout, an address-space limit (``RLIMIT_AS``,
    which macOS does not enforce), the sentinel output protocol, and an
    environment without the provider keys.
    """

    # Modules that expose file, network, or pickle I/O through public,
    # non-underscore APIs (e.g. numpy.DataSource().open(), numpy.load(...,
    # allow_pickle=True), pandas.read_pickle(url)). Importable only by code
    # that runs inside the OS sandbox: outside it they would hand the code
    # the host in one call.
    SANDBOXED_MODULES = {
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        # sympy.sympify / parse_expr evaluate arbitrary expressions and can be
        # coerced into Python code execution — a real OS sandbox must contain it.
        "sympy",
    }

    # The modules importable wherever the code runs. Not a boundary: they are
    # standard modules for computation and data handling, but some re-export
    # the host (re.enum.sys, json.codecs), so code that runs outside the OS
    # sandbox is approved per call.
    #
    # NOTE: ``operator`` and ``functools`` are deliberately excluded. Their
    # attrgetter/methodcaller (and reduce) take attribute names as *runtime
    # strings*, which bypasses the AST underscore-attribute filter and yields
    # host RCE (object.__subclasses__ → __init__.__globals__ → os.system).
    # Do not re-add them without the OS sandbox as the enforced boundary.
    CORE_MODULES = {
        # Math / science
        "math",
        "statistics",
        "cmath",
        # Collections and utilities
        "collections",
        "itertools",
        # Data handling
        "json",
        "re",
        "datetime",
        "decimal",
        "fractions",
        # Random (for simulations)
        "random",
    }

    # Every module a run may import: a run inside the OS sandbox gets all of
    # them, any other run CORE_MODULES only (see _choose_run).
    ALLOWED_MODULES = CORE_MODULES | SANDBOXED_MODULES

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
        self,
        default_timeout: int = 30,
        max_memory_mb: int = 512,
        os_sandbox_policy: OSSandboxPolicy | None = None,
        opt_out: str | None = None,
    ) -> None:
        """Initialize the executor.

        Args:
            default_timeout: Default execution timeout in seconds.
            max_memory_mb: Maximum memory for subprocess (MB, Unix only).
            os_sandbox_policy: OS-level sandbox policy. Enabled, the code runs
                inside the OS sandbox, or (strict, the default) not at all.
                None or disabled, it runs with the user's privileges.
            opt_out: How the user turns the sandbox off, as a refusal names it
                (``exec_policy.sandbox_opt_out``). Default: the user settings
                file of the policy's app (``default_opt_out``).
        """
        self.default_timeout = default_timeout
        self.max_memory_mb = max_memory_mb
        self.os_sandbox_policy = os_sandbox_policy
        self.opt_out = opt_out or default_opt_out(
            getattr(os_sandbox_policy, "app_name", None)
        )

    def validate_code(
        self, code: str, allowed_modules: frozenset[str] | set[str] | None = None
    ) -> tuple[bool, str]:
        """Check code against the executor's rules (not a security boundary).

        Args:
            code: Python code to validate.
            allowed_modules: The modules the code may import: those of the
                place it will run. Default: ``CORE_MODULES``, which every run
                allows.

        Returns:
            Tuple of (is_valid, error_message).
        """
        modules = self.CORE_MODULES if allowed_modules is None else allowed_modules
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
                            if alias.name.split(".")[0] not in modules:
                                return (
                                    False,
                                    f"Import of '{alias.name}' is not allowed",
                                )
                    elif isinstance(node, ast.ImportFrom):
                        if (
                            node.module
                            and node.module.split(".")[0] not in modules
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

            # Reject str.format()/format_map() field names that traverse private
            # attributes (e.g. "{0.__class__}"). The attribute path lives inside
            # the string literal, so the ast.Attribute check above never sees it.
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if self._format_field_accesses_private(node.value):
                    return (
                        False,
                        "Format string references a private attribute "
                        "(potential sandbox escape via str.format)",
                    )

            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BLOCKED_BUILTINS:
                        return False, f"Call to '{node.func.id}' is not allowed"

        return True, ""

    @staticmethod
    def _format_field_accesses_private(format_string: str) -> bool:
        """True if a ``str.format`` field references a ``_``-prefixed attribute.

        e.g. ``"{0.__class__}"`` or ``"{a.__class__.__base__}"``. Such fields
        traverse the object graph through the format machinery, bypassing the
        AST attribute filter. Item access (``{0[key]}``) is not an
        attribute-traversal vector and is left alone.
        """
        import string as _string

        try:
            parsed = list(_string.Formatter().parse(format_string))
        except (ValueError, IndexError):
            # Malformed format string: not our concern here — it will raise at
            # runtime inside the subprocess, it cannot escape.
            return False
        for _literal, field_name, format_spec, _conv in parsed:
            if field_name:
                for part in field_name.replace("[", ".").split("."):
                    if part.startswith("_"):
                        return True
            if format_spec and "{" in format_spec:
                # Nested replacement field inside the format spec.
                if SafePythonExecutor._format_field_accesses_private(format_spec):
                    return True
        return False

    @staticmethod
    def _subprocess_env() -> dict[str, str]:
        """Parent environment with provider secrets removed.

        The workflow manager exports API keys into ``os.environ``; a code
        execution escape must not be able to read them from the child process.
        """
        import os as _os
        import re as _re

        secret = _re.compile(r"(API_KEY|TOKEN|SECRET|PASSWORD)$", _re.IGNORECASE)
        return {k: v for k, v in _os.environ.items() if not secret.search(k)}

    def execute(
        self,
        code: str,
        context: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Execute Python code in a fresh subprocess.

        Args:
            code: Python code to execute.
            context: Optional variables to inject into namespace.
            timeout_seconds: Maximum execution time in seconds; ``None`` uses
                ``default_timeout``.

        Returns:
            Dict with execution results.
        """
        if timeout_seconds is not None and timeout_seconds <= 0:
            return {
                "success": False,
                "output": "",
                "result": None,
                "error": (
                    "timeout_seconds must be a positive number "
                    f"(got {timeout_seconds!r})."
                ),
                "execution_time_ms": 0,
            }
        timeout = self.default_timeout if timeout_seconds is None else timeout_seconds
        return self._execute_in_subprocess(code, context, timeout)

    def _sandbox_refused(self, message: str, start_time: float) -> dict[str, Any]:
        """Fail-closed result when strict sandboxing is required but unavailable."""
        logger.error("python_executor.os_sandbox_refused", message=message)
        elapsed = (time.time() - start_time) * 1000
        return {
            "success": False,
            "output": "",
            "result": None,
            "error": message,
            "execution_time_ms": round(elapsed, 2),
        }

    def _choose_run(self, context: dict[str, Any] | None) -> _Run:
        """Decide where this call's code runs: inside the OS sandbox, refused,
        or with the user's privileges (see the class docstring)."""
        policy = self.os_sandbox_policy
        if not (policy and policy.enabled):
            # No sandbox asked for: the caller's choice to run as the user.
            return self._unsandboxed_run(context)

        from agentic_cli.tools.shell.os_sandbox import get_os_sandbox

        strict = self._strict
        sandbox = get_os_sandbox()
        if sandbox.sandbox_type == "none":
            if strict:
                return _Run(refusal=(
                    "Refusing to run Python without the OS sandbox: no sandbox "
                    f"tool is available on this system. {how_to_run(self.opt_out)}"
                ))
            logger.warning(
                "python_executor.os_sandbox_unavailable_fallback",
                sandbox_type=sandbox.sandbox_type,
            )
            return self._unsandboxed_run(context)

        modules = frozenset(self.ALLOWED_MODULES)
        wrap = sandbox.wrap_python_command(
            [sys.executable, "-c", self._build_runner_script(context, modules)],
            Path.cwd(),
            policy,
        )
        if wrap.success:
            logger.debug("python_executor.os_sandbox_wrapped", sandbox_type=wrap.sandbox_type)
            return _Run(
                command=wrap.command,
                shell=True,
                modules=modules,
                sandboxed=True,
                sandbox_type=wrap.sandbox_type,
            )
        if strict:
            return _Run(refusal=(
                "Refusing to run Python without the OS sandbox: it could not be "
                f"set up ({wrap.sandbox_type}: {wrap.error}). {how_to_run(self.opt_out)}"
            ))
        logger.warning("python_executor.os_sandbox_wrap_failed_fallback", error=wrap.error)
        return self._unsandboxed_run(context)

    @property
    def _strict(self) -> bool:
        """Refuse, rather than fall back to running as the user, when the
        sandbox is unavailable (the policy's ``strict``, True by default)."""
        return bool(getattr(self.os_sandbox_policy, "strict", True))

    def _unsandboxed_run(self, context: dict[str, Any] | None) -> _Run:
        """A run with the user's privileges: ``CORE_MODULES`` only."""
        modules = frozenset(self.CORE_MODULES)
        script = self._build_runner_script(context, modules)
        return _Run(command=[sys.executable, "-c", script], modules=modules)

    def _execute_in_subprocess(
        self,
        code: str,
        context: dict[str, Any] | None,
        timeout: int,
    ) -> dict[str, Any]:
        """Choose where the code runs, validate it for there, and run it.

        Args:
            code: Python code to execute.
            context: Optional variables to inject.
            timeout: Execution timeout in seconds.

        Returns:
            Dict with execution results.
        """
        start_time = time.time()
        run = self._choose_run(context)
        if run.refusal is not None:
            return self._sandbox_refused(run.refusal, start_time)
        return self._validate_and_run(run, code, context, timeout, start_time)

    def _validate_and_run(
        self,
        run: _Run,
        code: str,
        context: dict[str, Any] | None,
        timeout: int,
        start_time: float,
    ) -> dict[str, Any]:
        """Validate the code for where ``run`` puts it, then run it there."""
        # The modules the code may import are those of the place it runs.
        logger.debug("python_executor.validation", code_length=len(code))
        is_valid, error = self.validate_code(code, allowed_modules=run.modules)
        if not is_valid:
            return {
                "success": False,
                "output": "",
                "result": None,
                "error": f"Validation error: {error}",
                "execution_time_ms": 0,
            }

        logger.debug(
            "python_executor.subprocess_start",
            timeout=timeout,
            max_memory_mb=self.max_memory_mb,
            sandboxed=run.sandboxed,
        )
        try:
            proc = subprocess.run(
                run.command,
                shell=run.shell,
                input=code,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._subprocess_env(),
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
        stderr_lines = (proc.stderr or "").splitlines()
        started = _STARTED_MARKER in stderr_lines
        stderr = "\n".join(line for line in stderr_lines if line != _STARTED_MARKER)

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

        # No sentinel found. A sandboxed run whose interpreter never started:
        # the sandbox tool failed (it is installed, but cannot set the sandbox
        # up here), so the code did not run. The sandbox is unavailable, as
        # when the wrap fails: refused, with its cause, or (not strict) run as
        # the user, checked again for that place's modules.
        if run.sandboxed and not started:
            cause = stderr.strip() or f"exit status {proc.returncode}"
            if self._strict:
                return self._sandbox_refused(
                    "Refusing to run Python without the OS sandbox: it did not start "
                    f"({run.sandbox_type}, exit status {proc.returncode}): {cause}. "
                    f"{how_to_run(self.opt_out)}",
                    start_time,
                )
            logger.warning(
                "python_executor.os_sandbox_start_failed_fallback",
                sandbox_type=run.sandbox_type,
                error=cause[:500],
            )
            return self._validate_and_run(
                self._unsandboxed_run(context), code, context, timeout, start_time
            )

        # The process crashed, was killed, or the code exited.
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
        stderr = stderr.strip()
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

    def _build_runner_script(
        self, context: dict[str, Any] | None, allowed_modules: frozenset[str]
    ) -> str:
        """Build the self-contained runner script for the subprocess.

        The script:
        1. Sets resource limits (Unix only)
        2. Reads code from stdin
        3. Sets up restricted namespace
        4. Executes code, capturing stdout
        5. Prints user output, sentinel, then JSON result

        Args:
            context: Optional variables to inject into the execution namespace.
            allowed_modules: The modules the run may import (its own import
                gate, and which of numpy/pandas are pre-bound).

        Returns:
            Python source code string for the runner script.
        """
        context_repr = repr(context) if context else "None"
        allowed_modules_repr = repr(set(allowed_modules))
        blocked_builtins_repr = repr(self.BLOCKED_BUILTINS)
        max_memory_bytes = self.max_memory_mb * 1024 * 1024

        return textwrap.dedent(f"""\
            import sys
            sys.stderr.write({_STARTED_MARKER!r} + "\\n")
            sys.stderr.flush()
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
                top_level = name.split(".")[0]
                if top_level not in ALLOWED_MODULES:
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
                # Pre-bind numpy/pandas as np/pd, but only when they are in the
                # effective allow-list. Otherwise they would be reachable without
                # an import statement, bypassing the import check entirely.
                for mod_name, alias in [("numpy", "np"), ("pandas", "pd")]:
                    if mod_name not in ALLOWED_MODULES:
                        continue
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

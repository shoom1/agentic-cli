"""Security regression tests for SafePythonExecutor (P0-1).

These encode sandbox-escape vectors that were reproducible against the
in-process AST filter. Each test asserts the escape is *blocked* — i.e. the
executor returns ``{"success": False, ...}`` rather than executing host code
or leaking the object graph. See docs/reviews/2026-07-03-* for the exploits.

The AST/name filter is defense-in-depth, not a hard boundary (the OS sandbox
is), but the *known* escape primitives must not be reachable from the default
(no-OS-sandbox) configuration.
"""

from __future__ import annotations

import subprocess

import pytest

from agentic_cli.tools.executor import SafePythonExecutor


@pytest.fixture
def executor() -> SafePythonExecutor:
    """Default executor: no OS sandbox, CORE_MODULES only."""
    return SafePythonExecutor()


# ---------------------------------------------------------------------------
# operator/functools string-getattr escape (confirmed RCE vector)
# ---------------------------------------------------------------------------

def test_operator_attrgetter_escape_is_blocked(executor: SafePythonExecutor) -> None:
    """operator.attrgetter/methodcaller take attr names as runtime strings,
    bypassing the AST underscore filter. Reaching object.__subclasses__ and a
    module's __globals__ gives host RCE. Must be blocked."""
    code = (
        "import operator\n"
        "cls_of = operator.attrgetter('__class__')\n"
        "base_of = operator.attrgetter('__base__')\n"
        "get_subs = operator.methodcaller('__subclasses__')\n"
        "get_globals = operator.attrgetter('__init__.__globals__')\n"
        "obj_cls = base_of(cls_of(()))\n"
        "hit = 'NO'\n"
        "for c in get_subs(obj_cls):\n"
        "    try:\n"
        "        gl = get_globals(c)\n"
        "        if 'os' in gl:\n"
        "            gl['os'].system('echo pwned')\n"
        "            hit = 'YES'\n"
        "            break\n"
        "    except Exception:\n"
        "        pass\n"
        "print(hit)\n"
    )
    result = executor.execute(code)
    assert result["success"] is False, f"escape was NOT blocked: {result}"


def test_import_operator_is_rejected(executor: SafePythonExecutor) -> None:
    result = executor.execute("import operator")
    assert result["success"] is False
    assert "operator" in result["error"]


def test_import_functools_is_rejected(executor: SafePythonExecutor) -> None:
    result = executor.execute("import functools")
    assert result["success"] is False
    assert "functools" in result["error"]


def test_from_operator_import_is_rejected(executor: SafePythonExecutor) -> None:
    result = executor.execute("from operator import attrgetter")
    assert result["success"] is False


# ---------------------------------------------------------------------------
# str.format object-graph traversal (info-leak vector, same root cause)
# ---------------------------------------------------------------------------

def test_str_format_dunder_traversal_is_blocked(
    executor: SafePythonExecutor,
) -> None:
    """str.format field names can reference dunder attributes (the path lives
    inside the format string, not as an AST attribute node), reaching the class
    hierarchy and __globals__. Must be blocked."""
    code = "print('{0.__class__.__base__.__subclasses__}'.format(()))"
    result = executor.execute(code)
    assert result["success"] is False, f"format escape not blocked: {result}"


def test_str_format_map_dunder_traversal_is_blocked(
    executor: SafePythonExecutor,
) -> None:
    code = "print('{a.__class__}'.format_map({'a': ()}))"
    result = executor.execute(code)
    assert result["success"] is False


def test_plain_str_format_still_allowed(executor: SafePythonExecutor) -> None:
    """The fix must not break ordinary format usage."""
    result = executor.execute("print('{} {name}'.format(1, name='x'))")
    assert result["success"] is True, result
    assert "1 x" in result["output"]


# ---------------------------------------------------------------------------
# sympy eval surface moved behind the OS sandbox
# ---------------------------------------------------------------------------

def test_sympy_import_rejected_without_os_sandbox(
    executor: SafePythonExecutor,
) -> None:
    """sympy.sympify/parse_expr can evaluate arbitrary expressions; sympy must
    require the OS sandbox like the other I/O-capable modules."""
    result = executor.execute("import sympy")
    assert result["success"] is False


# ---------------------------------------------------------------------------
# API keys must not leak into the execution subprocess environment
# ---------------------------------------------------------------------------

def test_api_keys_scrubbed_from_subprocess_env(
    executor: SafePythonExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any code that reaches the host must not find provider API keys in env."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("GOOGLE_API_KEY", "goog-secret")
    monkeypatch.setenv("SOME_TOKEN", "tok")
    monkeypatch.setenv("PATH", "/usr/bin")

    captured: dict = {}
    real_run = subprocess.run

    def _capture(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _capture)
    executor.execute("print(1)")

    env = captured["env"]
    assert env is not None, "executor must pass an explicit scrubbed env"
    assert "ANTHROPIC_API_KEY" not in env
    assert "GOOGLE_API_KEY" not in env
    assert "SOME_TOKEN" not in env
    assert env.get("PATH") == "/usr/bin", "non-secret env must be preserved"


# ---------------------------------------------------------------------------
# sanity: legitimate compute still works
# ---------------------------------------------------------------------------

def test_ordinary_computation_still_works(executor: SafePythonExecutor) -> None:
    result = executor.execute("import math\nprint(math.sqrt(16))")
    assert result["success"] is True, result
    assert "4.0" in result["output"]

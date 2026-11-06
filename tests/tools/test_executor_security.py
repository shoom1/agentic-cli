"""Security regression tests for SafePythonExecutor (P0-1, C2).

The first tests encode escape vectors that were reproducible against the AST
filter of a run outside the OS sandbox (``SafePythonExecutor()``, no policy):
each asserts that one is *blocked*. The filter is defense-in-depth, not a
boundary: other payloads pass it (see ``ESCAPES`` below), so the boundary is
the OS sandbox, and the last tests check that the default configuration never
runs code outside it.
"""

from __future__ import annotations

import subprocess

import pytest

from agentic_cli.tools.executor import SafePythonExecutor


@pytest.fixture
def executor() -> SafePythonExecutor:
    """An executor with no OS-sandbox policy: runs as the user, CORE_MODULES only."""
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


# ---------------------------------------------------------------------------
# The default configuration runs code only inside the OS sandbox
# ---------------------------------------------------------------------------
#
# CPython has no in-process sandbox. The standard modules the executor allows
# re-export the host through public attributes (``re.enum.sys``,
# ``re.enum.bltns``, ``json.codecs``), and a generator's frame reaches the
# runner's globals: the AST checks let every one of these through. So the
# question is never whether the checks stop them, but whether the code runs
# outside the OS sandbox without the user having said so.

#: Payloads that pass the executor's checks and reach the host when run
#: outside the sandbox. ``{target}`` is a file the payload writes, ``{probe}``
#: one it reads.
ESCAPES = {
    "os-getpid": 're.enum.sys.modules["os"].getpid()',
    "os-popen": 're.enum.sys.modules["os"].popen("echo shell-ran").read()',
    "builtins-open": "re.enum.bltns.open({probe!r}).readline()",
    "codecs-write": 'f = json.codecs.open({target!r}, "w")\nf.write("escaped")\nf.close()\n"ok"',
    "frame-walk": (
        "def f():\n"
        "    yield g.gi_frame.f_back\n"
        "g = f()\n"
        "fr = next(g)\n"
        'fr.f_back.f_globals["sys"].modules["os"].getpid() > 0'
    ),
}


class _Backend:
    """A stand-in OS sandbox: ``"none"`` is no backend; ``wrap_error`` makes
    the wrap fail; otherwise the "wrap" is the plain command."""

    def __init__(self, sandbox_type: str = "seatbelt", wrap_error: str | None = None) -> None:
        self.sandbox_type = sandbox_type
        self.wrap_error = wrap_error
        self.wrapped: list[list[str]] = []

    def wrap_python_command(self, argv, working_dir, policy):
        import shlex

        from agentic_cli.tools.shell.os_sandbox.base import OSSandboxResult

        self.wrapped.append(list(argv))
        if self.wrap_error:
            return OSSandboxResult(
                command="", success=False, error=self.wrap_error, sandbox_type=self.sandbox_type
            )
        return OSSandboxResult(command=shlex.join(argv), sandbox_type=self.sandbox_type)


def _use_backend(monkeypatch: pytest.MonkeyPatch, backend: _Backend) -> _Backend:
    monkeypatch.setattr(
        "agentic_cli.tools.shell.os_sandbox.get_os_sandbox", lambda *a, **k: backend
    )
    return backend


def _record_runs(monkeypatch: pytest.MonkeyPatch) -> list:
    """Record each subprocess the executor starts (and still start it)."""
    runs: list = []
    real_run = subprocess.run

    def _run(*args, **kwargs):
        runs.append((args, kwargs))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _run)
    return runs


def _runner_modules(script: str) -> set[str]:
    """The module set a runner script's own import gate allows."""
    import ast as _ast

    for line in script.splitlines():
        if line.strip().startswith("ALLOWED_MODULES = "):
            return set(_ast.literal_eval(line.split("=", 1)[1].strip()))
    raise AssertionError("no ALLOWED_MODULES line in the runner script")


@pytest.mark.parametrize("name", list(ESCAPES))
def test_core_module_reexports_do_not_reach_os(
    name: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no sandbox backend, the default configuration refuses the code
    rather than running it as the user."""
    from agentic_cli.tools.execution_tools import execute_python
    from tests.conftest import MockContext

    target = tmp_path / "written.txt"
    probe = tmp_path / "probe.txt"
    probe.write_text("a file the code should not read\n")
    code = ESCAPES[name].format(target=str(target), probe=str(probe))
    assert SafePythonExecutor().validate_code(code) == (True, "")  # no boundary
    _use_backend(monkeypatch, _Backend("none"))
    runs = _record_runs(monkeypatch)

    with MockContext():
        result = execute_python(code)

    assert result["success"] is False, result
    assert "os_sandbox_enabled" in result["error"]
    assert runs == []
    assert not target.exists()


def test_a_failed_wrap_is_refused_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from agentic_cli.tools.execution_tools import execute_python
    from tests.conftest import MockContext

    _use_backend(monkeypatch, _Backend("seatbelt", wrap_error="profile failed"))
    runs = _record_runs(monkeypatch)

    with MockContext():
        result = execute_python('re.enum.sys.modules["os"].getpid()')

    assert result["success"] is False, result
    assert "profile failed" in result["error"]
    assert runs == []


@pytest.mark.parametrize(
    "backend",
    [_Backend("none"), _Backend("bubblewrap", wrap_error="bwrap: setting up uid map")],
    ids=["no-backend", "failed-wrap"],
)
def test_the_refusal_says_how_to_get_a_sandbox_or_opt_out(
    backend: _Backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentic_cli.tools.execution_tools import execute_python
    from tests.conftest import MockContext

    _use_backend(monkeypatch, backend)
    with MockContext():
        error = execute_python("1 + 1")["error"]

    assert "sandbox-exec" in error and "bwrap" in error  # how to get one
    assert "os_sandbox_enabled" in error  # how to opt out
    assert "your privileges" in error  # and what opting out means


def test_a_failed_wrap_without_strict_runs_core_modules_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``os_sandbox_strict`` off: a failed wrap falls back to running without
    the sandbox, so with the unsandboxed module set, whatever was available
    when the executor was built."""
    from agentic_cli.tools.execution_tools import execute_python
    from tests.conftest import MockContext

    _use_backend(monkeypatch, _Backend("seatbelt", wrap_error="profile failed"))
    runs = _record_runs(monkeypatch)

    with MockContext(os_sandbox_strict=False):
        numpy = execute_python("import numpy\n'ran'")
        sqrt = execute_python("import math\nmath.sqrt(16)")

    assert numpy["success"] is False
    # Refused by the checks for where it runs, before any process starts.
    assert numpy["error"].startswith("Validation error"), numpy
    assert "not allowed" in numpy["error"], numpy
    assert sqrt["success"] is True, sqrt
    assert sqrt["result"] == "4.0"
    [(args, kwargs)] = runs  # only the math run started a process, unwrapped
    assert kwargs.get("shell") is not True
    assert _runner_modules(args[0][2]) == SafePythonExecutor.CORE_MODULES


def test_a_sandboxed_run_gets_the_sandboxed_module_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from agentic_cli.tools.execution_tools import execute_python
    from tests.conftest import MockContext

    backend = _use_backend(monkeypatch, _Backend("seatbelt"))
    with MockContext():
        assert execute_python("import math\nmath.sqrt(16)")["success"] is True

    [argv] = backend.wrapped
    assert _runner_modules(argv[2]) == (
        SafePythonExecutor.CORE_MODULES | SafePythonExecutor.SANDBOXED_MODULES
    )


def test_opting_out_runs_core_modules_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """``os_sandbox_enabled`` off is the user's explicit choice: the code runs
    as the user, without a wrap, with the unsandboxed module set."""
    from agentic_cli.tools.execution_tools import execute_python
    from tests.conftest import MockContext

    backend = _use_backend(monkeypatch, _Backend("seatbelt"))
    with MockContext(os_sandbox_enabled=False):
        numpy = execute_python("import numpy\n'ran'")
        sqrt = execute_python("import math\nmath.sqrt(16)")

    assert numpy["success"] is False and "not allowed" in numpy["error"]
    assert sqrt["success"] is True and sqrt["result"] == "4.0"
    assert backend.wrapped == []

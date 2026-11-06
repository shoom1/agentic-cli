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
    the wrap fail; ``command(argv)``, when given, is the wrapped command (one
    that fails before Python starts, say); otherwise the "wrap" is the plain
    command."""

    def __init__(
        self,
        sandbox_type: str = "seatbelt",
        wrap_error: str | None = None,
        command=None,
    ) -> None:
        self.sandbox_type = sandbox_type
        self.wrap_error = wrap_error
        self.command = command
        self.wrapped: list[list[str]] = []

    def wrap_python_command(self, argv, working_dir, policy):
        import shlex

        from agentic_cli.tools.shell.os_sandbox.base import OSSandboxResult

        self.wrapped.append(list(argv))
        if self.wrap_error:
            return OSSandboxResult(
                command="", success=False, error=self.wrap_error, sandbox_type=self.sandbox_type
            )
        command = self.command(argv) if self.command else shlex.join(argv)
        return OSSandboxResult(command=command, sandbox_type=self.sandbox_type)


def _fails_at_start(message: str, status: int):
    """A wrapped command that fails as a sandbox tool does when it cannot set
    the sandbox up: its message on stderr, Python never started."""
    import shlex

    return lambda argv: "sh -c " + shlex.quote(f"echo {shlex.quote(message)} >&2; exit {status}")


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


#: Each way the sandbox can be missing, and the cause its refusal must keep.
#: ``bwrap`` on PATH but blocked from user namespaces (Ubuntu 24.04's default
#: AppArmor policy), and a seatbelt profile the kernel rejects, fail only when
#: the wrapped command starts.
REFUSALS = {
    "no-backend": (lambda: _Backend("none"), "no sandbox tool is available"),
    "failed-wrap": (
        lambda: _Backend("bubblewrap", wrap_error="bwrap: profile failed"),
        "bwrap: profile failed",
    ),
    "bwrap-blocked-at-start": (
        lambda: _Backend(
            "bubblewrap",
            command=_fails_at_start("bwrap: setting up uid map: Permission denied", 1),
        ),
        "bwrap: setting up uid map: Permission denied",
    ),
    "seatbelt-rejected-at-start": (
        lambda: _Backend(
            "seatbelt",
            command=_fails_at_start("sandbox-exec: sandbox_apply: Operation not permitted", 65),
        ),
        "sandbox-exec: sandbox_apply: Operation not permitted",
    ),
}
REFUSAL_APP = "refusalapp"


def test_every_refusal_says_how_to_run_the_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """No backend, a failed wrap, or a sandbox tool that fails when it starts:
    each refusal keeps its cause and ends with the same remedy, how to get a
    working sandbox and where to turn it off (the setting is the user's: not
    the project file's, nor ``/settings``'s)."""
    from agentic_cli.tools.execution_tools import execute_python
    from tests.conftest import MockContext

    errors = {}
    for path, (make_backend, cause) in REFUSALS.items():
        _use_backend(monkeypatch, make_backend())
        runs = _record_runs(monkeypatch)
        with MockContext(app_name=REFUSAL_APP):
            result = execute_python('re.enum.sys.modules["os"].getpid()')
        assert result["success"] is False, (path, result)
        assert cause in result["error"], (path, result["error"])
        assert "To run it" in result["error"], (path, result["error"])
        # Refused; never retried without the sandbox.
        assert len(runs) == (1 if path.endswith("-at-start") else 0), path
        errors[path] = result["error"]

    tails = {error[error.index("To run it"):] for error in errors.values()}
    assert len(tails) == 1, errors
    [tail] = tails
    assert all(error.endswith(tail) for error in errors.values())
    # How to get a working sandbox, on each platform.
    assert "sandbox-exec" in tail and "macOS" in tail
    assert "bwrap" in tail and "Ubuntu 24.04" in tail and "user namespaces" in tail
    assert "Windows" in tail
    # How to opt out, where, and what that means.
    assert '"os_sandbox_enabled": false' in tail
    assert f"~/.{REFUSAL_APP}/settings.json" in tail
    assert "AGENTIC_OS_SANDBOX_ENABLED=false" in tail
    assert "your privileges" in tail


@pytest.mark.skipif(
    __import__("shutil").which("sandbox-exec") is None, reason="needs macOS's sandbox-exec"
)
def test_a_real_sandbox_that_will_not_start_python_is_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real sandbox-exec, with a profile that denies Python its start."""
    import shlex

    from agentic_cli.tools.execution_tools import execute_python
    from tests.conftest import MockContext

    _use_backend(monkeypatch, _Backend(
        "seatbelt",
        command=lambda argv: shlex.join(["sandbox-exec", "-p", "(version 1)(deny default)", *argv]),
    ))
    with MockContext():
        result = execute_python("1 + 1")

    assert result["success"] is False
    assert "sandbox-exec" in result["error"] and "Operation not permitted" in result["error"]
    assert "os_sandbox_enabled" in result["error"]


def test_without_strict_a_sandbox_that_fails_to_start_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``os_sandbox_strict`` off: a sandbox tool that fails as it starts is one
    more way the sandbox is unavailable, so the call falls back, as for a
    failed wrap: run as the user, with the unsandboxed module set (checked
    again, since the code was first checked for the sandbox)."""
    from agentic_cli.tools.execution_tools import execute_python
    from tests.conftest import MockContext

    _use_backend(monkeypatch, _Backend(
        "bubblewrap", command=_fails_at_start("bwrap: setting up uid map: Permission denied", 1)
    ))
    runs = _record_runs(monkeypatch)

    with MockContext(os_sandbox_strict=False):
        numpy = execute_python("import numpy\n'ran'")
        sqrt = execute_python("import math\nmath.sqrt(16)")

    assert numpy["success"] is False
    assert numpy["error"].startswith("Validation error"), numpy
    assert sqrt["success"] is True, sqrt
    assert sqrt["result"] == "4.0"
    # Each call tried the sandbox first; only the valid one then ran unwrapped.
    assert [kwargs.get("shell") is True for _, kwargs in runs] == [True, True, False]
    assert _runner_modules(runs[2][0][0][2]) == SafePythonExecutor.CORE_MODULES


def test_code_that_exits_in_the_sandbox_is_no_sandbox_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Code that exits before the result is written ran: its failure is its
    own, not a sandbox that did not start."""
    from agentic_cli.tools.execution_tools import execute_python
    from tests.conftest import MockContext

    _use_backend(monkeypatch, _Backend("seatbelt"))
    with MockContext():
        result = execute_python("re.enum.sys.exit(3)")

    assert result["success"] is False
    assert "os_sandbox_enabled" not in result["error"], result
    assert "exited with code 3" in result["error"], result


def test_a_failed_wrap_without_strict_runs_core_modules_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``os_sandbox_strict`` off: a failed wrap falls back to running without
    the sandbox, so with the unsandboxed module set: the set is chosen from
    where the call runs, not from what was available before the wrap."""
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

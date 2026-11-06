"""Where the framework's code-execution tools run, read from the settings.

CPython has no in-process sandbox: code that runs outside the OS sandbox (or a
container) runs with the user's privileges, whatever the executor checks
first. The permission engine approves such code call by call, and it learns
which capabilities that is from here: ``BaseWorkflowManager`` passes
``uncontained_exec_capabilities`` to the engine, which calls it at each check
(the engine imports nothing from ``tools/``).

* ``execute_python`` (``python.exec``) is contained only when the OS sandbox
  is on and strict (``os_sandbox_enabled`` and ``os_sandbox_strict``, the
  defaults): the code then runs inside the sandbox or not at all. Off, or not
  strict (a missing backend or a failed wrap falls back to running without
  one), it may run on the host.
* ``sandbox_execute`` (``python.exec.stateful``) runs on the host on the
  ``local`` backend, a Jupyter kernel with the user's privileges.

``python_sandbox_policy`` builds the OS-sandbox policy ``execute_python``
runs under from the same settings, so the tool and the engine read them
alike, and ``sandbox_opt_out`` names where the user turns the sandbox off.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic_cli.tools.os_sandbox.policy import OSSandboxPolicy

#: ``execute_python``'s capability.
PYTHON_EXEC = "python.exec"
#: ``sandbox_execute``'s capability.
STATEFUL_PYTHON_EXEC = "python.exec.stateful"


def _sandbox_enabled(settings: Any) -> bool:
    return bool(getattr(settings, "os_sandbox_enabled", False))


def _sandbox_strict(settings: Any) -> bool:
    return bool(getattr(settings, "os_sandbox_strict", True))


def python_sandbox_policy(settings: Any) -> "OSSandboxPolicy | None":
    """The OS-sandbox policy ``execute_python`` runs under, or None when the
    sandbox is off (``os_sandbox_enabled`` false: the code runs as the user)."""
    if not _sandbox_enabled(settings):
        return None
    from agentic_cli.tools.os_sandbox import OSSandboxPolicy

    return OSSandboxPolicy(
        enabled=True,
        writable_paths=list(getattr(settings, "os_sandbox_writable_paths", None) or []),
        allow_network=bool(getattr(settings, "os_sandbox_allow_network", False)),
        strict=_sandbox_strict(settings),
        app_name=getattr(settings, "app_name", None),
    )


def sandbox_opt_out(settings: Any) -> str:
    """Where the user turns the OS sandbox off, for a refusal to name.

    ``os_sandbox_enabled`` is the user's setting: a project file cannot set it,
    nor can ``/settings``. So: the user settings file, or the app's
    environment variable.
    """
    from agentic_cli.tools.executor import default_opt_out

    text = default_opt_out(getattr(settings, "app_name", None))
    config = getattr(type(settings), "model_config", None)
    prefix = config.get("env_prefix") if isinstance(config, dict) else None
    if prefix:
        text += f", or {prefix.upper()}OS_SANDBOX_ENABLED=false in the environment"
    return text


def uncontained_exec_capabilities(settings: Any) -> frozenset[str]:
    """The code-execution capabilities whose code may run on the host, with the
    user's privileges, under these settings."""
    uncontained: set[str] = set()
    if not (_sandbox_enabled(settings) and _sandbox_strict(settings)):
        uncontained.add(PYTHON_EXEC)
    if getattr(settings, "stateful_executor_backend", None) == "local":
        uncontained.add(STATEFUL_PYTHON_EXEC)
    return frozenset(uncontained)

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

import os
from pathlib import Path
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
    sandbox is off (``os_sandbox_enabled`` false: the code runs as the user).

    The memory folders are written only through the memory tools, so the
    policy protects them from the code (``_protect_memory_folders``).
    """
    if not _sandbox_enabled(settings):
        return None
    from agentic_cli.tools.os_sandbox import OSSandboxPolicy

    policy = OSSandboxPolicy(
        enabled=True,
        writable_paths=list(getattr(settings, "os_sandbox_writable_paths", None) or []),
        allow_network=bool(getattr(settings, "os_sandbox_allow_network", False)),
        strict=_sandbox_strict(settings),
        app_name=getattr(settings, "app_name", None),
    )
    _protect_memory_folders(policy, settings)
    return policy


def _protect_memory_folders(policy: "OSSandboxPolicy", settings: Any) -> None:
    """Hide the memory folders from the code, and keep it from writing them.

    Where ``workspace_dir`` sits in a place the sandbox may write (the working
    directory, an ``os_sandbox_writable_paths`` entry), hiding the folders is
    not enough: code could plant a file there that the next conversation reads
    as a memory. So, in this order:

    * The folders exist before the code runs (``<workspace_dir>/memory``,
      ``<workspace_dir>/projects``, and this project's scope), made 0700 as the
      store makes them. Code that found one missing could create it.
    * Both are denied for writes as subpaths: ``projects`` holds every
      project's scope, and a file planted in another project's folder is read
      by a later run there.
    * Seatbelt also denies renaming or removing the workspace and each
      ancestor the code may write (``deny_rename_paths``), or the code could
      move ``workspace_dir`` aside, write into the moved folder and move it
      back. Bubblewrap needs no such rule: a bind mount follows its directory
      when an ancestor is renamed, so a write through the new name still meets
      the read-only mount.
    * A folder that exists is denied for both backends; one that could not be
      made is denied by seatbelt alone (``deny_write_missing_paths``):
      bubblewrap binds only what exists, and a bind over a missing path
      leaves a stray file where the store later makes its folder.
    """
    workspace = getattr(settings, "workspace_dir", None)
    if workspace is None:
        return
    root = Path(workspace).expanduser().resolve(strict=False)
    folders = [root / "memory", root / "projects"]
    for folder in [*folders, _project_scope(settings)]:
        if folder is not None:
            _make_private(folder)
    policy.deny_read_paths += [str(folder) for folder in folders]
    policy.deny_write_paths += [str(folder) for folder in folders if folder.exists()]
    policy.deny_write_missing_paths += [str(folder) for folder in folders if not folder.exists()]
    writable = policy.resolved_writable_paths(Path.cwd())
    policy.deny_rename_paths += [
        str(path)
        for path in (root, *root.parents)
        if any(path == place or path.is_relative_to(place) for place in writable)
    ]


def _project_scope(settings: Any) -> Path | None:
    """This project's memory folder, where the store has a project scope."""
    from agentic_cli.workflow.memory_services import project_memory_dir

    try:
        return project_memory_dir(settings)
    except OSError:  # no working directory to name a project by
        return None


def _make_private(path: Path) -> None:
    """``path`` and any missing folder above it, each 0700, as the store makes
    them. A folder that cannot be made is left: the policy denies it anyway."""
    missing: list[Path] = []
    current = path
    while not os.path.lexists(current):
        missing.append(current)
        current = current.parent
    for folder in reversed(missing):
        try:
            folder.mkdir(mode=0o700)
        except FileExistsError:
            pass
        except OSError:
            return


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

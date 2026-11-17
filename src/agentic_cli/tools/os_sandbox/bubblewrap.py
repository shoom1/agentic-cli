"""Linux sandbox using bubblewrap (bwrap) with namespace isolation.

Wraps commands via:
    bwrap [args] -- /bin/bash -c '<command>'

Bubblewrap creates a lightweight user namespace with:
- Read-only root filesystem
- Writable bind mounts for allowed paths
- PID namespace isolation
- Network namespace isolation (no network)

A protected path that does not exist (``~/.bash_profile``, a project's
``.git`` before ``git init``) is bound over only where the code could create
it. Below the read-only root it needs nothing, and bwrap, which makes a
missing mount point itself, could make none there: it would exit with "Can't
create file at …: Read-only file system" before the code runs. In a folder
the code may write, bwrap would make the mount point on the host, an empty
file left behind after the run (an empty ``.git`` file breaks git: "invalid
gitfile format"), so the wrap makes a placeholder there instead: an empty
folder bound read-only, removed once the last run holding it is over
(``OSSandboxResult.release``).
"""

from __future__ import annotations

import contextlib
import os
import shlex
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from agentic_cli.tools.os_sandbox.base import OSSandbox, OSSandboxResult

if TYPE_CHECKING:
    from agentic_cli.tools.os_sandbox.policy import OSSandboxPolicy

#: A run's hold in a placeholder is a file named so. A placeholder is a folder
#: holding nothing else; each run that binds it holds it, in this process or
#: another, and the last to end removes it.
_HOLD = ".agentic-cli-sandbox-hold-"
#: The folder a placeholder is made in, with its first hold, before it is
#: renamed into place: it never appears without one.
_STAGING = ".agentic-cli-sandbox-staging-"
_HOLD_NOTE = "A sandboxed agentic-cli run keeps the folder this file is in until it ends.\n"


class BubblewrapSandbox(OSSandbox):
    """Linux sandbox using bubblewrap (bwrap) with namespace isolation."""

    def wrap_shell_command(
        self,
        command: str,
        working_dir: Path,
        policy: OSSandboxPolicy,
    ) -> OSSandboxResult:
        """Wrap a shell command with bwrap.

        Args:
            command: The raw shell command.
            working_dir: Working directory for execution.
            policy: Sandbox policy.

        Returns:
            OSSandboxResult with the wrapped command; call its ``release``
            once the command has run.
        """
        try:
            bwrap_args, release = self._prepare(working_dir, policy)
            escaped_command = command.replace("'", "'\\''")
            wrapped = f"{shlex.join(bwrap_args)} -- /bin/bash -c '{escaped_command}'"
            return OSSandboxResult(command=wrapped, sandbox_type="bubblewrap", release=release)
        except Exception as e:
            return OSSandboxResult(
                command=command,
                success=False,
                error=f"Failed to build bwrap arguments: {e}",
                sandbox_type="bubblewrap",
            )

    def wrap_python_command(
        self,
        python_args: list[str],
        working_dir: Path,
        policy: OSSandboxPolicy,
    ) -> OSSandboxResult:
        """Wrap a Python command with bwrap.

        Args:
            python_args: e.g. [sys.executable, "-c", script]
            working_dir: Working directory.
            policy: Sandbox policy.

        Returns:
            OSSandboxResult with the wrapped command; call its ``release``
            once the command has run.
        """
        try:
            bwrap_args, release = self._prepare(working_dir, policy)
            wrapped = shlex.join(bwrap_args + ["--"] + python_args)
            return OSSandboxResult(command=wrapped, sandbox_type="bubblewrap", release=release)
        except Exception as e:
            return OSSandboxResult(
                command=shlex.join(python_args),
                success=False,
                error=f"Failed to build bwrap arguments: {e}",
                sandbox_type="bubblewrap",
            )

    def _prepare(
        self, working_dir: Path, policy: OSSandboxPolicy
    ) -> tuple[list[str], Callable[[], None]]:
        """The bwrap arguments, with the placeholders they bind held, and the
        release of those holds."""
        release = _releaser(self._hold_placeholders(working_dir, policy))
        try:
            return self._build_bwrap_args(working_dir, policy), release
        except BaseException:
            release()
            raise

    def _hold_placeholders(
        self, working_dir: Path, policy: OSSandboxPolicy
    ) -> list[Path]:
        """Hold a placeholder at each missing protected path the code could
        create, made if no other run holds one there; returns this run's holds.

        Raises:
            RuntimeError: A placeholder could not be made (a folder of the
                user's that is read-only by its mode, which the code may
                chmod): the code could create the path, so no run.
        """
        writable = policy.resolved_writable_paths(working_dir)
        deny_writes = policy.resolved_deny_write_paths(working_dir)
        read_only = [d for d in deny_writes if os.path.lexists(d)]
        points = dict.fromkeys(p for p in map(_creation_point, deny_writes) if p is not None)
        holds: list[Path] = []
        for point in points:
            if not _could_create(point, writable, read_only):
                continue
            try:
                hold = _hold(point)
            except OSError as e:
                _releaser(holds)()
                raise RuntimeError(
                    f"no placeholder could be made at {point}, a protected path "
                    f"the code could otherwise create: {e.strerror or e}"
                ) from e
            if hold is not None:
                holds.append(hold)
        return holds

    def _build_bwrap_args(
        self, working_dir: Path, policy: OSSandboxPolicy
    ) -> list[str]:
        """Build bwrap command-line arguments from policy.

        Argument ordering matters — bwrap processes bind mounts in order,
        so more specific mounts must come after less specific ones.

        Structure:
        1. Isolation flags (session, die-with-parent, PID namespace)
        2. Network namespace (if blocking network)
        3. Read-only root (--ro-bind / /)
        4. Writable bind mounts for allowed paths
        5. Read-only overrides for denied paths within writable regions,
           and for the placeholders of missing ones the code could create
           (``_hold_placeholders`` makes them before bwrap runs)
        6. Hide denied read paths
        7. Device and proc filesystems

        Args:
            working_dir: Working directory (always writable).
            policy: Sandbox policy.

        Returns:
            List of bwrap arguments (without the trailing command).
        """
        args = [
            "bwrap",
            "--new-session",
            "--die-with-parent",
            "--unshare-pid",
        ]

        # Network isolation
        if not policy.allow_network:
            args.append("--unshare-net")

        # Read-only root filesystem
        args.extend(["--ro-bind", "/", "/"])

        # Writable bind mounts for allowed paths
        writable = policy.resolved_writable_paths(working_dir)
        for wp in writable:
            if wp.exists():
                args.extend(["--bind", str(wp), str(wp)])
            else:
                # Create a tmpfs for nonexistent writable paths so they
                # can be written to inside the sandbox
                args.extend(["--tmpfs", str(wp)])

        # Deny writes within writable regions: re-mount as read-only. A
        # missing one is bound at its placeholder where the code could create
        # it; elsewhere the read-only root covers it (see the module docstring).
        deny_writes = policy.resolved_deny_write_paths(working_dir)
        read_only = [d for d in deny_writes if os.path.lexists(d)]
        bound: set[Path] = set()
        for dp in deny_writes:
            target = _creation_point(dp) or dp
            if target in bound:
                continue
            if os.path.lexists(target) or _could_create(target, writable, read_only):
                args.extend(["--ro-bind", str(target), str(target)])
                bound.add(target)

        # Hide denied read paths with empty tmpfs overlays
        deny_reads = policy.resolved_deny_read_paths(working_dir)
        for drp in deny_reads:
            if drp.is_dir():
                args.extend(["--tmpfs", str(drp)])
            elif drp.is_file():
                args.extend(["--ro-bind", "/dev/null", str(drp)])

        # Standard device nodes and proc
        args.extend(["--dev", "/dev"])
        args.extend(["--proc", "/proc"])

        return args

    def is_available(self) -> bool:
        from agentic_cli.tools.os_sandbox.detect import (
            detect_sandbox_capabilities,
        )

        return detect_sandbox_capabilities().bubblewrap_available

    @property
    def sandbox_type(self) -> str:
        return "bubblewrap"


def _is_placeholder(path: Path) -> bool:
    """A folder holding nothing but holds: a placeholder some run keeps."""
    if path.is_symlink():
        return False
    try:
        with os.scandir(path) as entries:
            held = False
            for entry in entries:
                if not entry.name.startswith(_HOLD):
                    return False
                held = True
            return held
    except OSError:  # not a folder, or one this user cannot list
        return False


def _creation_point(path: Path) -> Path | None:
    """Where code would start creating ``path``: its first component that is
    missing or a placeholder (``path`` itself, or a missing folder above it).
    None when ``path`` exists."""
    for part in (*reversed(path.parents), path):
        if not os.path.lexists(part) or _is_placeholder(part):
            return part
    return None


def _could_create(point: Path, writable: list[Path], read_only: list[Path]) -> bool:
    """Whether the code could create ``point`` (missing, or a placeholder):
    it lies below a writable bind, and below no protected path bound
    read-only over it."""
    def below(place: Path) -> bool:
        return point != place and point.is_relative_to(place)

    return any(below(w) for w in writable if w.exists()) and not any(map(below, read_only))


def _hold(point: Path) -> Path | None:
    """This run's hold on the placeholder at ``point``, made if missing; None
    when a real file or folder took its place meanwhile (it is then bound as
    it is)."""
    name = f"{_HOLD}{os.getpid()}-{uuid.uuid4().hex}"
    for _ in range(5):
        if _is_placeholder(point):
            try:
                _write_hold(point / name)
                return point / name
            except FileNotFoundError:
                continue  # its last run removed it meanwhile: make it anew
        if os.path.lexists(point):
            return None
        staging = point.parent / f"{_STAGING}{uuid.uuid4().hex}"
        os.mkdir(staging, 0o700)
        try:
            _write_hold(staging / name)
            os.rename(staging, point)
            return point / name
        except OSError:
            _drop(staging / name)
            if not os.path.lexists(point):
                raise
            # Made meanwhile, by another run or for real: look again.
    raise OSError(f"{point} kept changing")


def _write_hold(path: Path) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, _HOLD_NOTE.encode())
    finally:
        os.close(fd)


def _drop(hold: Path) -> None:
    """Remove a hold, and its placeholder if no other run holds it (``rmdir``:
    only while it is empty, so whatever was put in it meanwhile stays)."""
    with contextlib.suppress(OSError):
        os.unlink(hold)
    with contextlib.suppress(OSError):
        os.rmdir(hold.parent)


def _releaser(holds: list[Path]) -> Callable[[], None]:
    """The release of ``holds``: each is dropped once, however often it is
    called."""

    def release() -> None:
        while holds:
            _drop(holds.pop())

    return release

"""Linux sandbox using bubblewrap (bwrap) with namespace isolation.

Wraps commands via:
    bwrap [args] -- /bin/bash -c '<command>'

Bubblewrap creates a lightweight user namespace with:
- Read-only root filesystem
- Writable bind mounts for allowed paths
- PID namespace isolation, the command its pid 1 (``--as-pid-1``): when it
  exits, the kernel kills and reaps everything it started before bwrap
  returns, so a run that returned has left nothing behind in the sandbox
- Network namespace isolation (no network)

A protected path that does not exist (``~/.bash_profile``, a project's
``.git`` before ``git init``) is bound over only where the code could create
it. Below the read-only root it needs nothing, and bwrap, which makes a
missing mount point itself, could make none there: it would exit with "Can't
create file at …: Read-only file system" before the code runs. In a folder
the code may write, bwrap would make the mount point on the host, an empty
file left behind after the run (an empty ``.git`` file breaks git: "invalid
gitfile format"), so the wrap makes a placeholder there instead: an empty
folder bound read-only. Each run holds it with a file of its own, in this
process or another, and ``OSSandboxResult.release`` drops the run's hold and
removes the folder once no hold is left. A run never removes a folder it did
not make: a protected folder that is there and empty is held and bound like a
placeholder (it may be one whose last run is removing it), but a hold in it
says no run made it, and the folder stays. A run that did not end on its own
(a timeout, a kill) is not released: its placeholder stays, empty and
protected, and later runs hold it in turn.
"""

from __future__ import annotations

import contextlib
import errno
import os
import shlex
import uuid
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from agentic_cli.file_utils import open_dir_no_follow
from agentic_cli.tools.os_sandbox.base import OSSandbox, OSSandboxResult

if TYPE_CHECKING:
    from agentic_cli.tools.os_sandbox.policy import OSSandboxPolicy

#: A run's hold in a placeholder is a file named so. A placeholder is a folder
#: below a writable bind holding nothing else; each run that binds it holds
#: it, and the last to release removes it.
_HOLD = ".agentic-cli-sandbox-hold-"
#: A hold in a folder no run made: one that was there and empty when the run
#: joined it, or one holding such a hold. No run removes it.
_KEPT = f"{_HOLD}kept-"
#: The folder a placeholder is made in, with its first hold, before it is
#: renamed into place: it never appears without one.
_STAGING = ".agentic-cli-sandbox-staging-"
_HOLD_NOTE = "A sandboxed agentic-cli run keeps the folder this file is in until it ends.\n"
#: How often the wrap looks again at a protected path that changed under it.
_TRIES = 5


@dataclass(frozen=True)
class _Hold:
    """A run's hold: the file ``name`` in the placeholder ``point``, which
    lies below the writable bind ``root``. ``removes``: the release may remove
    the folder (a run made it). ``name`` None: a protected folder no hold
    could go in, bound as it is, with nothing to drop."""

    root: Path
    point: Path
    name: str | None
    removes: bool = False


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
            once the command has ended on its own.
        """
        release: Callable[[], None] | None = None
        try:
            bwrap_args, release = self._prepare(working_dir, policy)
            escaped_command = command.replace("'", "'\\''")
            wrapped = f"{shlex.join(bwrap_args)} -- /bin/bash -c '{escaped_command}'"
            return OSSandboxResult(command=wrapped, sandbox_type="bubblewrap", release=release)
        except Exception as e:
            if release is not None:
                release()
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
            once the command has ended on its own.
        """
        release: Callable[[], None] | None = None
        try:
            bwrap_args, release = self._prepare(working_dir, policy)
            wrapped = shlex.join(bwrap_args + ["--"] + python_args)
            return OSSandboxResult(command=wrapped, sandbox_type="bubblewrap", release=release)
        except Exception as e:
            if release is not None:
                release()
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
        holds = self._hold_placeholders(working_dir, policy)
        release = _releaser(holds)
        try:
            held = [hold.point for hold in holds]
            return self._build_bwrap_args(working_dir, policy, held=held), release
        except BaseException:
            release()
            raise

    def _hold_placeholders(
        self, working_dir: Path, policy: OSSandboxPolicy
    ) -> list[_Hold]:
        """Hold a placeholder at each missing protected path the code could
        create (joining one another run holds, or one being removed), and
        return this run's holds.

        A path the code cannot create needs none: one below the read-only
        root, or where no placeholder can be made because the file system is
        read-only, or the folder is one the user may not write and does not
        own (so the code cannot give itself the right). A protected folder
        that is there but takes no hold is bound as it is.

        Raises:
            RuntimeError: Any other placeholder could not be made (a folder of
                the user's that is read-only by its mode, which the code may
                chmod; a folder swapped for a symlink), or a path kept
                changing: the code could create the path, so no run.
        """
        writable = _bound(policy, working_dir)
        deny_writes = policy.resolved_deny_write_paths(working_dir)
        read_only = _real(deny_writes, writable)
        holds: dict[Path, _Hold] = {}
        try:
            for path in deny_writes:
                self._hold_one(path, writable, read_only, holds)
        except BaseException:
            _releaser(list(holds.values()))()
            raise
        return list(holds.values())

    @staticmethod
    def _hold_one(
        path: Path, writable: list[Path], read_only: list[Path], holds: dict[Path, _Hold]
    ) -> None:
        """Hold the placeholder ``path`` needs, if any, adding it to ``holds``."""
        for _ in range(_TRIES):
            point = _creation_point(path, writable)
            root = None if point is None else _writable_root(point, writable, read_only)
            if point is None or root is None or point in holds:
                return
            try:
                hold = _hold(root, point, own=point == path)
            except OSError as e:
                if _code_cannot_create(e, point):
                    return
                raise RuntimeError(
                    f"no placeholder could be made at {point}, a protected path "
                    f"the code could otherwise create: {e.strerror or e}"
                ) from e
            if hold is not None:
                holds[point] = hold
                return
            # A real file or folder took its place, or another run's
            # placeholder came and went: look again.
        raise RuntimeError(f"{path} kept changing while its placeholder was made")

    def _build_bwrap_args(
        self, working_dir: Path, policy: OSSandboxPolicy, held: Collection[Path] = ()
    ) -> list[str]:
        """Build bwrap command-line arguments from policy.

        Argument ordering matters — bwrap processes bind mounts in order,
        so more specific mounts must come after less specific ones.

        Structure:
        1. Isolation flags (session, die-with-parent, PID namespace, the
           command as its pid 1)
        2. Network namespace (if blocking network)
        3. Read-only root (--ro-bind / /)
        4. Writable bind mounts for allowed paths
        5. Read-only overrides for denied paths that exist within writable
           regions, and for the placeholders this run holds
        6. Hide denied read paths
        7. Device and proc filesystems

        Args:
            working_dir: Working directory (always writable).
            policy: Sandbox policy.
            held: The placeholders this run holds (``_hold_placeholders``,
                which the wraps call first). A missing protected path the
                code could create is bound only through one of them.

        Returns:
            List of bwrap arguments (without the trailing command).
        """
        args = [
            "bwrap",
            "--new-session",
            "--die-with-parent",
            "--unshare-pid",
            "--as-pid-1",
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

        # Deny writes within writable regions: re-mount as read-only what
        # exists, and the placeholders of what does not; elsewhere the
        # read-only root covers a missing path (see the module docstring).
        deny_writes = policy.resolved_deny_write_paths(working_dir)
        for target in dict.fromkeys([*_real(deny_writes, _bound(policy, working_dir)), *held]):
            args.extend(["--ro-bind", str(target), str(target)])

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


def _bound(policy: OSSandboxPolicy, working_dir: Path) -> list[Path]:
    """The writable paths bound from the host (a missing one is a tmpfs,
    which nothing of the host's lies below)."""
    return [w for w in policy.resolved_writable_paths(working_dir) if w.exists()]


def _below(path: Path, place: Path) -> bool:
    return path != place and path.is_relative_to(place)


def _real(deny_writes: list[Path], writable: list[Path]) -> list[Path]:
    """The protected paths that exist as themselves, not as placeholders."""
    return [d for d in deny_writes if os.path.lexists(d) and _creation_point(d, writable) is None]


def _holds_in(fd_or_path: int | Path) -> list[str] | None:
    """The holds in a folder holding nothing but holds (none, when it is
    empty); None for anything else (a real folder, a file, one this user
    cannot list)."""
    try:
        with os.scandir(fd_or_path) as entries:
            names = [entry.name for entry in entries]
    except OSError:
        return None
    return names if all(name.startswith(_HOLD) for name in names) else None


def _joinable(holds: list[str] | None, own: bool) -> bool:
    """A placeholder a run may hold: one holding holds, or, at the protected
    path itself, an empty folder (a placeholder whose last run is removing
    it, or one that was there: either way joined, never taken for a real
    folder the release could remove from under the run). An empty folder
    above the path is the user's (a working directory, ``~/.config``)."""
    return holds is not None and (bool(holds) or own)


def _creation_point(path: Path, writable: list[Path]) -> Path | None:
    """Where code would start creating ``path``: its first component that is
    missing or, below a writable bind, a placeholder (``path`` itself, or a
    folder above it). None when ``path`` exists as itself."""
    for part in (*reversed(path.parents), path):
        if not os.path.lexists(part):
            return part
        if (
            any(_below(part, w) for w in writable)
            and not part.is_symlink()
            and _joinable(_holds_in(part), own=part == path)
        ):
            return part
    return None


def _writable_root(point: Path, writable: list[Path], read_only: list[Path]) -> Path | None:
    """The writable bind ``point`` lies below, when the code could create it
    there (below no protected path bound read-only over it); else None."""
    if any(_below(point, d) for d in read_only):
        return None
    roots = [w for w in writable if _below(point, w)]
    return max(roots, key=lambda w: len(w.parts), default=None)


def _code_cannot_create(error: OSError, point: Path) -> bool:
    """Whether making the placeholder failed where the code would fail too: a
    read-only file system, or a folder the user may not write and does not
    own (its owner could give the code the right with chmod)."""
    if error.errno == errno.EROFS:
        return True
    if error.errno in (errno.EACCES, errno.EPERM):
        try:
            return os.stat(point.parent).st_uid != os.geteuid()
        except OSError:
            return False
    return False


def _open_dir_below(root: Path, folder: Path) -> int:
    """Open ``folder``, ``root`` or below it, following no symlink below
    ``root``: code a run may write there could swap a folder for one. ``root``
    is a writable bind, a mount point in every sandbox, which the code cannot
    move."""
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in folder.relative_to(root).parts:
            inner = open_dir_no_follow(name, dir_fd=fd)
            os.close(fd)
            fd = inner
        return fd
    except BaseException:
        os.close(fd)
        raise


def _hold(root: Path, point: Path, own: bool) -> _Hold | None:
    """This run's hold on the placeholder at ``point`` (``own``: the protected
    path itself, not a folder above it), made if missing; None when the place
    changed meanwhile (a folder on the way went away, a real file or folder
    now stands there, the placeholder was removed before the hold was
    written, or another run made it first). Every step is taken in folders
    opened without following a symlink below ``root``."""
    run = f"{os.getpid()}-{uuid.uuid4().hex}"
    try:
        parent = _open_dir_below(root, point.parent)
    except FileNotFoundError:
        return None
    try:
        try:
            folder = open_dir_no_follow(point.name, dir_fd=parent)
        except FileNotFoundError:
            folder = None
        except OSError as e:
            if e.errno in (errno.ELOOP, errno.ENOTDIR):
                return None  # a symlink or a file: no placeholder of ours
            raise
        if folder is not None:
            try:
                return _join(root, point, folder, run, own)
            finally:
                os.close(folder)
        staging = f"{_STAGING}{uuid.uuid4().hex}"
        name = f"{_HOLD}{run}"
        try:
            os.mkdir(staging, 0o700, dir_fd=parent)
        except FileNotFoundError:
            return None  # the folder it goes in went away
        try:
            folder = open_dir_no_follow(staging, dir_fd=parent)
            try:
                _write_hold(name, folder)
            finally:
                os.close(folder)
            os.rename(staging, point.name, src_dir_fd=parent, dst_dir_fd=parent)
        except BaseException as error:
            _unhold(parent, staging, name)
            if not (isinstance(error, OSError) and _exists(point.name, parent)):
                raise
            return None  # made meanwhile, by another run or for real
        return _Hold(root, point, name, removes=True)
    finally:
        os.close(parent)


def _join(root: Path, point: Path, folder: int, run: str, own: bool) -> _Hold | None:
    """Hold the existing folder at ``point`` (open as ``folder``) as a
    placeholder. A folder that holds no hold, or a kept one, was not made by
    a run: the hold says so, and its release leaves the folder. One that
    takes no hold (the user may not write it) is bound as it is: the code
    could still move it aside."""
    holds = _holds_in(folder)
    if not _joinable(holds, own):
        return None  # a real folder now: looked at again, bound as it is
    kept = not holds or any(name.startswith(_KEPT) for name in holds)
    name = f"{_KEPT if kept else _HOLD}{run}"
    try:
        _write_hold(name, folder)
    except FileNotFoundError:
        return None  # removed by its last run before the hold was written
    except OSError:
        return _Hold(root, point, None)
    return _Hold(root, point, name, removes=not kept)


def _exists(name: str, parent: int) -> bool:
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _write_hold(name: str, folder: int) -> None:
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=folder)
    try:
        os.write(fd, _HOLD_NOTE.encode())
    finally:
        os.close(fd)


def _unhold(parent: int, folder_name: str, hold_name: str, *, remove: bool = True) -> None:
    """Remove the hold ``hold_name`` from the folder ``folder_name`` of
    ``parent``, then, if ``remove``, the folder if nothing is left in it
    (``rmdir``: another run's hold, or whatever the user put there meanwhile,
    keeps it)."""
    with contextlib.suppress(OSError):
        folder = open_dir_no_follow(folder_name, dir_fd=parent)
        try:
            os.unlink(hold_name, dir_fd=folder)
        finally:
            os.close(folder)
    if remove:
        with contextlib.suppress(OSError):
            os.rmdir(folder_name, dir_fd=parent)


def _drop(hold: _Hold) -> None:
    """Drop a hold, and its placeholder if a run made it and no other run
    holds it."""
    if hold.name is None:
        return
    try:
        parent = _open_dir_below(hold.root, hold.point.parent)
    except OSError:
        return  # moved or swapped meanwhile: left as it is
    try:
        _unhold(parent, hold.point.name, hold.name, remove=hold.removes)
    finally:
        os.close(parent)


def _releaser(holds: list[_Hold]) -> Callable[[], None]:
    """The release of ``holds``: each is dropped once, however often and from
    however many threads it is called."""

    def release() -> None:
        while True:
            try:
                hold = holds.pop()
            except IndexError:
                return
            _drop(hold)

    return release

"""Sandbox manager for stateful code execution.

Manages sandbox sessions and delegates execution to a pluggable backend
(default: JupyterLocalBackend).
"""

from __future__ import annotations

import atexit
import contextlib
import os
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, TYPE_CHECKING

from agentic_cli.logging import Loggers
from agentic_cli.file_utils import (
    copy_regular_file_no_follow, open_dir_no_follow, sanitize_filename,
)
from agentic_cli.tools.sandbox.models import ExecutionResult

#: A session ID names a directory and a container, so it is validated rather
#: than sanitized: sanitizing mapped "" to the directory holding every session
#: and made different IDs (``a.b``, ``a_b``) share one.
_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.tools.sandbox.backends.base import SandboxBackend

logger = Loggers.tools()


def _folded_parts(path: Path) -> tuple[str, ...]:
    return tuple(unicodedata.normalize("NFC", part).casefold() for part in path.parts)


def sandbox_root_for(settings) -> Path:
    """The directory that holds every sandbox session's directory."""
    return Path(settings.workspace_dir) / "sandbox"


def _runs_through(spellings: set[tuple[str, ...]], directory: Path) -> bool:
    for form in {Path(directory).absolute(), Path(os.path.realpath(directory))}:
        prefix = _folded_parts(form)
        if any(spelled[: len(prefix)] == prefix for spelled in spellings):
            return True
    return False


def _refuse_a_path_through_a_session(
    src: str, path: Path, session_dir: Path, sandbox_root: Path | None
) -> None:
    """Refuse an input whose path runs through a session directory: this
    session's, or, given ``sandbox_root``, any session's.

    Below a session directory every folder belongs to that session's kernel:
    it can swap one for a symlink between the permission check and the copy,
    and the host would stage the host file the kernel chose. This session
    already sees its own files; another session's outputs reach the
    artifacts directory, a host-only copy, after each successful call. The
    paths are compared as spelled (``..`` kept, since the OS walks it through
    the folder) and as resolved now, case-insensitively (APFS's default),
    against each directory's spelling and its real path."""
    spellings = {
        _folded_parts(Path(os.getcwd()) / path),
        _folded_parts(Path(os.path.realpath(path))),
    }
    if _runs_through(spellings, session_dir):
        raise ValueError(
            "input is inside this sandbox session's directory, which code in the "
            f"sandbox controls, and the sandbox already sees it: {src}"
        )
    if sandbox_root is not None and _runs_through(spellings, sandbox_root):
        raise ValueError(
            "input is inside another sandbox session's directory, which code in "
            f"that sandbox controls: {src}. Stage its copy from the artifacts "
            "directory instead: the files a session writes to outputs/ are copied "
            "there after each successful call."
        )


def stage_inputs(
    session_dir: Path, inputs: list[str], *, sandbox_root: Path | None = None
) -> None:
    """Copy each host path into session_dir/inputs/<basename> (the in-sandbox
    'inputs/<name>' contract). Raises ValueError on a missing file, a
    basename collision (v1 does not support renaming), or an input inside
    this session's directory or, given ``sandbox_root`` (see
    :func:`sandbox_root_for`), any session's."""
    if session_dir is None:
        raise ValueError("session_dir is required for staging inputs")
    if not inputs:
        return
    # The kernel can swap 'inputs' for a symlink to any host directory at any
    # moment, and O_NOFOLLOW guards only a file's own name. So 'inputs' is
    # opened once, without following a link, and every write goes through
    # that descriptor. The session directory itself is the container's
    # /workspace mount point, which the kernel cannot rename.
    Path(session_dir).mkdir(parents=True, exist_ok=True)
    session_fd = os.open(session_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            inputs_fd = open_dir_no_follow("inputs", dir_fd=session_fd, create=True)
        except OSError as exc:
            # 'inputs' is a symlink / not a real dir (kernel-plantable).
            raise ValueError(
                "inputs staging directory is not a real directory (symlink?)"
            ) from exc
    finally:
        os.close(session_fd)
    try:
        seen: set[str] = set()
        for src in inputs:
            p = Path(src).expanduser()
            _refuse_a_path_through_a_session(src, p, session_dir, sandbox_root)
            name = p.name
            if name in seen:
                raise ValueError(f"duplicate input basename {name!r}; rename one of the source files")
            seen.add(name)
            try:
                copy_regular_file_no_follow(p, name, dst_dir_fd=inputs_fd)
            except FileNotFoundError as exc:
                raise ValueError(f"input file not found: {src}") from exc
            except (OSError, ValueError) as exc:
                raise ValueError(f"input must be a regular file: {src}") from exc
    finally:
        os.close(inputs_fd)


def sandbox_disabled_reason(settings) -> str:
    """Backend-aware reason string for a disabled sandbox_execute. Shared by
    every entry point (module tool + factory tool) so the opt-in gate is
    applied consistently and worded accurately for the selected backend."""
    backend = getattr(settings, "stateful_executor_backend", "none")
    if backend == "docker":
        detail = "The 'docker' backend runs it in a network-isolated, memory/CPU/PID-capped container."
    elif backend == "local":
        detail = "The 'local' backend runs Python with host privileges and no OS sandbox."
    else:
        detail = "No stateful executor is configured."
    return (f"sandbox_execute is not available. {detail} "
            "Set stateful_executor_backend to 'docker' (isolated) or 'local' (unsandboxed) to use it.")


@dataclass
class SandboxSession:
    """Metadata for an active sandbox session."""

    session_id: str
    working_dir: Path
    execution_count: int = 0


class _SessionLock:
    """One session's lock, and how many calls hold it or wait for it: the
    entry is dropped when none does."""

    __slots__ = ("lock", "users")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.users = 0


class SandboxManager:
    """Manages sandbox sessions and delegates to a backend.

    Calls come from several threads at once (both backends run a model step's
    synchronous tool calls in worker threads), and a kernel runs one cell at a
    time through a client that is not thread-safe. So a call holds its
    session's lock from staging its inputs to collecting its outputs, and the
    manager's own lock covers the backend's creation, the session table and
    the session locks; it is never held across a call into a kernel.

    Args:
        settings: Application settings instance.
        backend: Optional backend for test injection. If None, created
            lazily from settings.stateful_executor_backend.
    """

    #: How much longer than its own timeout a call waits for its session
    #: while another call runs there, before it is told the session is busy.
    busy_margin: float = 60.0

    def __init__(
        self,
        settings: "BaseSettings",
        backend: "SandboxBackend | None" = None,
    ) -> None:
        self._settings = settings
        self._backend = backend
        self._sessions: dict[str, SandboxSession] = {}
        self._lock = threading.Lock()
        self._session_locks: dict[str, _SessionLock] = {}
        self._closed = False

        atexit.register(self._atexit_cleanup)
        logger.debug("sandbox_manager_created")

    def _ensure_backend(self) -> "SandboxBackend":
        """Lazily create the backend if not injected. The caller holds
        ``self._lock``, so two calls never create two."""
        if self._backend is None:
            self._backend = self._create_backend(self._settings.stateful_executor_backend)
        return self._backend

    def _create_backend(self, backend_name: str) -> "SandboxBackend":
        """Create a backend by name."""
        if backend_name == "local":
            from agentic_cli.tools.sandbox.backends.jupyter_local import JupyterLocalBackend
            return JupyterLocalBackend()
        if backend_name == "docker":
            from agentic_cli.tools.sandbox.backends.jupyter_docker import JupyterDockerBackend
            return JupyterDockerBackend(self._settings)
        raise ValueError(f"Unknown stateful_executor_backend: {backend_name!r}")

    def _get_session_dir(self, session_id: str) -> Path:
        """Get or create the working directory for a session."""
        safe_id = sanitize_filename(session_id)
        session_dir = sandbox_root_for(self._settings) / safe_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    @contextlib.contextmanager
    def _session_lock(self, session_id: str) -> Iterator[_SessionLock]:
        """``session_id``'s lock, counted as in use until the block ends."""
        with self._lock:
            slot = self._session_locks.setdefault(session_id, _SessionLock())
            slot.users += 1
        try:
            yield slot
        finally:
            with self._lock:
                slot.users -= 1
                if not slot.users:
                    del self._session_locks[session_id]

    def execute(
        self,
        code: str,
        session_id: str = "default",
        timeout_seconds: int | None = None,
        inputs: list[str] | None = None,
    ) -> ExecutionResult:
        """Execute code in a sandbox session.

        A call waits for one already running in the session, up to its own
        timeout plus ``busy_margin``.

        Args:
            code: Python code to execute.
            session_id: Session identifier (default: "default").
            timeout_seconds: Execution timeout (uses settings default if None).
            inputs: Optional list of host file paths to stage into
                inputs/<basename> inside the session before execution.

        Returns:
            ExecutionResult with output and metadata.
        """
        if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
            return ExecutionResult(
                success=False,
                error=(
                    "session_id must be 1-64 characters of letters, digits, '-' "
                    f"or '_' (got {session_id!r})."
                ),
            )
        if timeout_seconds is not None and timeout_seconds <= 0:
            return ExecutionResult(
                success=False,
                error=f"timeout_seconds must be a positive number (got {timeout_seconds!r}).",
            )

        if timeout_seconds is None:
            timeout_seconds = self._settings.sandbox_timeout

        # A lock cannot wait longer than TIMEOUT_MAX (about 292 years).
        wait = min(timeout_seconds + self.busy_margin, threading.TIMEOUT_MAX)
        with self._session_lock(session_id) as slot:
            if not slot.lock.acquire(timeout=wait):
                return ExecutionResult(
                    success=False,
                    error=(
                        f"Sandbox session {session_id!r} is busy: another call in it "
                        f"did not finish within {wait:g}s. Try again later, or use "
                        "another session_id."
                    ),
                )
            try:
                return self._execute_holding_lock(code, session_id, timeout_seconds, inputs)
            finally:
                self._release(slot, session_id)

    def _execute_holding_lock(
        self,
        code: str,
        session_id: str,
        timeout_seconds: int,
        inputs: list[str] | None,
    ) -> ExecutionResult:
        """Run one call; the caller holds the session's lock."""
        with self._lock:
            if self._closed:
                return ExecutionResult(success=False, error="The sandbox is shut down.")
            backend = self._ensure_backend()
            session = self._sessions.get(session_id)
            if session is None:
                max_sessions = self._settings.sandbox_max_sessions
                if len(self._sessions) >= max_sessions:
                    return ExecutionResult(
                        success=False,
                        error=f"Maximum sessions ({max_sessions}) reached. Reset an existing session first.",
                    )
                session = SandboxSession(
                    session_id=session_id,
                    working_dir=self._get_session_dir(session_id),
                )
                self._sessions[session_id] = session

        result = backend.execute(
            code=code,
            session_id=session_id,
            timeout_seconds=timeout_seconds,
            working_dir=session.working_dir,
            inputs=inputs,
        )

        # A failed start (docker down, startup error) must not leave phantom
        # session metadata occupying a max_sessions slot with no backend session.
        if not result.success and not backend.has_session(session_id):
            with self._lock:
                self._sessions.pop(session_id, None)
            return result

        session.execution_count += 1
        logger.debug(
            "sandbox_executed",
            session_id=session_id,
            execution_count=session.execution_count,
            success=result.success,
        )
        return result

    def _release(self, slot: _SessionLock, session_id: str) -> None:
        """Release a call's hold on its session. A session reset (or cleaned
        up) during the call is shut down first: the reset left that to the
        call. Checking and releasing under ``self._lock`` is what makes one
        of the two do it."""
        with self._lock:
            if session_id in self._sessions:
                slot.lock.release()
                return
            backend = self._backend
        try:
            self._shut_down(backend, session_id)
        finally:
            slot.lock.release()

    def _shut_down(self, backend: "SandboxBackend | None", session_id: str) -> None:
        """Shut a session's kernel down, if the backend has one. Never raises:
        the caller may be returning a call's result."""
        if backend is None or not backend.has_session(session_id):
            return
        try:
            backend.reset_session(session_id)
        except Exception as exc:  # noqa: BLE001 - a teardown must not fail the call
            logger.warning("sandbox_session_reset_failed", session_id=session_id, error=str(exc))

    def reset_session(self, session_id: str) -> bool:
        """Reset a sandbox session.

        Never waits for a call running in the session (the CLI's /sandbox
        runs on the event loop): that call shuts the kernel down as it ends,
        since its client is not thread-safe, and the session's next call
        starts afresh.

        Args:
            session_id: Session to reset.

        Returns:
            True if the session was active and reset, False if not found.
        """
        with self._session_lock(session_id) as slot:
            with self._lock:
                if self._sessions.pop(session_id, None) is None:
                    return False
                idle = slot.lock.acquire(blocking=False)
                backend = self._backend
            if idle:
                try:
                    if backend is not None:
                        backend.reset_session(session_id)
                finally:
                    slot.lock.release()
        logger.debug("sandbox_session_reset", session_id=session_id, deferred=not idle)
        return True

    def list_sessions(self) -> list[dict[str, Any]]:
        """List active sandbox sessions with live backend status."""
        with self._lock:
            sessions = list(self._sessions.values())
            backend = self._backend
        if not sessions or backend is None:
            return []
        rows: list[dict[str, Any]] = []
        for s in sessions:
            status = backend.session_status(s.session_id)
            rows.append({
                "session_id": s.session_id,
                "working_dir": str(s.working_dir),
                "execution_count": s.execution_count,
                "state": status.state,
            })
        return rows

    def cleanup(self) -> None:
        """Clean up all sessions and the backend.

        Never waits for a running call: its session is shut down as the call
        ends (see ``reset_session``), and a call still waiting for a session,
        or made later, is refused.
        """
        atexit.unregister(self._atexit_cleanup)
        with self._lock:
            self._closed = True
            session_ids = list(self._sessions)
        for session_id in session_ids:
            self.reset_session(session_id)
        with self._lock:
            # No call holds or waits for a session, and none can start: what
            # the backend still has belongs to no call.
            if not self._session_locks and self._backend is not None:
                self._backend.cleanup()
        logger.debug("sandbox_manager_cleaned_up")

    def _atexit_cleanup(self) -> None:
        """Safety net for process exit."""
        try:
            self.cleanup()
        except Exception:
            pass

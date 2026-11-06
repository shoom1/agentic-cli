"""Sandbox manager for stateful code execution.

Manages sandbox sessions and delegates execution to a pluggable backend
(default: JupyterLocalBackend).
"""

from __future__ import annotations

import atexit
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

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


def _refuse_a_path_through_the_session(src: str, path: Path, session_dir: Path) -> None:
    """Refuse an input whose path runs through the session directory.

    Below the session directory every folder is the kernel's: it can swap one
    for a symlink between the permission check and the copy, and the host
    would stage the host file the kernel chose. There is nothing to stage
    from there anyway: the sandbox already sees the file. The paths are
    compared as spelled (``..`` kept, since the OS walks it through the
    folder) and as resolved now, case-insensitively (APFS's default), against
    the session directory's spelling and its real path."""
    spellings = {
        _folded_parts(Path(os.getcwd()) / path),
        _folded_parts(Path(os.path.realpath(path))),
    }
    for session in {Path(session_dir).absolute(), Path(os.path.realpath(session_dir))}:
        prefix = _folded_parts(session)
        if any(spelled[: len(prefix)] == prefix for spelled in spellings):
            raise ValueError(
                f"input is inside the sandbox session, where the sandbox already sees it: {src}"
            )


def stage_inputs(session_dir: Path, inputs: list[str]) -> None:
    """Copy each host path into session_dir/inputs/<basename> (the in-sandbox
    'inputs/<name>' contract). Raises ValueError on a missing file or a
    basename collision (v1 does not support renaming)."""
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
            _refuse_a_path_through_the_session(src, p, session_dir)
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


class SandboxManager:
    """Manages sandbox sessions and delegates to a backend.

    Args:
        settings: Application settings instance.
        backend: Optional backend for test injection. If None, created
            lazily from settings.stateful_executor_backend.
    """

    def __init__(
        self,
        settings: "BaseSettings",
        backend: "SandboxBackend | None" = None,
    ) -> None:
        self._settings = settings
        self._backend = backend
        self._sessions: dict[str, SandboxSession] = {}

        atexit.register(self._atexit_cleanup)
        logger.debug("sandbox_manager_created")

    def _ensure_backend(self) -> "SandboxBackend":
        """Lazily create the backend if not injected."""
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
        base = Path(self._settings.workspace_dir)
        safe_id = sanitize_filename(session_id)
        session_dir = base / "sandbox" / safe_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def execute(
        self,
        code: str,
        session_id: str = "default",
        timeout_seconds: int | None = None,
        inputs: list[str] | None = None,
    ) -> ExecutionResult:
        """Execute code in a sandbox session.

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

        max_sessions = self._settings.sandbox_max_sessions
        if session_id not in self._sessions and len(self._sessions) >= max_sessions:
            return ExecutionResult(
                success=False,
                error=f"Maximum sessions ({max_sessions}) reached. Reset an existing session first.",
            )

        if timeout_seconds is None:
            timeout_seconds = self._settings.sandbox_timeout

        # Get or create session metadata
        if session_id not in self._sessions:
            working_dir = self._get_session_dir(session_id)
            self._sessions[session_id] = SandboxSession(
                session_id=session_id,
                working_dir=working_dir,
            )

        session = self._sessions[session_id]
        backend = self._ensure_backend()

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

    def reset_session(self, session_id: str) -> bool:
        """Reset a sandbox session.

        Args:
            session_id: Session to reset.

        Returns:
            True if the session was active and reset, False if not found.
        """
        backend = self._ensure_backend()
        was_active = session_id in self._sessions
        if was_active:
            del self._sessions[session_id]
            backend.reset_session(session_id)
            logger.debug("sandbox_session_reset", session_id=session_id)
        return was_active

    def list_sessions(self) -> list[dict[str, Any]]:
        """List active sandbox sessions with live backend status."""
        if not self._sessions:
            return []
        backend = self._ensure_backend()
        rows: list[dict[str, Any]] = []
        for s in self._sessions.values():
            status = backend.session_status(s.session_id)
            rows.append({
                "session_id": s.session_id,
                "working_dir": str(s.working_dir),
                "execution_count": s.execution_count,
                "state": status.state,
            })
        return rows

    def cleanup(self) -> None:
        """Clean up all sessions and the backend."""
        atexit.unregister(self._atexit_cleanup)
        if self._backend is not None:
            self._backend.cleanup()
        self._sessions.clear()
        logger.debug("sandbox_manager_cleaned_up")

    def _atexit_cleanup(self) -> None:
        """Safety net for process exit."""
        try:
            self.cleanup()
        except Exception:
            pass

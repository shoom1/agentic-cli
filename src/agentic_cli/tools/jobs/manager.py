"""JobManager — lifecycle, persistence, and concurrency for long-running jobs.

`JobManager` is **internal infrastructure, never an LLM-facing tool** (it is to
long-running tools what `SandboxManager` is to `sandbox_execute`). Typed
long-running tools call it from their bodies; the LLM only ever sees the tool.

Responsibilities: own `JobRecord`s, persist them under a base dir so jobs survive
turns and CLI restarts, reconcile state on read via the execution backends,
enforce a concurrency cap with a queue, and expose a uniform query/manage API.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from agentic_cli.file_utils import atomic_write_json
from agentic_cli.logging import Loggers
from agentic_cli.tools.jobs.backends import (
    TERMINAL_STATES,
    JobBackend,
    JobState,
    default_backends,
)

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

logger = Loggers.workflow()

# Persisted values of the terminal states, for comparing against raw JSON.
_TERMINAL_VALUES = frozenset(state.value for state in TERMINAL_STATES)


def _now() -> float:
    return time.time()


def _owner_token() -> str:
    """Identity of the process holding a resume claim (``<host>:<pid>``)."""
    return f"{socket.gethostname()}:{os.getpid()}"


def _claim_owner_is_live(owner: str | None) -> bool:
    """Whether the process that holds a claim is still running.

    Accepts both the two-part resume token (``<host>:<pid>``) and the
    three-part execution token (``<host>:<pid>:<instance>``).

    Startup recovery must fail only claims whose owner is *gone*: another CLI
    process delivering a result right now would otherwise have its claim
    yanked, and would then write ``DELIVERED`` over a record this process had
    already marked failed.

    Unknown is treated as "live" wherever we genuinely cannot tell (a claim
    from another host), because the failure mode of leaving a stale claim is a
    result you can still read with ``/jobs <id>``, while the failure mode of
    recovering a live one is a duplicated turn. A claim with no owner at all
    predates this bookkeeping and cannot belong to a running process.
    """
    if not owner:
        return False
    parts = owner.split(":")
    if len(parts) < 2:
        return False
    host, pid_text = parts[0], parts[1]
    if host and host != socket.gethostname():
        return True  # another machine — not ours to judge
    try:
        pid = int(pid_text)
    except ValueError:
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # exists but not signalable (different user) — still alive
    return True


class DurableRead(str, Enum):
    """Outcome of reading a job's persisted metadata.

    ``MISSING`` and ``UNREADABLE`` are deliberately separate: a record another
    manager cleaned away must be forgotten, while one that cannot be parsed
    must be left untouched (acting on it, or rewriting it, would destroy state
    we failed to read).
    """

    PRESENT = "present"
    MISSING = "missing"
    UNREADABLE = "unreadable"


class ClaimLockUnavailable(RuntimeError):
    """The cross-process lock over the jobs directory could not be taken.

    Raised rather than swallowed: without the lock, two processes can both
    decide a job is theirs to launch or deliver, and a write can lose what
    another process has just recorded. So without it no record is written: a
    launch or a delivery is not claimed, startup recovery does nothing, a
    cancel or a refresh leaves the record as it is (a read answers with the
    record as this manager has it), and recording a completed delivery
    records nothing, so the claim stays ``RESUMING`` until startup recovery
    fails it once its process is gone. Only a new job's first write goes
    ahead: there is nothing on disk yet for it to lose.
    """


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Exclusive lock over ``path``, held across processes.

    Every CLI of the app shares the jobs directory, so every read-modify-write
    of a job record happens inside this (``JobManager._mutate``).

    Raises:
        ClaimLockUnavailable: If no real lock could be taken — the lock file
            could not be created, or the platform offers no lock primitive.
            Silently continuing would leave the caller believing the section
            was serialized when it was not.
    """
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as exc:
        raise ClaimLockUnavailable(f"cannot open {path}: {exc}") from exc
    try:
        try:
            _acquire_lock(fd)
        except OSError as exc:  # any OS-level refusal is "no lock"
            raise ClaimLockUnavailable(f"cannot lock {path}: {exc}") from exc
        try:
            yield
        finally:
            _release_lock(fd)
    finally:
        os.close(fd)


def _acquire_lock(fd: int) -> None:
    """Take an exclusive lock on ``fd``, or raise ``ClaimLockUnavailable``."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX
        _acquire_lock_windows(fd)
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError as exc:
        raise ClaimLockUnavailable(f"flock failed: {exc}") from exc


def _acquire_lock_windows(fd: int) -> None:  # pragma: no cover - non-POSIX
    try:
        import msvcrt

        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
    except Exception as exc:  # noqa: BLE001
        raise ClaimLockUnavailable(f"no lock primitive: {exc}") from exc


def _release_lock(fd: int) -> None:
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
    except ImportError:  # pragma: no cover - non-POSIX
        try:
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except Exception:  # noqa: BLE001
            pass
    except OSError as exc:  # noqa: BLE001 - unlock failure is not actionable
        logger.debug("job_claim_unlock_failed", error=str(exc))


class ResumeStateError(RuntimeError):
    """An invalid resume-lifecycle transition was requested.

    Signals a coordinator bug (completing a delivery nobody claimed), not a
    user-facing condition — the valid "not mine to deliver" answer is
    ``begin_resume()`` returning False.
    """


class ResumeState(str, Enum):
    """Delivery lifecycle of a finished job's result back into the agent turn.

    ``PENDING`` → ``RESUMING`` → ``DELIVERED`` | ``FAILED``.

    The middle state exists so a crash can be told apart from a completed
    delivery: marking a job delivered *before* running the turn loses the
    result when the turn never ran, and marking it *after* re-delivers it if the
    process dies mid-turn. A record found in ``RESUMING`` at startup is
    therefore recovered as ``FAILED`` (its result is still readable via
    ``/jobs <id>``) rather than replayed, because the interrupted turn may
    already have executed tools.
    """

    PENDING = "pending"
    RESUMING = "resuming"
    DELIVERED = "delivered"
    FAILED = "failed"


@dataclass
class JobRecord:
    """One job's metadata. Persisted as ``<base_dir>/<job_id>/meta.json``.

    ``spec`` may hold live (non-serializable) objects in memory — e.g. an
    in-process callable; only JSON-safe entries are written to disk.
    """

    job_id: str
    tool: str
    backend: str
    name: str
    state: JobState
    spec: dict = field(default_factory=dict)
    backend_handle: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    tags: list[str] = field(default_factory=list)
    submitted_at: float = field(default_factory=_now)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    # --- Resume association (phase 2 push/resume; unset = fire-and-forget) ---
    session_id: str | None = None      # conversation that launched the job
    user_id: str | None = None
    resume_on_complete: bool = False   # wake the agent with the result when terminal
    call_id: str | None = None         # ADK function_call_id / LangChain tool_call_id
    call_name: str | None = None       # function name to answer on resume
    resume_state: str = ResumeState.PENDING.value   # see ResumeState
    resume_error: str | None = None    # why delivery failed, when it did
    resume_owner: str | None = None    # "<host>:<pid>" holding a RESUMING claim
    # "<host>:<pid>" that launched the job and owns its execution. Set when the
    # launch is claimed (at submit, for a job only its submitter can launch:
    # ``JobBackend.survives_restart`` False), so a second CLI sharing the jobs
    # directory neither relaunches a queued job nor writes off a running one
    # whose handle lives in the owner's memory.
    exec_owner: str | None = None
    # Set when a cancel could not stop the job (a running in-process thread, a
    # process that outlived SIGTERM): it stays RUNNING, holding its slot, and
    # is recorded CANCELLED when it ends.
    cancel_requested: bool = False

    @property
    def resumed(self) -> bool:
        """True once delivery reached a terminal state (kept for compatibility)."""
        return self.resume_state in (
            ResumeState.DELIVERED.value,
            ResumeState.FAILED.value,
        )

    def elapsed_s(self) -> float:
        start = self.started_at or self.submitted_at
        end = self.finished_at or _now()
        return round(max(0.0, end - start), 1)

    def to_dict(self) -> dict:
        """The record as ``meta.json`` holds it; it shares nothing with ``self``.

        Built field by field: ``dataclasses.asdict`` deep-copied the whole
        record first, ``spec`` included, so a live object JSON cannot hold
        (a bound method of a service holding a lock) failed the copy before
        ``_json_safe_spec`` could replace it.
        """
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["state"] = self.state.value
        d["spec"] = _json_safe_spec(self.spec)
        d["tags"] = list(self.tags)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "JobRecord":
        d = dict(d)
        d["state"] = JobState(d["state"])
        # Records written before the resume lifecycle carried a bool ``resumed``.
        legacy_resumed = d.pop("resumed", None)
        if "resume_state" not in d and legacy_resumed is not None:
            d["resume_state"] = (
                ResumeState.DELIVERED.value if legacy_resumed else ResumeState.PENDING.value
            )
        declared = cls.__dataclass_fields__  # type: ignore[attr-defined]
        # Only pass keys the record actually declares, so a missing optional
        # key falls back to its default instead of becoming None.
        return cls(**{k: v for k, v in d.items() if k in declared})

    def summary(self) -> dict:
        """Compact, JSON-safe view for tools / UI."""
        out = {
            "job_id": self.job_id,
            "tool": self.tool,
            "name": self.name,
            "backend": self.backend,
            "state": self.state.value,
            "elapsed_s": self.elapsed_s(),
            "exit_code": self.exit_code,
            "tags": self.tags,
        }
        if self.cancel_requested:
            out["cancel_requested"] = True
        if self.resume_on_complete:
            out["resume_state"] = self.resume_state
            if self.resume_error:
                out["resume_error"] = self.resume_error
        return out


def _json_safe_spec(spec: dict) -> dict:
    """Keep only JSON-serializable spec entries (drop live callables, etc.).

    A kept entry is a copy, so the result shares nothing with ``spec``; one
    JSON cannot hold becomes a placeholder and is never copied.
    """
    import copy
    import json

    safe: dict = {}
    for k, v in spec.items():
        try:
            json.dumps(v)
        except (TypeError, ValueError):
            safe[k] = f"<non-serializable: {type(v).__name__}>"
        else:
            safe[k] = copy.deepcopy(v)
    return safe


class _FieldRule(Enum):
    """How a manager reads a ``JobRecord`` field back from ``meta.json``.

    A manager's record is a copy; the file is shared by every manager over
    the jobs directory (another CLI of the app, or the manager an
    orchestrator switch builds while the old one is open). Before it decides
    anything, a manager reads each field back by its rule, so what another
    manager wrote is what it decides on, and what it writes.
    """

    # Set at submit and never changed: never read back.
    IMMUTABLE = "immutable"
    # Read back. A terminal state on disk is never replaced: how a job ended,
    # once recorded, is how it ended.
    STATE = "state"
    # What the launcher and the pollers record about the run: read back.
    EXECUTOR = "executor"
    # Which manager launches and runs the job: read back.
    OWNER = "owner"
    # The delivery claim: read back as one, from the legacy ``resumed`` in a
    # record written before the claim existed.
    CLAIM = "claim"
    # Once true on either side, true: a cancel requested by any manager.
    STICKY = "sticky"


# Every ``JobRecord`` field, with its rule. A field added to the record fails
# the test suite until it is placed here.
_FIELD_RULES: dict[str, _FieldRule] = {
    "job_id": _FieldRule.IMMUTABLE,
    "tool": _FieldRule.IMMUTABLE,
    "backend": _FieldRule.IMMUTABLE,
    "name": _FieldRule.IMMUTABLE,
    "state": _FieldRule.STATE,
    "spec": _FieldRule.IMMUTABLE,
    "backend_handle": _FieldRule.EXECUTOR,
    "pid": _FieldRule.EXECUTOR,
    "exit_code": _FieldRule.EXECUTOR,
    "tags": _FieldRule.IMMUTABLE,
    "submitted_at": _FieldRule.IMMUTABLE,
    "started_at": _FieldRule.EXECUTOR,
    "finished_at": _FieldRule.EXECUTOR,
    "error": _FieldRule.EXECUTOR,
    "session_id": _FieldRule.IMMUTABLE,
    "user_id": _FieldRule.IMMUTABLE,
    "resume_on_complete": _FieldRule.IMMUTABLE,
    "call_id": _FieldRule.IMMUTABLE,
    "call_name": _FieldRule.IMMUTABLE,
    "resume_state": _FieldRule.CLAIM,
    "resume_error": _FieldRule.CLAIM,
    "resume_owner": _FieldRule.CLAIM,
    "exec_owner": _FieldRule.OWNER,
    "cancel_requested": _FieldRule.STICKY,
}

# The fields a manager may change after submit: a transaction writes the
# record when its block changed one of them.
_CHANGING_FIELDS = tuple(
    name for name, rule in _FIELD_RULES.items() if rule is not _FieldRule.IMMUTABLE
)


def _claim_on_disk(data: dict) -> str | None:
    """The delivery state ``data`` records, or None when it records none."""
    state = data.get("resume_state")
    if state is None and data.get("resumed") is not None:
        # A record written before the lifecycle existed.
        state = (
            ResumeState.DELIVERED.value
            if data["resumed"]
            else ResumeState.PENDING.value
        )
    return state


def _reload_fields(rec: JobRecord, data: dict) -> None:
    """Read back into ``rec`` what ``data``, its ``meta.json``, records.

    Field by field, by ``_FIELD_RULES``. A key ``data`` does not have leaves
    the field as it is: the record was written by a version without it.
    """
    claim = _claim_on_disk(data)
    for name, rule in _FIELD_RULES.items():
        if rule is _FieldRule.IMMUTABLE:
            continue
        if rule is _FieldRule.STATE:
            try:
                setattr(rec, name, JobState(data[name]))
            except (KeyError, ValueError):  # pragma: no cover - none, or unknown
                pass
        elif rule is _FieldRule.STICKY:
            if data.get(name):
                setattr(rec, name, True)
        elif rule is _FieldRule.CLAIM:
            # The claim moves as one: its owner and error go with its state.
            if claim is not None:
                setattr(rec, name, claim if name == "resume_state" else data.get(name))
        elif name in data:  # EXECUTOR, OWNER
            setattr(rec, name, data[name])


class JobManager:
    """Lifecycle + persistence + concurrency for long-running jobs."""

    def __init__(
        self,
        settings: "BaseSettings | None" = None,
        *,
        base_dir: Path,
        max_concurrent: int = 4,
        backends: dict[str, JobBackend] | None = None,
    ) -> None:
        self._settings = settings
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._max_concurrent = max(1, int(max_concurrent))
        self._backends = backends or default_backends()
        self._lock = threading.RLock()
        # Depth of the cross-process claim transaction this manager holds.
        # POSIX ``flock`` is per file-description, so a second ``open`` in the
        # same process would block against our own lock — the transaction is
        # made re-entrant here instead (always entered under ``_lock``).
        self._claim_depth = 0
        # Execution ownership is per *manager*, not per process: the backend
        # handle for a running job lives in this instance, so a sibling manager
        # in the same process is as unable to poll it as another CLI would be.
        self._instance_id = uuid.uuid4().hex[:8]
        self._records: dict[str, JobRecord] = {}
        # Jobs whose unregistered backend has been logged at warning, which
        # happens once per job, when a caller asks for it (``_backend_of``).
        self._backendless_warned: set[str] = set()
        self._load_existing()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_backend(self, backend: JobBackend) -> None:
        """Add a backend under its ``name``, replacing one of the same name.

        An app registers its own after construction, when the constructor
        has already reconciled the jobs on disk: until then, a job on that
        backend was left as it is (``_backend_of``), and from now on this
        manager polls, launches and cancels it.
        """
        self._backends[backend.name] = backend

    def has_backend(self, name: str) -> bool:
        """Whether a backend of this ``name`` is registered with this manager.

        A job on a backend that is not is left as it is here: not polled,
        launched or cancelled. ``job_cancel`` and ``/jobs cancel`` ask, to
        say so.
        """
        return name in self._backends

    def submit(
        self,
        *,
        tool: str,
        backend: str,
        spec: dict,
        name: str | None = None,
        tags: list[str] | None = None,
        resume_on_complete: bool = False,
        call_id: str | None = None,
        call_name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> JobRecord:
        """Create a job; start it now if under the cap, else queue it.

        When ``resume_on_complete`` is set the job is flagged so the harness can
        auto-resume the agent with the result once it finishes (phase 2). The
        originating ``session_id``/``user_id`` are auto-filled from the active
        workflow turn when not supplied. ``call_id``/``call_name`` identify the
        tool call that started the job (ADK answers it on resume); without a
        ``call_id`` they come from the active tool call, which Deep Agents sets
        around each call, so a tool that reads its id from ADK's
        ``tool_context`` (None there) is linked all the same. A ``call_id`` or
        ``call_name`` the tool passes wins.

        The backend pins the spec now (``JobBackend.pin``): a queued job may
        be launched by another CLI, so a subprocess job's working directory
        is the one it was submitted in, whoever launches it. A job on a
        backend that does not survive a restart (``survives_restart`` False,
        the in-process one: its callable is not saved) is claimed for this
        manager now instead, so no other manager launches it; ``close()``
        records it FAILED if it has not started by then.
        """
        if backend not in self._backends:
            raise ValueError(
                f"unknown job backend {backend!r}; have {sorted(self._backends)}"
            )
        if resume_on_complete and (session_id is None or user_id is None):
            session_id, user_id = self._fill_turn_context(session_id, user_id)
        if resume_on_complete and call_id is None:
            call_id, call_name = self._fill_tool_call(call_name)
        runs_on = self._backends[backend]
        spec = runs_on.pin(dict(spec))
        with self._lock:
            job_id = uuid.uuid4().hex[:12]
            rec = JobRecord(
                job_id=job_id,
                tool=tool,
                backend=backend,
                name=name or tool,
                state=JobState.QUEUED,
                spec=spec,
                tags=list(tags or []),
                resume_on_complete=resume_on_complete,
                call_id=call_id,
                call_name=call_name or (tool if resume_on_complete else None),
                session_id=session_id,
                user_id=user_id,
                exec_owner=(
                    None if runs_on.survives_restart else self._exec_owner_token()
                ),
            )
            # Kept only once it is on disk: a record kept but never written
            # was launched by the next submit, after the tool was told this
            # one had failed.
            job_dir = self._job_dir(job_id)
            try:
                job_dir.mkdir(parents=True, exist_ok=True)
                self._create(rec)
            except BaseException:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise
            self._records[job_id] = rec
            self._maybe_start_queued()
            return rec

    @staticmethod
    def _fill_turn_context(
        session_id: str | None, user_id: str | None
    ) -> tuple[str | None, str | None]:
        """Best-effort fill session/user from the active workflow turn.

        Reads the ``WORKFLOW`` service from the registry (live during a tool
        call). Missing context is non-fatal: the job still runs, but it won't be
        resumable, so we warn.
        """
        from agentic_cli.workflow.service_registry import WORKFLOW, get_service

        wf = get_service(WORKFLOW)
        if wf is not None:
            session_id = session_id or getattr(wf, "active_session_id", None)
            user_id = user_id or getattr(wf, "active_user_id", None)
        if session_id is None or user_id is None:
            logger.warning(
                "job_resume_context_missing",
                have_session=session_id is not None,
                have_user=user_id is not None,
            )
        return session_id, user_id

    @staticmethod
    def _fill_tool_call(call_name: str | None) -> tuple[str | None, str | None]:
        """The active tool call's id, and its name unless the tool gave one.

        Outside a tool call (ADK, LangGraph, a job started by the harness)
        there is none, and both are left as they are.
        """
        from agentic_cli.workflow.sessions import get_active_tool_call

        active = get_active_tool_call()
        if active is None:
            return None, call_name
        return active.call_id, call_name or active.name

    def get(self, job_id: str) -> JobRecord | None:
        """The job, refreshed from durable state, or None if it is gone."""
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return None
            try:
                with self._claim_transaction():
                    if self._refresh(rec) is DurableRead.MISSING:
                        return None
                    self._maybe_start_queued()
            except ClaimLockUnavailable as exc:
                logger.warning(
                    "job_refresh_skipped_unlocked", job_id=job_id, error=str(exc)
                )
            return rec

    def list(
        self,
        *,
        state: JobState | None = None,
        tag: str | None = None,
        active_only: bool = False,
    ) -> list[JobRecord]:
        with self._lock:
            self.reconcile()
            recs = list(self._records.values())
        if active_only:
            recs = [r for r in recs if r.state in (JobState.QUEUED, JobState.RUNNING)]
        if state is not None:
            recs = [r for r in recs if r.state == state]
        if tag is not None:
            recs = [r for r in recs if tag in r.tags]
        return sorted(recs, key=lambda r: r.submitted_at, reverse=True)

    def known(self) -> list[JobRecord]:
        """The jobs this manager holds, as it last read them: no refresh, no lock.

        It never waits, neither for another CLI holding the jobs directory's
        lock nor for a thread of this manager's in a transaction or a cancel,
        so a caller on the event loop can ask (``workflow/interrupts.py``,
        answering a stopped turn's calls). For what a job never changes after
        submit (its call, its conversation, its name); its state here may be
        behind, where ``list`` refreshes it first.
        """
        return list(self._records.copy().values())

    def tail(self, job_id: str, n: int = 50, stream: str = "stdout") -> list[str]:
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return []
            backend = self._backend_of(rec, asked=True)
            if backend is None:
                return []
            return backend.logs(rec, self._job_dir(job_id), n, stream)

    def result(self, job_id: str) -> Any:
        """The job's result once it has ended, else None.

        The job is brought up to date first, as ``get`` does: its record
        read back, its backend polled and how it ended decided in one
        cross-process transaction, so a cancel another manager requested
        meanwhile is not decided past. Without the lock it is judged as this
        manager has it.
        """
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return None
            try:
                if self._refresh(rec) is DurableRead.MISSING:
                    return None
            except ClaimLockUnavailable as exc:
                logger.warning(
                    "job_refresh_skipped_unlocked", job_id=job_id, error=str(exc)
                )
            if rec.state not in TERMINAL_STATES:
                return None
            backend = self._backend_of(rec, asked=True)
            if backend is None:
                return None
            return backend.result(rec, self._job_dir(job_id))

    def cancel(self, job_id: str) -> JobRecord | None:
        """Cancel a job, if this manager can actually cancel it.

        Brings the record up to date first, as every read does (the durable
        record, then the backend's poll, in the transaction that records the
        cancel: ``_mutate``), and returns it as it is
        when that shows the job has ended: a job that has already finished
        stays finished, also when nobody had polled it since (terminal
        transitions are monotonic — neither a stale view nor a cancel that
        lands after the end may rewrite a success as CANCELLED). A job
        another live manager is executing is only cancelled when the backend
        can reach it from here (``cancels_foreign_jobs``); otherwise the
        record is returned unchanged rather than reported cancelled while it
        keeps running, as is a job whose backend this manager has not
        registered (``_backend_of``). A job the backend could not stop
        (``JobBackend.cancel`` returned False) stays RUNNING with
        ``cancel_requested``, holding its slot, and is recorded CANCELLED when
        it ends; the next queued job starts then, not now.

        Returns:
            The record — check ``state`` for what actually happened — or None
            if the job is unknown.
        """
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return None
            try:
                with self._claim_transaction():
                    with self._mutate(rec) as status:
                        if status is DurableRead.MISSING:
                            return None
                        if status is DurableRead.UNREADABLE:
                            logger.warning("job_cancel_unreadable", job_id=job_id)
                            return rec
                        self._poll(rec)
                        if rec.state not in TERMINAL_STATES:
                            backend = self._backend_of(rec, asked=True)
                            if backend is None:
                                return rec
                            foreign = self._execution_is_foreign(rec)
                            if foreign and not self._can_cancel_foreign(rec):
                                logger.warning(
                                    "job_cancel_not_reachable",
                                    job_id=job_id,
                                    owner=rec.exec_owner,
                                )
                                return rec
                            if backend.cancel(rec, self._job_dir(job_id)) is False:
                                rec.cancel_requested = True  # not stopped yet
                                return rec
                            rec.state = JobState.CANCELLED
                            rec.finished_at = _now()
                    self._maybe_start_queued()  # it has ended: its slot is free
                    return rec
            except ClaimLockUnavailable as exc:
                logger.warning(
                    "job_cancel_skipped_unlocked", job_id=job_id, error=str(exc)
                )
                return rec

    def reconcile(self) -> None:
        """Refresh non-terminal jobs from durable state and their backends.

        Runs inside the cross-process transaction: reloading a record, polling
        for it and writing the result is a read-modify-write of state other
        managers share. Skipped entirely if the lock is unavailable — stale
        reads are safe, unsynchronised writes are not.
        """
        with self._lock:
            try:
                with self._claim_transaction():
                    for rec in list(self._records.values()):
                        self._refresh(rec)  # forgets one another manager removed
                    self._maybe_start_queued()
            except ClaimLockUnavailable as exc:
                logger.warning("job_reconcile_skipped_unlocked", error=str(exc))

    def clean(self) -> int:
        """Remove terminal jobs (records + dirs). Returns the count removed.

        Each candidate is re-read under the cross-process lock first: deletion
        is irreversible, and a stale terminal snapshot would take a job another
        manager is still running with it.
        """
        with self._lock:
            self.reconcile()
            try:
                with self._claim_transaction():
                    removed: list[JobRecord] = []
                    for rec in list(self._records.values()):
                        with self._mutate(rec) as status:
                            pass  # read back; nothing changes, so nothing is written
                        if status is not DurableRead.PRESENT:
                            continue  # forgotten, or unreadable: never delete that
                        if rec.state not in TERMINAL_STATES:
                            continue
                        if rec.resume_state == ResumeState.RESUMING.value and (
                            _claim_owner_is_live(rec.resume_owner)
                        ):
                            # A delivery is in flight; removing the record now
                            # would strip the result out from under it.
                            logger.debug(
                                "job_clean_skipped_delivering", job_id=rec.job_id
                            )
                            continue
                        removed.append(rec)
                        self._records.pop(rec.job_id, None)
                        shutil.rmtree(self._job_dir(rec.job_id), ignore_errors=True)
                    return len(removed)
            except ClaimLockUnavailable as exc:
                logger.warning("job_clean_skipped_unlocked", error=str(exc))
                return 0

    def running_count(self) -> int:
        """Running jobs this manager knows, which count toward its cap.

        A job whose backend this manager has not registered is left out: it
        is never judged here, so it never ends here, and counted it would hold
        a slot for good once the app no longer registers its backend.
        """
        with self._lock:
            return sum(
                1
                for r in self._records.values()
                if r.state == JobState.RUNNING and r.backend in self._backends
            )

    def awaiting_resume(self) -> list[JobRecord]:
        """Terminal jobs flagged for resume whose result is still undelivered.

        Only ``PENDING`` records are returned: one already ``RESUMING`` is being
        delivered right now, and ``DELIVERED``/``FAILED`` are terminal.
        Reconciles first so freshly-finished jobs are included; sorted
        oldest-finished-first so the coordinator drains in completion order.
        """
        with self._lock:
            self.reconcile()
            recs = [
                r
                for r in self._records.values()
                if r.resume_on_complete
                and r.resume_state == ResumeState.PENDING.value
                and r.state in TERMINAL_STATES
            ]
        return sorted(recs, key=lambda r: r.finished_at or r.submitted_at)

    def begin_resume(self, job_id: str) -> bool:
        """Claim a job for delivery: ``PENDING`` → ``RESUMING``.

        The claim is atomic and durable **across processes**: two CLIs sharing
        one jobs directory (the default — it is user-scoped, not project-scoped)
        would otherwise both see ``PENDING`` in their own memory and deliver the
        same result into two conversations. The transition therefore happens
        under an inter-process lock, re-reading the record from disk so the
        decision is made on the shared state rather than this manager's cache.

        Only a job that is genuinely deliverable can be claimed: it must exist,
        be terminal, be flagged ``resume_on_complete``, and still be
        ``PENDING``.

        If the lock cannot be taken the claim **fails closed** (returns False):
        an undelivered result is still readable with ``/jobs <id>``, whereas an
        unsynchronised claim can deliver the same result into two conversations.

        Returns:
            True if this caller now owns delivery; False otherwise. A False
            result means "not yours to deliver" — never an error.
        """
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return False
            if not rec.resume_on_complete:
                return False
            try:
                # Terminality is judged on the durable record, not on this
                # manager's snapshot: a job that finished elsewhere is
                # deliverable even if we still have it as RUNNING.
                with self._mutate(rec) as status:
                    if status is DurableRead.MISSING:
                        return False
                    if status is DurableRead.UNREADABLE:
                        logger.warning("job_resume_claim_unreadable", job_id=job_id)
                        return False
                    if rec.state not in TERMINAL_STATES:
                        return False
                    if rec.resume_state != ResumeState.PENDING.value:
                        return False
                    rec.resume_state = ResumeState.RESUMING.value
                    rec.resume_owner = _owner_token()
                    rec.resume_error = None
            except ClaimLockUnavailable as exc:
                logger.warning(
                    "job_resume_claim_unlocked", job_id=job_id, error=str(exc)
                )
                return False
            return True

    def complete_resume(
        self, job_id: str, *, delivered: bool, error: str | None = None
    ) -> None:
        """Close out a claimed delivery: ``RESUMING`` → ``DELIVERED``/``FAILED``.

        Only the record *this process claimed* transitions. Anything else is a
        coordinator bug — closing out a job that was never claimed, or one
        another process is delivering, would rewrite state someone else owns —
        so it raises rather than silently overwriting.

        Args:
            job_id: The claimed job.
            delivered: True only when the resume turn actually ran to completion.
            error: Why delivery failed, recorded for ``/jobs``.

        If the lock cannot be taken nothing changes, on disk or in memory:
        the record cannot be written, so it does not pretend to have moved.
        It stays ``RESUMING``, owned by this process, as the disk has it.
        That is the failed-safe outcome: an unlocked rewrite could regress a
        delivery another process had just recorded, whereas a record stuck in
        ``RESUMING`` is recovered as failed once this process is gone, and is
        never re-delivered.

        Raises:
            ResumeStateError: If the job is unknown, was not claimed, or is
                claimed by someone else.
        """
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                raise ResumeStateError(f"Unknown job {job_id!r}: nothing to complete.")
            try:
                with self._mutate(rec) as status:
                    if status is not DurableRead.PRESENT:
                        # The record was cleaned away (or cannot be read) while
                        # the turn ran. Recording the outcome would recreate a
                        # deleted job; there is nothing left to deliver to.
                        logger.warning(
                            "job_resume_complete_record_gone",
                            job_id=job_id,
                            status=status.value,
                        )
                        return
                    if rec.resume_state != ResumeState.RESUMING.value:
                        raise ResumeStateError(
                            f"Job {job_id!r} is {rec.resume_state!r}, not "
                            f"{ResumeState.RESUMING.value!r}: complete_resume() "
                            "must follow a successful begin_resume()."
                        )
                    owner = _owner_token()
                    if rec.resume_owner not in (None, owner):
                        raise ResumeStateError(
                            f"Job {job_id!r} is being delivered by "
                            f"{rec.resume_owner!r}, not {owner!r}: only the "
                            "claiming process may complete it."
                        )
                    rec.resume_state = (
                        ResumeState.DELIVERED.value
                        if delivered
                        else ResumeState.FAILED.value
                    )
                    rec.resume_error = None if delivered else error
                    rec.resume_owner = None
            except ClaimLockUnavailable as exc:
                logger.warning(
                    "job_resume_complete_unpersisted",
                    job_id=job_id,
                    delivered=delivered,
                    error=str(exc),
                )

    def _recover_interrupted_resumes(self) -> None:
        """Fail records left mid-delivery by a *dead* owner (called on load).

        The interrupted turn may already have executed tools, so it is never
        replayed automatically; the result stays readable via ``/jobs <id>``.

        A claim whose owner is still running belongs to another live CLI
        delivering it right now (the jobs directory is shared per user) and is
        left alone — taking it over would duplicate the turn and race that
        process's ``complete_resume``.

        The decision is made on state re-read **inside** the lock: the records
        were loaded from disk before it was taken, and in that window another
        process may have claimed a job or finished delivering one. Acting on
        the pre-lock snapshot rewrote a completed delivery as failed. A record
        that window deleted is forgotten, and one it left unreadable is left
        as it is (``_mutate``): no verdict is written over either.

        If the lock is unavailable nothing is recovered — leaving a stale
        ``RESUMING`` record costs a result you can still read with
        ``/jobs <id>``, while a wrong recovery corrupts another process's
        state.
        """
        try:
            with self._claim_transaction():
                for rec in list(self._records.values()):
                    with self._mutate(rec) as status:
                        if status is DurableRead.UNREADABLE:
                            logger.warning(
                                "job_recovery_skipped_unreadable", job_id=rec.job_id
                            )
                        if status is not DurableRead.PRESENT:
                            continue
                        if rec.resume_state != ResumeState.RESUMING.value:
                            continue
                        if _claim_owner_is_live(rec.resume_owner):
                            logger.debug(
                                "job_resume_claim_active",
                                job_id=rec.job_id,
                                owner=rec.resume_owner,
                            )
                            continue
                        rec.resume_state = ResumeState.FAILED.value
                        rec.resume_error = (
                            "resume interrupted before the turn completed"
                        )
                        rec.resume_owner = None
                    logger.warning("job_resume_interrupted", job_id=rec.job_id)
        except ClaimLockUnavailable as exc:
            logger.warning("job_resume_recovery_skipped", error=str(exc))

    def close(self) -> None:
        """Release resources owned by the manager's backends.

        Idempotent. Jobs themselves are not cancelled: a detached subprocess is
        meant to outlive the CLI, and its state is recovered from disk on the
        next start. Only in-process resources (thread pools) are released.

        A job this manager claimed at submit (its backend does not survive a
        restart) and has not started yet never will: no other manager can
        launch it. It is recorded FAILED, saying so, and one submitted with
        ``resume_on_complete`` is then delivered as any failed job is. See
        :meth:`_fail_unstarted_claims`.
        """
        self._fail_unstarted_claims()
        for backend in self._backends.values():
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001 - shutdown must not fail
                logger.warning(
                    "job_backend_close_failed", backend=backend.name, error=str(exc)
                )

    def _fail_unstarted_claims(self) -> None:
        """Record FAILED the jobs this manager claimed but has not started.

        Only a job claimed at submit is still QUEUED under this manager's
        token: a launch claims and starts a job in one transaction. They are
        found in memory first, and the cross-process lock is taken only if
        there are any, so a close with nothing to fail neither waits behind
        another CLI nor warns of a lock it did not need. Each is judged on
        its record reloaded under the lock. Without the lock nothing is
        written: the job stays QUEUED under a claim that no other manager
        takes, and startup recovery fails it once this process is gone.
        """
        with self._lock:
            mine = self._exec_owner_token()
            claimed = [
                rec
                for rec in self._records.values()
                if rec.state == JobState.QUEUED and rec.exec_owner == mine
            ]
            if not claimed:
                return
            try:
                with self._claim_transaction():
                    for rec in claimed:
                        with self._mutate(rec) as status:
                            if status is not DurableRead.PRESENT:
                                continue
                            if rec.state != JobState.QUEUED or rec.exec_owner != mine:
                                continue
                            rec.state = JobState.FAILED
                            rec.error = (
                                "the app closed before the job started; a job on "
                                "this backend runs only in the app that submitted it"
                            )
                            rec.finished_at = _now()
                        logger.warning("job_closed_before_start", job_id=rec.job_id)
            except ClaimLockUnavailable as exc:
                logger.warning("job_close_skipped_unlocked", error=str(exc))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _job_dir(self, job_id: str) -> Path:
        return self.base_dir / job_id

    @property
    def _claim_lock_path(self) -> Path:
        """One lock for the whole jobs directory, held by every record's write."""
        return self.base_dir / ".resume.lock"

    @contextlib.contextmanager
    def _claim_transaction(self) -> Iterator[None]:
        """Serialize a read-modify-write of job records, cross-process.

        Re-entrant within this manager (see ``_claim_depth``) so a path that
        holds it (a reconcile, a cancel) can write a record (``_mutate``)
        without deadlocking against its own ``flock``.

        Raises:
            ClaimLockUnavailable: propagated from :func:`_file_lock`.
        """
        with self._lock:
            if self._claim_depth:
                self._claim_depth += 1
                try:
                    yield
                finally:
                    self._claim_depth -= 1
                return
            with _file_lock(self._claim_lock_path):
                self._claim_depth = 1
                try:
                    yield
                finally:
                    self._claim_depth = 0

    @contextlib.contextmanager
    def _mutate(
        self, rec: JobRecord, *, create: bool = False
    ) -> Iterator[DurableRead]:
        """One read-modify-write of a job's ``meta.json``.

        Inside the cross-process transaction (``_claim_transaction``), the
        record is read from disk and each field read back into ``rec`` by its
        rule (``_FIELD_RULES``): the block decides on what every manager has
        written, and changes ``rec``. When the block ends and has changed a
        field, ``rec`` is written onto the dict that was read, so a key this
        version does not declare (a later version's) is kept. A block that
        raises writes nothing.

        Yields what the read found, for the block to act on: ``PRESENT``;
        ``MISSING``, when another manager removed the record, which is then
        forgotten and never written back (unless ``create``: a new job's
        first write, of the whole record); ``UNREADABLE``, when nothing is
        read back or written, since writing would destroy state we failed to
        read.

        A terminal state on disk is never replaced: how a job ended, once
        recorded, is how it ended. A block that changed it is undone whole
        (every field as it was before the block, nothing written), and logged:
        what else it changed was decided on a view the disk contradicts.

        Raises:
            ClaimLockUnavailable: from ``_claim_transaction``; nothing was
                read or written.
        """
        with self._claim_transaction():
            status, data = self._read_persisted(rec.job_id)
            if status is DurableRead.PRESENT:
                _reload_fields(rec, data)
            elif status is DurableRead.MISSING and not create:
                self._forget(rec.job_id)
            before = [getattr(rec, name) for name in _CHANGING_FIELDS]
            yield status
            if status is DurableRead.MISSING and create:
                self._write(rec, {})
            if status is not DurableRead.PRESENT:
                return
            if [getattr(rec, name) for name in _CHANGING_FIELDS] == before:
                return
            durable = data.get("state")
            if durable in _TERMINAL_VALUES and durable != rec.state.value:
                for name, value in zip(_CHANGING_FIELDS, before):
                    setattr(rec, name, value)
                logger.info(
                    "job_terminal_state_preserved", job_id=rec.job_id, state=durable
                )
                return
            self._write(rec, data)

    def _create(self, rec: JobRecord) -> None:
        """Write a new job's first ``meta.json``, through ``_mutate``.

        Without the lock it is written all the same: a record that is not on
        disk yet holds nothing another manager wrote, for this write to lose.
        One that is there after all is left as it is.
        """
        try:
            with self._mutate(rec, create=True):
                pass
        except ClaimLockUnavailable as exc:
            if (self._job_dir(rec.job_id) / "meta.json").exists():
                logger.warning(
                    "job_persist_skipped_unlocked", job_id=rec.job_id, error=str(exc)
                )
                return
            logger.warning(
                "job_persist_created_unlocked", job_id=rec.job_id, error=str(exc)
            )
            self._write(rec, {})

    def _read_persisted(self, job_id: str) -> "tuple[DurableRead, dict | None]":
        """Read a record from disk, distinguishing *gone* from *unreadable*.

        The two demand opposite responses: a record another manager cleaned
        away must be forgotten (recreating it would resurrect a deleted job),
        while one we simply cannot parse must be left exactly as it is.
        """
        import json

        meta = self._job_dir(job_id) / "meta.json"
        try:
            text = meta.read_text()
        except FileNotFoundError:
            return DurableRead.MISSING, None
        except OSError as exc:
            logger.warning("job_meta_unreadable", job_id=job_id, error=str(exc))
            return DurableRead.UNREADABLE, None
        try:
            data = json.loads(text)
        except ValueError as exc:
            logger.warning("job_meta_corrupt", job_id=job_id, error=str(exc))
            return DurableRead.UNREADABLE, None
        if not isinstance(data, dict):
            logger.warning("job_meta_corrupt", job_id=job_id, error="not an object")
            return DurableRead.UNREADABLE, None
        return DurableRead.PRESENT, data

    def _forget(self, job_id: str) -> None:
        """Drop a record another manager deleted. Never recreate it."""
        if self._records.pop(job_id, None) is not None:
            logger.info("job_record_deleted_elsewhere", job_id=job_id)

    def _write(self, rec: JobRecord, read: dict) -> None:
        """Write ``rec`` onto ``read``, what its ``meta.json`` held.

        A key this version does not declare is kept: a later version's, which
        it would otherwise erase. The legacy ``resumed`` is not: the
        ``resume_state`` written with it replaces it, and every version since
        the delivery lifecycle reads that first.
        """
        data = {**read, **rec.to_dict()}
        data.pop("resumed", None)
        atomic_write_json(self._job_dir(rec.job_id) / "meta.json", data)

    def _backend_of(
        self, rec: JobRecord, *, asked: bool = False
    ) -> JobBackend | None:
        """The job's backend, or None when this manager has not registered it.

        Such a job is left exactly as it is: neither polled, judged, launched
        (a queued one is not claimed) nor cancelled here, and nothing is
        written. Without the backend this manager cannot tell a running job
        from an ended one, and an app registers its own backends after the
        constructor has reconciled (``register_backend``): writing a verdict
        then ended a live job, and handed it to a resume as finished.

        Logged at debug on the passes every start and orchestrator switch
        makes (refresh, scheduling), which meet such a job before the app has
        registered its backend; at warning, once per job, when a caller
        ``asked`` for it by name (cancel, tail, result).
        """
        backend = self._backends.get(rec.backend)
        if backend is None:
            warn = asked and rec.job_id not in self._backendless_warned
            if warn:
                self._backendless_warned.add(rec.job_id)
            log = logger.warning if warn else logger.debug
            log("job_backend_not_registered", job_id=rec.job_id, backend=rec.backend)
        return backend

    def _can_cancel_foreign(self, rec: JobRecord) -> bool:
        """Whether this manager could actually cancel a job it did not start.

        A declared backend capability (``cancels_foreign_jobs``), not an
        inference: publishing a readable outcome and being remotely
        controllable are different properties, and conflating them would mark a
        record CANCELLED while the job kept running in the owning process.
        """
        backend = self._backends.get(rec.backend)
        return bool(backend is not None and backend.cancels_foreign_jobs)

    def _exec_owner_token(self) -> str:
        """This manager's execution identity (``<host>:<pid>:<instance>``)."""
        return f"{_owner_token()}:{self._instance_id}"

    def _execution_is_foreign(self, rec: JobRecord) -> bool:
        """Whether a live manager *other than this one* owns the execution."""
        owner = rec.exec_owner
        return (
            bool(owner)
            and owner != self._exec_owner_token()
            and _claim_owner_is_live(owner)
        )

    def _refresh(self, rec: JobRecord) -> DurableRead:
        """Bring a job up to date: durable state first, then the backend.

        One transaction (``_mutate``): the record read back, the backend
        polled and what it says recorded, so the decision is made on what
        every manager has written. The durable record comes first for *every*
        record, terminal ones included — a finished job's delivery lifecycle
        keeps moving, and that is state another manager owns.

        Returns:
            The durable read status. A record another manager deleted
            (``MISSING``) has been forgotten.

        Raises:
            ClaimLockUnavailable: nothing was read or written.
        """
        with self._mutate(rec) as status:
            if status is DurableRead.PRESENT:
                self._poll(rec)
        return status

    def _poll(self, rec: JobRecord) -> None:
        """Ask the backend how a job is doing, and record what it says.

        Caller holds ``_mutate`` on the record, which writes the change.

        A foreign job is still polled: backends publish their outcome durably
        (the ``exit_code`` sentinel), so the observer can read it. What it must
        not do is believe ``UNKNOWN`` — that answer means "I hold no handle for
        this", which is true of every job another manager started, and taking
        it at face value would mark a healthy job terminal and hand its
        "result" to a resume. Nor does a manager without the job's backend
        judge it: the job is left as it is (``_backend_of``).
        """
        if rec.state in TERMINAL_STATES or rec.state == JobState.QUEUED:
            return
        backend = self._backend_of(rec)
        if backend is None:
            return
        new_state = backend.poll(rec, self._job_dir(rec.job_id))
        if new_state == rec.state:
            return
        if new_state is JobState.UNKNOWN and self._execution_is_foreign(rec):
            logger.debug(
                "job_poll_unknown_foreign",
                job_id=rec.job_id,
                owner=rec.exec_owner,
            )
            return
        if new_state in TERMINAL_STATES and rec.cancel_requested:
            new_state = JobState.CANCELLED  # it ended after it was cancelled
        rec.state = new_state
        if new_state in TERMINAL_STATES and rec.finished_at is None:
            rec.finished_at = _now()

    def _maybe_start_queued(self) -> None:
        """Start queued jobs up to the concurrency cap (caller holds the lock)."""
        if self.running_count() >= self._max_concurrent:
            return
        queued = sorted(
            (r for r in self._records.values() if r.state == JobState.QUEUED),
            key=lambda r: r.submitted_at,
        )
        for rec in queued:
            if self.running_count() >= self._max_concurrent:
                break
            self._start(rec)

    def _start(self, rec: JobRecord) -> None:
        """Launch a queued job — at most once across every process.

        The jobs directory is user-scoped, so two CLIs can hold the same queued
        record. Execution is therefore claimed the same way delivery is: under
        the cross-process lock, against the record *on disk*, recording the
        owner durably **before** the backend is touched. If the lock is
        unavailable the launch is skipped — a late job beats two of them, and
        so is a job whose backend this manager has not registered: it is not
        claimed, and stays queued for a manager that has it.
        """
        backend = self._backend_of(rec)
        if backend is None:
            return
        try:
            with self._claim_transaction():
                with self._mutate(rec) as status:
                    claimed = self._claim_execution(rec, status)
                if not claimed:
                    return
                # The claim is on disk; start the job only as claimed.
                claim = (JobState.QUEUED, self._exec_owner_token())
                with self._mutate(rec) as status:
                    as_claimed = (rec.state, rec.exec_owner) == claim
                    if status is DurableRead.PRESENT and as_claimed:
                        self._launch(rec, backend)
        except ClaimLockUnavailable as exc:
            logger.warning(
                "job_launch_skipped_unlocked", job_id=rec.job_id, error=str(exc)
            )

    def _claim_execution(self, rec: JobRecord, status: DurableRead) -> bool:
        """Take ownership of a queued job's launch.

        Decided on ``rec`` as ``_mutate`` read it back (``status``: what the
        read found); the caller holds that transaction, which writes the
        claim before the backend is touched.
        """
        if status is DurableRead.UNREADABLE:
            logger.warning("job_launch_skipped_unreadable", job_id=rec.job_id)
            return False
        if status is not DurableRead.PRESENT:
            return False  # another manager removed it
        if rec.state != JobState.QUEUED:
            return False
        mine = self._exec_owner_token()
        if rec.exec_owner == mine:
            return True  # claimed at submit: only this manager can run it
        if rec.exec_owner is not None:
            # Already claimed: either it is running elsewhere, it waits for
            # the manager that submitted it, or its launcher died and startup
            # recovery will fail it. Never launch it twice.
            logger.debug(
                "job_launch_owned_elsewhere", job_id=rec.job_id, owner=rec.exec_owner
            )
            return False
        rec.exec_owner = mine
        return True

    def _launch(self, rec: JobRecord, backend: JobBackend) -> None:
        """Start the backend for a claimed job.

        The caller holds ``_mutate`` on the record, which writes the outcome.
        """
        try:
            backend.start(rec, self._job_dir(rec.job_id))
            rec.state = JobState.RUNNING
            rec.started_at = _now()
        except Exception as exc:  # noqa: BLE001 - surface launch failures as failed jobs
            rec.state = JobState.FAILED
            rec.error = f"launch failed: {exc}"
            rec.finished_at = _now()
            logger.warning("job_launch_failed", job_id=rec.job_id, error=str(exc))

    def _recover_interrupted_launches(self) -> None:
        """Fail queued jobs whose launcher died mid-claim (called on load).

        The claim is written before the backend is touched, so a record left
        QUEUED with a dead owner may or may not have started something. It is
        never relaunched — a duplicate side effect is worse than a job that
        must be resubmitted — and is failed with that stated plainly.

        Only records the durable read still vouches for are judged: one
        another manager deleted is forgotten, and one that cannot be read is
        left as it is (``_mutate``).
        """
        try:
            with self._claim_transaction():
                for rec in list(self._records.values()):
                    with self._mutate(rec) as status:
                        if status is DurableRead.UNREADABLE:
                            logger.warning(
                                "job_recovery_skipped_unreadable", job_id=rec.job_id
                            )
                        if status is not DurableRead.PRESENT:
                            continue
                        if rec.state != JobState.QUEUED or rec.exec_owner is None:
                            continue
                        if _claim_owner_is_live(rec.exec_owner):
                            continue
                        rec.state = JobState.FAILED
                        rec.error = "launch interrupted before the job started"
                        rec.finished_at = _now()
                    logger.warning("job_launch_interrupted", job_id=rec.job_id)
        except ClaimLockUnavailable as exc:
            logger.warning("job_launch_recovery_skipped", error=str(exc))

    def _load_existing(self) -> None:
        """Load persisted job records on startup and reconcile their state."""
        import json

        if not self.base_dir.exists():
            return
        with self._lock:
            for meta in self.base_dir.glob("*/meta.json"):
                try:
                    rec = JobRecord.from_dict(json.loads(meta.read_text()))
                except (ValueError, OSError, TypeError, KeyError):
                    continue
                # In-memory handles (Popen / Future) are gone after a restart,
                # so non-restart-safe running jobs become UNKNOWN — unless a
                # live process still owns their execution (see _poll).
                self._records[rec.job_id] = rec
            # Recover before reconciling: reconcile() starts queued jobs, and
            # an interrupted launch must never be one of them.
            self._recover_interrupted_launches()
            self._recover_interrupted_resumes()
            self.reconcile()

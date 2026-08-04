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
import socket
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
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
    """The cross-process resume lock could not be established.

    Raised rather than swallowed: without the lock, two processes can both
    decide a job is theirs to deliver. Callers that could *cause* a duplicate
    delivery (claiming, startup recovery) must treat this as "not mine" and do
    nothing; callers that only *prevent* one (recording a completed delivery)
    proceed best-effort.
    """


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Exclusive lock over ``path``, held across processes.

    Resume claims are the one piece of job state two CLI processes contend
    for, so every read-modify-write of the resume fields happens inside this.

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
    call_id: str | None = None         # ADK function_call_id / LangGraph tool_call_id
    call_name: str | None = None       # function name to answer on resume
    resume_state: str = ResumeState.PENDING.value   # see ResumeState
    resume_error: str | None = None    # why delivery failed, when it did
    resume_owner: str | None = None    # "<host>:<pid>" holding a RESUMING claim
    # "<host>:<pid>" that launched the job and owns its execution. Set when the
    # launch is claimed, so a second CLI sharing the jobs directory neither
    # relaunches a queued job nor writes off a running one whose handle lives
    # in the owner's memory.
    exec_owner: str | None = None

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
        d = asdict(self)
        d["state"] = self.state.value
        d["spec"] = _json_safe_spec(self.spec)
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
        fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
        # Only pass keys the record actually declares, so a missing optional
        # key falls back to its default instead of becoming None.
        return cls(**{k: v for k, v in d.items() if k in fields})

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
        if self.resume_on_complete:
            out["resume_state"] = self.resume_state
            if self.resume_error:
                out["resume_error"] = self.resume_error
        return out


def _json_safe_spec(spec: dict) -> dict:
    """Keep only JSON-serializable spec entries (drop live callables, etc.)."""
    import json

    safe: dict = {}
    for k, v in spec.items():
        try:
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError):
            safe[k] = f"<non-serializable: {type(v).__name__}>"
    return safe


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
        # Job ids this manager has seen on disk. A record that was persisted
        # and is now missing was *deleted* by another manager; one that was
        # never persisted is simply new. Writing must tell them apart.
        self._persisted: set[str] = set()
        self._load_existing()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_backend(self, backend: JobBackend) -> None:
        self._backends[backend.name] = backend

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
        workflow turn when not supplied; ``call_id``/``call_name`` (which only
        the calling tool knows) identify the pending tool call to answer.
        """
        if backend not in self._backends:
            raise ValueError(
                f"unknown job backend {backend!r}; have {sorted(self._backends)}"
            )
        if resume_on_complete and (session_id is None or user_id is None):
            session_id, user_id = self._fill_turn_context(session_id, user_id)
        with self._lock:
            job_id = uuid.uuid4().hex[:12]
            rec = JobRecord(
                job_id=job_id,
                tool=tool,
                backend=backend,
                name=name or tool,
                state=JobState.QUEUED,
                spec=dict(spec),
                tags=list(tags or []),
                resume_on_complete=resume_on_complete,
                call_id=call_id,
                call_name=call_name or (tool if resume_on_complete else None),
                session_id=session_id,
                user_id=user_id,
            )
            self._records[job_id] = rec
            self._job_dir(job_id).mkdir(parents=True, exist_ok=True)
            self._persist(rec)
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

    def get(self, job_id: str) -> JobRecord | None:
        """The job, refreshed from durable state, or None if it is gone."""
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return None
            try:
                with self._claim_transaction():
                    if self._refresh(rec) is DurableRead.MISSING:
                        self._forget(job_id)
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

    def tail(self, job_id: str, n: int = 50, stream: str = "stdout") -> list[str]:
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return []
            return self._backends[rec.backend].logs(rec, self._job_dir(job_id), n, stream)

    def result(self, job_id: str) -> Any:
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return None
            self._refresh(rec)
            if rec.state not in TERMINAL_STATES:
                return None
            return self._backends[rec.backend].result(rec, self._job_dir(job_id))

    def cancel(self, job_id: str) -> JobRecord | None:
        """Cancel a job, if this manager can actually cancel it.

        Reloads the durable record first: a job that has already finished stays
        finished (terminal transitions are monotonic — a stale view must not
        rewrite a recorded success as CANCELLED). A job another live manager is
        executing is only cancelled when the backend can reach it from here
        (``cancels_foreign_jobs``); otherwise the record is returned unchanged
        rather than reported cancelled while it keeps running.

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
                    status = self._reload_durable(rec)
                    if status is DurableRead.MISSING:
                        self._forget(job_id)
                        return None
                    if status is DurableRead.UNREADABLE:
                        logger.warning("job_cancel_unreadable", job_id=job_id)
                        return rec
                    if rec.state in TERMINAL_STATES:
                        return rec
                    if self._execution_is_foreign(rec) and not self._can_cancel_foreign(
                        rec
                    ):
                        logger.warning(
                            "job_cancel_not_reachable",
                            job_id=job_id,
                            owner=rec.exec_owner,
                        )
                        return rec
                    self._backends[rec.backend].cancel(rec, self._job_dir(job_id))
                    rec.state = JobState.CANCELLED
                    rec.finished_at = _now()
                    self._persist(rec, owns_shared_state=True)
                    self._maybe_start_queued()
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
                    gone = [
                        rec.job_id
                        for rec in list(self._records.values())
                        if self._refresh(rec) is DurableRead.MISSING
                    ]
                    for job_id in gone:
                        self._forget(job_id)
                    self._maybe_start_queued()
            except ClaimLockUnavailable as exc:
                logger.warning("job_reconcile_skipped_unlocked", error=str(exc))

    def clean(self) -> int:
        """Remove terminal jobs (records + dirs). Returns the count removed.

        Each candidate is re-read under the cross-process lock first: deletion
        is irreversible, and a stale terminal snapshot would take a job another
        manager is still running with it.
        """
        import shutil

        with self._lock:
            self.reconcile()
            try:
                with self._claim_transaction():
                    removed: list[JobRecord] = []
                    for rec in list(self._records.values()):
                        status = self._reload_durable(rec)
                        if status is DurableRead.MISSING:
                            self._forget(rec.job_id)
                            continue
                        if status is DurableRead.UNREADABLE:
                            continue  # never delete state we could not read
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
                        self._persisted.discard(rec.job_id)
                        shutil.rmtree(self._job_dir(rec.job_id), ignore_errors=True)
                    return len(removed)
            except ClaimLockUnavailable as exc:
                logger.warning("job_clean_skipped_unlocked", error=str(exc))
                return 0

    def running_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._records.values() if r.state == JobState.RUNNING)

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
                with self._claim_transaction():
                    # Terminality is judged on the durable record, not on this
                    # manager's snapshot: a job that finished elsewhere is
                    # deliverable even if we still have it as RUNNING.
                    status = self._reload_durable(rec)
                    if status is DurableRead.MISSING:
                        self._forget(job_id)
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
                    self._persist(rec, owns_shared_state=True)
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

        If the lock cannot be taken the durable record is **left as it is** —
        still ``RESUMING``, owned by this process. That is the failed-safe
        outcome: an unlocked rewrite could regress a delivery another process
        had just recorded, whereas a record stuck in ``RESUMING`` is recovered
        as failed once this process is gone, and is never re-delivered. The
        in-memory record still transitions, so this manager's own ``/jobs``
        view is accurate.

        Raises:
            ResumeStateError: If the job is unknown, was not claimed, or is
                claimed by someone else.
        """
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                raise ResumeStateError(f"Unknown job {job_id!r}: nothing to complete.")
            try:
                with self._claim_transaction():
                    self._complete_resume_locked(rec, delivered, error)
            except ClaimLockUnavailable as exc:
                logger.warning(
                    "job_resume_complete_unpersisted",
                    job_id=job_id,
                    delivered=delivered,
                    error=str(exc),
                )
                rec.resume_state = (
                    ResumeState.DELIVERED.value
                    if delivered
                    else ResumeState.FAILED.value
                )
                rec.resume_error = None if delivered else error
                rec.resume_owner = None

    def _complete_resume_locked(
        self, rec: JobRecord, delivered: bool, error: str | None
    ) -> None:
        """Body of :meth:`complete_resume`; the caller holds the transaction."""
        status = self._reload_durable(rec)
        if status is not DurableRead.PRESENT:
            # The record was cleaned away (or cannot be read) while the turn
            # ran. Recording the outcome would recreate a deleted job; there is
            # nothing left to deliver to.
            logger.warning(
                "job_resume_complete_record_gone",
                job_id=rec.job_id,
                status=status.value,
            )
            if status is DurableRead.MISSING:
                self._forget(rec.job_id)
            return
        if rec.resume_state != ResumeState.RESUMING.value:
            raise ResumeStateError(
                f"Job {rec.job_id!r} is {rec.resume_state!r}, not "
                f"{ResumeState.RESUMING.value!r}: complete_resume() must "
                "follow a successful begin_resume()."
            )
        owner = _owner_token()
        if rec.resume_owner not in (None, owner):
            raise ResumeStateError(
                f"Job {rec.job_id!r} is being delivered by {rec.resume_owner!r}, "
                f"not {owner!r}: only the claiming process may complete it."
            )
        rec.resume_state = (
            ResumeState.DELIVERED.value if delivered else ResumeState.FAILED.value
        )
        rec.resume_error = None if delivered else error
        rec.resume_owner = None
        self._persist(rec, owns_shared_state=True)

    def mark_resumed(self, job_id: str) -> None:
        """Mark a job's result delivered (durably) so it is never resumed twice.

        Deprecated compatibility shim for the pre-lifecycle API: it claims and
        completes in one step. Prefer ``begin_resume()``/``complete_resume()``,
        which survives a crash between claiming and delivering. A job that
        cannot be claimed is left alone.
        """
        if self.begin_resume(job_id):
            self.complete_resume(job_id, delivered=True)

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
        that window deleted, or left unreadable, is skipped outright — see
        :meth:`_recovery_target`.

        If the lock is unavailable nothing is recovered — leaving a stale
        ``RESUMING`` record costs a result you can still read with
        ``/jobs <id>``, while a wrong recovery corrupts another process's
        state.
        """
        try:
            with self._claim_transaction():
                for rec in list(self._records.values()):
                    if not self._recovery_target(rec):
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
                    rec.resume_error = "resume interrupted before the turn completed"
                    rec.resume_owner = None
                    self._persist(rec, owns_shared_state=True)
                    logger.warning("job_resume_interrupted", job_id=rec.job_id)
        except ClaimLockUnavailable as exc:
            logger.warning("job_resume_recovery_skipped", error=str(exc))

    def close(self) -> None:
        """Release resources owned by the manager's backends.

        Idempotent. Jobs themselves are not cancelled: a detached subprocess is
        meant to outlive the CLI, and its state is recovered from disk on the
        next start. Only in-process resources (thread pools) are released.
        """
        for backend in self._backends.values():
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001 - shutdown must not fail
                logger.warning(
                    "job_backend_close_failed", backend=backend.name, error=str(exc)
                )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _job_dir(self, job_id: str) -> Path:
        return self.base_dir / job_id

    @property
    def _claim_lock_path(self) -> Path:
        """One lock for the whole jobs directory; claims are rare and brief."""
        return self.base_dir / ".resume.lock"

    @contextlib.contextmanager
    def _claim_transaction(self) -> Iterator[None]:
        """Serialize a read-modify-write of the resume fields, cross-process.

        Re-entrant within this manager (see ``_claim_depth``) so a claim path
        can persist without deadlocking against its own ``flock``.

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

    def _persist(self, rec: JobRecord, *, owns_shared_state: bool = False) -> None:
        """Write a record's metadata, never clobbering a newer delivery state.

        ``meta.json`` holds both this process's job bookkeeping *and* the
        cross-process resume fields. A plain state write (a poll that finds the
        job finished, a cancel, a launch) carries whatever resume fields this
        manager last read, which may be older than what another process has
        since written — persisting them verbatim erased that process's claim,
        and the job could then be claimed and delivered twice.

        So unless the caller owns the resume fields (the claim transitions,
        which set them under the lock), they are re-read from disk first and
        adopted into ``rec``.

        Args:
            rec: The record to write.
            owns_shared_state: True only for a caller holding the claim
                transaction that just set these fields.
        """
        if owns_shared_state:
            self._write(rec)
            return
        try:
            with self._claim_transaction():
                status, data = self._read_persisted(rec.job_id)
                if status is DurableRead.UNREADABLE:
                    logger.warning("job_persist_skipped_unreadable", job_id=rec.job_id)
                    return
                if status is DurableRead.MISSING and rec.job_id in self._persisted:
                    # Another manager cleaned this job away; writing it back
                    # would resurrect a deleted record.
                    self._forget(rec.job_id)
                    return
                if data is not None:
                    durable = data.get("state")
                    if durable in _TERMINAL_VALUES and durable != rec.state.value:
                        # Terminal transitions are monotonic: some manager has
                        # already recorded how this job ended, and a snapshot
                        # taken before that cannot say otherwise. Adopt it.
                        self._reload_durable(rec)
                        logger.info(
                            "job_terminal_state_preserved",
                            job_id=rec.job_id,
                            state=durable,
                        )
                        return
                    self._adopt_shared_fields(rec, data)
                self._write(rec)
        except ClaimLockUnavailable as exc:
            # No lock means no safe read-modify-write: re-reading and rewriting
            # is exactly the race the lock prevents, so an interleaved delivery
            # would be regressed. Skip the write entirely — the in-memory
            # change is retried by the next reconcile — unless there is nothing
            # on disk yet to clobber, in which case creating the record is safe.
            meta = self._job_dir(rec.job_id) / "meta.json"
            if meta.exists() or rec.job_id in self._persisted:
                logger.warning(
                    "job_persist_skipped_unlocked", job_id=rec.job_id, error=str(exc)
                )
                return
            logger.warning(
                "job_persist_created_unlocked", job_id=rec.job_id, error=str(exc)
            )
            self._write(rec)

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
        self._persisted.discard(job_id)
        if self._records.pop(job_id, None) is not None:
            logger.info("job_record_deleted_elsewhere", job_id=job_id)

    # The lifecycle a record's *executor* owns and persists. Every manager
    # reads these back before acting: an in-memory copy is only ever a snapshot
    # of what some manager last wrote.
    _DURABLE_LIFECYCLE_FIELDS = (
        "exit_code",
        "error",
        "started_at",
        "finished_at",
        "pid",
        "backend_handle",
    )

    def _adopt_shared_fields(self, rec: JobRecord, data: dict) -> None:
        """Adopt the ownership/delivery fields from a persisted snapshot."""
        if "exec_owner" in data:
            rec.exec_owner = data["exec_owner"]
        state = data.get("resume_state")
        if state is None:
            # A record written before the lifecycle existed.
            legacy = data.get("resumed")
            if legacy is None:
                return
            state = (
                ResumeState.DELIVERED.value if legacy else ResumeState.PENDING.value
            )
        rec.resume_state = state
        rec.resume_error = data.get("resume_error")
        rec.resume_owner = data.get("resume_owner")

    def _write(self, rec: JobRecord) -> None:
        """Write the record and remember that it now exists on disk."""
        atomic_write_json(self._job_dir(rec.job_id) / "meta.json", rec.to_dict())
        self._persisted.add(rec.job_id)

    def _reload_durable(self, rec: JobRecord) -> DurableRead:
        """Reload the whole persisted lifecycle into ``rec``.

        A manager's in-memory record is a snapshot; the file is the shared
        truth. Every mutator reloads before deciding, so a stale view cannot
        overwrite an outcome another manager already recorded, an observer
        learns that a foreign job finished, and a *terminal* record still picks
        up the delivery lifecycle that moves after it.

        Returns:
            What the read found. ``MISSING`` means another manager deleted the
            record and the caller must forget it rather than write it back;
            ``UNREADABLE`` means the caller must not act on it at all.
        """
        status, data = self._read_persisted(rec.job_id)
        if status is not DurableRead.PRESENT:
            return status
        raw_state = data.get("state")
        if raw_state is not None:
            try:
                rec.state = JobState(raw_state)
            except ValueError:  # pragma: no cover - unknown state on disk
                pass
        for name in self._DURABLE_LIFECYCLE_FIELDS:
            if name in data:
                setattr(rec, name, data[name])
        self._adopt_shared_fields(rec, data)
        return status

    def _recovery_target(self, rec: JobRecord) -> bool:
        """Whether startup recovery may judge — and rewrite — this record.

        The durable read is the *decision*, not a side effect. Recovery writes
        with ``owns_shared_state=True``, which by design does not re-check the
        file, so both non-PRESENT answers have to be handled here:

        - ``MISSING`` — another manager cleaned the job away between our load
          and now. Forget it; writing a verdict would resurrect a deleted job.
        - ``UNREADABLE`` — we have no basis for a verdict at all, and rewriting
          would destroy exactly the state we failed to read. Leave it alone.

        Returns:
            True only if the record was reloaded and may be acted on.
        """
        status = self._reload_durable(rec)
        if status is DurableRead.PRESENT:
            return True
        if status is DurableRead.MISSING:
            self._forget(rec.job_id)
        else:
            logger.warning("job_recovery_skipped_unreadable", job_id=rec.job_id)
        return False

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

        The durable record comes first for *every* record, terminal ones
        included — a finished job's delivery lifecycle keeps moving, and that
        is state another manager owns.

        A foreign job is still polled: backends publish their outcome durably
        (the ``exit_code`` sentinel), so the observer can read it. What it must
        not do is believe ``UNKNOWN`` — that answer means "I hold no handle for
        this", which is true of every job another manager started, and taking
        it at face value would mark a healthy job terminal and hand its
        "result" to a resume.

        Returns:
            The durable read status, so callers can forget a deleted record.
        """
        status = self._reload_durable(rec)
        if status is not DurableRead.PRESENT:
            return status
        if rec.state in TERMINAL_STATES or rec.state == JobState.QUEUED:
            return status
        backend = self._backends.get(rec.backend)
        if backend is None:
            rec.state = JobState.UNKNOWN
            self._persist(rec)
            return status
        new_state = backend.poll(rec, self._job_dir(rec.job_id))
        if new_state == rec.state:
            return status
        if new_state is JobState.UNKNOWN and self._execution_is_foreign(rec):
            logger.debug(
                "job_poll_unknown_foreign",
                job_id=rec.job_id,
                owner=rec.exec_owner,
            )
            return status
        rec.state = new_state
        if new_state in TERMINAL_STATES and rec.finished_at is None:
            rec.finished_at = _now()
        self._persist(rec)
        return status

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
        unavailable the launch is skipped — a late job beats two of them.
        """
        try:
            with self._claim_transaction():
                if not self._claim_execution(rec):
                    return
                self._launch(rec)
        except ClaimLockUnavailable as exc:
            logger.warning(
                "job_launch_skipped_unlocked", job_id=rec.job_id, error=str(exc)
            )

    def _claim_execution(self, rec: JobRecord) -> bool:
        """Take ownership of a queued job's launch. Caller holds the transaction."""
        status, data = self._read_persisted(rec.job_id)
        if status is DurableRead.UNREADABLE:
            logger.warning("job_launch_skipped_unreadable", job_id=rec.job_id)
            return False
        if status is DurableRead.MISSING and rec.job_id in self._persisted:
            self._forget(rec.job_id)
            return False
        if data is not None:
            try:
                rec.state = JobState(data.get("state", rec.state))
            except ValueError:  # pragma: no cover - unknown state on disk
                pass
            rec.exec_owner = data.get("exec_owner")
        if rec.state != JobState.QUEUED:
            return False
        if rec.exec_owner is not None:
            # Already claimed: either it is running elsewhere, or its launcher
            # died and startup recovery will fail it. Never launch it twice.
            logger.debug(
                "job_launch_owned_elsewhere", job_id=rec.job_id, owner=rec.exec_owner
            )
            return False
        rec.exec_owner = self._exec_owner_token()
        self._persist(rec, owns_shared_state=True)
        return True

    def _launch(self, rec: JobRecord) -> None:
        """Start the backend for a claimed job. Caller holds the transaction."""
        backend = self._backends[rec.backend]
        try:
            backend.start(rec, self._job_dir(rec.job_id))
            rec.state = JobState.RUNNING
            rec.started_at = _now()
        except Exception as exc:  # noqa: BLE001 - surface launch failures as failed jobs
            rec.state = JobState.FAILED
            rec.error = f"launch failed: {exc}"
            rec.finished_at = _now()
            logger.warning("job_launch_failed", job_id=rec.job_id, error=str(exc))
        self._persist(rec, owns_shared_state=True)

    def _recover_interrupted_launches(self) -> None:
        """Fail queued jobs whose launcher died mid-claim (called on load).

        The claim is written before the backend is touched, so a record left
        QUEUED with a dead owner may or may not have started something. It is
        never relaunched — a duplicate side effect is worse than a job that
        must be resubmitted — and is failed with that stated plainly.

        Only records the durable read still vouches for are judged; see
        :meth:`_recovery_target`.
        """
        try:
            with self._claim_transaction():
                for rec in list(self._records.values()):
                    if not self._recovery_target(rec):
                        continue
                    if rec.state != JobState.QUEUED or rec.exec_owner is None:
                        continue
                    if _claim_owner_is_live(rec.exec_owner):
                        continue
                    rec.state = JobState.FAILED
                    rec.error = "launch interrupted before the job started"
                    rec.finished_at = _now()
                    self._persist(rec, owns_shared_state=True)
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
                # live process still owns their execution (see _refresh).
                self._records[rec.job_id] = rec
                self._persisted.add(rec.job_id)
            # Recover before reconciling: reconcile() starts queued jobs, and
            # an interrupted launch must never be one of them.
            self._recover_interrupted_launches()
            self._recover_interrupted_resumes()
            self.reconcile()

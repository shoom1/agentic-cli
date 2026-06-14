"""JobManager — lifecycle, persistence, and concurrency for long-running jobs.

`JobManager` is **internal infrastructure, never an LLM-facing tool** (it is to
long-running tools what `SandboxManager` is to `sandbox_execute`). Typed
long-running tools call it from their bodies; the LLM only ever sees the tool.

Responsibilities: own `JobRecord`s, persist them under a base dir so jobs survive
turns and CLI restarts, reconcile state on read via the execution backends,
enforce a concurrency cap with a queue, and expose a uniform query/manage API.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def _now() -> float:
    return time.time()


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
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})  # type: ignore[attr-defined]

    def summary(self) -> dict:
        """Compact, JSON-safe view for tools / UI."""
        return {
            "job_id": self.job_id,
            "tool": self.tool,
            "name": self.name,
            "backend": self.backend,
            "state": self.state.value,
            "elapsed_s": self.elapsed_s(),
            "exit_code": self.exit_code,
            "tags": self.tags,
        }


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
        self._records: dict[str, JobRecord] = {}
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
    ) -> JobRecord:
        """Create a job; start it now if under the cap, else queue it."""
        if backend not in self._backends:
            raise ValueError(
                f"unknown job backend {backend!r}; have {sorted(self._backends)}"
            )
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
            )
            self._records[job_id] = rec
            self._job_dir(job_id).mkdir(parents=True, exist_ok=True)
            self._persist(rec)
            self._maybe_start_queued()
            return rec

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return None
            self._refresh(rec)
            self._maybe_start_queued()
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
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return None
            if rec.state in TERMINAL_STATES:
                return rec
            self._backends[rec.backend].cancel(rec, self._job_dir(job_id))
            rec.state = JobState.CANCELLED
            rec.finished_at = _now()
            self._persist(rec)
            self._maybe_start_queued()
            return rec

    def reconcile(self) -> None:
        """Refresh non-terminal jobs from their backends, then promote queued."""
        with self._lock:
            for rec in self._records.values():
                if rec.state not in TERMINAL_STATES:
                    self._refresh(rec)
            self._maybe_start_queued()

    def clean(self) -> int:
        """Remove terminal jobs (records + dirs). Returns the count removed."""
        import shutil

        with self._lock:
            self.reconcile()
            removed = [r for r in self._records.values() if r.state in TERMINAL_STATES]
            for rec in removed:
                self._records.pop(rec.job_id, None)
                shutil.rmtree(self._job_dir(rec.job_id), ignore_errors=True)
            return len(removed)

    def running_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._records.values() if r.state == JobState.RUNNING)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _job_dir(self, job_id: str) -> Path:
        return self.base_dir / job_id

    def _persist(self, rec: JobRecord) -> None:
        atomic_write_json(self._job_dir(rec.job_id) / "meta.json", rec.to_dict())

    def _refresh(self, rec: JobRecord) -> None:
        """Poll the backend for a non-terminal job and persist any change."""
        if rec.state in TERMINAL_STATES or rec.state == JobState.QUEUED:
            return
        backend = self._backends.get(rec.backend)
        if backend is None:
            rec.state = JobState.UNKNOWN
            self._persist(rec)
            return
        new_state = backend.poll(rec, self._job_dir(rec.job_id))
        if new_state != rec.state:
            rec.state = new_state
            if new_state in TERMINAL_STATES and rec.finished_at is None:
                rec.finished_at = _now()
            self._persist(rec)

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
        self._persist(rec)

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
                # so non-restart-safe running jobs become UNKNOWN.
                self._records[rec.job_id] = rec
            self.reconcile()

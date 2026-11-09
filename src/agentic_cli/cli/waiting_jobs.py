"""Which finished jobs wait to be told in the open conversation.

One rule, for the three places that say it: the line an opened conversation
says (``BaseCLIApp._name_its_waiting_jobs``), the status bar's ``↻N``
(``JobMonitor``), and the deliveries (``BaseCLIApp.resume_finished_jobs``:
at a turn's end, after ``/resume``, and by ``/jobs resume``).

A job waits for delivery when it has finished, was submitted with
``resume_on_complete``, and its delivery is still ``PENDING``
(``JobRecord.awaiting_resume``, what ``JobManager.awaiting_resume`` returns):
one being delivered (``RESUMING``) waits no more. Such a job is told in the
open conversation when it is that conversation's (its session, and its user
or none), when it has no conversation, or when its conversation is gone
(``can_resume`` says it cannot be resumed: deleted, or kept in memory by an
earlier run); the last two are told nowhere else, as unavailable. Another
conversation's job waits until that conversation is opened.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from agentic_cli.logging import Loggers

logger = Loggers.cli()

#: The open conversation: its user and its session.
Here = tuple[str | None, str | None]

#: ``workflow.can_resume``: whether a job's conversation can be resumed now.
CanResume = Callable[[Any], Awaitable[bool]]


def is_its_job(record: Any, here: Here | None) -> bool:
    """Whether ``record`` is the open conversation's job, or no conversation's.

    With no conversation given (``here`` None), every job is.
    """
    session_id = getattr(record, "session_id", None)
    if here is None or session_id is None:
        return True
    user, session = here
    return session_id == session and getattr(record, "user_id", None) in (None, user)


async def told_here(record: Any, here: Here | None, can_resume: CanResume | None) -> bool:
    """Whether ``record``, a job that waits for delivery, is told in the open conversation."""
    return is_its_job(record, here) or not await _can_be_opened(record, can_resume)


async def waiting_here(
    records: Iterable[Any], here: Here | None, can_resume: CanResume | None
) -> list[Any]:
    """The jobs among ``records`` (each waiting for delivery) told here, in their order."""
    return [record for record in records if await told_here(record, here, can_resume)]


async def _can_be_opened(record: Any, can_resume: CanResume | None) -> bool:
    """Whether ``record``'s conversation can still be opened, to be told there.

    Unknown counts as yes (a lookup that fails, or a workflow that cannot
    say): the job waits for its conversation rather than being told here as
    unavailable.
    """
    if can_resume is None:
        return True
    try:
        return bool(await can_resume(record))
    except Exception:  # noqa: BLE001 - unknown: it waits for its conversation
        logger.debug(
            "job_conversation_check_failed",
            job_id=getattr(record, "job_id", None), exc_info=True,
        )
        return True

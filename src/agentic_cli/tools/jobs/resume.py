"""What a finished job tells the agent that started it.

When a job submitted with ``resume_on_complete`` finishes, the backend resumes
the agent with this payload: a summary and a pointer, not the raw result, so
the resumed turn's context stays lean. The agent pulls the whole result or
the logs with ``job_result`` / ``job_logs`` by the job's id.

ADK delivers the payload as the pending call's ``FunctionResponse`` when the
root agent made the call. A call that is already answered cannot be answered
again (Anthropic and OpenAI accept one result per call): on Deep Agents every
job's call has its "pending" result, and on ADK so does a sub-agent's, whose
starter is an ordinary call (``workflow/adk/delegation.py``). Those jobs are
told to the root agent in a message of its own, ``job_notice``, which opens
with ``JOB_NOTICE_HEADER``.
"""

from __future__ import annotations

import json
from typing import Any

#: Opens a job notification, so the model can tell it from what the user
#: wrote.
JOB_NOTICE_HEADER = "[Automatic job notification — not written by the user]"

#: Longest result text the payload carries.
_RESULT_SUMMARY_LIMIT = 1000


def job_result_payload(record: Any, result: Any) -> dict:
    """The payload that reports ``record``'s outcome to the agent.

    Args:
        record: The finished ``JobRecord``.
        result: The job's result, or None when it has none.

    Returns:
        ``job_id``, ``tool`` and ``state``; ``exit_code`` and ``error`` when
        the job has them; ``result_summary`` (the result as text, cut to
        1,000 characters) when there is a result; and a ``hint`` naming the
        tools that read the rest.
    """
    payload: dict[str, Any] = {
        "job_id": record.job_id,
        "tool": record.tool,
        "state": record.state.value,
    }
    if record.exit_code is not None:
        payload["exit_code"] = record.exit_code
    if record.error:
        payload["error"] = record.error
    summary = _summarize_result(result)
    if summary is not None:
        payload["result_summary"] = summary
    payload["hint"] = (
        f"Full result via job_result('{record.job_id}'); "
        f"logs via job_logs('{record.job_id}')."
    )
    return payload


def job_notice(record: Any, result: Any) -> str:
    """The text that tells the root agent ``record``'s job finished.

    ``JOB_NOTICE_HEADER``, then ``job_result_payload`` as JSON: what both
    backends send when the job's call cannot take the payload as its result.

    Args:
        record: The finished ``JobRecord``.
        result: The job's result, or None.
    """
    payload = job_result_payload(record, result)
    return f"{JOB_NOTICE_HEADER}\n{json.dumps(payload, ensure_ascii=False)}"


def _summarize_result(result: Any) -> str | None:
    """Render a job result as a short string for the resume payload.

    Truncates to keep the resumed turn's context lean; the agent pulls the full
    result via the ``job_result`` tool when it needs it.
    """
    if result is None:
        return None
    text = result if isinstance(result, str) else repr(result)
    if len(text) > _RESULT_SUMMARY_LIMIT:
        return text[:_RESULT_SUMMARY_LIMIT] + f"… (+{len(text) - _RESULT_SUMMARY_LIMIT} chars)"
    return text

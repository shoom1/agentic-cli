"""What a finished job's notice asks of the root agent: who started the job.

A job a sub-agent started is told to the root (that run has ended), so the
notice names the sub-agent instead of "you" (spec 9).
"""

from __future__ import annotations

from types import SimpleNamespace

from agentic_cli.tools.jobs.backends import JobState
from agentic_cli.tools.jobs.resume import (
    JOB_NOTICE_ASK,
    JOB_NOTICE_HEADER,
    job_notice,
    job_notice_ask,
)


def _record(**fields) -> SimpleNamespace:
    return SimpleNamespace(
        job_id="j1", tool="run_shell_job", state=JobState.SUCCEEDED, exit_code=0, error=None,
        **fields,
    )


class TestTheAsk:
    def test_the_roots_own_job_is_one_it_started(self):
        assert job_notice_ask(_record(started_by=None)) == JOB_NOTICE_ASK == (
            "A background job you started has finished. Tell the user what it means."
        )

    def test_a_sub_agents_job_names_the_sub_agent(self):
        assert job_notice_ask(_record(started_by="researcher-2")) == (
            "A background job researcher-2 started has finished. Tell the user what it means."
        )

    def test_a_record_without_the_field_is_the_roots(self):
        assert job_notice_ask(_record()) == JOB_NOTICE_ASK

    def test_the_notice_asks_on_its_second_line(self):
        lines = job_notice(_record(started_by="researcher-2"), "built ok").splitlines()

        assert lines[0] == JOB_NOTICE_HEADER
        assert lines[1] == job_notice_ask(_record(started_by="researcher-2"))

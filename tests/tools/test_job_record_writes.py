"""Every write of a job record keeps what it did not change.

A job's ``meta.json`` is shared by every manager over the jobs directory:
another CLI of the app, or the manager an orchestrator switch builds while
the old one is open. Each field of a record has a declared rule
(``_FIELD_RULES``) saying how a manager reads it back before it writes.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from agentic_cli.tools.jobs import JobRecord


# A value of each field a manager may change after submit, as another
# manager would have written it.
_ELSEWHERE: dict[str, Any] = {
    "state": "cancelled",
    "backend_handle": "handle-elsewhere",
    "pid": 4242,
    "exit_code": 7,
    "started_at": 2.0,
    "finished_at": 3.0,
    "error": "written elsewhere",
    "exec_owner": "otherhost:1:abc",
    "resume_state": "failed",
    "resume_error": "written elsewhere",
    "resume_owner": "otherhost:9",
    "cancel_requested": True,
}


class TestTheFieldTable:
    def test_every_field_of_a_record_has_a_rule(self):
        """A field added to ``JobRecord`` fails here until it has one."""
        from agentic_cli.tools.jobs.manager import _FIELD_RULES

        assert set(_FIELD_RULES) == {f.name for f in dataclasses.fields(JobRecord)}

    def test_another_managers_values_cover_every_field_that_changes(self):
        from agentic_cli.tools.jobs.manager import _FIELD_RULES, _FieldRule

        assert set(_ELSEWHERE) == {
            name for name, rule in _FIELD_RULES.items() if rule is not _FieldRule.IMMUTABLE
        }

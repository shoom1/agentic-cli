"""A delegation's result envelope (``workflow/agents/envelope.py``).

A blocking delegation returns its run's envelope: whether it succeeded (only
a completed run did, partial or not), its instance and run, its status,
whether its answer is partial, its final text and a one-line error, and for
a cancelled run who cancelled it, and for a run that did not complete fully
the tasks it left open. ``read_envelope`` finds one in a
delegation's result as either backend's ``TOOL_RESULT`` carries it (a dict,
or the JSON text of one), and never raises.
"""

from __future__ import annotations

import json

import pytest

from agentic_cli.workflow.agents.envelope import (
    CANCELLED_BY,
    INTERRUPTED_ERROR,
    STEP_LIMIT_ERROR,
    TIME_LIMIT_ERROR,
    TOKEN_LIMIT_ERROR,
    envelope,
    open_task_descriptions,
    read_envelope,
    view_notice,
)
from agentic_cli.workflow.agents.runs import (
    RUN_CANCELLED,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_INTERRUPTED,
    RUN_REFUSED,
    RUN_STARTED,
)
from agentic_cli.workflow.sessions import RunRef, root_run

RESEARCHER = RunRef("researcher", "researcher-2", "researcher-2#1", None, 1)


class TestTheTexts:
    def test_the_errors(self):
        assert STEP_LIMIT_ERROR == "step limit reached"
        assert TIME_LIMIT_ERROR == "time limit reached"
        assert TOKEN_LIMIT_ERROR == "token limit reached"
        assert INTERRUPTED_ERROR == "interrupted before it finished"

    def test_who_cancels(self):
        assert CANCELLED_BY == frozenset({"parent", "user", "limit"})


class TestEnvelope:
    def test_a_completed_run(self):
        assert envelope(RESEARCHER, RUN_COMPLETED, result="Found it.") == {
            "success": True, "agent": "researcher-2", "run": "researcher-2#1",
            "status": "completed", "partial": False, "result": "Found it.", "error": None,
        }

    def test_the_keys_in_their_order(self):
        assert list(envelope(RESEARCHER, RUN_COMPLETED)) == [
            "success", "agent", "run", "status", "partial", "result", "error",
        ]

    def test_a_partial_answer_is_a_success(self):
        value = envelope(RESEARCHER, RUN_COMPLETED, result="So far…", partial=True,
                         error=STEP_LIMIT_ERROR)
        assert (value["success"], value["partial"], value["error"]) == (
            True, True, "step limit reached",
        )

    @pytest.mark.parametrize("status", [RUN_FAILED, RUN_REFUSED, RUN_INTERRUPTED])
    def test_any_other_end_is_no_success(self, status):
        value = envelope(RESEARCHER, status, result="half", error="why")
        assert (value["success"], value["status"], value["result"], value["error"]) == (
            False, status, "half", "why",
        )
        assert "cancelled_by" not in value

    @pytest.mark.parametrize("by", ["parent", "user", "limit"])
    def test_a_cancelled_run_says_who_cancelled_it(self, by):
        value = envelope(RESEARCHER, RUN_CANCELLED, result="so far", error=TIME_LIMIT_ERROR,
                         cancelled_by=by)
        assert value == {
            "success": False, "agent": "researcher-2", "run": "researcher-2#1",
            "status": "cancelled", "partial": False, "result": "so far",
            "error": "time limit reached", "cancelled_by": by,
        }

    def test_the_error_is_one_line(self):
        value = envelope(RESEARCHER, RUN_FAILED, error="400 Bad Request:\n  {\"error\":\n \"bad\"}\n")
        assert value["error"] == '400 Bad Request: {"error": "bad"}'

    def test_it_is_json(self):
        value = envelope(RESEARCHER, RUN_CANCELLED, cancelled_by="user", error="stopped")
        assert json.loads(json.dumps(value)) == value

    def test_the_root_has_none(self):
        with pytest.raises(ValueError, match="sub-agent run"):
            envelope(root_run("lead"), RUN_COMPLETED)

    @pytest.mark.parametrize("status", [RUN_STARTED, "running", "done"])
    def test_only_a_runs_end(self, status):
        with pytest.raises(ValueError, match=repr(status)):
            envelope(RESEARCHER, status)

    def test_a_cancelled_run_needs_who_cancelled_it(self):
        with pytest.raises(ValueError, match="cancelled_by"):
            envelope(RESEARCHER, RUN_CANCELLED)

    def test_who_cancelled_is_one_of_three(self):
        with pytest.raises(ValueError, match="'model'"):
            envelope(RESEARCHER, RUN_CANCELLED, cancelled_by="model")

    def test_only_a_cancelled_run_names_who_cancelled_it(self):
        with pytest.raises(ValueError, match="cancelled_by"):
            envelope(RESEARCHER, RUN_FAILED, cancelled_by="user")


class TestOpenTasks:
    """A run that did not complete fully names the tasks it left open, so its
    parent can continue it with a follow-up."""

    def test_they_come_last_when_given(self):
        value = envelope(RESEARCHER, RUN_COMPLETED, result="So far...", partial=True,
                         error=STEP_LIMIT_ERROR, open_tasks=["find Y", "find Z"])
        assert list(value)[-1] == "open_tasks"
        assert value["open_tasks"] == ["find Y", "find Z"]

    def test_after_who_cancelled_it(self):
        value = envelope(RESEARCHER, RUN_CANCELLED, error="stopped", cancelled_by="parent",
                         open_tasks=("find Y",))
        assert list(value)[-2:] == ["cancelled_by", "open_tasks"]
        assert value["open_tasks"] == ["find Y"]

    @pytest.mark.parametrize("given", [None, [], ()])
    def test_none_without_any(self, given):
        assert "open_tasks" not in envelope(
            RESEARCHER, RUN_FAILED, error="why", open_tasks=given
        )

    def test_the_descriptions_of_the_tasks_not_completed_in_their_order(self):
        tasks = [
            {"description": "find X", "status": "completed"},
            {"description": "find Y", "status": "in_progress"},
            {"description": "find Z"},  # the state tools' default: pending
            {"description": "skip W", "status": "cancelled"},
            {"status": "pending"},  # no description: nothing to name
            {"description": 7, "status": "pending"},
        ]
        assert open_task_descriptions(tasks) == ["find Y", "find Z", "skip W"]

    def test_none_for_none(self):
        assert open_task_descriptions([]) == []


class TestReadEnvelope:
    def test_a_dict(self):
        value = envelope(RESEARCHER, RUN_COMPLETED, result="Found it.")
        assert read_envelope(value) == value

    def test_its_json_text(self):
        value = envelope(RESEARCHER, RUN_CANCELLED, cancelled_by="parent", error="stopped")
        assert read_envelope(json.dumps(value)) == value
        assert read_envelope(f"  {json.dumps(value)}\n") == value

    def test_a_copy(self):
        value = envelope(RESEARCHER, RUN_COMPLETED)
        read = read_envelope(value)
        assert read == value and read is not value

    def test_a_cut_envelope_is_still_one(self):
        # The tool result cap keeps a dict's keys and adds its note.
        value = {**envelope(RESEARCHER, RUN_COMPLETED, result="…"), "truncated": "see file"}
        assert read_envelope(value) == value

    @pytest.mark.parametrize("value", [
        {"success": False, "error": "task needs agent_type: one of researcher."},
        {"result": "Found it."},
        "Found it.",
        "{not json",
        "[1, 2]",
        "null",
        "",
        None,
        42,
        ["success", "agent", "run", "status", "partial", "result", "error"],
        b'{"success": true}',
        "[" * 100_000,
    ])
    def test_anything_else_is_none(self, value):
        assert read_envelope(value) is None


class TestTheViewNotice:
    """What the root is told once the user has worked with one of its
    sub-agents directly: the instance and its latest envelope, as JSON."""

    def test_it_names_the_instance_and_carries_its_envelope_as_json(self):
        latest = envelope(RESEARCHER, RUN_COMPLETED, result="Y is 7; ünïcode kept.")

        notice = view_notice("researcher-2", latest)

        assert notice == (
            "The user worked with researcher-2 directly; its latest result: "
            '{"success": true, "agent": "researcher-2", "run": "researcher-2#1", '
            '"status": "completed", "partial": false, "result": "Y is 7; ünïcode kept.", '
            '"error": null}'
        )
        assert json.loads(notice.split("its latest result: ", 1)[1]) == latest

"""A delegation's brief and its task message (``workflow/agents/brief.py``).

A brief has a title and an objective, and may say what to deliver, the
constraints, what is already known and when the task is done. The task
message lays the filled fields out under one heading each, and an inheriting
type's ends with ``INHERITED_CONTEXT_NOTE``.
"""

from __future__ import annotations

import dataclasses

import pytest

from agentic_cli.workflow.agents.brief import (
    BRIEF_FIELDS,
    INHERITED_CONTEXT_NOTE,
    Brief,
    task_message,
)


class TestTheBrief:
    def test_the_fields_in_their_order(self):
        assert BRIEF_FIELDS == ("objective", "deliverable", "constraints", "known", "done_when")

    def test_only_the_title_and_the_objective_are_needed(self):
        brief = Brief(title="Find the paper", objective="Find it.")
        assert (brief.deliverable, brief.constraints, brief.known, brief.done_when) == (
            None, None, None, None,
        )

    def test_it_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            Brief(title="t", objective="o").title = "u"  # type: ignore[misc]

    def test_fields_are_the_filled_ones_title_first(self):
        brief = Brief(title="T", objective="O", known="K", deliverable="D")
        assert list(brief.fields().items()) == [
            ("title", "T"), ("objective", "O"), ("deliverable", "D"), ("known", "K"),
        ]

    def test_an_empty_field_is_not_filled(self):
        assert Brief(title="T", objective="O", constraints="").fields() == {
            "title": "T", "objective": "O",
        }


class TestTheTaskMessage:
    def test_a_title_and_an_objective(self):
        message = task_message(Brief(title="Find the paper", objective="Find the 2017 paper."))
        assert message == "# Task: Find the paper\n\n## Objective\nFind the 2017 paper."

    def test_every_section_in_its_order(self):
        brief = Brief(
            title="T", objective="O", deliverable="D", constraints="C", known="K", done_when="W",
        )
        assert task_message(brief) == (
            "# Task: T\n\n"
            "## Objective\nO\n\n"
            "## Deliverable\nD\n\n"
            "## Constraints\nC\n\n"
            "## What is already known\nK\n\n"
            "## Done when\nW"
        )

    def test_a_section_only_when_filled(self):
        message = task_message(Brief(title="T", objective="O", done_when="W"))
        assert message == "# Task: T\n\n## Objective\nO\n\n## Done when\nW"
        assert "## Deliverable" not in message and "## Constraints" not in message

    def test_a_field_of_several_lines_is_kept_as_written(self):
        message = task_message(Brief(title="T", objective="O", known="- a\n- b"))
        assert message.endswith("## What is already known\n- a\n- b")

    def test_an_inheriting_types_message_ends_with_the_note(self):
        message = task_message(Brief(title="T", objective="O"), inherit=True)
        assert message == f"# Task: T\n\n## Objective\nO\n\n{INHERITED_CONTEXT_NOTE}"

    def test_a_fresh_types_message_has_no_note(self):
        assert INHERITED_CONTEXT_NOTE not in task_message(Brief(title="T", objective="O"))


class TestTheInheritedNote:
    def test_adk_delegation_uses_this_one(self):
        pytest.importorskip("google.adk")
        from agentic_cli.workflow.adk import delegation

        assert delegation.INHERITED_CONTEXT_NOTE is INHERITED_CONTEXT_NOTE

    def test_its_text(self):
        assert INHERITED_CONTEXT_NOTE.startswith(
            "You are a sub-agent: another agent has handed you a task. The conversation "
            "before the task is that agent's"
        )
        assert INHERITED_CONTEXT_NOTE.endswith(
            "use the specific facts, figures and identifiers already established in the "
            "conversation rather than answering generically."
        )

"""A delegation's brief, its task message and the task tool's text (``workflow/agents/brief.py``).

A brief has a title and an objective, and may say what to deliver, the
constraints, what is already known and when the task is done. ``read_brief``
reads one from a ``task`` call's arguments, against the agent types the
caller may start, and a brief that cannot start a run is a ``BriefError``
whose text is the call's error result. The task message lays the filled
fields out under one heading each, and an inheriting type's ends with
``INHERITED_CONTEXT_NOTE``. ``task_parameters`` and ``task_description`` are
the tool's schema and description, which both backends build ``task`` from;
``follow_up_parameters`` and ``follow_up_description`` are the parent's other
tools'.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from agentic_cli.workflow.agents import brief as brief_module
from agentic_cli.workflow.agents.brief import (
    BRIEF_FIELDS,
    DEPTH_SENTENCE,
    INHERITED_CONTEXT_NOTE,
    Brief,
    BriefError,
    TaskOffer,
    follow_up_description,
    follow_up_parameters,
    read_brief,
    task_description,
    task_message,
    task_offer,
    task_parameters,
)
from agentic_cli.workflow.agents.policy import SubAgentPolicy
from agentic_cli.workflow.config import AgentConfig

RESEARCHER = AgentConfig(name="researcher", prompt="p", description="Finds facts.")
WRITER = AgentConfig(
    name="writer", prompt="p", description="Writes the report.",
    brief_required=("objective", "done_when", "deliverable"),
)
NOTES = AgentConfig(name="notes", prompt="p", inherit_context=True)
TYPES = {"researcher": RESEARCHER, "writer": WRITER}


@pytest.fixture
def logged(monkeypatch) -> list[dict]:
    """Every log call of ``agents/brief.py``, with its level."""
    events: list[dict] = []

    class Recorder:
        def __getattr__(self, level: str):
            def log(event: str, **fields) -> None:
                events.append({"event": event, "log_level": level, **fields})

            return log

    monkeypatch.setattr(brief_module, "logger", Recorder())
    return events


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
    def test_its_text(self):
        assert INHERITED_CONTEXT_NOTE.startswith(
            "You are a sub-agent: another agent has handed you a task. The conversation "
            "before the task is that agent's"
        )
        assert INHERITED_CONTEXT_NOTE.endswith(
            "use the specific facts, figures and identifiers already established in the "
            "conversation rather than answering generically."
        )
        # True on both backends: Deep Agents' conversation leaves out the call
        # that handed the run its task, which ADK's keeps.
        assert "what that agent did is quoted. Do not carry out" in INHERITED_CONTEXT_NOTE


class TestReadBrief:
    def test_a_brief_for_a_type(self):
        config, brief = read_brief(
            {"agent_type": "researcher", "title": "Find it", "objective": "Find the paper."}, TYPES,
        )
        assert config is RESEARCHER
        assert brief == Brief(title="Find it", objective="Find the paper.")

    def test_each_text_is_stripped_and_an_empty_one_left_out(self):
        _, brief = read_brief({
            "agent_type": " researcher ", "title": "  Find it\n", "objective": "\tFind the paper. ",
            "known": "  ", "constraints": None, "done_when": " Found. ",
        }, TYPES)
        assert brief == Brief(title="Find it", objective="Find the paper.", done_when="Found.")

    def test_the_arguments_are_left_as_they_were(self):
        arguments = {"agent_type": "researcher", "title": " T ", "objective": " O "}
        read_brief(arguments, TYPES)
        assert arguments == {"agent_type": "researcher", "title": " T ", "objective": " O "}

    def test_it_is_a_value_error(self):
        assert issubclass(BriefError, ValueError)

    @pytest.mark.parametrize("arguments", [
        {"title": "T", "objective": "O"},
        {"agent_type": "", "title": "T", "objective": "O"},
        {"agent_type": "  ", "title": "T", "objective": "O"},
        {"agent_type": 3, "title": "T", "objective": "O"},
        {"agent_type": None, "title": "T", "objective": "O"},
    ])
    def test_no_agent_type(self, arguments):
        with pytest.raises(BriefError) as exc:
            read_brief(arguments, TYPES)
        assert str(exc.value) == "task needs agent_type: one of researcher, writer."

    def test_an_unknown_agent_type(self):
        with pytest.raises(BriefError) as exc:
            read_brief({"agent_type": "coder", "title": "T", "objective": "O"}, TYPES)
        assert str(exc.value) == (
            "Unknown agent_type 'coder': the agents you can delegate to are researcher, writer."
        )

    def test_a_type_the_caller_may_not_start_is_unknown(self):
        # ``types`` are the caller's own: another agent's sub-agent is not one.
        with pytest.raises(BriefError, match="Unknown agent_type 'writer'"):
            read_brief({"agent_type": "writer", "title": "T", "objective": "O"},
                       {"researcher": RESEARCHER})

    def test_an_argument_task_does_not_have(self):
        with pytest.raises(BriefError) as exc:
            read_brief({"agent_type": "researcher", "title": "T", "objective": "O",
                        "subagent_type": "researcher", "prompt": "x"}, TYPES)
        assert str(exc.value) == "task has no argument 'subagent_type'."

    @pytest.mark.parametrize("field", ["title", "objective", "known"])
    @pytest.mark.parametrize("value", [3, ["a"], {"a": 1}, True])
    def test_a_field_that_is_not_text(self, field, value):
        arguments = {"agent_type": "researcher", "title": "T", "objective": "O", field: value}
        with pytest.raises(BriefError) as exc:
            read_brief(arguments, TYPES)
        assert str(exc.value) == f"The brief's {field} must be text."

    def test_the_title_and_the_objective_are_always_needed(self):
        with pytest.raises(BriefError) as exc:
            read_brief({"agent_type": "researcher", "objective": "  "}, TYPES)
        assert str(exc.value) == "The brief for 'researcher' is missing: title, objective."

    def test_a_type_asks_for_more_named_in_the_schemas_order(self):
        with pytest.raises(BriefError) as exc:
            read_brief({"agent_type": "writer", "title": "T", "objective": "O", "known": "K"}, TYPES)
        assert str(exc.value) == "The brief for 'writer' is missing: deliverable, done_when."

    def test_a_type_that_asks_for_nothing_more_still_needs_both(self):
        types = {"free": AgentConfig(name="free", prompt="p", brief_required=())}
        with pytest.raises(BriefError, match="missing: title, objective"):
            read_brief({"agent_type": "free"}, types)

    def test_the_agent_type_is_read_first(self):
        with pytest.raises(BriefError, match="Unknown agent_type"):
            read_brief({"agent_type": "coder", "nope": 1}, TYPES)

    def test_an_unknown_argument_before_a_missing_field(self):
        with pytest.raises(BriefError, match="no argument 'nope'"):
            read_brief({"agent_type": "researcher", "nope": 1}, TYPES)


class TestTaskParameters:
    def test_the_schema(self):
        schema = task_parameters({"researcher": RESEARCHER, "writer": WRITER, "notes": NOTES})
        assert schema == {
            "type": "object",
            "properties": {
                "agent_type": {
                    "type": "string",
                    "enum": ["researcher", "writer", "notes"],
                    "description": (
                        "The agent to hand the task to:\n"
                        "- researcher: Finds facts.\n"
                        "- writer (its brief also needs: deliverable, done_when): Writes the report.\n"
                        "- notes (sees this conversation, so the brief need not repeat it): "
                        "The notes agent."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "A few words naming the task, shown to the user while it runs.",
                },
                "objective": {"type": "string", "description": "What the sub-agent is to do."},
                "deliverable": {"type": "string", "description": "What to return and in what form."},
                "constraints": {"type": "string", "description": "Limits on how to do it."},
                "known": {
                    "type": "string",
                    "description": (
                        "What you already know that the sub-agent needs: facts found, decisions "
                        "made, file paths and URLs, what was tried."
                    ),
                },
                "done_when": {"type": "string", "description": "The criteria for the task to be done."},
            },
            "required": ["agent_type", "title", "objective"],
        }

    def test_it_is_json(self):
        schema = task_parameters(TYPES)
        assert json.loads(json.dumps(schema)) == schema

    def test_its_arguments_are_the_ones_read_brief_reads(self):
        properties = task_parameters(TYPES)["properties"]
        assert list(properties) == ["agent_type", "title", *BRIEF_FIELDS]

    def test_it_warns_of_nothing(self, logged):
        task_parameters({"notes": NOTES})
        assert logged == []


class TestTaskDescription:
    def test_the_description(self, logged):
        text = task_description(TYPES, SubAgentPolicy())
        assert text == (
            "Hand a task to a sub-agent. The sub-agent works on it on its own, with its own "
            "tools, and returns its result to you as this call's result; the user does not "
            "see it, so tell the user what it means yourself. Several task calls in one "
            "response run in parallel.\n\n"
            "A sub-agent sees only the brief you write, unless its line below says it sees "
            "this conversation: put in the brief everything it needs to know.\n\n"
            "At most 4 sub-agents work at once; more calls in one response wait for a free "
            "slot. At most 20 runs per turn.\n\n"
            "The agents you can hand a task to:\n"
            "- researcher: Finds facts.\n"
            "- writer (its brief also needs: deliverable, done_when): Writes the report."
        )
        assert logged == []

    def test_an_inheriting_type_says_so(self):
        text = task_description({"notes": NOTES, "researcher": RESEARCHER}, SubAgentPolicy())
        assert "- notes (sees this conversation, so the brief need not repeat it): " in text
        assert "- researcher: Finds facts." in text

    def test_a_type_without_a_description_is_named_and_warned_of_once(self, logged):
        text = task_description({"notes": NOTES, "researcher": RESEARCHER}, SubAgentPolicy())
        assert text.endswith(": The notes agent.\n- researcher: Finds facts.")
        assert logged == [
            {"event": "subagent_without_description", "log_level": "warning", "agent": "notes"},
        ]

    def test_a_blank_description_is_none(self, logged):
        blank = dataclasses.replace(RESEARCHER, description="   ")
        assert task_description({"researcher": blank}, SubAgentPolicy()).endswith(
            "- researcher: The researcher agent."
        )
        assert [e["agent"] for e in logged] == ["researcher"]

    def test_it_states_the_working_slots_then_the_runs_a_turn_may_start_with_the_policys_numbers(
        self,
    ):
        policy = SubAgentPolicy(max_active=2, max_runs_per_turn=3)

        assert brief_module.limit_sentences(policy) == [
            "At most 2 sub-agents work at once; more calls in one response wait for a free slot.",
            "At most 3 runs per turn.",
        ]
        assert task_description(TYPES, policy).split("\n\n")[2] == (
            "At most 2 sub-agents work at once; more calls in one response wait for a free "
            "slot. At most 3 runs per turn."
        )

    def test_the_limits_in_force_are_one_paragraph_before_the_agents(self, monkeypatch):
        seen: list[SubAgentPolicy] = []

        def sentences(policy: SubAgentPolicy) -> list[str]:
            seen.append(policy)
            return [f"At most {policy.max_active} at once.", "One more limit."]

        monkeypatch.setattr(brief_module, "limit_sentences", sentences)
        policy = SubAgentPolicy(max_active=2)
        paragraphs = task_description(TYPES, policy).split("\n\n")
        assert paragraphs[2] == "At most 2 at once. One more limit."
        assert paragraphs[3].startswith("The agents you can hand a task to:")
        assert seen == [policy]

    def test_the_depth_sentence_follows_the_limits(self, logged):
        text = task_description(TYPES, SubAgentPolicy(), depth_sentence=True)
        assert DEPTH_SENTENCE == "Sub-agents cannot start sub-agents."
        assert text.split("\n\n")[2] == (
            "At most 4 sub-agents work at once; more calls in one response wait for a free "
            "slot. At most 20 runs per turn. Sub-agents cannot start sub-agents."
        )
        assert text.replace(" Sub-agents cannot start sub-agents.", "") == (
            task_description(TYPES, SubAgentPolicy())
        )

    def test_it_can_be_made_without_a_warning(self, logged):
        text = task_description({"notes": NOTES}, SubAgentPolicy(), warn=False)
        assert text.endswith(": The notes agent.")
        assert logged == []


class TestTaskOffer:
    """What ``task`` offers one run: the types, the schema and the description,
    made together from the same types."""

    def test_its_parts(self, logged):
        offer = task_offer(TYPES, SubAgentPolicy(max_active=2))

        assert offer == TaskOffer(
            types=TYPES, parameters=task_parameters(TYPES),
            description=task_description(TYPES, SubAgentPolicy(max_active=2)),
        )
        assert logged == []  # made per run: it warns only when asked

    def test_with_the_depth_sentence_and_the_warning(self, logged):
        offer = task_offer({"notes": NOTES}, SubAgentPolicy(), depth_sentence=True, warn=True)

        assert DEPTH_SENTENCE in offer.description
        assert [e["event"] for e in logged] == ["subagent_without_description"]


AGENT_ID = "The sub-agent's id, such as researcher-2."


class TestTheParentsTools:
    """``message_agent``, ``stop_agent`` and ``check_agents``: their
    descriptions and schemas, which both backends build the tools from."""

    def test_the_descriptions(self):
        assert follow_up_description("message_agent") == (
            "Send a message to one of your sub-agents that already ran (its id, such as "
            "researcher-2, is in the result task gave you, or in check_agents). An idle one "
            "starts a new run with your message as its input, keeping everything it did "
            "before, and you get its result as task gives it. A running one gets your "
            "message before its next step; interrupt=true stops its run and starts a new one "
            "with your message. One the user stopped or a closed one cannot be messaged."
        )
        assert follow_up_description("stop_agent") == (
            "Stop the running run of one of your sub-agents. close=true also closes it and "
            "its own sub-agents: a closed agent cannot be messaged again, and its work stays "
            "readable to the user."
        )
        assert follow_up_description("check_agents") == (
            "The state of your sub-agents, or of one: each one's status, its current or last "
            "run, the first lines of its last result and the tokens it used. A run's result "
            "comes back to you on its own: do not call this to wait for one."
        )

    def test_message_agents_schema(self):
        assert follow_up_parameters("message_agent") == {
            "type": "object",
            "properties": {
                "agent": {"type": "string", "description": AGENT_ID},
                "message": {"type": "string", "description": "What to tell it."},
                "interrupt": {
                    "type": "boolean",
                    "description": (
                        "For a running one: stop its run and start a new one with your "
                        "message, rather than give it the message before its next step."
                    ),
                },
            },
            "required": ["agent", "message"],
        }

    def test_stop_agents_schema(self):
        assert follow_up_parameters("stop_agent") == {
            "type": "object",
            "properties": {
                "agent": {"type": "string", "description": AGENT_ID},
                "close": {
                    "type": "boolean",
                    "description": "Also close it and its own sub-agents.",
                },
            },
            "required": ["agent"],
        }

    def test_check_agents_schema(self):
        assert follow_up_parameters("check_agents") == {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "One sub-agent's id; all of yours when left out.",
                },
            },
            "required": [],
        }

    def test_each_schema_is_a_copy(self):
        schema = follow_up_parameters("stop_agent")
        schema["properties"].clear()

        assert follow_up_parameters("stop_agent")["properties"]

    def test_no_other_tool_has_one(self):
        with pytest.raises(KeyError):
            follow_up_parameters("task")
        with pytest.raises(KeyError):
            follow_up_description("task")

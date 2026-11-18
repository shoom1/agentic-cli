"""A delegation's brief, the task message a run starts from, and the task tool's text.

A parent delegates with a brief: a title (a few words, for ``/agents`` and
the status bar) and an objective, and, where they help, what to deliver,
the constraints, what is already known and when the task is done
(``BRIEF_FIELDS``). ``read_brief`` reads one from a ``task`` call's
arguments, against the agent types the caller may start; a brief that cannot
start a run is a ``BriefError``, whose text is the call's error result.
``task_message`` lays a brief out as the first message of the run it starts
(the system prompt stays the type's own); an inheriting type's ends with
``INHERITED_CONTEXT_NOTE``. ``task_parameters`` and ``task_description`` are
``task``'s schema and description, the same for both backends.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.workflow.agents.policy import SubAgentPolicy
    from agentic_cli.workflow.config import AgentConfig

logger = Loggers.workflow()

#: A brief's fields besides its title, in the order a task message lays them out.
BRIEF_FIELDS = ("objective", "deliverable", "constraints", "known", "done_when")

#: ``task``'s arguments, in the order its schema lists them.
_ARGUMENTS = ("agent_type", "title", *BRIEF_FIELDS)

#: What ``task``'s schema tells the model of each field of the brief.
_FIELD_DESCRIPTIONS = {
    "title": "A few words naming the task, shown to the user while it runs.",
    "objective": "What the sub-agent is to do.",
    "deliverable": "What to return and in what form.",
    "constraints": "Limits on how to do it.",
    "known": (
        "What you already know that the sub-agent needs: facts found, decisions "
        "made, file paths and URLs, what was tried."
    ),
    "done_when": "The criteria for the task to be done.",
}

#: The opening of ``task``'s description: what a delegation is.
_DELEGATION = (
    "Hand a task to a sub-agent. The sub-agent works on it on its own, with its own "
    "tools, and returns its result to you as this call's result; the user does not "
    "see it, so tell the user what it means yourself. Several task calls in one "
    "response run in parallel."
)

#: What a sub-agent sees, in ``task``'s description.
_WHAT_IT_SEES = (
    "A sub-agent sees only the brief you write, unless its line below says it sees "
    "this conversation: put in the brief everything it needs to know."
)

#: The heading of each field's section in a task message.
_HEADINGS = {
    "objective": "Objective",
    "deliverable": "Deliverable",
    "constraints": "Constraints",
    "known": "What is already known",
    "done_when": "Done when",
}

#: What an inheriting sub-agent is told of the conversation it reads: whose it
#: is, and which message is its task (Deep Agents' ``_FORK_TASK_PREAMBLE``,
#: told the way ADK shows that conversation). ``task_message`` ends an
#: inheriting type's task message with it, on both backends: the run reads
#: its parent's conversation before that message. ADK's copy of it keeps the
#: call that handed the run its task and Deep Agents' leaves it out, so the
#: note says nothing of that call.
INHERITED_CONTEXT_NOTE = (
    "You are a sub-agent: another agent has handed you a task. The conversation "
    "before the task is that agent's, shown so you know what has been established: "
    "the user's messages in it were addressed to that agent, not to you, and what "
    "that agent did is quoted. Do not carry out those messages yourself or answer "
    "as that agent. Your task is the request that follows the conversation: "
    "complete it directly, and use the specific facts, figures and identifiers "
    "already established in the conversation rather than answering generically."
)


class BriefError(ValueError):
    """A ``task`` call whose brief cannot start a run; its text is the call's error result."""


@dataclass(frozen=True)
class Brief:
    """What a parent gives a sub-agent run to work from.

    Attributes:
        title: A few words naming the task, for ``/agents`` and the status bar.
        objective: What the run is to do.
        deliverable: What to return, and in what form.
        constraints: Limits on how to do it.
        known: What the parent already knows that the run needs: facts
            found, decisions made, file paths and URLs, what was tried.
        done_when: The criteria for the task being done.
    """

    title: str
    objective: str
    deliverable: str | None = None
    constraints: str | None = None
    known: str | None = None
    done_when: str | None = None

    def fields(self) -> dict[str, str]:
        """The filled fields (None and empty ones left out): the title first,
        then ``BRIEF_FIELDS``' order."""
        return {
            name: value
            for name in ("title", *BRIEF_FIELDS)
            if (value := getattr(self, name))
        }


def task_message(brief: Brief, *, inherit: bool = False) -> str:
    """The first message of the run ``brief`` starts.

    ``# Task: <title>``, then a section for each filled field (its heading
    line, then its text), one blank line apart. An inheriting type's
    (``inherit``) ends with ``INHERITED_CONTEXT_NOTE``: that run reads its
    parent's conversation before this message.
    """
    sections = [f"# Task: {brief.title}"]
    sections += [
        f"## {_HEADINGS[name]}\n{text}"
        for name, text in brief.fields().items()
        if name != "title"
    ]
    if inherit:
        sections.append(INHERITED_CONTEXT_NOTE)
    return "\n\n".join(sections)


def read_brief(
    arguments: Mapping[str, Any], types: Mapping[str, AgentConfig]
) -> tuple[AgentConfig, Brief]:
    """The agent type and the brief of a ``task`` call.

    ``types`` are the agents the caller may start, by name (never empty: an
    agent without any has no ``task``). Each text is stripped, and an empty
    one counts as missing. The title and the objective are always needed;
    the type's ``brief_required`` may ask for more.

    Raises:
        BriefError: ``agent_type`` is missing or not one of ``types``, an
            argument ``task`` does not have, a field that is not text, or a
            field the type needs is missing: in that order, and the fields
            named in the order ``task``'s schema lists them.
    """
    names = ", ".join(types)
    agent_type = arguments.get("agent_type")
    if not isinstance(agent_type, str) or not agent_type.strip():
        raise BriefError(f"task needs agent_type: one of {names}.")
    agent_type = agent_type.strip()
    config = types.get(agent_type)
    if config is None:
        raise BriefError(
            f"Unknown agent_type '{agent_type}': the agents you can delegate to are {names}."
        )
    for name in arguments:
        if name not in _ARGUMENTS:
            raise BriefError(f"task has no argument '{name}'.")

    texts: dict[str, str] = {}
    for name in ("title", *BRIEF_FIELDS):
        value = arguments.get(name)
        if value is None:
            continue
        if not isinstance(value, str):
            raise BriefError(f"The brief's {name} must be text.")
        if value.strip():
            texts[name] = value.strip()
    needed = {"title", "objective", *config.brief_required}
    missing = [name for name in ("title", *BRIEF_FIELDS) if name in needed and name not in texts]
    if missing:
        raise BriefError(f"The brief for '{agent_type}' is missing: {', '.join(missing)}.")
    return config, Brief(**texts)


def task_parameters(types: Mapping[str, AgentConfig]) -> dict[str, Any]:
    """``task``'s arguments, as a JSON schema object: both backends build its schema from this.

    ``agent_type`` is one of ``types`` (the agents the caller may start), each
    named in its description with what it does; then the brief's title and
    fields. The type, the title and the objective are required: what else a
    type needs of the brief is said in its line, and ``read_brief`` checks it.
    """
    lines = "\n".join(_type_line(name, config) for name, config in types.items())
    properties: dict[str, Any] = {
        "agent_type": {
            "type": "string",
            "enum": list(types),
            "description": f"The agent to hand the task to:\n{lines}",
        },
    }
    for name in ("title", *BRIEF_FIELDS):
        properties[name] = {"type": "string", "description": _FIELD_DESCRIPTIONS[name]}
    return {
        "type": "object",
        "properties": properties,
        "required": ["agent_type", "title", "objective"],
    }


def task_description(types: Mapping[str, AgentConfig], policy: SubAgentPolicy) -> str:
    """``task``'s description, as the model reads it.

    What a delegation is (the sub-agent works on its own and returns its
    result; several calls in one response run in parallel), what a sub-agent
    sees (the brief alone, unless its type inherits the conversation), the
    limits in force (``limit_sentences``), and each of ``types`` with its
    description. A type without a description is offered as "The <name>
    agent.", with one warning logged for it.
    """
    paragraphs = [_DELEGATION, _WHAT_IT_SEES]
    limits = limit_sentences(policy)
    if limits:
        paragraphs.append(" ".join(limits))
    lines = "\n".join(_type_line(name, config, warn=True) for name, config in types.items())
    paragraphs.append(f"The agents you can hand a task to:\n{lines}")
    return "\n\n".join(paragraphs)


def limit_sentences(policy: SubAgentPolicy) -> list[str]:
    """The sentences of ``task``'s description that state the limits a delegation enforces.

    One sentence for each limit the framework enforces on a delegation, with
    the number ``policy`` sets, in the order they apply: the runs that work
    at once (``max_active``: more wait for a free working slot), then the
    runs one turn may start (``max_runs_per_turn``).
    """
    return [
        f"At most {policy.max_active} sub-agents work at once; more calls in one response "
        "wait for a free slot.",
        f"At most {policy.max_runs_per_turn} runs per turn.",
    ]


def _type_line(name: str, config: AgentConfig, *, warn: bool = False) -> str:
    """How ``task`` offers ``config``: ``- <name> (<notes>): <description>``.

    The notes say that an inheriting type sees the caller's conversation and
    which brief fields the type needs beyond the title and the objective. A
    type without a description is "The <name> agent.", logged when ``warn``.
    """
    description = config.description.strip()
    if not description:
        if warn:
            logger.warning("subagent_without_description", agent=name)
        description = f"The {name} agent."
    notes = []
    if config.inherit_context:
        notes.append("sees this conversation, so the brief need not repeat it")
    more = [field for field in BRIEF_FIELDS[1:] if field in config.brief_required]
    if more:
        notes.append(f"its brief also needs: {', '.join(more)}")
    head = f"- {name} ({'; '.join(notes)})" if notes else f"- {name}"
    return f"{head}: {description}"

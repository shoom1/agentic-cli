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
``task``'s schema and description, the same for both backends, made for the
types one run may start (``task_offer``, ``Delegations.offered``);
``follow_up_parameters`` and ``follow_up_description`` are the parent's other
tools' (``message_agent``, ``stop_agent``, ``check_agents``). Every call may
narrow its run's tools (``tools``), and one to a type with model choices may
pick one (``model``), which ``read_choices`` checks.
"""

from __future__ import annotations

import copy
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.workflow.agents.policy import SubAgentPolicy
    from agentic_cli.workflow.config import AgentConfig

logger = Loggers.workflow()

#: A brief's fields besides its title, in the order a task message lays them out.
BRIEF_FIELDS = ("objective", "deliverable", "constraints", "known", "done_when")

#: ``task``'s arguments, in the order its schema lists them (``model`` only for
#: types with model choices).
_ARGUMENTS = ("agent_type", "title", *BRIEF_FIELDS, "tools", "model")

#: What ``task``'s schema tells the model of ``tools``.
TOOLS_DESCRIPTION = (
    "Optional: names of the agent's own tools to give this task, to narrow it; "
    "leave it out for all of them."
)

#: What ``task``'s schema tells the model of ``model``, when a type has choices.
MODEL_DESCRIPTION = (
    "Optional: one of the agent's model choices (named with it below) to run this "
    "task on; leave it out for its own model."
)

#: What a call is told whose ``tools`` are not some of the type's own.
TOOLS_ERROR = "The tools for '{agent_type}' must be some of its own: {names}."

#: What a call is told that narrows the tools of a type that has none of its own.
NO_TOOLS_ERROR = "'{agent_type}' has no tools of its own; leave tools out."

#: What a call is told whose ``model`` is none of the type's choices.
MODEL_ERROR = "The model for '{agent_type}' must be one of its model choices: {names}."

#: What a call is told that names a model for a type with no choices.
NO_MODEL_CHOICES_ERROR = "'{agent_type}' has no model choices; leave model out."

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

#: Added to ``task``'s limits when what a run starts could not start more.
DEPTH_SENTENCE = "Sub-agents cannot start sub-agents."

#: The parent's other tools' descriptions, by tool name.
_PARENT_TOOL_DESCRIPTIONS = {
    "message_agent": (
        "Send a message to one of your sub-agents that already ran (its id, such as "
        "researcher-2, is in the result task gave you, or in check_agents). An idle one "
        "starts a new run with your message as its input, keeping everything it did "
        "before, and you get its result as task gives it. A running one gets your "
        "message before its next step; interrupt=true stops its run and starts a new one "
        "with your message. One the user stopped or a closed one cannot be messaged."
    ),
    "stop_agent": (
        "Stop the running run of one of your sub-agents. close=true also closes it and "
        "its own sub-agents: a closed agent cannot be messaged again, and its work stays "
        "readable to the user."
    ),
    "check_agents": (
        "The state of your sub-agents, or of one: each one's status, its current or last "
        "run, the first lines of its last result and the tokens it used. A run's result "
        "comes back to you on its own: do not call this to wait for one."
    ),
}

_AGENT_ID = {"type": "string", "description": "The sub-agent's id, such as researcher-2."}

#: The parent's other tools' schemas, by tool name.
_FOLLOW_UP_PARAMETERS: dict[str, dict[str, Any]] = {
    "message_agent": {
        "type": "object",
        "properties": {
            "agent": _AGENT_ID,
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
    },
    "stop_agent": {
        "type": "object",
        "properties": {
            "agent": _AGENT_ID,
            "close": {
                "type": "boolean",
                "description": "Also close it and its own sub-agents.",
            },
        },
        "required": ["agent"],
    },
    "check_agents": {
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "description": "One sub-agent's id; all of yours when left out.",
            },
        },
        "required": [],
    },
}

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


def read_choices(
    arguments: Mapping[str, Any],
    config: AgentConfig,
    own_tools: Collection[str] = (),
) -> tuple[frozenset[str] | None, str | None]:
    """The tools and the model a ``task`` call gives its run: None each when left out (or null).

    ``tools`` are some of the type's own: ``own_tools`` are the names its
    backend gives it (the state and delegation tools left out), and none at
    all is a narrowing too. ``model`` is one of the type's ``model_choices``,
    as written (stripped). Read after ``read_brief``, before the run counts;
    the tools are read first.

    Raises:
        BriefError: ``tools`` are not a list of the names of the type's own
            tools (the message lists them), or the type has none; ``model`` is not
            one of its choices (the message lists them), or it has none.
    """
    agent_type = config.name
    tools: frozenset[str] | None = None
    given = arguments.get("tools")
    if given is not None:
        if not own_tools and given != []:
            raise BriefError(NO_TOOLS_ERROR.format(agent_type=agent_type))
        if not isinstance(given, list) or not all(
            isinstance(name, str) and name in own_tools for name in given
        ):
            raise BriefError(
                TOOLS_ERROR.format(agent_type=agent_type, names=", ".join(sorted(own_tools)))
            )
        tools = frozenset(given)

    model: str | None = None
    chosen = arguments.get("model")
    if chosen is not None:
        if not config.model_choices:
            raise BriefError(NO_MODEL_CHOICES_ERROR.format(agent_type=agent_type))
        if not isinstance(chosen, str) or chosen.strip() not in config.model_choices:
            raise BriefError(
                MODEL_ERROR.format(agent_type=agent_type, names=", ".join(config.model_choices))
            )
        model = chosen.strip()
    return tools, model


def task_parameters(types: Mapping[str, AgentConfig]) -> dict[str, Any]:
    """``task``'s arguments, as a JSON schema object: both backends build its schema from this.

    ``agent_type`` is one of ``types`` (the agents the caller may start), each
    named in its description with what it does; then the brief's title and
    fields. The type, the title and the objective are required: what else a
    type needs of the brief is said in its line, and ``read_brief`` checks it.
    Then ``tools``, which every call may give, and ``model``, only when a type
    in ``types`` has model choices (``read_choices`` checks both).
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
    properties["tools"] = {
        "type": "array",
        "items": {"type": "string"},
        "description": TOOLS_DESCRIPTION,
    }
    if any(config.model_choices for config in types.values()):
        properties["model"] = {"type": "string", "description": MODEL_DESCRIPTION}
    return {
        "type": "object",
        "properties": properties,
        "required": ["agent_type", "title", "objective"],
    }


def task_description(
    types: Mapping[str, AgentConfig],
    policy: SubAgentPolicy,
    *,
    depth_sentence: bool = False,
    warn: bool = True,
) -> str:
    """``task``'s description, as the model reads it.

    What a delegation is (the sub-agent works on its own and returns its
    result; several calls in one response run in parallel), what a sub-agent
    sees (the brief alone, unless its type inherits the conversation), the
    limits in force (``limit_sentences``), and each of ``types`` with its
    description. A type without a description is offered as "The <name>
    agent.", with one warning logged for it (none when ``warn`` is False).
    ``depth_sentence`` ends the limits with ``DEPTH_SENTENCE``.
    """
    paragraphs = [_DELEGATION, _WHAT_IT_SEES]
    limits = limit_sentences(policy)
    if depth_sentence:
        limits = [*limits, DEPTH_SENTENCE]
    if limits:
        paragraphs.append(" ".join(limits))
    lines = "\n".join(_type_line(name, config, warn=warn) for name, config in types.items())
    paragraphs.append(f"The agents you can hand a task to:\n{lines}")
    return "\n\n".join(paragraphs)


@dataclass(frozen=True)
class TaskOffer:
    """What ``task`` offers one run: the types it may start, the schema and the description."""

    types: Mapping[str, AgentConfig]
    parameters: dict[str, Any]
    description: str


def task_offer(
    types: Mapping[str, AgentConfig],
    policy: SubAgentPolicy,
    *,
    depth_sentence: bool = False,
    warn: bool = False,
) -> TaskOffer:
    """The offer for ``types``: schema and description made together from the same types.

    ``warn`` logs a type without a description once; an offer made for each
    run leaves it off.
    """
    return TaskOffer(
        types=dict(types),
        parameters=task_parameters(types),
        description=task_description(
            types, policy, depth_sentence=depth_sentence, warn=warn
        ),
    )


def follow_up_description(name: str) -> str:
    """The description of the parent's tool ``name`` (``message_agent``,
    ``stop_agent`` or ``check_agents``), the same for both backends.

    Raises:
        KeyError: ``name`` is not one of them.
    """
    return _PARENT_TOOL_DESCRIPTIONS[name]


def follow_up_parameters(name: str) -> dict[str, Any]:
    """The JSON schema of the parent's tool ``name``'s arguments: a copy each call.

    Raises:
        KeyError: ``name`` is not one of the parent's tools.
    """
    return copy.deepcopy(_FOLLOW_UP_PARAMETERS[name])


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
    which brief fields the type needs beyond the title and the objective, and
    names its model choices, last. A type without a description is "The <name>
    agent.", logged when ``warn``.
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
    if config.model_choices:
        notes.append(f"model choices: {', '.join(config.model_choices)}")
    head = f"- {name} ({'; '.join(notes)})" if notes else f"- {name}"
    return f"{head}: {description}"

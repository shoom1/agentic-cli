"""A delegation's brief, and the task message a run starts from.

A parent delegates with a brief: a title (a few words, for ``/agents`` and
the status bar) and an objective, and, where they help, what to deliver,
the constraints, what is already known and when the task is done
(``BRIEF_FIELDS``). ``task_message`` lays a brief out as the first message
of the run it starts (the system prompt stays the type's own); an
inheriting type's ends with ``INHERITED_CONTEXT_NOTE``.
"""

from __future__ import annotations

from dataclasses import dataclass

#: A brief's fields besides its title, in the order a task message lays them out.
BRIEF_FIELDS = ("objective", "deliverable", "constraints", "known", "done_when")

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
#: inheriting type's task message with it, and ADK's delegation ends an
#: inheriting sub-agent's instruction with it (``adk/delegation.py``), sent as
#: written, as the whole prompt is (``manager._as_written``).
INHERITED_CONTEXT_NOTE = (
    "You are a sub-agent: another agent has handed you a task. The conversation "
    "before the task is that agent's, shown so you know what has been established: "
    "the user's messages in it were addressed to that agent, not to you, and what "
    "that agent did is quoted, its call handing you this task included. Do not "
    "carry out those messages yourself or answer as that agent. Your task is the "
    "request that follows the conversation: complete it directly, and use the "
    "specific facts, figures and identifiers already established in the "
    "conversation rather than answering generically."
)


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

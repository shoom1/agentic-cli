"""Outside content: what marks a conversation, and the mark each one keeps (spec §9).

A conversation is marked once a tool that brings content from outside the
user and the project (a web page, a document, a program's output, an MCP
server's answer) has been allowed to run in it. Content read stays in the
model's context after its turn, so the mark never clears: from then on a
memory write in that conversation is approved per call.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from agentic_cli.logging import Loggers
from agentic_cli.workflow.sessions import SessionRef

if TYPE_CHECKING:
    from agentic_cli.tools.registry import ToolDefinition

logger = Loggers.workflow()

#: ADK's session-state key for the mark.
OUTSIDE_CONTENT_KEY = "agentic_cli.outside_content"
#: The mark of a conversation that has turns but no record (ADK, from before 0.7.0).
BEFORE_0_7_0 = "a conversation from before 0.7.0"
#: The mark of every LangGraph conversation: that backend tracks none.
LANGGRAPH_MARK = "the LangGraph backend, which tracks no outside content"
#: The mark a job's notice turn leaves.
JOB_RESULT = "job_result"


def brings_outside_content(definition: "ToolDefinition | None", args: Mapping[str, Any]) -> bool:
    """Whether a call of ``definition`` with ``args`` brings outside content.

    A flag that is a function decides from the arguments; one that raises
    counts as outside content (the safe answer).
    """
    flag = getattr(definition, "outside_content", False)
    if callable(flag):
        try:
            return bool(flag(dict(args)))
        except Exception:
            logger.debug("outside_content_check_failed", exc_info=True)
            return True
    return bool(flag)


def outside_content_note(tool: str) -> str:
    """The permission prompt's note for a memory write in a marked conversation."""
    return (
        f"(This conversation has read outside content ({tool}). A memory written now is "
        "read in every later conversation, so approval applies to this call only.)"
    )


class OutsideContentMarks:
    """The mark of each conversation this process runs turns of, by ``SessionRef``.

    Seeded from the conversation's own state when a turn starts; the gates
    mark it when they allow a call that brings outside content. The first
    tool wins and a mark never clears.

    Args:
        always: A mark every conversation has (the LangGraph backend's).
    """

    def __init__(self, *, always: str | None = None) -> None:
        self._always = always
        self._marks: dict[SessionRef, str] = {}
        self._lock = threading.Lock()

    def mark(self, ref: SessionRef, tool: str) -> bool:
        """Mark ``ref``'s conversation with ``tool``; True when it was not marked."""
        with self._lock:
            if ref in self._marks:
                return False
            self._marks[ref] = tool
            return True

    def seed(self, ref: SessionRef, value: str) -> None:
        """What the conversation's state says (``""`` for unmarked); a mark already held stays."""
        if value:
            self.mark(ref, value)

    def get(self, ref: SessionRef | None) -> str | None:
        """The tool that marked ``ref``'s conversation, or None."""
        if self._always is not None:
            return self._always
        if ref is None:
            return None
        with self._lock:
            return self._marks.get(ref)

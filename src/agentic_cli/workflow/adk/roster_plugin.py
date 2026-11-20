"""The main agent's roster of its open sub-agents, first in its requests.

The roster is a snapshot per conversation (``BaseWorkflowManager._roster_now``,
``workflow/agents/roster.py``), inserted when each request is built and never
stored in the session. Registered before ``MemoryIndexPlugin``, which puts the
memory index ahead of it, and before the overflow handler and the fit guard,
which keep what opens a request (``protected_head``: the index, the roster,
then a summary) and fold it into the first user text.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest


def roster_content(text: str) -> types.Content:
    """The roster as a user content of its own."""
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


class RosterPlugin(BasePlugin):
    """Puts the roster of open sub-agents first in the main agent's requests.

    Args:
        root: The main agent's name; sub-agents get no roster.
        roster_for: The turn's conversation's roster (``""`` for none). It
            reads the records in memory only.
    """

    def __init__(self, *, root: str, roster_for: Callable[[], str]) -> None:
        super().__init__(name="roster")
        self._root = root
        self._roster_for = roster_for

    async def before_model_callback(
        self, *, callback_context: "CallbackContext", llm_request: "LlmRequest"
    ) -> None:
        if callback_context.agent_name != self._root:
            return None
        text = self._roster_for()
        if text:
            llm_request.contents = [roster_content(text), *llm_request.contents]
        return None

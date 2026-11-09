"""The memory index first in each of the main agent's requests (spec §11.2).

The index is a snapshot per conversation (``BaseWorkflowManager._memory_index_now``),
inserted when each request is built and never stored in the session, so a
summary cannot drop it and a resumed conversation gets the current one.
Registered before the overflow handler and the fit guard: the guard keeps it
(``protected_head``), counts it in the budget and folds it into the first
user text, so no provider sees two user messages in a row.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest


def index_content(text: str) -> types.Content:
    """The index as a user content of its own."""
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


class MemoryIndexPlugin(BasePlugin):
    """Puts the memory index first in the main agent's requests.

    Args:
        root: The main agent's name; sub-agents get no index.
        index_for: The turn's conversation's index (``""`` for none). It reads
            files on its first call, so it runs on a worker thread.
    """

    def __init__(self, *, root: str, index_for: Callable[[], str]) -> None:
        super().__init__(name="memory_index")
        self._root = root
        self._index_for = index_for

    async def before_model_callback(
        self, *, callback_context: "CallbackContext", llm_request: "LlmRequest"
    ) -> None:
        if callback_context.agent_name != self._root:
            return None
        text = await asyncio.to_thread(self._index_for)
        if text:
            llm_request.contents = [index_content(text), *llm_request.contents]
        return None

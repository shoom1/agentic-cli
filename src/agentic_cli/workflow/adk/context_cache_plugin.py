"""At most one try a turn at a Gemini context cache.

ADK's context cache (the app's ``ContextCacheConfig``) keeps a Gemini cache per
agent. When creating one fails or times out, ADK keeps the prompt's
fingerprint and tries again at the agent's next model call, even within the
same turn. A turn of tool steps against a cache endpoint that keeps failing,
or hangs, would pay a failed create (up to its 60-second bound) at every step.
``ContextCachePlugin`` ends caching for the rest of an agent's turn once a call
that could have made a cache ends without one; the next turn tries again.
"""

from __future__ import annotations

from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin


class ContextCachePlugin(BasePlugin):
    """Stops an agent's Gemini cache attempts for the turn after one comes to nothing.

    A call could have made a cache when its request carried ADK's cache
    metadata from an earlier call: an agent's first call only records the
    prompt's fingerprint, and a tool-using first turn caches from its second
    call. A Gemini response carries metadata without a cache name when no
    cache was made (the create failed, timed out, or the prompt was too small
    for one). Claude's responses carry no cache metadata, so its cache
    breakpoints are never touched.
    """

    def __init__(self) -> None:
        super().__init__(name="context_cache")
        # (invocation, agent) → whether its model call in flight could make a cache.
        self._could_cache: dict[tuple[str, str], bool] = {}
        # The (invocation, agent) pairs that cache no more this turn.
        self._spent: set[tuple[str, str]] = set()

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        key = (callback_context.invocation_id, callback_context.agent_name)
        if key in self._spent:
            llm_request.cache_config = None
        elif llm_request.cache_config is not None:
            self._could_cache[key] = llm_request.cache_metadata is not None
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        if llm_response.partial:
            return None
        key = (callback_context.invocation_id, callback_context.agent_name)
        metadata = llm_response.cache_metadata
        if (
            self._could_cache.pop(key, False)
            and metadata is not None
            and metadata.cache_name is None
        ):
            self._spent.add(key)
        return None

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        invocation = invocation_context.invocation_id
        self._spent = {key for key in self._spent if key[0] != invocation}
        self._could_cache = {
            key: value for key, value in self._could_cache.items() if key[0] != invocation
        }

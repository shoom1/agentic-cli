"""No model is sent another provider's thoughts.

ADK sends an agent's own earlier events back with every request, thoughts
included, as if one model wrote them all. A ``/settings`` switch that keeps the
session breaks that. After Gemini, Claude rejects a thinking block it did not
sign ("Invalid `signature` in `thinking` block"); after Claude, Gemini 3
rejects a thought signature it did not issue ("Corrupted thought signature").
Either way every later turn of the session failed.

``ForeignReasoningPlugin`` tags each model response with the provider of the
agent's model, in the event's ``custom_metadata``. Before each model call it
leaves out of the request every thought that an event of another provider, or
an untagged event (one saved before this plugin), holds. The session keeps
them all, and a model's own thoughts still go back to it: Claude needs its
signed thinking within a tool loop.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

#: Where an event records the provider of the model that wrote it.
PROVIDER_KEY = "agentic_cli_provider"

_CONTENT_FIELDS = (
    "text", "inline_data", "file_data", "function_call", "function_response",
    "executable_code", "code_execution_result",
)


class ForeignReasoningPlugin(BasePlugin):
    """Leaves another provider's thoughts out of each agent's requests.

    Args:
        providers: Each agent's provider (``ProviderKind`` value), by agent
            name. An agent not named here (a native ADK YAML app's) is left
            as ADK sends it.
    """

    def __init__(self, providers: Mapping[str, str]) -> None:
        super().__init__(name="foreign_reasoning")
        self._providers = dict(providers)

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        provider = self._providers.get(callback_context.agent_name)
        if provider is not None:
            llm_response.custom_metadata = {
                **(llm_response.custom_metadata or {}),
                PROVIDER_KEY: provider,
            }
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        agent = callback_context.agent_name
        provider = self._providers.get(agent)
        if provider is None:
            return None
        own = _thoughts_of(callback_context.session.events, agent, provider)
        contents: list[types.Content] = []
        for content in llm_request.contents:
            parts = content.parts or []
            kept = [part for part in parts if not _is_reasoning(part) or _key(part) in own]
            if len(kept) == len(parts):
                contents.append(content)
            elif kept:
                contents.append(content.model_copy(update={"parts": kept}))
        llm_request.contents = contents
        return None


def _is_reasoning(part: types.Part) -> bool:
    """A thought, or a thought signature with nothing beside it.

    A function call or response is never left out, even one marked as a
    thought: the conversation needs it.
    """
    if part.function_call or part.function_response:
        return False
    if part.thought:
        return True
    return bool(part.thought_signature) and not any(
        getattr(part, name, None) for name in _CONTENT_FIELDS
    )


def _key(part: types.Part) -> tuple[str | None, bytes | None]:
    return part.text, part.thought_signature


def _thoughts_of(
    events: Iterable[Event], agent: str, provider: str
) -> set[tuple[str | None, bytes | None]]:
    """The thoughts that ``agent``'s events tagged with ``provider`` hold."""
    return {
        _key(part)
        for event in events
        if event.author == agent
        and event.content is not None
        and (event.custom_metadata or {}).get(PROVIDER_KEY) == provider
        for part in event.content.parts or ()
        if _is_reasoning(part)
    }

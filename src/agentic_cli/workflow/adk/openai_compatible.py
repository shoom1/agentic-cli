"""OpenRouter and OpenAI-compatible servers (vLLM and the like) on ADK.

``OpenAICompatibleLlm`` is ADK's Chat Completions ``OpenAILlm`` with three
additions:

- Reasoning text. These servers send a model's reasoning beside its answer,
  as ``reasoning`` (OpenRouter, newer vLLM) or ``reasoning_content`` (older
  vLLM), on streamed deltas and on the final message. ADK's class drops it;
  this one yields it as ``thought=True`` parts, before the answer, so the
  thinking box fills for open models.
- ``extra_body``, sent with every request: the provider's reasoning
  parameter (``workflow/adk/model_factory.py`` decides it).
- ``max_tokens`` may be None, which leaves the limit to the server; ADK's
  class always sends one (4096 by default).

The factory gives each instance its client, with the base URL, the key, the
timeout and retries and, for OpenRouter, the attribution headers.

ADK's class builds the request and sends an unstreamed one itself, handing
only a streamed one to ``_generate_content_streaming``. This class asks for
every request to be handed there, and sends it streamed or not as its caller
asked. The streaming loop (``_stream``) is a copy of ADK's
(``OpenAILlm._generate_content_streaming``, google-adk 2.12) with the
reasoning lines added; ``tests/workflow/adk/test_openai_compatible.py`` fails
when ADK's loop changes.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from google.adk.integrations.openai import OpenAILlm
from google.adk.integrations.openai._openai_llm import (
    _map_finish_reason,
    _response_to_llm_response,
    _usage_metadata,
)
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.utils import streaming_utils
from google.genai import types
from pydantic import Field, PrivateAttr

from agentic_cli.logging import Loggers

logger = Loggers.workflow()

# Where these servers put reasoning text, in the order it is looked for.
_REASONING_FIELDS = ("reasoning", "reasoning_content")


class OpenAICompatibleLlm(OpenAILlm):
    """A Chat Completions model on OpenRouter or an OpenAI-compatible server.

    Attributes:
        max_tokens: The output limit; None leaves it to the server.
        extra_body: Fields added to every request's body.
    """

    max_tokens: int | None = None  # type: ignore[assignment]
    extra_body: dict[str, Any] = Field(default_factory=dict)

    # Whether this call's caller asked for a stream (see generate_content_async).
    _streamed: bool = PrivateAttr(default=True)

    @classmethod
    def supported_models(cls) -> list[str]:
        # Built by the model factory, never resolved from a model string.
        return []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        # ADK's OpenAILlm builds the request, then hands a streamed one to
        # _generate_content_streaming. It is asked to stream, from a copy of
        # this model that remembers what the caller asked, so every request
        # reaches _generate_content_streaming and concurrent calls never share
        # the answer. The copy shares this model's client.
        self._openai_client  # noqa: B018 - cached here, so every copy has it
        call = self.model_copy()
        call._streamed = stream
        async for response in super(OpenAICompatibleLlm, call).generate_content_async(
            _without_thoughts(llm_request), stream=True
        ):
            yield response

    async def _generate_content_streaming(
        self, kwargs: dict[str, Any]
    ) -> AsyncGenerator[LlmResponse, None]:
        """Send the request ADK built: streamed or not, as the caller asked."""
        for limit in ("max_tokens", "max_completion_tokens"):
            if kwargs.get(limit) is None:
                kwargs.pop(limit, None)
        if self.extra_body:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **self.extra_body}
        if self._streamed:
            async for response in self._stream(kwargs):
                yield response
            return
        response = await self._openai_client.chat.completions.create(**kwargs)
        message = response.choices[0].message if response.choices else None
        yield _with_reasoning(_response_to_llm_response(response), _reasoning_of(message))

    async def _stream(self, kwargs: dict[str, Any]) -> AsyncGenerator[LlmResponse, None]:
        """ADK's ``OpenAILlm._generate_content_streaming``, reading reasoning too.

        The lines marked ``# reasoning`` are this class's; the rest is ADK's.
        """
        kwargs["stream"] = True
        # Ask for the trailing usage-only chunk. Backends that ignore this simply
        # never send it, so the metadata stays absent rather than the call failing.
        kwargs["stream_options"] = {"include_usage": True}
        raw_stream = await self._openai_client.chat.completions.create(**kwargs)

        reasoning_accumulated = ""  # reasoning
        text_accumulated = ""
        tool_calls_accumulated: dict[int, dict[str, Any]] = {}
        usage = None
        finish_reason: str | None = None

        async for chunk in raw_stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
            delta = choice.delta

            reasoning = _reasoning_of(delta)  # reasoning
            if reasoning:  # reasoning
                reasoning_accumulated += reasoning  # reasoning
                yield LlmResponse(  # reasoning
                    content=types.Content(role="model", parts=[_thought(reasoning)]),
                    partial=True,
                )

            if delta.content:
                text_accumulated += delta.content
                yield LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=delta.content)],
                    ),
                    partial=True,
                )

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    index = tc_delta.index
                    if index not in tool_calls_accumulated:
                        tool_calls_accumulated[index] = {
                            "id": tc_delta.id,
                            "name": tc_delta.function.name if tc_delta.function else None,
                            "arguments": "",
                        }
                    else:
                        if tc_delta.id:
                            tool_calls_accumulated[index]["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            tool_calls_accumulated[index]["name"] = tc_delta.function.name

                    arguments_delta = tc_delta.function.arguments if tc_delta.function else None
                    partial_args = None
                    if arguments_delta:
                        tool_calls_accumulated[index]["arguments"] += arguments_delta
                        tracker = tool_calls_accumulated[index].setdefault(
                            "tracker", streaming_utils._JsonPathTracker()
                        )
                        partial_args = tracker.handle_chunk(arguments_delta)

                    yield LlmResponse(
                        partial=True,
                        content=types.Content(
                            role="model",
                            parts=[
                                types.Part(
                                    function_call=types.FunctionCall(
                                        id=tool_calls_accumulated[index]["id"],
                                        name=tool_calls_accumulated[index]["name"] or None,
                                        partial_args=partial_args or None,
                                        will_continue=True,
                                    )
                                )
                            ],
                        ),
                    )

        # Yield final response with all accumulated content
        parts = []
        if text_accumulated:
            parts.append(types.Part.from_text(text=text_accumulated))

        for index in sorted(tool_calls_accumulated.keys()):
            acc = tool_calls_accumulated[index]
            args = {}
            if acc["arguments"]:
                try:
                    args = json.loads(acc["arguments"])
                except json.JSONDecodeError:
                    logger.warning("tool_call_arguments_not_json", tool=acc["name"])

            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        name=acc["name"], args=args, id=acc["id"]
                    )
                )
            )

        mapped_finish_reason = _map_finish_reason(finish_reason)

        if not parts and mapped_finish_reason not in (None, types.FinishReason.STOP):
            # Nothing but (perhaps) reasoning, and an abnormal stop: an error, as
            # unstreamed (ADK's comment: mirror the non-streaming path).
            yield LlmResponse(
                error_code=mapped_finish_reason,
                error_message=(
                    "OpenAI streaming response finished with reason"
                    f" {finish_reason!r} and no content."
                ),
                finish_reason=mapped_finish_reason,
                usage_metadata=_usage_metadata(usage),
            )
            return

        if reasoning_accumulated:  # reasoning
            parts.insert(0, _thought(reasoning_accumulated))  # reasoning

        # Final chunk: always a non-partial closing response, so the trailing
        # usage-only chunk still delivers usage_metadata (ADK's comment).
        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            partial=False,
            usage_metadata=_usage_metadata(usage),
            finish_reason=mapped_finish_reason,
        )


def _without_thoughts(llm_request: LlmRequest) -> LlmRequest:
    """The request without the model's earlier reasoning.

    ADK keeps a model's thought parts in the session, and its Chat Completions
    converter sends each one back as answer text ("Thought: ..."), so every
    later turn and every step of a tool loop would replay the reasoning. It is
    not part of the conversation: thought parts are left out of the request
    (the session keeps them), and a turn that was only reasoning is dropped.
    """
    contents: list[types.Content] = []
    changed = False
    for content in llm_request.contents:
        parts = content.parts or []
        kept = [
            part
            for part in parts
            if not part.thought or part.function_call or part.function_response
        ]
        if len(kept) == len(parts):
            contents.append(content)
            continue
        changed = True
        if kept:
            contents.append(content.model_copy(update={"parts": kept}))
    if not changed:
        return llm_request
    return llm_request.model_copy(update={"contents": contents})


def _reasoning_of(message: Any) -> str | None:
    """The reasoning text a delta or message carries beside its answer, if any."""
    for field in _REASONING_FIELDS:
        value = getattr(message, field, None)
        if isinstance(value, str) and value:
            return value
    return None


def _thought(text: str) -> types.Part:
    return types.Part(text=text, thought=True)


def _with_reasoning(response: LlmResponse, reasoning: str | None) -> LlmResponse:
    """``response`` with ``reasoning`` as a thought before its answer.

    An error response (no answer, an abnormal stop) is left as it is.
    """
    if not reasoning or response.error_code is not None:
        return response
    if response.content is None:
        response.content = types.Content(role="model", parts=[_thought(reasoning)])
    else:
        response.content.parts = [_thought(reasoning), *(response.content.parts or [])]
    return response

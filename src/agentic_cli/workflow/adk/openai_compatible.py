"""OpenRouter and OpenAI-compatible servers (vLLM and the like) on ADK.

``OpenAICompatibleLlm`` is ADK's Chat Completions ``OpenAILlm`` with five
additions:

- Reasoning text. These servers send a model's reasoning beside its answer,
  as ``reasoning`` (OpenRouter, newer vLLM) or ``reasoning_content`` (older
  vLLM), on streamed deltas and on the final message. ADK's class drops it;
  this one yields it as ``thought=True`` parts, before the answer, so the
  thinking box fills for open models. The thoughts stay out of later
  requests (``_without_thoughts``): ADK would send them as answer text.
- Reasoning details. OpenRouter also sends ``reasoning_details``, the
  structured reasoning it asks to have back on the assistant message that
  called a tool: Claude's signed thinking, Gemini's thought signatures,
  OpenAI's encrypted reasoning. Streamed, they come in pieces, merged here
  much as ChatOpenRouter merges them (``_merged`` says how they differ).
  They are kept on the thought part, as its
  ``thought_signature`` (JSON naming the model that wrote them), so the
  session holds them; a later request to that model carries them on that
  turn's assistant message (``_put_back_details``), and never one to
  another model.
- ``extra_body``, sent with every request: the provider's reasoning
  parameter (``workflow/adk/model_factory.py`` decides it).
- ``max_tokens`` may be None, which leaves the limit to the server; ADK's
  class always sends one (4096 by default).
- No field is sent as null. ADK's class sets ``tools``, ``tool_choice`` and
  ``response_format`` to None when a request has none, and the SDK writes
  None as null, which a server that validates the request (Synthetic)
  refuses.

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
from collections.abc import Sequence
from typing import Any, AsyncGenerator

from google.adk.integrations.openai import OpenAILlm
from google.adk.integrations.openai._openai_llm import (
    _content_to_openai_messages,
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

# The fields of a streamed reasoning detail whose pieces are joined: the
# reasoning text, and a summary's text. Any other field keeps its last value.
_JOINED_DETAIL_FIELDS = ("text", "summary")

# Each content of a request as ADK's converter sends it (its messages) and
# the reasoning details its assistant message carries, if any.
_Replayed = Sequence[tuple[list[dict[str, Any]], list[dict[str, Any]] | None]]


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
    # The request's contents with the details to put back (_put_back_details);
    # empty when no turn has any.
    _replayed: _Replayed = PrivateAttr(default=())

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
        request, details = _without_thoughts(llm_request, self.model)
        if any(details):
            call._replayed = [
                (_content_to_openai_messages(content), turn_details)
                for content, turn_details in zip(request.contents, details)
            ]
        async for response in super(OpenAICompatibleLlm, call).generate_content_async(
            request, stream=True
        ):
            yield response

    async def _generate_content_streaming(
        self, kwargs: dict[str, Any]
    ) -> AsyncGenerator[LlmResponse, None]:
        """Send the request ADK built: streamed or not, as the caller asked."""
        # An unset max_tokens, and the tools, tool_choice and response_format
        # of a request without them: left out, never sent as null. ADK passes
        # only body fields here, no request option such as timeout (where None
        # means no timeout at all).
        for field in [name for name, value in kwargs.items() if value is None]:
            del kwargs[field]
        if self.extra_body:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **self.extra_body}
        if self._replayed and not _put_back_details(kwargs["messages"], self._replayed):
            logger.warning("reasoning_details_not_sent", model=self.model)
        if self._streamed:
            async for response in self._stream(kwargs):
                yield response
            return
        response = await self._openai_client.chat.completions.create(**kwargs)
        message = response.choices[0].message if response.choices else None
        yield _with_reasoning(
            _response_to_llm_response(response),
            _reasoning_of(message),
            _signature(self.model, _details_of(message)),
        )

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
        details_accumulated: list[dict[str, Any]] = []  # reasoning
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

            details_accumulated += _details_of(delta) or []  # reasoning
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

        signature = _signature(self.model, _merged(details_accumulated))  # reasoning
        if reasoning_accumulated or signature:  # reasoning
            thought = _thought(reasoning_accumulated or None, signature)  # reasoning
            parts.insert(0, thought)  # reasoning

        # Final chunk: always a non-partial closing response, so the trailing
        # usage-only chunk still delivers usage_metadata (ADK's comment).
        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            partial=False,
            usage_metadata=_usage_metadata(usage),
            finish_reason=mapped_finish_reason,
        )


def _without_thoughts(
    llm_request: LlmRequest, model: str
) -> tuple[LlmRequest, list[list[dict[str, Any]] | None]]:
    """The request without the model's earlier reasoning, and the reasoning
    details to send back with each of its contents.

    ADK keeps a model's thought parts in the session, and its Chat Completions
    converter sends each one back as answer text ("Thought: ..."), so every
    later turn and every step of a tool loop would replay the reasoning. It is
    not part of the conversation: thought parts are left out of the request
    (the session keeps them), and a turn that was only reasoning is dropped.
    The reasoning details ``model`` wrote, which its thought parts hold, go
    back on that turn's message instead.
    """
    contents: list[types.Content] = []
    details: list[list[dict[str, Any]] | None] = []
    changed = False
    for content in llm_request.contents:
        parts = content.parts or []
        kept = [part for part in parts if not _is_thought(part)]
        if len(kept) == len(parts):
            contents.append(content)
            details.append(None)
            continue
        changed = True
        if kept:
            contents.append(content.model_copy(update={"parts": kept}))
            found = [
                detail
                for part in parts if _is_thought(part)
                for detail in _details_in(part, model) or []
            ]
            details.append(found or None)
    if not changed:
        return llm_request, details
    return llm_request.model_copy(update={"contents": contents}), details


def _is_thought(part: types.Part) -> bool:
    """Reasoning alone: a thought that is no function call or response."""
    return bool(part.thought) and not (part.function_call or part.function_response)


def _put_back_details(messages: list[dict[str, Any]], replayed: _Replayed) -> bool:
    """Put each turn's reasoning details on its assistant message.

    ``messages`` is the request ADK built: the system instruction, then each
    content's messages in order, as ``replayed`` holds them (each converted
    again with ADK's converter). False, and nothing changed, when they are
    not that: ADK built them some other way.
    """
    expected = [message for sent, _ in replayed for message in sent]
    start = len(messages) - len(expected)
    if start < 0 or messages[start:] != expected:
        return False
    for sent, details in replayed:
        for offset, message in enumerate(sent):
            if details and message["role"] == "assistant":
                assistant = messages[start + offset]
                assistant["reasoning_details"] = _without_ephemeral_ids(details)
        start += len(sent)
    return True


def _reasoning_of(message: Any) -> str | None:
    """The reasoning text a delta or message carries beside its answer, if any."""
    for field in _REASONING_FIELDS:
        value = getattr(message, field, None)
        if isinstance(value, str) and value:
            return value
    return None


def _details_of(message: Any) -> list[dict[str, Any]] | None:
    """The reasoning details a delta or message carries, if any (OpenRouter)."""
    value = getattr(message, "reasoning_details", None)
    if isinstance(value, list) and value:
        return [detail for detail in value if isinstance(detail, dict)]
    return None


def _merged(pieces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Streamed reasoning details, each entry whole again.

    OpenRouter streams an entry in pieces tagged with its ``index``, and the
    entry is the pieces joined in order. As in ChatOpenRouter's
    ``_merge_reasoning_details``, consecutive pieces of one type and index
    are one entry: its text is joined, every other field keeps its last value
    that is not null (a signature comes in the last piece), and a piece
    without an index is an entry of its own. Two differences, so that what
    goes back is what OpenRouter sent: a summary's text is joined too
    (``_JOINED_DETAIL_FIELDS``), and a field every piece leaves null stays,
    as null.
    """
    merged: list[dict[str, Any]] = []
    for piece in pieces:
        entry = (piece.get("type"), piece.get("index"))
        last = merged[-1] if merged else {}
        if entry[1] is None or entry != (last.get("type"), last.get("index")):
            merged.append(dict(piece))
            continue
        for field, value in piece.items():
            if field in _JOINED_DETAIL_FIELDS and isinstance(value, str):
                last[field] = (last.get(field) or "") + value
            elif value is not None or field not in last:
                last[field] = value
    return merged


def _signature(model: str, details: list[dict[str, Any]] | None) -> bytes | None:
    """What a thought part keeps of ``model``'s reasoning details."""
    if not details:
        return None
    return json.dumps({"model": model, "reasoning_details": details}).encode()


def _details_in(part: types.Part, model: str) -> list[dict[str, Any]] | None:
    """The reasoning details ``part`` keeps, if ``model`` wrote them."""
    if not part.thought_signature:
        return None
    try:
        kept = json.loads(part.thought_signature)
    except ValueError:  # not ours: another model class's signature
        return None
    # Another model's (a /settings switch within OpenRouter): it would take
    # them for its own, and Claude's signature means nothing to Gemini.
    if not isinstance(kept, dict) or kept.get("model") != model:
        return None
    details = kept.get("reasoning_details")
    return details if isinstance(details, list) else None


def _without_ephemeral_ids(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``details`` without OpenAI's reasoning item ids (``rs_``).

    They name items OpenAI did not store, which a later request may not
    refer to; ChatOpenRouter leaves them out too. Every other field goes
    back as it came.
    """
    def ephemeral(field: str, value: Any) -> bool:
        return field == "id" and isinstance(value, str) and value.startswith("rs_")

    return [
        {field: value for field, value in detail.items() if not ephemeral(field, value)}
        for detail in details
    ]


def _thought(text: str | None, signature: bytes | None = None) -> types.Part:
    return types.Part(text=text, thought=True, thought_signature=signature)


def _with_reasoning(
    response: LlmResponse, reasoning: str | None, signature: bytes | None = None
) -> LlmResponse:
    """``response`` with ``reasoning`` as a thought before its answer, the
    thought keeping the reasoning details (``signature``).

    An error response (no answer, an abnormal stop) is left as it is.
    """
    if not (reasoning or signature) or response.error_code is not None:
        return response
    thought = _thought(reasoning, signature)
    if response.content is None:
        response.content = types.Content(role="model", parts=[thought])
    else:
        response.content.parts = [thought, *(response.content.parts or [])]
    return response

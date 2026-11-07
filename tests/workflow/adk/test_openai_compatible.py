"""``OpenAICompatibleLlm``: OpenRouter and vLLM-style servers on ADK, with their reasoning text.

A fake OpenAI-compatible server (``httpx.MockTransport`` behind the
``AsyncOpenAI`` client) answers a streamed request with SSE chunks and an
unstreamed one with a single completion. Reasoning arrives the way those
servers send it, as ``reasoning`` (OpenRouter, newer vLLM) or
``reasoning_content`` (older vLLM) beside the answer, and on OpenRouter with
``reasoning_details``, which go back to the model with its tool results.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any

import httpx
import pytest

pytest.importorskip("google.adk")

from google.adk.integrations.openai import OpenAILlm  # noqa: E402
from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.genai import types  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from agentic_cli.workflow.adk.openai_compatible import OpenAICompatibleLlm  # noqa: E402
from tests.workflow.adk.fakes import user  # noqa: E402

MODEL = "qwen/qwen3-235b-a22b"


class Server:
    """An OpenAI-compatible chat completions endpoint.

    A streamed request gets ``deltas`` as SSE chunks (then the finish reason
    and a usage chunk); an unstreamed one gets ``message`` as one completion.
    """

    def __init__(
        self,
        *,
        deltas: list[dict[str, Any]] = (),
        message: dict[str, Any] | None = None,
        finish_reason: str = "stop",
    ) -> None:
        self.deltas = list(deltas)
        self.message = message or {"role": "assistant", "content": "ok"}
        self.finish_reason = finish_reason
        self.requests: list[httpx.Request] = []

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]

    def client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key="r-test-key",
            base_url="https://compatible.test/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self._handle)),
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if json.loads(request.content).get("stream"):
            return self._stream()
        return httpx.Response(200, json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": MODEL,
            "choices": [
                {"index": 0, "message": self.message, "finish_reason": self.finish_reason}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        })

    def _stream(self) -> httpx.Response:
        def chunk(choices, **fields):
            return {
                "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0,
                "model": MODEL, "choices": choices, **fields,
            }

        chunks = [
            chunk([{"index": 0, "delta": delta, "finish_reason": None}]) for delta in self.deltas
        ]
        chunks.append(chunk([{"index": 0, "delta": {}, "finish_reason": self.finish_reason}]))
        chunks.append(
            chunk([], usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})
        )
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body.encode()
        )


def request() -> LlmRequest:
    return LlmRequest(model=MODEL, contents=[user("Hi")], config=types.GenerateContentConfig())


async def responses(model, *, stream: bool) -> list:
    return [r async for r in model.generate_content_async(request(), stream=stream)]


def compatible(server: Server, **fields) -> OpenAICompatibleLlm:
    return OpenAICompatibleLlm(model=MODEL, client=server.client(), **fields)


def parts(response) -> list[tuple[str, Any]]:
    """A response's parts as (kind, value) pairs, in order."""
    out = []
    for part in response.content.parts:
        if part.function_call is not None:
            call = part.function_call
            out.append(("call", (call.id, call.name, call.args)))
        elif part.thought:
            out.append(("thought", part.text))
        else:
            out.append(("text", part.text))
    return out


def tool_call_deltas() -> list[dict[str, Any]]:
    """A get_weather call whose arguments arrive in pieces."""
    return [
        {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                         "function": {"name": "get_weather", "arguments": ""}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": '{"city": '}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]},
    ]


TOOL_CALL_MESSAGE = {
    "role": "assistant",
    "content": None,
    "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}],
}


# --- reasoning text becomes thoughts, before the answer -------------------------------


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
async def test_streamed_reasoning_arrives_as_thoughts_before_the_answer(field):
    server = Server(deltas=[
        {"role": "assistant", field: "Let me "},
        {field: "think."},
        {"content": "The "},
        {"content": "answer."},
    ])

    *partials, final = await responses(compatible(server), stream=True)

    assert [parts(p) for p in partials] == [
        [("thought", "Let me ")], [("thought", "think.")], [("text", "The ")], [("text", "answer.")],
    ]
    assert all(p.partial for p in partials)
    assert parts(final) == [("thought", "Let me think."), ("text", "The answer.")]
    assert final.partial is False
    assert final.usage_metadata.total_token_count == 5


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
async def test_a_final_message_s_reasoning_arrives_as_a_thought_before_the_answer(field):
    server = Server(message={"role": "assistant", field: "Because.", "content": "Answer."})

    (response,) = await responses(compatible(server), stream=False)

    assert parts(response) == [("thought", "Because."), ("text", "Answer.")]
    assert response.usage_metadata.total_token_count == 5
    assert not server.bodies[0].get("stream")  # sent unstreamed, as asked


async def test_a_streamed_tool_call_after_reasoning():
    server = Server(
        deltas=[{"role": "assistant", "reasoning": "I need the weather."}, *tool_call_deltas()],
        finish_reason="tool_calls",
    )

    *partials, final = await responses(compatible(server), stream=True)

    assert parts(partials[0]) == [("thought", "I need the weather.")]
    assert [p.content.parts[0].function_call.name for p in partials[1:]] == ["get_weather"] * 3
    assert parts(final) == [
        ("thought", "I need the weather."),
        ("call", ("call_1", "get_weather", {"city": "Paris"})),
    ]
    assert final.finish_reason == types.FinishReason.STOP


async def test_an_unstreamed_tool_call_after_reasoning():
    server = Server(
        message={**TOOL_CALL_MESSAGE, "reasoning": "I need the weather."},
        finish_reason="tool_calls",
    )

    (response,) = await responses(compatible(server), stream=False)

    assert parts(response) == [
        ("thought", "I need the weather."),
        ("call", ("call_1", "get_weather", {"city": "Paris"})),
    ]


async def test_reasoning_alone_is_still_a_response():
    server = Server(message={"role": "assistant", "reasoning": "Hmm.", "content": None})

    (response,) = await responses(compatible(server), stream=False)

    assert parts(response) == [("thought", "Hmm.")]


@pytest.mark.parametrize("stream", [True, False])
async def test_a_reply_cut_off_before_any_answer_is_an_error_without_its_reasoning(stream):
    """As in ADK's class, no answer and an abnormal finish is an error."""
    server = Server(
        deltas=[{"role": "assistant", "reasoning": "Long thoughts"}],
        message={"role": "assistant", "reasoning": "Long thoughts", "content": None},
        finish_reason="length",
    )

    final = (await responses(compatible(server), stream=stream))[-1]

    assert final.error_code == types.FinishReason.MAX_TOKENS
    assert final.content is None


# --- the request -------------------------------------------------------------------


@pytest.mark.parametrize("stream", [True, False])
async def test_extra_body_is_sent_with_every_request(stream):
    server = Server(deltas=[{"content": "ok"}])
    model = compatible(server, extra_body={"reasoning": {"effort": "high"}})

    await responses(model, stream=stream)

    assert server.bodies[0]["reasoning"] == {"effort": "high"}


@pytest.mark.parametrize("stream", [True, False])
async def test_an_unset_max_tokens_is_left_to_the_server(stream):
    """ADK's class always sends a max_tokens (4096 by default)."""
    server = Server(deltas=[{"content": "ok"}])

    await responses(compatible(server), stream=stream)
    await responses(compatible(server, max_tokens=1000), stream=stream)

    unset, given = server.bodies
    assert "max_tokens" not in unset
    assert given["max_tokens"] == 1000


@pytest.mark.parametrize("stream", [True, False])
async def test_a_request_sends_no_null_fields(stream):
    """ADK's class sets ``tools``, ``tool_choice`` and ``response_format`` to
    None when the request has none, and the SDK writes None as JSON null: a
    server that validates the request (Synthetic) refuses a null
    ``response_format``."""
    server = Server(deltas=[{"content": "ok"}])

    await responses(compatible(server), stream=stream)

    (body,) = server.bodies
    assert sorted(key for key, value in body.items() if value is None) == []


@pytest.mark.parametrize("stream", [True, False])
async def test_a_request_s_tools_are_still_sent(stream):
    server = Server(deltas=[{"content": "ok"}])
    tool = types.Tool(function_declarations=[
        types.FunctionDeclaration(name="lookup", description="Look a word up.")
    ])
    asked = LlmRequest(
        model=MODEL, contents=[user("Hi")], config=types.GenerateContentConfig(tools=[tool])
    )

    _ = [r async for r in compatible(server).generate_content_async(asked, stream=stream)]

    (body,) = server.bodies
    assert [t["function"]["name"] for t in body["tools"]] == ["lookup"]
    assert body["tool_choice"] == "auto"


@pytest.mark.parametrize("stream", [True, False])
async def test_a_response_format_is_still_sent(stream):
    server = Server(deltas=[{"content": "{}"}], message={"role": "assistant", "content": "{}"})
    asked = LlmRequest(
        model=MODEL,
        contents=[user("Hi")],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    _ = [r async for r in compatible(server).generate_content_async(asked, stream=stream)]

    (body,) = server.bodies
    assert body["response_format"] == {"type": "json_object"}


async def test_concurrent_streamed_and_unstreamed_requests_keep_their_own_mode():
    import asyncio

    server = Server(deltas=[{"content": "a"}, {"content": "b"}])
    model = compatible(server)

    streamed, unstreamed = await asyncio.gather(
        responses(model, stream=True), responses(model, stream=False)
    )

    assert len(streamed) == 3 and streamed[0].partial
    assert len(unstreamed) == 1 and not unstreamed[0].partial
    assert sorted(bool(body.get("stream")) for body in server.bodies) == [False, True]


# --- the copied loop is still ADK's ---------------------------------------------------

# The hashes of ADK's OpenAILlm._generate_content_streaming, which
# openai_compatible.py copies, in each supported google-adk: 2.12.0, and 2.13.0,
# which builds a call's part in one expression (the copy follows it; both build
# the same part).
ADK_LOOP_SHA256 = {
    "46bdee101bd7155a34772afbb239045d29729e3dda6964f46e779cc24d88fdd3",
    "a229e0eb02e4e84d00b0bde40939c834f422b6fdf75edbaaf5e7e71badb27436",
}

PLAIN_STREAMS = {
    "text": dict(deltas=[{"role": "assistant", "content": "Hel"}, {"content": "lo."}]),
    "tool call": dict(deltas=tool_call_deltas(), finish_reason="tool_calls"),
    "text then two tool calls": dict(
        deltas=[
            {"role": "assistant", "content": "Checking."},
            *tool_call_deltas(),
            {"tool_calls": [{"index": 1, "id": "call_2", "type": "function",
                             "function": {"name": "get_time", "arguments": "{}"}}]},
        ],
        finish_reason="tool_calls",
    ),
    "cut off before any answer": dict(deltas=[{"role": "assistant"}], finish_reason="length"),
    "filtered": dict(deltas=[{"role": "assistant", "content": "Par"}], finish_reason="content_filter"),
}


@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "unstreamed"])
@pytest.mark.parametrize("case", PLAIN_STREAMS)
async def test_without_reasoning_the_responses_are_adk_s_own(case, stream):
    """Without reasoning text, OpenAICompatibleLlm yields exactly what ADK's
    OpenAILlm yields. An ADK upgrade that changes its streaming loop, which
    OpenAICompatibleLlm copies, fails here."""
    message = {"role": "assistant", "content": "Hello."}
    adk = Server(**PLAIN_STREAMS[case], message=message)
    ours = Server(**PLAIN_STREAMS[case], message=message)

    expected = await responses(OpenAILlm(model=MODEL, client=adk.client()), stream=stream)
    actual = await responses(compatible(ours), stream=stream)

    assert [r.model_dump() for r in actual] == [r.model_dump() for r in expected]


def test_adk_s_streaming_loop_is_the_one_copied():
    source = inspect.getsource(OpenAILlm._generate_content_streaming)

    assert hashlib.sha256(source.encode()).hexdigest() in ADK_LOOP_SHA256, (
        "ADK's OpenAILlm._generate_content_streaming changed. Copy the new loop into "
        "OpenAICompatibleLlm._stream (agentic_cli/workflow/adk/openai_compatible.py), "
        "keeping its reasoning lines, then add its hash to ADK_LOOP_SHA256."
    )


# --- earlier reasoning is not sent back -----------------------------------------------
#
# ADK keeps a model's thought parts in the session, and its Chat Completions
# converter would send each one back as answer text ("Thought: ..."): every
# later turn, and every step of a tool loop, would replay the reasoning.


def _model_turn(*parts: types.Part) -> types.Content:
    return types.Content(role="model", parts=list(parts))


def _history() -> LlmRequest:
    weather = {"id": "call_1", "name": "get_weather"}
    return LlmRequest(model=MODEL, config=types.GenerateContentConfig(), contents=[
        user("What is the weather in Paris?"),
        _model_turn(
            types.Part(text="I need the weather first.", thought=True),
            types.Part(function_call=types.FunctionCall(**weather, args={"city": "Paris"})),
        ),
        types.Content(role="user", parts=[types.Part(
            function_response=types.FunctionResponse(**weather, response={"result": "sunny"})
        )]),
        _model_turn(types.Part(text="Let me phrase it.", thought=True), types.Part(text="It is sunny.")),
        _model_turn(types.Part(text="Only thinking here.", thought=True)),
        user("Thanks. And tomorrow?"),
    ])


@pytest.mark.parametrize("stream", [False, True])
async def test_earlier_reasoning_is_not_sent_back(stream):
    server = Server()
    history = _history()

    [r async for r in compatible(server).generate_content_async(history, stream=stream)]

    body = server.bodies[0]
    assert "Thought" not in json.dumps(body)
    assistant = [m for m in body["messages"] if m["role"] == "assistant"]
    assert len(assistant) == 2  # the thinking-only turn sends nothing
    assert assistant[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert not assistant[0].get("content")
    assert assistant[1]["content"] == "It is sunny."
    # The caller's request, and so the session, keeps every thought.
    assert [len(c.parts) for c in history.contents] == [1, 2, 1, 2, 1, 1]


async def test_a_second_turn_does_not_replay_the_first_turn_s_reasoning():
    from google.adk import Runner
    from google.adk.agents import LlmAgent
    from google.adk.apps import App
    from google.adk.sessions import InMemorySessionService

    server = Server(message={
        "role": "assistant", "reasoning": "Let me think hard about this.", "content": "The answer.",
    })
    agent = LlmAgent(name="writer", model=compatible(server), instruction="Write.")
    runner = Runner(
        app=App(name="compatible_test", root_agent=agent), session_service=InMemorySessionService()
    )
    session = await runner.session_service.create_session(app_name="compatible_test", user_id="u")
    for text in ("First question", "Second question"):
        [e async for e in runner.run_async(user_id="u", session_id=session.id, new_message=user(text))]

    second = server.bodies[1]["messages"]
    assert [m["content"] for m in second if m["role"] == "assistant"] == ["The answer."]


# --- OpenRouter's reasoning details go back with the tool results ---------------------
#
# Beside its reasoning text OpenRouter sends ``reasoning_details``: Claude's
# signed thinking, Gemini's thought signatures, OpenAI's encrypted reasoning.
# The model needs them back, exactly as it sent them, on the assistant message
# that called a tool; the thought text itself still stays out of the request.

CLAUDE = "anthropic/claude-sonnet-4.5"
GEMINI = "google/gemini-3-pro-preview"

#: Claude's reasoning as OpenRouter sends it whole: its signed thinking, then
#: a redacted (encrypted) block.
SIGNED = [
    {"type": "reasoning.text", "text": "I need the weather.", "signature": "sig-1",
     "format": "anthropic-claude-v1", "index": 0},
    {"type": "reasoning.encrypted", "data": "c2VjcmV0", "format": "anthropic-claude-v1",
     "index": 1},
]


def signed_deltas() -> list[dict[str, Any]]:
    """``SIGNED`` streamed: the thinking in pieces of one entry (``index`` 0),
    its signature last, then the redacted block, then the call."""
    def piece(**fields):
        return {"type": "reasoning.text", "format": "anthropic-claude-v1", "index": 0,
                **fields}

    return [
        {"role": "assistant", "reasoning": "I need ",
         "reasoning_details": [piece(text="I need ")]},
        {"reasoning": "the weather.", "reasoning_details": [piece(text="the weather.")]},
        {"reasoning_details": [piece(text="", signature="sig-1"), SIGNED[1]]},
        *tool_call_deltas(),
    ]


class Scripted(Server):
    """A server answering each request with the next reply (``Server``'s keywords)."""

    def __init__(self, *replies: dict[str, Any]) -> None:
        super().__init__()
        self.replies = list(replies)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        reply = self.replies.pop(0)
        self.deltas = list(reply.get("deltas", ()))
        self.message = reply.get("message") or {"role": "assistant", "content": "ok"}
        self.finish_reason = reply.get("finish_reason", "stop")
        return super()._handle(request)


def calls_the_tool(**reasoning: Any) -> dict[str, Any]:
    """A reply that calls get_weather, streamed or not, ``reasoning`` on its message."""
    return dict(
        deltas=tool_call_deltas(), message={**TOOL_CALL_MESSAGE, **reasoning},
        finish_reason="tool_calls",
    )


SUNNY = dict(deltas=[{"role": "assistant", "content": "It is sunny."}],
             message={"role": "assistant", "content": "It is sunny."})


def get_weather(city: str) -> dict:
    """The weather in a city."""
    return {"result": "sunny"}


async def tool_loop(model: OpenAICompatibleLlm, *, stream: bool):
    """One turn of an agent on ``model`` through ADK's runner: it calls
    get_weather, then answers. Returns the session."""
    from google.adk import Runner
    from google.adk.agents import LlmAgent, RunConfig
    from google.adk.agents.run_config import StreamingMode
    from google.adk.apps import App
    from google.adk.sessions import InMemorySessionService

    agent = LlmAgent(name="writer", model=model, instruction="Write.", tools=[get_weather])
    app = App(name="compatible_test", root_agent=agent)
    runner = Runner(app=app, session_service=InMemorySessionService())
    session = await runner.session_service.create_session(app_name=app.name, user_id="u")
    config = RunConfig(streaming_mode=StreamingMode.SSE if stream else StreamingMode.NONE)
    asked = user("What is the weather in Paris?")
    [e async for e in runner.run_async(
        user_id="u", session_id=session.id, new_message=asked, run_config=config
    )]
    return await runner.session_service.get_session(
        app_name=app.name, user_id="u", session_id=session.id
    )


def on(model: str, server: Server) -> OpenAICompatibleLlm:
    """``model`` on OpenRouter, here ``server``."""
    return OpenAICompatibleLlm(model=model, client=server.client())


def replayed(body: dict[str, Any]) -> dict[str, Any]:
    """The request's one assistant message: the call the model made."""
    (assistant,) = [m for m in body["messages"] if m["role"] == "assistant"]
    return assistant


def thoughts(session) -> list[str | None]:
    return [
        part.text
        for event in session.events if event.content and event.author == "writer"
        for part in event.content.parts or () if part.thought
    ]


@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "unstreamed"])
async def test_reasoning_details_go_back_across_a_tool_call(stream):
    """The ADK twin of the LangChain adapter's test: Claude and Gemini through
    OpenRouter need their signed reasoning back with the tool result."""
    first = calls_the_tool(reasoning="I need the weather.", reasoning_details=SIGNED)
    server = Scripted({**first, "deltas": signed_deltas()}, SUNNY)

    session = await tool_loop(on(CLAUDE, server), stream=stream)

    call = replayed(server.bodies[1])
    assert call["reasoning_details"] == SIGNED  # merged, in order, every field
    assert call["tool_calls"][0]["id"] == "call_1"
    assert not call.get("content") and "reasoning" not in call  # no thought text
    assert "Thought" not in json.dumps(server.bodies[1])
    assert thoughts(session) == ["I need the weather."]  # the thinking box's text


async def test_reasoning_details_without_reasoning_text_go_back():
    """Gemini's thought signature can come alone, as an encrypted entry."""
    signature = [{"type": "reasoning.encrypted", "data": "Z2VtaW5p", "id": "call_1",
                  "format": "google-gemini-v1", "index": 0}]
    server = Scripted(calls_the_tool(reasoning_details=signature), SUNNY)

    session = await tool_loop(on(GEMINI, server), stream=False)

    assert replayed(server.bodies[1])["reasoning_details"] == signature
    assert thoughts(session) == [None]  # nothing for the thinking box


async def test_a_streamed_summary_is_joined_and_its_null_fields_kept():
    """A summary streams in pieces like the reasoning text; a field OpenRouter
    sends as null goes back as null."""
    def piece(summary: str) -> dict[str, Any]:
        return {"type": "reasoning.summary", "summary": summary, "id": None,
                "format": "openai-responses-v1", "index": 0}

    first = calls_the_tool()
    first["deltas"] = [
        {"role": "assistant", "reasoning_details": [piece("Look the ")]},
        {"reasoning_details": [piece("weather up.")]},
        *tool_call_deltas(),
    ]
    server = Scripted(first, SUNNY)

    await tool_loop(on("openai/gpt-5", server), stream=True)

    assert replayed(server.bodies[1])["reasoning_details"] == [piece("Look the weather up.")]


async def test_openai_s_ephemeral_reasoning_ids_are_not_sent_back():
    """OpenAI's ``rs_`` item ids name reasoning OpenRouter does not store
    (``store`` false); the details go back without them, as ChatOpenRouter
    sends them on Deep Agents."""
    encrypted = {"type": "reasoning.encrypted", "data": "gAAAA", "id": "rs_1",
                 "format": "openai-responses-v1", "index": 0}
    server = Scripted(calls_the_tool(reasoning_details=[encrypted]), SUNNY)

    await tool_loop(on("openai/gpt-5", server), stream=False)

    assert replayed(server.bodies[1])["reasoning_details"] == [
        {key: value for key, value in encrypted.items() if key != "id"}
    ]


@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "unstreamed"])
async def test_another_model_is_not_sent_them(stream):
    """A model switch within OpenRouter keeps the provider: the details go
    back only to the model that wrote them."""
    reply = calls_the_tool(reasoning="I need the weather.", reasoning_details=SIGNED)
    first = Scripted({**reply, "deltas": signed_deltas()})
    asked = LlmRequest(model=CLAUDE, contents=[user("What is the weather in Paris?")],
                       config=types.GenerateContentConfig())
    *_, response = [
        r async for r in on(CLAUDE, first).generate_content_async(asked, stream=stream)
    ]
    result = types.FunctionResponse(id="call_1", name="get_weather", response={"result": "sunny"})
    history = [
        *asked.contents,
        response.content,
        types.Content(role="user", parts=[types.Part(function_response=result)]),
    ]
    server = Server(deltas=SUNNY["deltas"], message=SUNNY["message"])

    for model in (GEMINI, CLAUDE):
        again = LlmRequest(model=model, contents=history, config=types.GenerateContentConfig())
        [r async for r in on(model, server).generate_content_async(again, stream=stream)]

    to_gemini, to_claude = (replayed(body) for body in server.bodies)
    assert "reasoning_details" not in to_gemini
    assert to_gemini["tool_calls"] == to_claude["tool_calls"]
    assert to_claude["reasoning_details"] == SIGNED


async def test_reasoning_text_alone_leaves_the_request_as_it_was():
    """A server that sends only reasoning text (vLLM, Synthetic, qwen on
    OpenRouter): the call goes back exactly as ADK sends it."""
    server = Scripted(calls_the_tool(reasoning="I need the weather."), SUNNY)

    await tool_loop(compatible(server), stream=False)

    assert replayed(server.bodies[1]) == {
        "role": "assistant",
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}],
    }


async def test_adk_builds_the_messages_one_content_at_a_time():
    """OpenAICompatibleLlm finds a model turn's assistant message in the
    request ADK built by converting each content again with ADK's own
    ``_content_to_openai_messages``: the system instruction, then each
    content's messages in order, a model turn's call in one assistant message.
    An ADK upgrade that builds the messages otherwise fails here."""
    from google.adk.integrations.openai._openai_llm import _content_to_openai_messages

    server = Server()
    contents = _history().contents
    asked = LlmRequest(model=MODEL, contents=contents,
                       config=types.GenerateContentConfig(system_instruction="Be brief."))

    adk = OpenAILlm(model=MODEL, client=server.client())
    [r async for r in adk.generate_content_async(asked)]

    assert server.bodies[0]["messages"] == [
        {"role": "system", "content": "Be brief."},
        *(m for content in contents for m in _content_to_openai_messages(content)),
    ]
    assert [m["role"] for m in _content_to_openai_messages(contents[1])] == ["assistant"]


async def test_a_request_adk_built_otherwise_goes_without_the_details(monkeypatch):
    """Should ADK build the messages some other way, the details are left out
    (with a warning) rather than put on the wrong message."""
    import structlog

    from agentic_cli.workflow.adk import openai_compatible

    server = Scripted(calls_the_tool(reasoning_details=SIGNED), SUNNY)
    monkeypatch.setattr(
        openai_compatible, "_content_to_openai_messages", lambda content: [{"role": "other"}]
    )

    with structlog.testing.capture_logs() as logs:
        await tool_loop(on(CLAUDE, server), stream=False)

    assert "reasoning_details" not in replayed(server.bodies[1])
    assert [log["event"] for log in logs if log["log_level"] == "warning"] == [
        "reasoning_details_not_sent"
    ]

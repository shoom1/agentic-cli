"""``OpenAICompatibleLlm``: OpenRouter and vLLM-style servers on ADK, with their reasoning text.

A fake OpenAI-compatible server (``httpx.MockTransport`` behind the
``AsyncOpenAI`` client) answers a streamed request with SSE chunks and an
unstreamed one with a single completion. Reasoning arrives the way those
servers send it, as ``reasoning`` (OpenRouter, newer vLLM) or
``reasoning_content`` (older vLLM) beside the answer.
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

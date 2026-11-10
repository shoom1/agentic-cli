"""Fake provider APIs for the ADK adapter tests, and the request a built model sends.

A built model keeps the client the factory gave it (its key, timeout and
retries); ``serve`` only points that client's HTTP at a fake API, so what the
tests read is the request ADK's own model class builds.
"""

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Callable
from types import ModuleType
from typing import Any

import httpx
from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions import InMemorySessionService
from google.genai import types

Answer = dict[str, Any] | httpx.Response | Callable[[httpx.Request], Any]


class FakeAPI:
    """One provider's API: records every request and answers with ``answer``.

    ``answer`` is a JSON payload, a response, or a function of the request
    returning either, or a coroutine of either (to hold a request open, as a
    slow provider would).
    """

    def __init__(self, answer: Answer) -> None:
        self.answer = answer
        self.requests: list[httpx.Request] = []

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]

    def sent_key(self, index: int = 0) -> str | None:
        """The API key request ``index`` carried: anthropic 1.x sends it as
        ``x-api-key``, 2.x as an ``Authorization`` bearer token."""
        headers = self.requests[index].headers
        if "x-api-key" in headers:
            return headers["x-api-key"]
        scheme, _, token = headers.get("authorization", "").partition(" ")
        return token if scheme.lower() == "bearer" else None

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handler(httpx))

    def http_client(self, library: ModuleType = httpx) -> Any:
        """An async client of ``library`` (``httpx`` or ``httpx2``) that sends here."""
        return library.AsyncClient(transport=library.MockTransport(self._handler(library)))

    def _handler(self, library: ModuleType) -> Callable[[Any], Any]:
        def respond(answer: Any) -> Any:
            if isinstance(answer, library.Response):
                return answer
            if isinstance(answer, httpx.Response):  # a client of another library asked
                return library.Response(
                    answer.status_code,
                    headers=answer.headers.multi_items(),
                    content=answer.content,
                )
            return library.Response(200, json=answer)

        def handle(request: Any) -> Any:
            self.requests.append(request)
            answer = self.answer(request) if callable(self.answer) else self.answer
            if inspect.isawaitable(answer):
                # The mock transport awaits what is not a response.
                async def later() -> Any:
                    return respond(await answer)

                return later()
            return respond(answer)

        return handle


def http_library(client: Any) -> ModuleType:
    """The HTTP package an SDK client (or client class) takes its ``http_client`` from.

    anthropic 1.x runs on ``httpx2`` and rejects ``httpx`` objects; openai, and
    anthropic before 1.0, take ``httpx``.
    """
    cls = client if isinstance(client, type) else type(client)
    base = importlib.import_module(f"{cls.__module__.partition('.')[0]}._base_client")
    return getattr(base, "httpx2", httpx)


def serve(model: Any, api: FakeAPI) -> None:
    """Send ``model``'s requests to ``api``, keeping its client's key, timeout and retries."""
    model.client = model.client.with_options(
        http_client=api.http_client(http_library(model.client))
    )


#: A local port nothing listens on.
DEAD_URL = "http://127.0.0.1:9"


def dead_base_urls(monkeypatch) -> None:
    """Point every provider SDK's base URL at ``DEAD_URL``: a client a test did
    not route fails at once instead of reaching a provider."""
    for name in (
        "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "GOOGLE_GEMINI_BASE_URL",
        "GOOGLE_VERTEX_BASE_URL", "OPENROUTER_BASE_URL",
    ):
        monkeypatch.setenv(name, DEAD_URL)


def route_anthropic(monkeypatch, api: FakeAPI) -> None:
    """Send every Anthropic client built from now on to ``api``, keeping its other arguments.

    Both bindings: the framework's ``anthropic.AsyncAnthropic`` and the one
    ADK's own ``AnthropicLlm`` module bound at import, which a native ADK YAML
    app's bare ``claude-*`` model builds. Each is replaced by a subclass, since
    ADK names the class in a type union at run time.
    """
    import anthropic
    from google.adk.models import anthropic_llm

    for module in (anthropic, anthropic_llm):
        real = module.AsyncAnthropic
        library = http_library(real)

        class Routed(real):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, _library: ModuleType = library, **kwargs: Any) -> None:
                super().__init__(*args, **{**kwargs, "http_client": api.http_client(_library)})

        monkeypatch.setattr(module, "AsyncAnthropic", Routed)


def anthropic_message(*blocks: dict[str, Any], model: str = "claude-sonnet-4-5") -> dict[str, Any]:
    """An Anthropic Messages API response; one text block by default."""
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": list(blocks) or [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }


def openai_response(text: str = "ok", *, model: str = "gpt-5.5") -> dict[str, Any]:
    """An OpenAI Responses API response holding one message."""
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 3,
            "output_tokens": 1,
            "total_tokens": 4,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def chat_completion(text: str = "ok", *, model: str = "qwen/qwen3-235b-a22b") -> dict[str, Any]:
    """An OpenAI Chat Completions response (OpenRouter, vLLM and the like)."""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


def user(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


def request_for(built: Any, text: str = "Hi") -> LlmRequest:
    """The request ADK's flow would build for an agent on ``built``: the agent's
    config, copied, with the planner's thinking config applied."""
    request = LlmRequest(
        model=built.model.model,
        contents=[user(text)],
        config=built.generate_content_config.model_copy(deep=True),
    )
    if built.planner is not None:
        built.planner.apply_thinking_config(request)
    return request


async def generate(built: Any, text: str = "Hi", *, stream: bool = False) -> list[Any]:
    """Every response ``built.model`` yields for one user message."""
    return [
        response
        async for response in built.model.generate_content_async(
            request_for(built, text), stream=stream
        )
    ]


async def run_agent(built: Any, text: str = "Hi", *, name: str = "writer") -> list[Any]:
    """Run one turn of an agent on ``built`` through ADK's own runner.

    The agent takes the model, planner and config exactly as the ADK manager
    gives them to ``LlmAgent``; ADK builds the request from there.
    """
    agent = LlmAgent(
        name=name,
        model=built.model,
        instruction="Write.",
        planner=built.planner,
        generate_content_config=built.generate_content_config,
    )
    runner = Runner(
        app=App(name="model_factory_test", root_agent=agent),
        session_service=InMemorySessionService(),
    )
    session = await runner.session_service.create_session(
        app_name="model_factory_test", user_id="u"
    )
    return [
        event
        async for event in runner.run_async(
            user_id="u", session_id=session.id, new_message=user(text)
        )
    ]

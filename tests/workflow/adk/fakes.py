"""Fake provider APIs for the ADK adapter tests, and the request a built model sends.

A built model keeps the client the factory gave it (its key, timeout and
retries); ``serve`` only points that client's HTTP at a fake API, so what the
tests read is the request ADK's own model class builds.
"""

from __future__ import annotations

import json
from collections.abc import Callable
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
    returning either.
    """

    def __init__(self, answer: Answer) -> None:
        self.answer = answer
        self.requests: list[httpx.Request] = []

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]

    def http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answer(request) if callable(self.answer) else self.answer
        return answer if isinstance(answer, httpx.Response) else httpx.Response(200, json=answer)


def serve(model: Any, api: FakeAPI) -> None:
    """Send ``model``'s requests to ``api``, keeping its client's key, timeout and retries."""
    model.client = model.client.with_options(http_client=api.http_client())


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

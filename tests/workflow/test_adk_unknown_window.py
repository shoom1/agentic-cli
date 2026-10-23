"""A switch to an endpoint model declared without its window, smaller than the conversation.

The budget assumes 128,000 tokens for such a model, so the summary call
sized to that room is refused by a model whose real window is 32,000. The
summary call is then sent once more, sized to the limit the provider's
error states (else half the window): the summary is written, and the
conversation goes on. Without that, every turn on the small model was
refused four times and ended in the overflow message, with ``/compact`` no
help either.
"""

from __future__ import annotations

import json

import httpx
import openai
import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import EventType
from tests.workflow.adk.fakes import FakeAPI, chat_completion
from tests.workflow.test_manager_models import route

SMALL_WINDOW = 32_000


class Server:
    """``local:big`` takes anything; ``local:small`` refuses a request over 32,000
    tokens (4 characters a token), saying so as vLLM does. Usage is the request's size."""

    def __init__(self) -> None:
        self.api = FakeAPI(self._answer)
        self.log: list[tuple[str, str, int, bool]] = []

    def _answer(self, request):
        body = json.loads(request.content)
        size = len(request.content) // 4
        kind = "summary" if b"Context Extraction Assistant" in request.content else "agent"
        refused = body["model"] == "small" and size > SMALL_WINDOW
        self.log.append((body["model"], kind, size, refused))
        if refused:
            return httpx.Response(400, json={"error": {
                "message": f"This model's maximum context length is {SMALL_WINDOW} tokens. "
                           f"However, you requested about {size} tokens.", "code": 400}})
        answer = chat_completion("A summary of the notes." if kind == "summary" else "Noted.")
        answer["usage"] = {"prompt_tokens": size, "completion_tokens": 5,
                           "total_tokens": size + 5}
        return answer


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    for variable in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    return tmp_path / "project"


async def test_after_a_switch_to_a_small_undeclared_window_the_conversation_goes_on(
    project, monkeypatch
):
    server = Server()
    route(monkeypatch, openai, "AsyncOpenAI", server.api)
    manager = GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt="You help.", include_state_tools=False)],
        settings=BaseSettings(
            session_store="memory", default_model="local:big",
            model_endpoints={"local": {"base_url": "http://localhost:8000/v1", "models": {
                "big": {"context_window": 1_000_000}, "small": {}}}},
        ),
    )
    note = "word " * 12_000  # 60,000 characters, about 15,000 tokens
    turns = []
    try:
        for n in range(1, 11):  # about 150,000 tokens on the big model
            [e async for e in manager.process(f"Note {n}: {note}", "u", session_id="s")]
        before_switch = len(server.log)
        await manager.reinitialize(model="local:small")
        for text in ["Next?", "And now?", "Still there?"]:
            turns.append([e async for e in manager.process(text, "u", session_id="s")])
    finally:
        await manager.cleanup()

    for events in turns:
        assert [e.content for e in events if e.type is EventType.TEXT] == ["Noted."]
        assert [e.content for e in events if e.type is EventType.ERROR] == []
    # The first summary on the small model is refused, then sent once more
    # sized to the limit the error stated, and written; nothing else is refused.
    summaries = [row for row in server.log[before_switch:] if row[1] == "summary"]
    assert [refused for *_, refused in summaries][:2] == [True, False]
    assert summaries[1][2] <= SMALL_WINDOW
    assert sum(refused for *_, refused in server.log[before_switch:]) == 1


@pytest.mark.parametrize("message, window, expected", [
    # vLLM, OpenRouter and OpenAI-compatible servers.
    ("This model's maximum context length is 32000 tokens. However, ...", 128_000, 32_000),
    ("maximum context length is 32,768 tokens", 128_000, 32_768),
    # Claude.
    ("prompt is too long: 250123 tokens > 200000 maximum", 1_000_000, 200_000),
    # Gemini.
    ("The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).",
     2_000_000, 1_048_576),
    # No limit stated: half the window.
    ("context_length_exceeded", 128_000, 64_000),
    # A stated limit no smaller than the window it was sized to: half the window.
    ("maximum context length is 200000 tokens", 128_000, 64_000),
])
def test_the_window_a_refused_summary_is_sized_to_next(message, window, expected):
    from agentic_cli.workflow.adk.compaction import _summary_window

    assert _summary_window(message, window) == expected

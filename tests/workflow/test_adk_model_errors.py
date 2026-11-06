"""A main agent's reply that is no answer ends its turn with an ``ERROR``.

ADK turns a Gemini reply with no parts and a finish reason other than STOP
(``SAFETY``, ``MALFORMED_FUNCTION_CALL``, ``MAX_TOKENS``), or a prompt Gemini
blocked, into a response with an error code and no content; a STOP reply with
no parts, or only thoughts, gets ``MODEL_RETURNED_NO_CONTENT``; a Claude
refusal or a reply cut at its token limit has only its finish reason. None of
them raises. Only a sub-agent's such reply, or a context overflow, became an
``ERROR``: the main agent's turn ended with no text and no error. A model
call that fails is not this: ADK records an error event for it too, and the
run then raises, which reports it once.
"""

from __future__ import annotations

import httpx
import pytest

pytest.importorskip("google.adk")

from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
import anthropic  # noqa: E402

from tests.workflow.adk.fakes import FakeAPI, anthropic_message  # noqa: E402
from tests.workflow.test_adk_context_cache import route_gemini  # noqa: E402
from tests.workflow.test_manager_models import KEYS, adk, keys, route, settings  # noqa: E402

USAGE = {"promptTokenCount": 12, "candidatesTokenCount": 0, "totalTokenCount": 12}


@pytest.fixture(autouse=True)
def _hermetic(tmp_path, monkeypatch):
    """A hermetic cwd and HOME, with only a Google key."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in KEYS:
        monkeypatch.delenv(variable, raising=False)
    keys(monkeypatch, "google", "anthropic")


def finished(reason: str, message: str | None = None, *parts: dict) -> dict:
    """A Gemini reply whose one candidate has ``parts`` (none by default)."""
    candidate = {"finishReason": reason, "index": 0}
    if parts:
        candidate["content"] = {"role": "model", "parts": list(parts)}
    if message is not None:
        candidate["finishMessage"] = message
    return {"candidates": [candidate], "usageMetadata": USAGE}


THOUGHT = {"text": "Let me think about this.", "thought": True}


def claude_stopped(stop_reason: str, *blocks: dict) -> dict:
    """A Claude reply that ended for ``stop_reason`` with ``blocks`` (none)."""
    message = anthropic_message(model="claude-sonnet-4-6")
    message.update(content=list(blocks), stop_reason=stop_reason)
    return message


BLOCKED = {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}, "usageMetadata": USAGE}


async def turn_on(monkeypatch, answer, model: str = "gemini-2.5-flash") -> list:
    """One turn of a main agent on ``model``, its model answering ``answer``."""
    if model.startswith("claude"):
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answer))
    else:
        route_gemini(monkeypatch, FakeAPI(answer))
    manager = adk([AgentConfig(name="solo", prompt="Answer.")],
                  settings(default_model=model))
    try:
        return [event async for event in manager.process("Hello", user_id="u", session_id="s")]
    finally:
        await manager.cleanup()


def errors(events) -> list:
    return [event for event in events if event.type is EventType.ERROR]


@pytest.mark.parametrize(
    "answer, model, code",
    [
        pytest.param(finished("SAFETY"), "gemini-2.5-flash", "SAFETY", id="safety"),
        pytest.param(BLOCKED, "gemini-2.5-flash", "PROHIBITED_CONTENT", id="blocked-prompt"),
        # Gemini's finishMessage does not reach ADK (google-genai drops it on
        # this path), so the code says why.
        pytest.param(
            finished("MALFORMED_FUNCTION_CALL", "Malformed function call: print(1"),
            "gemini-2.5-flash", "MALFORMED_FUNCTION_CALL", id="malformed-call",
        ),
        pytest.param(finished("MAX_TOKENS"), "gemini-2.5-flash", "MAX_TOKENS", id="max-tokens"),
        pytest.param(
            finished("MAX_TOKENS", None, THOUGHT), "gemini-2.5-flash", "MAX_TOKENS",
            id="max-tokens-thinking",
        ),
        pytest.param(
            finished("STOP"), "gemini-2.5-flash", "MODEL_RETURNED_NO_CONTENT", id="stop-empty",
        ),
        pytest.param(
            finished("STOP", None, THOUGHT), "gemini-2.5-flash", "MODEL_RETURNED_NO_CONTENT",
            id="stop-thinking-only",
        ),
        pytest.param(claude_stopped("refusal"), "claude-sonnet-4-6", "SAFETY", id="claude-refusal"),
        pytest.param(
            claude_stopped("max_tokens"), "claude-sonnet-4-6", "MAX_TOKENS", id="claude-max-tokens",
        ),
    ],
)
async def test_no_answer_ends_the_turn_with_an_error(monkeypatch, answer, model, code):
    events = await turn_on(monkeypatch, answer, model)

    (error,) = errors(events)
    assert error.content.startswith(f"{'anthropic' if 'claude' in model else 'google'}:{model}"
                                    " returned no answer")
    assert error.metadata["error_code"] == code
    assert error.metadata["recoverable"] is False
    assert not [event for event in events if event.type is EventType.TEXT]


async def test_a_reply_cut_at_its_limit_with_text_is_an_answer(monkeypatch):
    """Text before the limit is the answer, shown as it is."""
    events = await turn_on(
        monkeypatch, finished("MAX_TOKENS", None, {"text": "The answer is 4"})
    )

    assert errors(events) == []
    assert [event.content for event in events if event.type is EventType.TEXT] == [
        "The answer is 4"
    ]


async def test_a_failed_model_call_is_reported_once_by_the_run(monkeypatch):
    """ADK's own error event for the failed call is not a second report."""
    refused = httpx.Response(
        400, json={"error": {"code": 400, "message": "bad request", "status": "INVALID_ARGUMENT"}}
    )

    with pytest.raises(Exception, match="bad request"):
        events = []
        route_gemini(monkeypatch, FakeAPI(refused))
        manager = adk([AgentConfig(name="solo", prompt="Answer.")],
                      settings(default_model="gemini-2.5-flash"))
        try:
            async for event in manager.process("Hello", user_id="u", session_id="s"):
                events.append(event)
        finally:
            await manager.cleanup()

    assert errors(events) == []

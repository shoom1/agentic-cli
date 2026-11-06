"""A main agent's answer that is no answer ends its turn with an ``ERROR``.

ADK turns a Gemini reply with no parts and a finish reason other than STOP
(``SAFETY``, ``MALFORMED_FUNCTION_CALL``, a ``MAX_TOKENS`` spent thinking), or
a prompt Gemini blocked, into a response with an error code and no content,
and raises nothing. Only a sub-agent's such reply, or a context overflow,
became an ``ERROR``: the main agent's turn ended with no text and no error.
A model call that fails is not this: ADK records an error event for it too,
and the run then raises, which reports it once.
"""

from __future__ import annotations

import httpx
import pytest

pytest.importorskip("google.adk")

from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI  # noqa: E402
from tests.workflow.test_adk_context_cache import route_gemini  # noqa: E402
from tests.workflow.test_manager_models import KEYS, adk, keys, settings  # noqa: E402

USAGE = {"promptTokenCount": 12, "candidatesTokenCount": 0, "totalTokenCount": 12}


@pytest.fixture(autouse=True)
def _hermetic(tmp_path, monkeypatch):
    """A hermetic cwd and HOME, with only a Google key."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in KEYS:
        monkeypatch.delenv(variable, raising=False)
    keys(monkeypatch, "google")


def finished(reason: str, message: str | None = None) -> dict:
    """A Gemini reply whose one candidate has no parts."""
    candidate = {"finishReason": reason, "index": 0}
    if message is not None:
        candidate["finishMessage"] = message
    return {"candidates": [candidate], "usageMetadata": USAGE}


BLOCKED = {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}, "usageMetadata": USAGE}


async def turn_on(monkeypatch, answer) -> list:
    """One turn of a Gemini main agent, its model answering ``answer``."""
    route_gemini(monkeypatch, FakeAPI(answer))
    manager = adk([AgentConfig(name="solo", prompt="Answer.")],
                  settings(default_model="gemini-2.5-flash"))
    try:
        return [event async for event in manager.process("Hello", user_id="u", session_id="s")]
    finally:
        await manager.cleanup()


def errors(events) -> list:
    return [event for event in events if event.type is EventType.ERROR]


@pytest.mark.parametrize(
    "answer, said",
    [
        pytest.param(finished("SAFETY"), "SAFETY", id="safety"),
        pytest.param(BLOCKED, "PROHIBITED_CONTENT", id="blocked-prompt"),
        # Gemini's finishMessage does not reach ADK (google-genai drops it on
        # this path), so the reason is the code.
        pytest.param(
            finished("MALFORMED_FUNCTION_CALL", "Malformed function call: print(1"),
            "MALFORMED_FUNCTION_CALL", id="malformed-call",
        ),
        pytest.param(finished("MAX_TOKENS"), "MAX_TOKENS", id="max-tokens"),
    ],
)
async def test_no_answer_ends_the_turn_with_an_error(monkeypatch, answer, said):
    events = await turn_on(monkeypatch, answer)

    (error,) = errors(events)
    assert error.content == f"google:gemini-2.5-flash returned no answer: {said}"
    assert error.metadata["recoverable"] is False
    assert not [event for event in events if event.type is EventType.TEXT]


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

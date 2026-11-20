"""A main agent's reply that is no answer ends its turn with an ``ERROR``.

ADK turns a Gemini reply with no parts and a finish reason other than STOP
(``SAFETY``, ``MALFORMED_FUNCTION_CALL``, ``MAX_TOKENS``), or a prompt Gemini
blocked, into a response with an error code and no content; a STOP reply with
no parts, or only thoughts, gets ``MODEL_RETURNED_NO_CONTENT``; a Claude
refusal or a reply cut at its token limit with nothing in it has its finish
reason, and from google-adk 2.15 that reason as its error code too, with
ADK's own words for it. None of them raises. Only a sub-agent's such reply,
or a context overflow, became an ``ERROR``: the main agent's turn ended with
no text and no error. A model call that fails is not this: ADK records an
error event for it too, and the run then raises, which the manager reports
once, as the turn's ``ERROR``.
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
    # The fields of every failed turn's ERROR; no exception ended this one.
    assert error.metadata["rate_limited"] is False
    assert error.metadata["error_type"] == code
    assert not [event for event in events if event.type is EventType.TEXT]


@pytest.mark.parametrize("stop_reason, code", [("refusal", "SAFETY"), ("max_tokens", "MAX_TOKENS")])
async def test_a_claude_reply_with_nothing_in_it_reads_the_same_on_every_adk(
    monkeypatch, stop_reason, code
):
    """ADK's own words for it ("Claude stopped with SAFETY ('refusal') and no
    content.", from google-adk 2.15) say no more than the code: the error is
    the model's and "returned no answer", as on an older ADK."""
    events = await turn_on(monkeypatch, claude_stopped(stop_reason), "claude-sonnet-4-6")

    (error,) = errors(events)
    assert error.content == "anthropic:claude-sonnet-4-6 returned no answer"
    assert (error.metadata["error_code"], error.metadata["error_type"]) == (code, code)


@pytest.mark.parametrize("stop_reason, code", [("refusal", "SAFETY"), ("max_tokens", "MAX_TOKENS")])
def test_adks_claude_adapter_words_a_reply_with_nothing_in_it_so(stop_reason, code):
    """The contract the event processor reads: ADK's Claude adapter gives such a
    reply its finish reason, and either no error (before google-adk 2.15) or
    that reason as its error code, with words the processor leaves out."""
    from google.adk.models.anthropic_llm import message_to_generate_content_response

    from agentic_cli.workflow.adk.event_processor import ADK_CLAUDE_NO_CONTENT

    message = anthropic.types.Message.model_validate(claude_stopped(stop_reason))
    response = message_to_generate_content_response(message)

    assert response.finish_reason.value == code
    if response.error_code is None:
        assert response.error_message is None
    else:
        assert getattr(response.error_code, "value", response.error_code) == code
        assert ADK_CLAUDE_NO_CONTENT.fullmatch(response.error_message), response.error_message


async def test_a_sub_agents_reply_that_is_no_answer_fails_its_run_warned_with_its_code(
    monkeypatch,
):
    """The run fails, its parent reads why, and the user's warning carries
    the reply's code, as the main agent's failed turn does."""
    import json

    from agentic_cli.workflow.agents import RUN_FAILED

    def coordinator(request) -> dict:
        body = json.loads(request.content)
        answer = anthropic_message(model="claude-sonnet-4-6")
        if len(body["messages"]) == 1:
            answer.update(stop_reason="tool_use", content=[{
                "type": "tool_use", "id": "c1", "name": "task",
                "input": {"agent_type": "researcher", "title": "Find X", "objective": "Find X."},
            }])
        else:
            answer["content"] = [{"type": "text", "text": "The researcher could not."}]
        return answer

    route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(coordinator))
    route_gemini(monkeypatch, FakeAPI(finished("MALFORMED_FUNCTION_CALL")))
    manager = adk(
        [AgentConfig(name="coordinator", prompt="Delegate.", sub_agents=["researcher"]),
         AgentConfig(name="researcher", prompt="Find.", description="Finds.",
                     model="gemini-2.5-flash")],
        settings(default_model="claude-sonnet-4-6"),
    )
    try:
        events = [e async for e in manager.process("Find X.", user_id="u", session_id="s")]
    finally:
        await manager.cleanup()

    (warning,) = errors(events)
    assert warning.content.startswith(
        "researcher-1 failed: google:gemini-2.5-flash returned no answer"
    )
    assert (warning.metadata["recoverable"], warning.metadata["error_code"]) == (
        True, "MALFORMED_FUNCTION_CALL"
    )
    (record,) = manager.agent_runs.records(manager.session_ref("s", "u"))
    assert record.status == RUN_FAILED


async def test_a_failing_state_read_keeps_a_no_answer_runs_own_error(monkeypatch):
    """The run's child session is read for its plan and tasks once it ended;
    a read that raises after a no-answer reply leaves them empty and the run
    failed with the reply's error and code, not the store's."""
    import json

    from agentic_cli.workflow.agents import RUN_FAILED

    def coordinator(request) -> dict:
        body = json.loads(request.content)
        answer = anthropic_message(model="claude-sonnet-4-6")
        if len(body["messages"]) == 1:
            answer.update(stop_reason="tool_use", content=[{
                "type": "tool_use", "id": "c1", "name": "task",
                "input": {"agent_type": "researcher", "title": "Find X", "objective": "Find X."},
            }])
        else:
            answer["content"] = [{"type": "text", "text": "The researcher could not."}]
        return answer

    route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(coordinator))
    route_gemini(monkeypatch, FakeAPI(finished("MALFORMED_FUNCTION_CALL")))
    manager = adk(
        [AgentConfig(name="coordinator", prompt="Delegate.", sub_agents=["researcher"]),
         AgentConfig(name="researcher", prompt="Find.", description="Finds.",
                     model="gemini-2.5-flash")],
        settings(default_model="claude-sonnet-4-6"),
    )
    await manager._ensure_initialized()
    service = manager._session_service
    read = service.get_session

    async def locked(**kwargs):
        config = kwargs.get("config")
        if kwargs["app_name"].endswith("--agents") and config is not None \
                and config.num_recent_events == 0:
            raise RuntimeError("database is locked")
        return await read(**kwargs)

    service.get_session = locked
    try:
        events = [e async for e in manager.process("Find X.", user_id="u", session_id="s")]
    finally:
        await manager.cleanup()

    (warning,) = errors(events)
    assert warning.content.startswith(
        "researcher-1 failed: google:gemini-2.5-flash returned no answer"
    )
    assert warning.metadata["error_code"] == "MALFORMED_FUNCTION_CALL"
    (record,) = manager.agent_runs.records(manager.session_ref("s", "u"))
    assert (record.status, record.plan, record.tasks) == (RUN_FAILED, "", [])


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
    """ADK's own error event for the failed call is not a second report: the
    turn's one ``ERROR`` is the run's failure, and the turn raises nothing."""
    refused = httpx.Response(
        400, json={"error": {"code": 400, "message": "bad request", "status": "INVALID_ARGUMENT"}}
    )

    events = await turn_on(monkeypatch, refused)

    (error,) = errors(events)
    assert "bad request" in error.content
    assert error.metadata["recoverable"] is False
    assert error.metadata["error_type"] == "ClientError"
    assert error.metadata["rate_limited"] is False

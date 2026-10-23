"""On ADK a provider's context overflow is recovered once, or ends in one clear error.

A model call the provider refuses as too long (``is_context_overflow``: each
provider's own status, code or wording) reaches ``OverflowPlugin``
(``workflow/adk/overflow_plugin.py``). For the main agent it summarizes the
conversation now (a compaction event, as before a model call) and retries
the request once with its oldest turns left out and the summary folded in;
for a sub-agent it retries with the oldest turns left out. A retry that
overflows too, or a request with nothing to leave out, ends the main agent's
turn with one ``ERROR`` (``CONTEXT_OVERFLOW``): "This conversation no longer
fits <model>'s context window. Use /compact or /new.", and no reply is
stored. A sub-agent answers its coordinator that it could not do the task,
and the user gets one warning. Any other error is ADK's as before.

Turns run the real manager on ``FakeAPI``, one test per provider's way of
saying it: Claude's 400 and 413, OpenAI's ``context_length_exceeded``,
OpenRouter's integer 400 and Gemini's 400.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import anthropic
import httpx
import openai
import pytest

pytest.importorskip("google.adk")

from google.adk import Runner  # noqa: E402
from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.apps import App  # noqa: E402
from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.adk.plugins.base_plugin import BasePlugin  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402
from pydantic import PrivateAttr  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.models import ProviderKind  # noqa: E402
from agentic_cli.workflow.adk.fit_guard_plugin import FIT_NOTE  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.context import OVERFLOW_ERROR_CODE, overflow_message  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.summary_prompt import SUMMARY_PREFIX  # noqa: E402
from tests.workflow.adk.fakes import (  # noqa: E402
    FakeAPI,
    anthropic_message,
    chat_completion,
    openai_response,
)
from tests.workflow.test_adk_context_cache import route_gemini  # noqa: E402
from tests.workflow.test_manager_models import route  # noqa: E402


def gemini_answer(text: str) -> dict:
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]},
                        "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5,
                          "totalTokenCount": 15},
    }


#: Each provider: its model, the environment key, how it answers, and how a
#: request is routed to a FakeAPI.
PROVIDERS = {
    "anthropic": ("claude-sonnet-4-6", "ANTHROPIC_API_KEY",
                  lambda text: anthropic_message({"type": "text", "text": text}),
                  lambda mp, api: route(mp, anthropic, "AsyncAnthropic", api)),
    "openai": ("openai:gpt-5.5", "OPENAI_API_KEY", openai_response,
               lambda mp, api: route(mp, openai, "AsyncOpenAI", api)),
    "openrouter": ("openrouter:qwen/qwen3-235b-a22b", "OPENROUTER_API_KEY", chat_completion,
                   lambda mp, api: route(mp, openai, "AsyncOpenAI", api)),
    "gemini": ("gemini-2.5-flash", "GOOGLE_API_KEY", gemini_answer,
               lambda mp, api: route_gemini(mp, api)),
}

#: Each provider's overflow, as its API sends it (the published wording).
SHAPES = {
    "claude-400": ("anthropic", 400, {"type": "error", "error": {
        "type": "invalid_request_error",
        "message": "prompt is too long: 208310 tokens > 200000 maximum"}}),
    "claude-413": ("anthropic", 413, {"type": "error", "error": {
        "type": "request_too_large",
        "message": "Request exceeds the maximum allowed number of bytes."}}),
    "openai": ("openai", 400, {"error": {
        "message": "Your input exceeds the context window of this model. Please adjust your "
                   "input and try again.", "type": "invalid_request_error", "param": "input",
        "code": "context_length_exceeded"}}),
    "openrouter": ("openrouter", 400, {"error": {
        "message": "This endpoint's maximum context length is 131072 tokens. However, you "
                   "requested about 150000 tokens (150000 of text input).", "code": 400}}),
    "gemini": ("gemini", 400, {"error": {
        "code": 400, "message": "The input token count (1200000) exceeds the maximum number "
                                "of tokens allowed (1048576).", "status": "INVALID_ARGUMENT"}}),
}


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    for variable in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
                     "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    return tmp_path / "project"


class Scripted(FakeAPI):
    """A provider answering from ``script``, in order."""

    def __init__(self, *script) -> None:
        self.script = list(script)
        super().__init__(lambda request: self.script.pop(0))

    @property
    def sent(self) -> list[str]:
        return [request.content.decode() for request in self.requests]


def on(monkeypatch, provider: str, *script) -> tuple[Scripted, GoogleADKWorkflowManager]:
    model, key, _, routing = PROVIDERS[provider]
    monkeypatch.setenv(key, "test-key")
    api = Scripted(*script)
    routing(monkeypatch, api)
    manager = GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt="You help.", include_state_tools=False)],
        settings=BaseSettings(session_store="memory", default_model=model,
                              prompt_caching_enabled=False),
    )
    return api, manager


def answer(provider: str, text: str):
    return PROVIDERS[provider][2](text)


async def turn(manager, text: str) -> list:
    return [event async for event in manager.process(text, "u", session_id="s")]


def texts(events) -> list[str]:
    return [event.content for event in events if event.type is EventType.TEXT]


def errors(events) -> list:
    return [event for event in events if event.type is EventType.ERROR]


async def stored(manager):
    session = await manager._session_service.get_session(
        app_name=manager.app_name, user_id="u", session_id="s"
    )
    return session.events


@pytest.mark.parametrize("shape", list(SHAPES))
class TestEachProvidersOverflow:
    async def test_it_is_retried_once_without_the_oldest_turns(
        self, project, monkeypatch, shape
    ):
        provider, status, body = SHAPES[shape]
        api, manager = on(monkeypatch, provider, answer(provider, "A1."),
                          answer(provider, "A2."), httpx.Response(status, json=body),
                          answer(provider, "A3."))
        try:
            for text in ["One.", "Two."]:
                await turn(manager, text)
            third = await turn(manager, "Three?")
        finally:
            await manager.cleanup()

        assert texts(third) == ["A3."]
        assert errors(third) == []
        assert len(api.requests) == 4  # the refused request, and one retry
        refused, retry = api.sent[2], api.sent[3]
        assert "One." in refused and FIT_NOTE not in refused
        assert "One." not in retry and FIT_NOTE in retry and "Three?" in retry

    async def test_a_second_overflow_is_one_clear_error_and_no_reply(
        self, project, monkeypatch, shape
    ):
        provider, status, body = SHAPES[shape]
        refused = httpx.Response(status, json=body)
        api, manager = on(monkeypatch, provider, answer(provider, "A1."),
                          answer(provider, "A2."), refused, httpx.Response(status, json=body))
        try:
            for text in ["One.", "Two."]:
                await turn(manager, text)
            third = await turn(manager, "Three?")
            model = str(manager.agent_models["solo"].ref)
            events = await stored(manager)
        finally:
            await manager.cleanup()

        assert len(api.requests) == 4
        assert texts(third) == []
        (error,) = errors(third)
        assert error.content == overflow_message(model)
        assert error.content.endswith("Use /compact or /new.")
        assert error.metadata == {"recoverable": False, "error_code": OVERFLOW_ERROR_CODE}
        # The main agent's last event is the error, holding no reply.
        last = [e for e in events if e.author == "solo"][-1]
        assert (last.error_code, last.content) == (OVERFLOW_ERROR_CODE, None)


class TestOtherErrors:
    async def test_a_400_that_is_no_overflow_raises_as_before(self, project, monkeypatch):
        refused = httpx.Response(400, json={"type": "error", "error": {
            "type": "invalid_request_error", "message": "max_tokens: 100000 > 64000"}})
        api, manager = on(monkeypatch, "anthropic", refused)
        try:
            with pytest.raises(anthropic.BadRequestError, match="max_tokens"):
                await turn(manager, "One.")
        finally:
            await manager.cleanup()

        assert len(api.requests) == 1  # no retry

    async def test_a_retry_that_fails_otherwise_is_the_turns_error(self, project, monkeypatch):
        """The provider's own error, as for any failed call: ADK's plugin
        manager wraps what a plugin raises, and the manager unwraps it."""
        provider, status, body = SHAPES["claude-400"]
        unavailable = httpx.Response(400, json={"type": "error", "error": {
            "type": "invalid_request_error", "message": "max_tokens: 100000 > 64000"}})
        api, manager = on(monkeypatch, provider, answer(provider, "A1."),
                          httpx.Response(status, json=body), unavailable)
        try:
            await turn(manager, "One.")
            with pytest.raises(anthropic.BadRequestError, match="max_tokens"):
                await turn(manager, "Two?")
        finally:
            await manager.cleanup()

        assert len(api.requests) == 3

    def test_only_the_overflow_handlers_wrapper_is_unwrapped(self):
        """Another plugin's failure stays ADK's error, its cause chained."""
        from agentic_cli.workflow.adk.manager import _plugin_cause

        cause = anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
        ours = RuntimeError("Error in plugin 'context_overflow' during "
                            "'on_model_error_callback' callback: Connection error.")
        ours.__cause__ = cause
        other = RuntimeError("Error in plugin 'llm_logging' during 'after_model_callback' "
                             "callback: Connection error.")
        other.__cause__ = cause

        assert _plugin_cause(ours) is cause
        assert _plugin_cause(other) is other
        assert _plugin_cause(RuntimeError("Error in plugin 'context_overflow'")).args

    async def test_a_request_with_nothing_to_leave_out_is_the_clear_error_at_once(
        self, project, monkeypatch
    ):
        provider, status, body = SHAPES["claude-400"]
        api, manager = on(monkeypatch, provider, httpx.Response(status, json=body))
        try:
            first = await turn(manager, "One huge turn.")
        finally:
            await manager.cleanup()

        assert len(api.requests) == 1
        (error,) = errors(first)
        assert error.metadata["error_code"] == OVERFLOW_ERROR_CODE

    async def test_an_overflow_reported_in_a_response_is_the_clear_error(
        self, project, monkeypatch
    ):
        """OpenAI Responses can answer 200 with ``status: failed``: no exception,
        an error response instead."""
        failed = openai_response("")
        failed.update(status="failed", output=[], error={
            "code": "context_length_exceeded",
            "message": "Your input exceeds the context window of this model.",
        })
        api, manager = on(monkeypatch, "openai", answer("openai", "A1."), failed)
        try:
            await turn(manager, "One.")
            second = await turn(manager, "Two?")
        finally:
            await manager.cleanup()

        assert len(api.requests) == 2  # no retry
        (error,) = errors(second)
        assert error.content == overflow_message("openai:gpt-5.5")
        assert error.metadata["error_code"] == OVERFLOW_ERROR_CODE


# --- the main agent: a summary first ---------------------------------------------------------

LONG = "word " * 2_400
SUMMARY = "The user wrote twice at length."


def claude_says(text: str, prompt_tokens: int = 4_000) -> dict:
    message = anthropic_message({"type": "text", "text": text})
    message["usage"] = {"input_tokens": prompt_tokens, "output_tokens": 5}
    return message


def blocks(request) -> list[list[str]]:
    """Each message's text blocks."""
    return [
        [block["text"] for block in message["content"] if block["type"] == "text"]
        for message in json.loads(request.content)["messages"]
    ]


class TestTheMainAgentsOverflow:
    async def test_it_is_summarized_now_and_the_retry_carries_the_summary(
        self, project, monkeypatch
    ):
        provider, status, body = SHAPES["claude-400"]
        script = [claude_says("A1."), claude_says("A2."), httpx.Response(status, json=body),
                  claude_says("A3."), claude_says("A4.")]
        asked: list[str] = []

        def claude(request):
            if "system" not in json.loads(request.content):
                asked.append("summary")
                return claude_says(SUMMARY, 7_000)
            asked.append("solo")
            return script.pop(0)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        api = FakeAPI(claude)
        route(monkeypatch, anthropic, "AsyncAnthropic", api)
        # A 16,000-token ceiling keeps 1,920 tokens: the first turn is worth summarizing.
        manager = GoogleADKWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt="You help.",
                                       include_state_tools=False)],
            settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6",
                                  context_max_tokens=16_000, prompt_caching_enabled=False),
        )
        try:
            for text in [f"One: {LONG}", f"Two: {LONG}"]:
                await turn(manager, text)
            third = await turn(manager, "Three?")
            fourth = await turn(manager, "Four?")
            summaries = [e for e in await stored(manager) if e.actions and e.actions.compaction]
        finally:
            await manager.cleanup()

        assert asked == ["solo", "solo", "solo", "summary", "solo", "solo"]
        assert len(summaries) == 1
        summarized = f"{SUMMARY_PREFIX}\n\n{SUMMARY}"
        retry = blocks(api.requests[4])
        assert retry[0] == [summarized, FIT_NOTE, f"Two: {LONG}"]
        assert [e.type for e in third][:3] == [
            EventType.CONTEXT_SUMMARIZING, EventType.LLM_USAGE, EventType.CONTEXT_TRIMMED
        ]
        # The summary the overflow asked for is the main agent's, as any summary.
        summary_usage = third[1]
        assert (summary_usage.metadata["purpose"], summary_usage.metadata["agent"],
                summary_usage.metadata["context_trigger"],
                summary_usage.metadata["window_known"]) == ("summarization", "solo", 16_000, True)
        assert texts(third) == ["A3."]
        # From then on the stored summary stands for the first turn, in the
        # first kept user message.
        assert blocks(api.requests[5])[0] == [summarized, f"Two: {LONG}"]
        assert texts(fourth) == ["A4."]


class RealWindow:
    """An endpoint model declared without a window (128,000 assumed) whose real
    window is 32,000 tokens: it refuses any request longer (4 characters a
    token), as vLLM does."""

    def __init__(self, real_window: int = 32_000) -> None:
        self.real = real_window
        self.api = FakeAPI(self._answer)
        self.log: list[tuple[str, int, bool]] = []

    def _answer(self, request):
        size = len(request.content)
        kind = "summary" if b"Context Extraction Assistant" in request.content else "agent"
        refused = size / 4 > self.real
        self.log.append((kind, size, refused))
        if refused:
            return httpx.Response(400, json={"error": {
                "message": f"This model's maximum context length is {self.real} tokens. "
                           f"However, you requested about {size // 4} tokens.", "code": 400}})
        if kind == "summary":  # about 3,000 tokens
            return chat_completion("The user sent long notes. " + "detail " * 1_700)
        return chat_completion("Noted.")


class TestAnOverflowTheSummaryCanFix:
    async def test_the_retry_is_the_summary_and_the_turns_it_kept(self, project, monkeypatch):
        """Our budget for the model is wrong (no window declared): the main
        agent overflows, is summarized now, and the retry is what every later
        call sends, the summary and the turns it kept, not half the budget of
        the refused request with the summarized turns still in it."""
        server = RealWindow()
        route(monkeypatch, openai, "AsyncOpenAI", server.api)
        note = "word " * 1_600  # 8,000 characters, about 2,000 tokens
        manager = GoogleADKWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt="You help.",
                                       include_state_tools=False)],
            settings=BaseSettings(session_store="memory", default_model="local:unknown",
                                  model_endpoints={"local": {
                                      "base_url": "http://localhost:8000/v1",
                                      "models": {"unknown": {}}}}),
        )
        try:
            failed = []
            for n in range(1, 21):
                failed += errors(await turn(manager, f"Note {n}: {note}"))
            summaries = [e for e in await stored(manager) if e.actions and e.actions.compaction]
        finally:
            await manager.cleanup()

        assert failed == []
        assert len(summaries) >= 1
        # The retry (the next agent call after the first refused one, a
        # summary call between) is smaller than what was refused.
        first = next(i for i, (kind, _, was_refused) in enumerate(server.log)
                     if was_refused and kind == "agent")
        assert [kind for kind, _, _ in server.log[first + 1:first + 3]] == ["summary", "agent"]
        assert server.log[first + 2][1] < server.log[first][1]


class Holding(FakeAPI):
    """Claude answering agent calls from ``script`` and summary calls with
    ``summary``; the first summary call is held until ``release``."""

    def __init__(self, script: list, summary) -> None:
        self.script = list(script)
        self.summary = summary
        self.held = asyncio.Event()
        self.release = asyncio.Event()
        super().__init__(self._answer)

    def _answer(self, request):
        if "system" not in json.loads(request.content):
            if not self.held.is_set():
                return self._hold()
            return self.summary
        return self.script.pop(0)

    async def _hold(self):
        self.held.set()
        await self.release.wait()
        return self.summary


def small_ceiling() -> BaseSettings:
    """A 16,000-token ceiling keeps 1,920: a long first turn is worth a summary."""
    return BaseSettings(session_store="memory", default_model="claude-sonnet-4-6",
                        context_max_tokens=16_000, prompt_caching_enabled=False)


class TestTheSummaryAnOverflowAsksFor:
    async def test_ctrl_c_during_it_writes_nothing_and_the_next_turn_answers(
        self, project, monkeypatch
    ):
        _, status, body = SHAPES["claude-400"]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        api = Holding([claude_says("A1."), claude_says("A2."), httpx.Response(status, json=body),
                       claude_says("A3.")], claude_says(SUMMARY, 7_000))
        route(monkeypatch, anthropic, "AsyncAnthropic", api)
        manager = GoogleADKWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt="You help.",
                                       include_state_tools=False)],
            settings=small_ceiling(),
        )
        try:
            for text in [f"One: {LONG}", f"Two: {LONG}"]:
                await turn(manager, text)
            consumer = asyncio.create_task(turn(manager, "Three?"))
            await asyncio.wait_for(api.held.wait(), 5)
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):  # not a TimeoutError: no hang
                await asyncio.wait_for(consumer, 5)
            after_cancel = [e for e in await stored(manager)
                            if e.actions and e.actions.compaction]
            api.release.set()
            again = await asyncio.wait_for(turn(manager, "Three, again?"), 5)
        finally:
            api.release.set()
            await manager.cleanup()

        assert after_cancel == []
        assert texts(again) == ["A3."]

    async def test_one_that_fails_is_one_warning_and_the_retry_trims(
        self, project, monkeypatch
    ):
        provider, status, body = SHAPES["claude-400"]
        refused_summary = httpx.Response(400, json={"type": "error", "error": {
            "type": "invalid_request_error", "message": "max_tokens: 100000 > 64000"}})

        def claude(request):
            if "system" not in json.loads(request.content):
                return refused_summary
            return script.pop(0)

        script = [claude_says("A1."), claude_says("A2."), httpx.Response(status, json=body),
                  claude_says("A3.")]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        api = FakeAPI(claude)
        route(monkeypatch, anthropic, "AsyncAnthropic", api)
        manager = GoogleADKWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt="You help.",
                                       include_state_tools=False)],
            settings=small_ceiling(),
        )
        try:
            for text in [f"One: {LONG}", f"Two: {LONG}"]:
                await turn(manager, text)
            third = await turn(manager, "Three?")
        finally:
            await manager.cleanup()

        (warning,) = errors(third)
        assert (warning.metadata["error_code"], warning.metadata["recoverable"]) == (
            "summary_failed", True
        )
        assert texts(third) == ["A3."]
        assert FIT_NOTE in api.requests[-1].content.decode()


# --- a caching Gemini agent: the retry is built from the request as it was -----------------

GEMINI_CACHE = "cachedContents/fake-cache"


class TestACachingGeminiAgent:
    async def test_its_retry_is_the_request_before_the_cache_rewrote_it(
        self, project, monkeypatch
    ):
        """With an active cache, ADK's Gemini rewrites the request in place:
        the system instruction, the tools and the cached contents move into
        the cache. The retry is built from the request as the fit guard saw
        it, and sent without the cache."""
        _, status, body = SHAPES["gemini"]
        script = [gemini_answer("ok"), gemini_answer("ok"), httpx.Response(status, json=body),
                  gemini_answer("A3.")]

        def gemini(request):
            if request.url.path.endswith("/cachedContents"):
                return {"name": GEMINI_CACHE, "model": "models/gemini-2.5-flash"}
            answer = script.pop(0)
            if isinstance(answer, dict):  # a prompt that clears the cache's floor
                answer["usageMetadata"]["promptTokenCount"] = 5_000
            return answer

        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        api = FakeAPI(gemini)
        route_gemini(monkeypatch, api)
        manager = GoogleADKWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt="You help.",
                                       include_state_tools=False)],
            settings=BaseSettings(session_store="memory", default_model="gemini-2.5-flash"),
        )
        try:
            for text in ["Hello", "Again"]:
                await turn(manager, text)
            third = await turn(manager, "Three?")
        finally:
            await manager.cleanup()

        refused, retry = api.bodies[-2], api.bodies[-1]
        assert refused["cachedContent"] == GEMINI_CACHE  # the cache was in use
        assert "cachedContent" not in retry
        assert retry["systemInstruction"]["parts"][0]["text"].startswith("You help.")
        assert [part["text"] for content in retry["contents"] for part in content["parts"]] == [
            FIT_NOTE, "Again", "ok", "Three?"
        ]
        assert texts(third) == ["A3."]


# --- a sub-agent: its answer, and one warning ------------------------------------------------


def team(context: str | None) -> list[AgentConfig]:
    return [
        AgentConfig(name="coordinator", prompt="You coordinate.", sub_agents=["researcher"],
                    description="Leads.", include_state_tools=False),
        AgentConfig(name="researcher", prompt="You research.", description="Finds facts.",
                    context=context, include_state_tools=False),
    ]


def delegate() -> dict:
    message = claude_says("")
    message["content"] = [{"type": "tool_use", "id": "toolu_c1", "name": "researcher",
                           "input": {"request": "Find X."}}]
    message["stop_reason"] = "tool_use"
    return message


class Team(FakeAPI):
    """Claude for a coordinator and its researcher, told apart by their prompts."""

    def __init__(self, coordinator: list, researcher: list) -> None:
        self.scripts = {"You coordinate.": list(coordinator), "You research.": list(researcher)}
        self.asked: list[str] = []
        super().__init__(self._answer)

    def _answer(self, request):
        system = json.dumps(json.loads(request.content).get("system"))
        prompt = next(p for p in self.scripts if p in system)
        self.asked.append(prompt)
        return self.scripts[prompt].pop(0)


class TestASubAgentsOverflow:
    async def test_it_answers_that_it_could_not_and_the_user_is_warned_once(
        self, project, monkeypatch
    ):
        _, status, body = SHAPES["claude-400"]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        api = Team(coordinator=[delegate(), claude_says("The researcher could not.")],
                   researcher=[httpx.Response(status, json=body)])
        route(monkeypatch, anthropic, "AsyncAnthropic", api)
        manager = GoogleADKWorkflowManager(
            agent_configs=team("fresh"),
            settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6",
                                  prompt_caching_enabled=False),
        )
        try:
            events = await turn(manager, "Find X, please.")
        finally:
            await manager.cleanup()

        model = "anthropic:claude-sonnet-4-6"
        # Its brief alone is the request: nothing to leave out, no retry.
        assert api.asked == ["You coordinate.", "You research.", "You coordinate."]
        (error,) = errors(events)
        assert error.content == f"researcher: the request no longer fits {model}'s context window."
        assert error.metadata == {"recoverable": True, "error_code": OVERFLOW_ERROR_CODE,
                                  "agent": "researcher"}
        result = json.loads(api.requests[2].content)["messages"][-1]["content"][0]
        assert f"I could not do this task: the request no longer fits {model}'s context window." \
            in json.dumps(result)
        assert texts(events)[-1] == "The researcher could not."

    async def test_an_inheriting_one_is_retried_without_the_oldest_turns(
        self, project, monkeypatch
    ):
        _, status, body = SHAPES["claude-400"]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        api = Team(
            coordinator=[claude_says("Noted."), claude_says("Noted again."), delegate(),
                         claude_says("X is 42.")],
            researcher=[httpx.Response(status, json=body), claude_says("X is 42.")],
        )
        route(monkeypatch, anthropic, "AsyncAnthropic", api)
        # The coordinator's first turn would be worth a summary (a 16,000-token
        # ceiling keeps 1,920): a sub-agent's overflow must not write one.
        manager = GoogleADKWorkflowManager(
            agent_configs=team("inherit"),
            settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6",
                                  context_max_tokens=16_000, prompt_caching_enabled=False),
        )
        try:
            await turn(manager, f"One: {LONG}")
            await turn(manager, f"Two: {LONG}")
            events = await turn(manager, "Find X, please.")
            summaries = [e for e in await stored(manager) if e.actions and e.actions.compaction]
        finally:
            await manager.cleanup()

        refused, retry = (r.content.decode() for r in api.requests[3:5])
        assert "One: word" in refused
        assert "One: word" not in retry and FIT_NOTE in retry and "Find X." in retry
        assert errors(events) == []
        assert summaries == []  # a sub-agent never summarizes the shared session
        assert texts(events)[-1] == "X is 42."
        # Its retry left turns out: reported, once, as the fit guard reports.
        (trimmed,) = [e for e in events if e.type is EventType.CONTEXT_TRIMMED]
        assert (trimmed.metadata["source"], trimmed.metadata["agent"]) == (
            "fit_guard", "researcher"
        )
        assert trimmed.metadata["turns_dropped"] >= 1

    async def test_an_inheriting_ones_second_overflow_is_still_one_warning(
        self, project, monkeypatch
    ):
        _, status, body = SHAPES["claude-400"]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        api = Team(
            coordinator=[claude_says("Noted."), claude_says("Noted again."), delegate(),
                         claude_says("It could not.")],
            researcher=[httpx.Response(status, json=body), httpx.Response(status, json=body)],
        )
        route(monkeypatch, anthropic, "AsyncAnthropic", api)
        manager = GoogleADKWorkflowManager(
            agent_configs=team("inherit"),
            settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6",
                                  prompt_caching_enabled=False),
        )
        try:
            for text in [f"One: {LONG}", f"Two: {LONG}"]:
                await turn(manager, text)
            events = await turn(manager, "Find X, please.")
        finally:
            await manager.cleanup()

        (error,) = errors(events)
        assert (error.metadata["recoverable"], error.metadata["agent"]) == (True, "researcher")
        assert len([e for e in events if e.type is EventType.CONTEXT_TRIMMED]) == 1
        assert texts(events)[-1] == "It could not."


async def test_the_handler_comes_before_the_fit_guard(project, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    manager = GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt="You help.")],
        settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6"),
    )
    try:
        await manager.initialize_services()
        names = [plugin.name for plugin in manager._runner.app.plugins]
    finally:
        await manager.cleanup()

    # Last but the fit guard: a retry is the request the guard already fitted.
    assert names[-2:] == ["context_overflow", "fit_guard"]


# --- nothing prints over the UI ----------------------------------------------------------------

#: ADK's loggers whose failure tracebacks the CLI logs at debug.
FAILURE_LOGGERS = (
    "google_adk.google.adk.apps.compaction", "google_adk.google.adk.workflow._node_runner",
    "google_adk.google.adk.runners", "google_adk.google.adk.plugins.plugin_manager",
)


@pytest.fixture
def notice_hook(monkeypatch):
    """The CLI's hook, as BaseCLIApp installs it, taken off after the test."""
    import logging
    import warnings

    from agentic_cli.cli.app import _install_adk_notice_hook

    for name in FAILURE_LOGGERS:
        adk_logger = logging.getLogger(name)
        monkeypatch.setattr(adk_logger, "filters", list(adk_logger.filters))
    monkeypatch.setattr(warnings, "showwarning", warnings.showwarning)
    _install_adk_notice_hook()


def over_the_ui(caplog) -> list[tuple[str, str]]:
    import logging

    return [(r.name, r.getMessage()[:80]) for r in caplog.records
            if r.name.startswith("google_adk") and r.levelno >= logging.WARNING]


class TestNothingPrintsOverTheUI:
    async def test_a_plain_model_error_on_the_main_agent(
        self, project, monkeypatch, caplog, notice_hook
    ):
        import logging

        refused = httpx.Response(400, json={"type": "error", "error": {
            "type": "invalid_request_error", "message": "max_tokens: 100000 > 64000"}})
        _, manager = on(monkeypatch, "anthropic", refused)
        try:
            with caplog.at_level(logging.DEBUG):
                with pytest.raises(anthropic.BadRequestError):
                    await turn(manager, "One.")
        finally:
            await manager.cleanup()

        assert over_the_ui(caplog) == []

    async def test_a_retry_that_fails_otherwise(self, project, monkeypatch, caplog, notice_hook):
        import logging

        _, status, body = SHAPES["claude-400"]
        other = httpx.Response(400, json={"type": "error", "error": {
            "type": "invalid_request_error", "message": "max_tokens: 100000 > 64000"}})
        _, manager = on(monkeypatch, "anthropic", answer("anthropic", "A1."),
                        httpx.Response(status, json=body), other)
        try:
            await turn(manager, "One.")
            with caplog.at_level(logging.DEBUG):
                with pytest.raises(anthropic.BadRequestError):
                    await turn(manager, "Two?")
        finally:
            await manager.cleanup()

        assert over_the_ui(caplog) == []



# --- each model call's own copy: two parallel calls of one sub-agent -----------------------
#
# The fit guard keeps each model call's request for the overflow handler's
# retry. Two parallel delegations to one sub-agent run under one invocation
# and one agent name; ADK gives each model call an ``EventActions`` of its
# own, which reaches all three of its callbacks: that object is the key.


def _call(name: str, call_id: str, **args) -> types.Part:
    return types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args))


def _texts_of(contents) -> str:
    return " ".join(part.text for content in contents for part in content.parts or []
                    if part.text)


class _Delegating(BaseLlm):
    """A coordinator that delegates twice in one response, then answers."""

    model: str = "coordinator"
    calls: int = 0

    async def generate_content_async(self, llm_request, stream: bool = False):
        self.calls += 1
        parts = ([_call("researcher", "c1", request="Task A."),
                  _call("researcher", "c2", request="Task B.")]
                 if self.calls == 1 else [types.Part.from_text(text="Both done.")])
        yield LlmResponse(content=types.Content(role="model", parts=parts))


class _Refusing(BaseLlm):
    """The sub-agent's model: Task A's call fails once Task B's has started,
    and Task B's answers once Task A's failure was handled."""

    model: str = "researcher"
    _b_started: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _a_failed: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)

    async def generate_content_async(self, llm_request, stream: bool = False):
        if "Task A." in _texts_of(llm_request.contents):
            await asyncio.wait_for(self._b_started.wait(), 5)
            raise ValueError("refused")
        self._b_started.set()
        await asyncio.wait_for(self._a_failed.wait(), 5)
        yield LlmResponse(content=types.Content(role="model", parts=[
            types.Part.from_text(text="B done.")
        ]))


class _Recording(BasePlugin):
    """Before the fit guard, where the overflow handler is: each callback's
    ``EventActions``, and at an error what the guard holds for the call."""

    def __init__(self, guard, model: _Refusing) -> None:
        super().__init__(name="recording")
        self.guard, self.model = guard, model
        self.seen: list[tuple[str, str, int, str]] = []

    async def before_model_callback(self, *, callback_context, llm_request):
        self.seen.append(("before", callback_context.agent_name,
                          id(callback_context.actions), _texts_of(llm_request.contents)))

    async def after_model_callback(self, *, callback_context, llm_response):
        self.seen.append(("after", callback_context.agent_name,
                          id(callback_context.actions), ""))

    async def on_model_error_callback(self, *, callback_context, llm_request, error):
        copy = self.guard.request_as_seen(callback_context)
        self.seen.append(("error", callback_context.agent_name, id(callback_context.actions),
                          _texts_of(copy[0]) if copy else "<no copy>"))
        self.model._a_failed.set()
        return LlmResponse(content=types.Content(role="model", parts=[
            types.Part.from_text(text="A recovered.")
        ]))


async def test_one_model_call_is_one_event_actions_whose_copy_is_its_own():
    from agentic_cli.workflow.adk.fit_guard_plugin import FitGuardPlugin
    from agentic_cli.workflow.context import ContextBudget

    roomy = ContextBudget(window=1_000_000, window_known=True, budget=900_000,
                          trigger=800_000, keep=1_000)
    guard = FitGuardPlugin({"coordinator": roomy, "researcher": roomy},
                           {"coordinator": ProviderKind.ANTHROPIC,
                            "researcher": ProviderKind.ANTHROPIC})
    model = _Refusing()
    recording = _Recording(guard, model)
    researcher = LlmAgent(name="researcher", model=model, instruction="Research.",
                          description="Finds.", mode="single_turn",
                          disallow_transfer_to_parent=True, disallow_transfer_to_peers=True)
    root = LlmAgent(name="coordinator", model=_Delegating(), instruction="Coordinate.",
                    sub_agents=[researcher])
    runner = Runner(app=App(name="contract", root_agent=root, plugins=[recording, guard]),
                    session_service=InMemorySessionService())
    session = await runner.session_service.create_session(app_name="contract", user_id="u")
    async for _ in runner.run_async(user_id="u", session_id=session.id,
                                    new_message=types.Content(role="user", parts=[
                                        types.Part.from_text(text="Do both.")])):
        pass

    calls = [row for row in recording.seen if row[1] == "researcher"]
    (a,) = [actions for kind, _, actions, sent in calls if kind == "before" and "Task A." in sent]
    (b,) = [actions for kind, _, actions, sent in calls if kind == "before" and "Task B." in sent]
    assert a != b  # each of the parallel calls has its own
    (error,) = [row for row in calls if row[0] == "error"]
    assert error[2] == a  # the failed call's, at its error
    assert sorted(actions for kind, _, actions, _ in calls if kind == "after") == sorted([a, b])
    # What the guard holds for the failed call is its own request, not the other's.
    assert "Task A." in error[3] and "Task B." not in error[3]
    assert guard._seen == {}  # both released


def _last_text(body: dict) -> str:
    return " ".join(block["text"] for block in body["messages"][-1]["content"]
                    if block["type"] == "text")


async def test_a_parallel_delegations_retry_carries_its_own_task(project, monkeypatch):
    _, status, refusal = SHAPES["claude-400"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    coordinator = [claude_says("Noted."), claude_says(""), claude_says("Both done.")]
    coordinator[1]["content"] = [
        {"type": "tool_use", "id": "toolu_c1", "name": "researcher",
         "input": {"request": "Task A."}},
        {"type": "tool_use", "id": "toolu_c2", "name": "researcher",
         "input": {"request": "Task B."}},
    ]
    coordinator[1]["stop_reason"] = "tool_use"
    b_started, retried = asyncio.Event(), asyncio.Event()
    retries: list[str] = []

    async def researcher(body: dict):
        task = _last_text(body)
        if FIT_NOTE in json.dumps(body["messages"]):  # a retry
            retries.append(task)
            retried.set()
            return claude_says(f"Did {task}")
        if task == "Task A.":  # refused while Task B's call is in flight
            await asyncio.wait_for(b_started.wait(), 5)
            return httpx.Response(status, json=refusal)
        b_started.set()
        await asyncio.wait_for(retried.wait(), 5)
        return claude_says(f"Did {task}")

    def claude(request):
        body = json.loads(request.content)
        if "You research." in json.dumps(body.get("system")):
            return researcher(body)
        return coordinator.pop(0)

    api = FakeAPI(claude)
    route(monkeypatch, anthropic, "AsyncAnthropic", api)
    manager = GoogleADKWorkflowManager(
        agent_configs=team("inherit"),
        settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6",
                              prompt_caching_enabled=False),
    )
    try:
        await turn(manager, "One: the background.")
        events = await turn(manager, "Do both tasks, please.")
    finally:
        await manager.cleanup()

    # Both requests quote both calls; a call's own task is its brief, last.
    assert retries == ["Task A."]
    final = json.loads(api.requests[-1].content)["messages"][-1]["content"]
    results = {block["tool_use_id"]: json.dumps(block["content"]) for block in final
               if block["type"] == "tool_result"}
    assert "Did Task A." in results["toolu_c1"] and "Did Task B." in results["toolu_c2"]
    assert errors(events) == []
    assert texts(events)[-1] == "Both done."


async def test_a_cancelled_call_leaves_no_copy_past_the_next_turn(project, monkeypatch):
    from agentic_cli.workflow.adk.fit_guard_plugin import FitGuardPlugin

    started, never = asyncio.Event(), asyncio.Event()

    async def held(reply):
        started.set()
        await never.wait()
        return reply

    api, manager = on(monkeypatch, "anthropic")
    api.script = [held(answer("anthropic", "Late.")), answer("anthropic", "Again.")]
    try:
        await manager.initialize_services()
        (guard,) = [p for p in manager._runner.app.plugins if isinstance(p, FitGuardPlugin)]
        consumer = asyncio.create_task(turn(manager, "First."))
        await asyncio.wait_for(started.wait(), 5)
        consumer.cancel()  # mid-request: neither release fires
        with pytest.raises(asyncio.CancelledError):
            await consumer
        left_by_the_cancel = len(guard._seen)
        events = await turn(manager, "Again.")
        after_the_next_turn = len(guard._seen)
    finally:
        await manager.cleanup()

    assert texts(events) == ["Again."]
    assert left_by_the_cancel == 1
    assert after_the_next_turn == 0

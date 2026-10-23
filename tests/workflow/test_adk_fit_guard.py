"""On ADK a request over its agent's budget goes out under the trigger, the session intact.

The summary is triggered by the provider's count of the previous prompt, so
it lags one call; an inheriting sub-agent may run on a smaller model, and
never compacts the shared session; an unknown window counts as 128,000
tokens. The fit guard (``workflow/adk/fit_guard_plugin.py``), the last of the
manager's plugins, catches all three: it drops the oldest whole turns from
the request, folds a note into the first kept user content, and reports it
once a turn. These tests run the real manager on ``FakeAPI``: an endpoint
model (Chat Completions) whose usage is scripted small, so only the guard
acts, and Claude for a coordinator.
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic
import httpx
import openai
import pytest

pytest.importorskip("google.adk")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.adk.fit_guard_plugin import FIT_NOTE, FitGuardPlugin  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.summary_prompt import SUMMARY_PREFIX  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message, chat_completion  # noqa: E402
from tests.workflow.test_adk_context_cache import route_gemini  # noqa: E402
from tests.workflow.test_manager_models import route  # noqa: E402

#: The user's server: one model of unknown window (128,000 assumed), one of 20,000.
ENDPOINTS = {
    "local": {
        "base_url": "http://localhost:8000/v1",
        "models": {"unknown": {}, "small": {"context_window": 20_000}},
    }
}
HUGE = "word " * 24_000  # 120,000 characters, about 33,000 tokens with the margin
BIG = "word " * 8_000  # 40,000 characters


def answered(text: str, prompt_tokens: int = 10, *, tool: str | None = None) -> dict:
    message = chat_completion(text)
    message["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": 5,
                        "total_tokens": prompt_tokens + 5}
    if tool is not None:
        message["choices"][0]["message"] = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": tool, "arguments": "{}"}}
        ]}
        message["choices"][0]["finish_reason"] = "tool_calls"
    return message


class Server:
    """The user's server: answers from ``script``, records each request's body."""

    def __init__(self, *script: dict) -> None:
        self.script = list(script)
        self.api = FakeAPI(self._answer)

    def _answer(self, request) -> dict:
        return self.script.pop(0) if self.script else answered("[no scripted answer]")

    @property
    def bodies(self) -> list[dict]:
        return self.api.bodies


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    for variable in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
    return tmp_path / "project"


def serve(monkeypatch, server: Server) -> Server:
    route(monkeypatch, openai, "AsyncOpenAI", server.api)
    return server


def solo(model: str = "local:unknown", **values) -> GoogleADKWorkflowManager:
    return GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt="You help.",
                                   include_state_tools=values.pop("state_tools", False))],
        settings=BaseSettings(session_store="memory", model_endpoints=ENDPOINTS,
                              default_model=model, **values),
    )


async def turn(manager, text: str) -> list:
    return [event async for event in manager.process(text, "u", session_id="s")]


def user_texts(body: dict) -> list[str]:
    return [m["content"] for m in body["messages"] if m["role"] == "user"]


def trims(events) -> list:
    return [e for e in events if e.type is EventType.CONTEXT_TRIMMED
            and e.metadata["source"] == "fit_guard"]


async def stored_texts(manager) -> list[str]:
    session = await manager._session_service.get_session(
        app_name=manager.app_name, user_id="u", session_id="s"
    )
    return [part.text for event in session.events if event.author == "user"
            for part in (event.content.parts if event.content else []) if part.text]


TURNS = [f"{n}: {HUGE}" for n in ("One", "Two", "Three", "Four")]


class TestOverTheBudget:
    async def test_an_unknown_window_goes_out_under_the_trigger_with_the_note(
        self, project, monkeypatch
    ):
        server = serve(monkeypatch, Server(*(answered(f"A{n}.") for n in range(1, 5))))
        manager = solo()
        try:
            events = [await turn(manager, text) for text in TURNS]
            budget = manager._budgets["solo"]
            stored = await stored_texts(manager)
        finally:
            await manager.cleanup()

        assert (budget.window, budget.window_known) == (128_000, False)
        third, fourth = server.bodies[2], server.bodies[3]
        # Three turns fit the budget; the fourth request would not.
        assert user_texts(third) == TURNS[:3]
        assert user_texts(fourth) == [f"{FIT_NOTE}\n{TURNS[1]}", *TURNS[2:]]
        assert [m["content"] for m in fourth["messages"] if m["role"] == "assistant"] == [
            "A2.", "A3."
        ]
        # The session keeps every turn as it was.
        assert stored == TURNS
        (trimmed,) = trims(events[3])
        assert (trimmed.metadata["agent"], trimmed.metadata["turns_dropped"]) == ("solo", 1)
        assert all(trims(e) == [] for e in events[:3])

    async def test_it_is_reported_once_a_turn_across_its_model_calls(self, project, monkeypatch):
        server = serve(monkeypatch, Server(
            answered("A1."), answered("A2."), answered("A3."),
            answered("", tool="get_plan"), answered("A4."), answered("A5."),
        ))
        manager = solo(state_tools=True)
        try:
            for text in TURNS[:3]:
                await turn(manager, text)
            fourth = await turn(manager, TURNS[3])
            fifth = await turn(manager, "Five?")
        finally:
            await manager.cleanup()

        # Both of the fourth turn's calls, and the fifth's, were trimmed.
        for body in server.bodies[3:]:
            assert user_texts(body)[0].startswith(FIT_NOTE)
        assert len(trims(fourth)) == 1
        assert len(trims(fifth)) == 1

    async def test_a_newest_turn_over_the_budget_alone_goes_as_it_is(self, project, monkeypatch):
        server = serve(monkeypatch, Server(answered("A1."), answered("A2.")))
        manager = solo()
        try:
            await turn(manager, "One.")
            events = await turn(manager, HUGE * 5)
        finally:
            await manager.cleanup()

        assert user_texts(server.bodies[1]) == ["One.", HUGE * 5]
        assert trims(events) == []


# --- a sub-agent on a smaller model inheriting a large conversation (Review Focus 4) --------


class TestASmallerSubAgent:
    async def test_its_request_is_trimmed_and_the_session_never_compacted(
        self, project, monkeypatch
    ):
        def says(text: str, prompt_tokens: int = 100) -> dict:
            message = anthropic_message({"type": "text", "text": text})
            message["usage"] = {"input_tokens": prompt_tokens, "output_tokens": 5}
            return message

        delegate = says("")
        delegate["content"] = [{"type": "tool_use", "id": "toolu_c1", "name": "researcher",
                                "input": {"request": "Find X."}}]
        delegate["stop_reason"] = "tool_use"
        coordinator = [says("Noted."), says("Noted again."), delegate, says("X is 42.")]
        claude = FakeAPI(lambda request: coordinator.pop(0))
        route(monkeypatch, anthropic, "AsyncAnthropic", claude)
        # The researcher's prompt passes its own trigger too: still no summary.
        server = serve(monkeypatch, Server(answered("X is 42.", prompt_tokens=18_000)))
        manager = GoogleADKWorkflowManager(
            agent_configs=[
                AgentConfig(name="coordinator", prompt="You coordinate.",
                            sub_agents=["researcher"], description="Leads.",
                            include_state_tools=False),
                AgentConfig(name="researcher", prompt="You research.", description="Finds.",
                            model="local:small", context="inherit", include_state_tools=False),
            ],
            settings=BaseSettings(session_store="memory", model_endpoints=ENDPOINTS,
                                  default_model="claude-sonnet-4-6"),
        )
        try:
            await turn(manager, f"One: {BIG}")
            await turn(manager, f"Two: {BIG}")
            events = await turn(manager, "Find X, please.")
            session = await manager._session_service.get_session(
                app_name=manager.app_name, user_id="u", session_id="s"
            )
            budget = manager._budgets["researcher"]
        finally:
            await manager.cleanup()

        (body,) = server.bodies
        sent = user_texts(body)
        assert sent[0].startswith(FIT_NOTE)
        assert not any(text.startswith("One:") for text in sent)
        assert any("Two:" in text for text in sent)
        assert sent[-1].startswith("Find X.")
        (trimmed,) = trims(events)
        assert trimmed.metadata["agent"] == "researcher"
        # The coordinator's own requests went whole, and nothing was summarized.
        assert not any(FIT_NOTE in json.dumps(b) for b in claude.bodies)
        assert len(claude.requests) == 4
        assert [e for e in session.events if e.actions and e.actions.compaction] == []
        assert budget.budget == 19_000


# --- a switch to a smaller window after a summary (Review Focus 3) ---------------------------


class TestASwitchToASmallerWindow:
    async def test_the_summary_stays_first_and_the_oldest_turns_after_it_go(
        self, project, monkeypatch
    ):
        def says(text: str, prompt_tokens: int = 4_000) -> dict:
            message = anthropic_message({"type": "text", "text": text})
            message["usage"] = {"input_tokens": prompt_tokens, "output_tokens": 5}
            return message

        # The second turn's prompt passes Claude's 16,000-token ceiling: the
        # third is summarized first.
        script = [says("A1."), says("A2.", 17_000), says("A3."), says("A4."), says("A5.")]

        def claude(request):
            body = json.loads(request.content)
            if "system" not in body:  # the summary call
                return says("The user wrote at length.", 7_000)
            return script.pop(0)

        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(claude))
        server = serve(monkeypatch, Server(answered("From the small model.")))
        manager = solo(model="claude-sonnet-4-6", context_max_tokens=16_000)
        long = "word " * 2_400
        try:
            for text in [f"One: {long}", f"Two: {long}", "Three?", f"Four: {BIG}",
                         f"Five: {BIG}"]:
                await turn(manager, text)
            await manager.reinitialize(model="local:small")
            budget = manager._budgets["solo"]
            events = await turn(manager, "Six?")
        finally:
            await manager.cleanup()

        assert (budget.window, budget.budget) == (20_000, 19_000)
        # The summary, and the kept turns after it ("Two", "Three?", "Four")
        # until the request is under the small model's trigger; the summary
        # rides in the first kept user message, not a message of its own.
        (body,) = server.bodies
        assert user_texts(body) == [
            f"{SUMMARY_PREFIX}\n\nThe user wrote at length.\n{FIT_NOTE}\nFive: {BIG}", "Six?"
        ]
        (trimmed,) = trims(events)
        assert (trimmed.metadata["turns_dropped"], trimmed.metadata["messages_after"]) == (3, 4)


class Refusing:
    """A 20,000-token model behind the user's endpoint that refuses, as vLLM
    does, a request longer than its window (4 characters a token); summary
    calls are told apart by their prompt."""

    def __init__(self, *answers: dict) -> None:
        self.answers = list(answers)
        self.api = FakeAPI(self._answer)
        self.summaries: list[int] = []
        self.refused: list[str] = []

    def _answer(self, request):
        size = len(request.content)
        kind = "summary" if b"Context Extraction Assistant" in request.content else "agent"
        if kind == "summary":
            self.summaries.append(size)
        if size / 4 > 20_000:
            self.refused.append(kind)
            return httpx.Response(400, json={"error": {
                "message": "This model's maximum context length is 20000 tokens. However, "
                           f"you requested about {size // 4} tokens.", "code": 400}})
        if kind == "summary":
            return answered("The user wrote three long texts.", 9_000)
        return self.answers.pop(0) if self.answers else answered("[none]", 100)


class TestTheSummaryAfterASwitch:
    async def test_it_is_written_on_the_small_model_and_fits_it(self, project, monkeypatch):
        """Under Claude's trigger the conversation grew past the small model's
        window. After the switch the summary call is sized to the small model
        (the middle of the range left out): it is written, and no turn warns."""
        def says(text: str) -> dict:
            message = anthropic_message({"type": "text", "text": text})
            message["usage"] = {"input_tokens": 72_000, "output_tokens": 5}
            return message

        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(lambda request: says("Noted.")))
        small = Refusing(answered("A4.", 17_000), answered("A5.", 17_000))
        serve(monkeypatch, small)
        huge = "word " * 16_000  # 80,000 characters
        manager = solo(model="claude-sonnet-4-6", context_max_tokens=0,
                       prompt_caching_enabled=False)
        try:
            for n in ("One", "Two", "Three"):
                await turn(manager, f"{n}: {huge}")
            await manager.reinitialize(model="local:small")
            fourth, fifth = await turn(manager, "Four?"), await turn(manager, "Five?")
            session = await manager._session_service.get_session(
                app_name=manager.app_name, user_id="u", session_id="s"
            )
        finally:
            await manager.cleanup()

        assert small.summaries and all(size / 4 <= 20_000 for size in small.summaries)
        assert small.refused == []
        assert len([e for e in session.events if e.actions and e.actions.compaction]) == 1
        for events in (fourth, fifth):
            assert [e for e in events if e.type is EventType.ERROR] == []


def gemini_says(text: str, prompt_tokens: int) -> dict:
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]},
                        "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": prompt_tokens, "candidatesTokenCount": 5,
                          "totalTokenCount": prompt_tokens + 5},
    }


async def test_a_gemini_turn_after_a_summary_sends_no_two_user_contents_in_a_row(
    project, monkeypatch
):
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test-key")
    script = [gemini_says("A1.", 4_000), gemini_says("A2.", 17_000), gemini_says("A3.", 4_000)]

    def gemini(request):
        if b"systemInstruction" not in request.content:  # the summary call
            return gemini_says("The user wrote twice.", 7_000)
        return script.pop(0)

    api = FakeAPI(gemini)
    route_gemini(monkeypatch, api)
    text = "word " * 3_000  # 15,000 characters
    manager = solo(model="gemini-2.5-flash", context_max_tokens=16_000,
                   prompt_caching_enabled=False)
    try:
        for message in [f"One: {text}", f"Two: {text}", "Three?"]:
            await turn(manager, message)
    finally:
        await manager.cleanup()

    body = api.bodies[-1]  # the third turn's, after the summary
    assert [content["role"] for content in body["contents"]] == ["user", "model", "user"]
    assert [part["text"] for part in body["contents"][0]["parts"]] == [
        f"{SUMMARY_PREFIX}\n\nThe user wrote twice.", f"Two: {text}"
    ]


async def test_the_raw_log_records_the_request_as_sent(project, monkeypatch):
    """The request the fit guard trimmed, not the one ADK built: someone
    reading the log after an overflow reads what went out."""
    serve(monkeypatch, Server(*(answered(f"A{n}.") for n in range(1, 5))))
    manager = solo(raw_llm_logging=True)
    try:
        events = [await turn(manager, text) for text in TURNS]
        names = [plugin.name for plugin in manager._runner.app.plugins]
    finally:
        await manager.cleanup()

    (logged,) = [e for e in events[3] if e.type is EventType.LLM_REQUEST]
    sent = json.dumps(logged.metadata["messages"])
    assert FIT_NOTE in sent and "One:" not in sent and "Four:" in sent
    # Recorded after the guard; the response still ahead of any plugin
    # that could replace it.
    assert names[-2:] == ["fit_guard", "llm_logging_sent"]
    assert names.index("llm_logging") < names.index("fit_guard")


async def test_the_guard_is_the_last_plugin(project):
    """After every plugin that edits the request: it measures what is sent."""
    manager = solo()
    try:
        await manager.initialize_services()
        plugins, budgets = list(manager._runner.app.plugins), dict(manager._budgets)
    finally:
        await manager.cleanup()

    assert isinstance(plugins[-1], FitGuardPlugin)
    assert plugins[-1]._budgets == budgets

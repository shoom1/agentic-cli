"""Prompt caching on ADK: the app's context cache, and what it puts on the wire.

ADK caches through one ``ContextCacheConfig`` on the ``App``, for every agent
in it. The ADK manager sets it when ``prompt_caching_enabled`` is on and any
agent runs on a model that caches: Claude, whose requests ADK then marks with
cache breakpoints from the first turn, and Gemini, for which ADK creates a
server-side cache from the second turn. GPT, OpenRouter and endpoint models
cache nothing through ADK, so a tree of them gets no config. A native ADK YAML
app has no resolved agents, so its default model decides.

The config keeps ADK's defaults (a 1800-second TTL, a Gemini cache used for
at most 10 invocations) but bounds the creation of a Gemini cache at 60
seconds: a create that fails or hangs costs that turn its cache, never the
turn, because ADK then sends the request uncached.

Model requests go to the fakes in ``tests/workflow/adk/fakes.py``, and the
conftest guard gives every manager an unlisted catalog (records from the
shipped table), so nothing reaches the network.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import textwrap
import time

import httpx
import pytest

pytest.importorskip("google.adk")

import anthropic  # noqa: E402
import google.adk.runners as adk_runners  # noqa: E402
from google import genai  # noqa: E402
from google.adk.agents.context_cache_config import ContextCacheConfig  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_cli.workflow.adk import manager as adk_manager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message  # noqa: E402
from tests.workflow.test_manager_models import (  # noqa: E402
    KEYS,
    LOCAL,
    adk,
    keys,
    route,
    settings,
    team,
)

ADK_RUNNER_LOGGER = "google_adk.google.adk.runners"
ADVICE = "has no context_cache_config"


@pytest.fixture(autouse=True)
def _hermetic(tmp_path, monkeypatch):
    """A hermetic cwd and HOME, and no provider key unless a test sets one."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in KEYS:
        monkeypatch.delenv(variable, raising=False)


def cache_config(manager) -> ContextCacheConfig | None:
    """The context cache of the app the manager's runner runs."""
    return manager._runner.app.context_cache_config


def assert_caches(config: ContextCacheConfig | None) -> None:
    """ADK's defaults, with Gemini cache creation bounded at 60 seconds."""
    assert isinstance(config, ContextCacheConfig)
    assert (config.ttl_seconds, config.cache_intervals, config.min_tokens) == (1800, 10, 0)
    assert config.create_http_options == types.HttpOptions(timeout=60_000)


async def turn(manager, text: str) -> list:
    """One user turn, in one session for the whole test."""
    return [event async for event in manager.process(text, user_id="u", session_id="s")]


# --- when the app caches -----------------------------------------------------------------


@pytest.mark.parametrize(
    "providers, default_model, writer_model, caches",
    [
        pytest.param(["anthropic"], "claude-sonnet-4-5", None, True, id="claude"),
        pytest.param(["google"], "gemini-2.5-flash", None, True, id="gemini"),
        pytest.param(["google"], "gemini-3-flash-preview", None, True, id="gemini-3"),
        pytest.param(["openai"], "openai:gpt-5.5", None, False, id="all-openai"),
        pytest.param(
            ["openai", "anthropic"], "openai:gpt-5.5", "claude-sonnet-4-5", True,
            id="one-claude-agent-among-openai",
        ),
        pytest.param(
            ["openrouter"], "openrouter:qwen/qwen3-235b-a22b", "local:Qwen/Qwen3-32B", False,
            id="openrouter-and-endpoint",
        ),
    ],
)
async def test_the_app_caches_when_a_model_does(
    monkeypatch, providers, default_model, writer_model, caches
):
    keys(monkeypatch, *providers)
    manager = adk(team(writer_model), settings(default_model=default_model, model_endpoints=LOCAL))

    await manager.initialize_services()
    try:
        if caches:
            assert_caches(cache_config(manager))
        else:
            assert cache_config(manager) is None
    finally:
        await manager.cleanup()


async def test_prompt_caching_off_turns_it_off_for_every_model(monkeypatch):
    keys(monkeypatch, "google", "anthropic")
    manager = adk(
        team("claude-sonnet-4-5"),
        settings(default_model="gemini-2.5-flash", prompt_caching_enabled=False),
    )

    await manager.initialize_services()
    try:
        assert cache_config(manager) is None
    finally:
        await manager.cleanup()


async def test_a_default_model_no_agent_runs_on_does_not_decide(monkeypatch):
    """Every agent names a model that does not cache: the Claude default
    model, which none of them runs on, does not make the app cache."""
    keys(monkeypatch, "anthropic", "openai")
    configs = [
        AgentConfig(name="coordinator", prompt="Coordinate.", sub_agents=["writer"],
                    model="openai:gpt-5.5"),
        AgentConfig(name="writer", prompt="Write.", model="openai:gpt-5.5"),
    ]
    manager = adk(configs, settings(default_model="claude-sonnet-4-5"))

    await manager.initialize_services()
    try:
        assert manager.model == "anthropic:claude-sonnet-4-5"
        assert cache_config(manager) is None
    finally:
        await manager.cleanup()


def _native_yaml(tmp_path, model: str, *, helper: bool = False) -> str:
    """A native ADK YAML app on ``model``; with ``helper``, a root and a
    sub-agent it can transfer to."""
    path = tmp_path / "root_agent.yaml"
    sub_agents = "sub_agents:\n  - config_path: helper.yaml\n" if helper else ""
    path.write_text(
        textwrap.dedent(
            f"""
            agent_class: LlmAgent
            name: root
            model: {model}
            instruction: You are root.
            """
        ) + sub_agents,
        encoding="utf-8",
    )
    if helper:
        (tmp_path / "helper.yaml").write_text(
            textwrap.dedent(
                f"""
                agent_class: LlmAgent
                name: helper
                model: {model}
                instruction: You help.
                """
            ),
            encoding="utf-8",
        )
    return str(path)


@pytest.mark.parametrize(
    "provider, default_model, caches",
    [
        pytest.param("anthropic", "claude-sonnet-4-5", True, id="claude"),
        pytest.param("google", "gemini-2.5-flash", True, id="gemini"),
        pytest.param("openai", "gpt-5.5", False, id="openai"),
    ],
)
async def test_a_native_adk_app_follows_the_default_model(
    tmp_path, monkeypatch, provider, default_model, caches
):
    """A native ADK YAML app (``adk_config_path``) has no resolved agents."""
    keys(monkeypatch, provider)
    manager = adk(
        [], settings(default_model=default_model),
        adk_config_path=_native_yaml(tmp_path, default_model),
    )

    await manager.initialize_services()
    try:
        assert manager._resolved_models == {}
        if caches:
            assert_caches(cache_config(manager))
        else:
            assert cache_config(manager) is None
    finally:
        await manager.cleanup()


async def test_a_model_switch_decides_again(monkeypatch):
    """Claude to GPT and back, keeping the conversation: the app is built
    again, and caches only while a model caches."""
    keys(monkeypatch, "anthropic", "openai")
    manager = adk(team(), settings(default_model="claude-sonnet-4-5"))
    await manager.initialize_services()
    try:
        assert_caches(cache_config(manager))
        service = manager._session_service

        await manager.reinitialize(model="openai:gpt-5.5", preserve_sessions=True)
        assert cache_config(manager) is None

        await manager.reinitialize(model="anthropic:claude-sonnet-4-5", preserve_sessions=True)
        assert_caches(cache_config(manager))
        assert manager._session_service is service
    finally:
        await manager.cleanup()


@pytest.mark.parametrize(
    "native, caching, advice",
    [
        pytest.param(True, True, [], id="native-caching"),
        pytest.param(True, False, ["WARNING"], id="native-not-caching"),
        pytest.param(False, True, [], id="configs-caching"),
        pytest.param(False, False, [], id="configs-not-caching"),
    ],
)
async def test_adk_advises_a_context_cache_only_while_the_app_has_none(
    tmp_path, monkeypatch, caplog, native, caching, advice
):
    """ADK's runner advises a context cache for an app whose agents transfer
    to each other and that has none: a native ADK YAML app with sub-agents,
    since the agents built from configs delegate instead, and never
    transfer. With caching on there is nothing to advise; with it off the
    advice stays, and the CLI logs it at debug
    (``tests/cli/test_adk_notice_hook.py``)."""
    monkeypatch.setattr(adk_runners, "_UNCACHED_TRANSFER_APPS", set())  # ADK advises once per app
    # Whatever another test's CLI left on ADK's runner logger, it is not this test's.
    monkeypatch.setattr(logging.getLogger(ADK_RUNNER_LOGGER), "filters", [])
    keys(monkeypatch, "anthropic")
    s = settings(default_model="claude-sonnet-4-5", prompt_caching_enabled=caching)
    if native:
        manager = adk(
            [], s, adk_config_path=_native_yaml(tmp_path, "claude-sonnet-4-5", helper=True)
        )
    else:
        manager = adk(team(), s)

    with caplog.at_level(logging.DEBUG, logger=ADK_RUNNER_LOGGER):
        await manager.initialize_services()
    await manager.cleanup()

    given = [r.levelname for r in caplog.records if ADVICE in r.getMessage()]
    assert given == advice


# --- what caching sends --------------------------------------------------------------------


class TestClaudeOnTheWire:
    """ADK marks three cache breakpoints on every Claude request, from the
    first turn: the last tool, the system instruction and the last message."""

    async def test_a_request_marks_the_tools_the_system_and_the_conversation(self, monkeypatch):
        keys(monkeypatch, "anthropic")
        api = FakeAPI(anthropic_message())
        route(monkeypatch, anthropic, "AsyncAnthropic", api)
        manager = adk(team(), settings(default_model="claude-sonnet-4-5"))
        try:
            await turn(manager, "Hello")
        finally:
            await manager.cleanup()

        (body,) = api.bodies
        # Claude's default, 5-minute cache: ADK asks for its dearer hour-long
        # one only for a TTL of an hour or more.
        ephemeral = {"type": "ephemeral"}
        assert [tool.get("cache_control") for tool in body["tools"]][-2:] == [None, ephemeral]
        (system,) = body["system"]
        assert system["type"] == "text"
        assert system["text"].startswith("Coordinate.")
        assert system["cache_control"] == ephemeral
        assert body["messages"][-1]["content"][-1] == {
            "type": "text", "text": "Hello", "cache_control": ephemeral,
        }

    async def test_without_caching_nothing_is_marked(self, monkeypatch):
        keys(monkeypatch, "anthropic")
        api = FakeAPI(anthropic_message())
        route(monkeypatch, anthropic, "AsyncAnthropic", api)
        manager = adk(team(), settings(default_model="claude-sonnet-4-5",
                                       prompt_caching_enabled=False))
        try:
            await turn(manager, "Hello")
        finally:
            await manager.cleanup()

        (request,) = api.requests
        assert b"cache_control" not in request.content
        assert isinstance(api.bodies[0]["system"], str)


CACHE_NAME = "cachedContents/fake-cache"
GENERATE = "/v1beta/models/gemini-2.5-flash:generateContent"
CREATE = "/v1beta/cachedContents"


def gemini_api(request: httpx.Request) -> dict:
    """Google's Gemini API: a cache it creates, or an answer whose prompt
    (5000 tokens) clears Gemini 2.5's 2048-token floor for a cache."""
    if request.url.path == CREATE:
        return {"name": CACHE_NAME, "model": "models/gemini-2.5-flash"}
    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 5000, "candidatesTokenCount": 1,
                          "totalTokenCount": 5001},
    }


def refused_create(request: httpx.Request) -> dict | httpx.Response:
    """The Gemini API, failing every cache creation with a server error."""
    if request.url.path == CREATE:
        return httpx.Response(500, json={"error": {"code": 500, "status": "INTERNAL"}})
    return gemini_api(request)


class UnansweredCreate(httpx.AsyncBaseTransport):
    """The Gemini API behind ``api``, except that a cache creation is never
    answered. Like httpx's own transport, this one gives up at the request's
    read timeout; it records each deadline it was given, and fails a request
    that has none at once (a real server that never answers would hold it
    forever)."""

    def __init__(self, api: FakeAPI) -> None:
        self.api = api
        self.deadlines: list[float | None] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != CREATE:
            return await self.api.transport.handle_async_request(request)
        self.api.requests.append(request)
        deadline = request.extensions["timeout"]["read"]
        self.deadlines.append(deadline)
        if deadline is not None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.Event().wait(), deadline)
        raise httpx.ReadTimeout("timed out", request=request)


def route_gemini(monkeypatch, api: FakeAPI, transport: httpx.AsyncBaseTransport | None = None):
    """Send the requests of every ``genai.Client`` ADK's ``Gemini`` builds to
    ``api`` (through ``transport``, if given), keeping the key, headers and
    retries ADK gives it."""
    real = genai.Client

    def client(**kwargs):
        options = kwargs.get("http_options") or types.HttpOptions()
        http_client = httpx.AsyncClient(transport=transport or api.transport)
        kwargs["http_options"] = options.model_copy(update={"httpx_async_client": http_client})
        return real(**kwargs)

    monkeypatch.setattr(genai, "Client", client)


def solo_gemini(**values):
    return adk([AgentConfig(name="solo", prompt="Write.")],
               settings(default_model="gemini-2.5-flash", **values))


class TestGeminiOnTheWire:
    """ADK caches a Gemini agent's prompt on Google's side: never on a first
    request, and from the second once the previous prompt's size clears the
    model's floor. The cache holds the system instruction, the tools and the
    conversation before the new message; the request names it and sends the
    rest. Creating it may take 60 seconds; a create that fails or hangs leaves
    that turn's request uncached. Whether Google accepts the cache, and what it
    bills for it, only a live run shows."""

    async def test_the_second_turn_creates_a_cache_and_runs_on_it(self, monkeypatch):
        keys(monkeypatch, "google")
        api = FakeAPI(gemini_api)
        route_gemini(monkeypatch, api)
        manager = solo_gemini()
        try:
            await turn(manager, "Hello")
            await turn(manager, "Again")
        finally:
            await manager.cleanup()

        assert [r.url.path for r in api.requests] == [GENERATE, CREATE, GENERATE]
        first, cache, second = api.bodies
        assert "cachedContent" not in first
        assert first["systemInstruction"]["parts"][0]["text"].startswith("Write.")
        # The cache: what the first turn sent and answered, kept for 1800 s.
        assert cache["ttl"] == "1800s"
        assert cache["systemInstruction"] == first["systemInstruction"]
        assert cache["tools"] == first["tools"]
        assert cache["contents"] == [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "ok"}]},
        ]
        # The second turn names the cache and sends only the new message.
        assert second["cachedContent"] == CACHE_NAME
        assert "systemInstruction" not in second and "tools" not in second
        assert second["contents"] == [{"role": "user", "parts": [{"text": "Again"}]}]
        # Creating the cache may take 60 s; generating, the model's 600 s.
        deadlines = [
            (r.extensions["timeout"]["read"], r.headers.get("x-server-timeout"))
            for r in api.requests
        ]
        assert deadlines == [(600.0, "600"), (60.0, "60"), (600.0, "600")]

    @pytest.mark.parametrize("failure", ["refused", "hung"])
    async def test_a_failed_cache_creation_costs_the_turn_only_its_cache(
        self, monkeypatch, failure
    ):
        keys(monkeypatch, "google")
        if failure == "refused":
            api = FakeAPI(refused_create)
            route_gemini(monkeypatch, api)
        else:
            # The 60-second bound, shortened so that the test waits it out.
            monkeypatch.setattr(adk_manager, "_CACHE_CREATE_TIMEOUT_MS", 200, raising=False)
            api = FakeAPI(gemini_api)
            unanswered = UnansweredCreate(api)
            route_gemini(monkeypatch, api, unanswered)
        manager = solo_gemini()
        try:
            await turn(manager, "Hello")
            started = time.monotonic()
            second_turn = await turn(manager, "Again")
            elapsed = time.monotonic() - started
        finally:
            await manager.cleanup()

        # One attempt to create the cache, then the turn's request, whole.
        assert [r.url.path for r in api.requests] == [GENERATE, CREATE, GENERATE]
        second = api.bodies[-1]
        assert "cachedContent" not in second
        assert second["systemInstruction"]["parts"][0]["text"].startswith("Write.")
        assert [c["parts"][0]["text"] for c in second["contents"]] == ["Hello", "ok", "Again"]
        assert [e.content for e in second_turn if e.type == EventType.TEXT] == ["ok"]
        if failure == "hung":
            assert unanswered.deadlines == [0.2]
            assert elapsed < 5

    async def test_without_caching_no_cache_is_made(self, monkeypatch):
        keys(monkeypatch, "google")
        api = FakeAPI(gemini_api)
        route_gemini(monkeypatch, api)
        manager = solo_gemini(prompt_caching_enabled=False)
        try:
            await turn(manager, "Hello")
            await turn(manager, "Again")
        finally:
            await manager.cleanup()

        assert [r.url.path for r in api.requests] == [GENERATE, GENERATE]
        assert all("cachedContent" not in body for body in api.bodies)


# --- what a cache failure may cost a turn ---------------------------------------------

DELETE = "/v1beta/" + CACHE_NAME


def expired_create(request: httpx.Request) -> dict:
    """The Gemini API, whose caches have expired by the next turn (as after a
    pause of half an hour), so ADK deletes each one before it makes the next."""
    if request.url.path == CREATE:
        return {"name": CACHE_NAME, "model": "models/gemini-2.5-flash",
                "expireTime": "2020-01-01T00:00:00Z"}
    return gemini_api(request)


class UnansweredDelete(httpx.AsyncBaseTransport):
    """The Gemini API behind ``api``, except that deleting a cache is never
    answered: given up at the request's read timeout, and at once without one
    (a real server would hold it forever). Records each deadline."""

    def __init__(self, api: FakeAPI) -> None:
        self.api = api
        self.deadlines: list[float | None] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method != "DELETE":
            return await self.api.transport.handle_async_request(request)
        self.api.requests.append(request)
        deadline = request.extensions["timeout"]["read"]
        self.deadlines.append(deadline)
        if deadline is not None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.Event().wait(), deadline)
        raise httpx.ReadTimeout("timed out", request=request)


class ToolSteps:
    """The Gemini API for turns of ``steps`` tool calls and then an answer;
    ``create`` answers a cache creation."""

    def __init__(self, steps: int, create) -> None:
        self.steps, self.create, self.generated = steps, create, 0

    def __call__(self, request: httpx.Request) -> dict | httpx.Response:
        if request.url.path == CREATE:
            return self.create(request)
        self.generated += 1
        if (self.generated - 1) % (self.steps + 1) < self.steps:
            part = {"functionCall": {"name": "get_plan", "args": {}}}
        else:
            part = {"text": "ok"}
        answer = gemini_api(request)
        answer["candidates"][0]["content"]["parts"] = [part]
        return answer


def paths_per_turn(api: FakeAPI, marks: list[int]) -> list[list[str]]:
    edges = [*marks, len(api.requests)]
    return [[r.url.path for r in api.requests[a:b]] for a, b in zip(edges, edges[1:])]


class TestCacheFailuresCostAtMostTheirTurn:
    """A cache that cannot be made or removed costs at most one bounded attempt
    a turn: never the turn, and not one attempt per tool step."""

    async def test_deleting_an_expired_cache_is_bounded(self, monkeypatch):
        keys(monkeypatch, "google")
        # The Gemini client's own bound, shortened so that the test waits it out.
        from agentic_cli.workflow.adk import model_factory

        monkeypatch.setattr(model_factory, "_GEMINI_CLIENT_TIMEOUT_MS", 200, raising=False)
        api = FakeAPI(expired_create)
        unanswered = UnansweredDelete(api)
        route_gemini(monkeypatch, api, unanswered)
        manager = solo_gemini()
        try:
            await turn(manager, "Hello")
            await turn(manager, "Again")
            started = time.monotonic()
            third = await turn(manager, "Once more")
            elapsed = time.monotonic() - started
        finally:
            await manager.cleanup()

        assert unanswered.deadlines == [0.2]
        assert [e.content for e in third if e.type == EventType.TEXT] == ["ok"]
        assert elapsed < 5

    @pytest.mark.parametrize("failure", ["refused", "hung"])
    async def test_a_failing_cache_creation_is_tried_once_a_turn(self, monkeypatch, failure):
        keys(monkeypatch, "google")
        if failure == "refused":
            api = FakeAPI(ToolSteps(3, refused_create))
            route_gemini(monkeypatch, api)
        else:
            monkeypatch.setattr(adk_manager, "_CACHE_CREATE_TIMEOUT_MS", 200, raising=False)
            api = FakeAPI(ToolSteps(3, gemini_api))
            route_gemini(monkeypatch, api, UnansweredCreate(api))
        manager = solo_gemini()
        marks = []
        try:
            for text in ("Hello", "Again"):
                marks.append(len(api.requests))
                answer = await turn(manager, text)
                assert [e.content for e in answer if e.type == EventType.TEXT] == ["ok"]
        finally:
            await manager.cleanup()

        assert [turn.count(CREATE) for turn in paths_per_turn(api, marks)] == [1, 1]
        assert all("cachedContent" not in body for body in api.bodies if "contents" in body)

    async def test_a_tool_using_first_turn_still_caches_from_its_second_call(self, monkeypatch):
        keys(monkeypatch, "google")
        api = FakeAPI(ToolSteps(3, gemini_api))
        route_gemini(monkeypatch, api)
        manager = solo_gemini()
        try:
            await turn(manager, "Hello")
        finally:
            await manager.cleanup()

        assert [r.url.path for r in api.requests] == [GENERATE, CREATE, *[GENERATE] * 3]
        assert [body.get("cachedContent") for body in api.bodies[2:]] == [CACHE_NAME] * 3

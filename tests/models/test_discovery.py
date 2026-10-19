"""The listers: each provider's model list over its REST API, through a fake transport.

Every request goes through ``httpx.MockTransport``; nothing reaches the
network. The payloads under ``tests/fixtures/model_listings/`` are trimmed
recordings of each API's real response shape.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from agentic_cli.models import EndpointSpec
from agentic_cli.models.discovery import ListedModel, Listing, list_models
from tests.models.fakes import endpoints

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "model_listings"

KEYS = {
    "google": "AIza-test-GOOGLE-SECRET",
    "anthropic": "sk-ant-test-ANTHROPIC-SECRET",
    "openai": "sk-test-OPENAI-SECRET",
    "openrouter": "sk-or-test-OPENROUTER-SECRET",
    "local-vllm": "vllm-test-ENDPOINT-SECRET",
}
VLLM_SPEC = EndpointSpec(base_url="http://gpu-box:8000/v1", api_key_env="VLLM_KEY")


def payload(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def endpoint(provider: str, *, keyed: bool = True):
    """The endpoint for ``provider`` (a built-in, or ``local-vllm``), with its test key."""
    builtins = ("google", "anthropic", "openai", "openrouter")
    keys = {name: KEYS[name] for name in builtins} if keyed else {}
    env = {"VLLM_KEY": KEYS["local-vllm"]} if keyed else {}
    return endpoints(specs={"local-vllm": VLLM_SPEC}, env=env, **keys)[provider]


class Server:
    """A fake provider API: answers from ``handler`` and records every request."""

    def __init__(self, handler) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = handler

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self._handler(request)
        if asyncio.iscoroutine(response):
            response = await response
        return response


def serving(body: dict, status: int = 200) -> Server:
    return Server(lambda request: httpx.Response(status, json=body))


FIXTURE_FOR = {
    "google": "google",
    "anthropic": "anthropic",
    "openai": "openai",
    "openrouter": "openrouter",
    "local-vllm": "vllm",
}


async def listed(provider: str, server: Server, **kwargs) -> Listing:
    return await list_models(endpoint(provider, **kwargs), transport=server.transport)


def by_id(listing: Listing) -> dict[str, ListedModel]:
    return {model.id: model for model in listing.models}


# --- requests: URL, headers, the key's one place -----------------------------

# Where each provider's key goes: (header, how its value is spelled).
KEY_HEADERS = {
    "google": ("x-goog-api-key", "{key}"),
    "anthropic": ("x-api-key", "{key}"),
    "openai": ("authorization", "Bearer {key}"),
    "openrouter": ("authorization", "Bearer {key}"),
    "local-vllm": ("authorization", "Bearer {key}"),
}
URLS = {
    "google": "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
    "anthropic": "https://api.anthropic.com/v1/models?limit=1000",
    "openai": "https://api.openai.com/v1/models",
    "openrouter": "https://openrouter.ai/api/v1/models",
    "local-vllm": "http://gpu-box:8000/v1/models",
}


@pytest.mark.parametrize("provider", list(URLS))
async def test_the_request_url(provider):
    server = serving(payload(FIXTURE_FOR[provider]))

    listing = await listed(provider, server)

    assert listing.ok, listing.error
    assert [str(r.url) for r in server.requests] == [URLS[provider]]
    assert all(r.method == "GET" for r in server.requests)


@pytest.mark.parametrize("provider", list(KEY_HEADERS))
async def test_the_key_goes_in_its_header_and_nowhere_else(provider):
    server = serving(payload(FIXTURE_FOR[provider]))
    key = KEYS[provider]

    await listed(provider, server)

    (request,) = server.requests
    header, spelling = KEY_HEADERS[provider]
    assert request.headers[header] == spelling.format(key=key)
    carrying = [name.lower() for name, value in request.headers.items() if key in value]
    assert carrying == [header]
    assert key not in str(request.url)
    assert key.encode() not in request.content


async def test_anthropic_sends_the_api_version():
    server = serving(payload("anthropic"))

    await listed("anthropic", server)

    assert server.requests[0].headers["anthropic-version"] == "2023-06-01"


async def test_openrouter_is_listed_without_a_key():
    server = serving(payload("openrouter"))

    listing = await listed("openrouter", server, keyed=False)

    assert listing.ok
    assert "authorization" not in server.requests[0].headers


async def test_an_endpoint_without_a_key_sends_a_placeholder():
    server = serving(payload("vllm"))

    listing = await listed("local-vllm", server, keyed=False)

    assert listing.ok
    assert server.requests[0].headers["authorization"] == "Bearer EMPTY"


# --- pagination --------------------------------------------------------------


async def test_google_follows_the_page_token():
    models = payload("google")["models"]
    pages = {
        None: {"models": models[:3], "nextPageToken": "page-2"},
        "page-2": {"models": models[3:6], "nextPageToken": "page-3"},
        "page-3": {"models": models[6:]},
    }
    server = Server(lambda r: httpx.Response(200, json=pages[r.url.params.get("pageToken")]))

    listing = await listed("google", server)

    assert listing.ok
    assert [r.url.params.get("pageToken") for r in server.requests] == [None, "page-2", "page-3"]
    assert all(r.url.params["pageSize"] == "1000" for r in server.requests)
    assert set(by_id(listing)) == set(by_id(await listed("google", serving(payload("google")))))


async def test_anthropic_follows_has_more_and_last_id():
    data = payload("anthropic")["data"]
    pages = {
        None: {"data": data[:2], "has_more": True, "last_id": data[1]["id"]},
        data[1]["id"]: {"data": data[2:], "has_more": False, "last_id": data[-1]["id"]},
    }
    server = Server(lambda r: httpx.Response(200, json=pages[r.url.params.get("after_id")]))

    listing = await listed("anthropic", server)

    assert listing.ok
    assert [r.url.params.get("after_id") for r in server.requests] == [None, "claude-opus-4-7"]
    assert len(listing.models) == len(data)


@pytest.mark.parametrize(
    "provider, page",
    [
        ("google", {"models": [], "nextPageToken": "again"}),
        ("anthropic", {"data": [], "has_more": True, "last_id": "again"}),
    ],
)
async def test_pagination_that_never_ends_is_a_failure(provider, page):
    """A cursor that repeats would loop forever, and a partial list must not
    become the authority that rejects a model."""
    server = serving(page)

    listing = await listed(provider, server)

    assert not listing.ok
    assert listing.models == ()
    assert "pagination" in listing.error
    assert len(server.requests) <= 25


# --- field mapping -----------------------------------------------------------


async def test_google_fields():
    models = by_id(await listed("google", serving(payload("google"))))

    assert models["gemini-2.5-flash"] == ListedModel(
        id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        context_window=1048576,
        max_output=65536,
        reasoning=True,
    )
    # No thinking flag in the payload: unknown, not "no".
    assert models["gemini-2.0-flash"].reasoning is None
    assert models["gemini-2.0-flash"].max_output == 8192


async def test_google_keeps_the_text_models_as_0_6_did():
    """``generateContent`` and ``createCachedContent``, ``gemini-`` only, no
    point releases, ``-latest`` aliases, dated snapshots, ``-customtools`` or
    ``-exp-`` variants."""
    listing = await listed("google", serving(payload("google")))

    assert [model.id for model in listing.models] == [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.0-flash",
    ]


async def test_anthropic_fields():
    models = by_id(await listed("anthropic", serving(payload("anthropic"))))

    assert models["claude-opus-4-7"] == ListedModel(
        id="claude-opus-4-7",
        display_name="Claude Opus 4.7",
        context_window=1000000,
        max_output=128000,
        image_input=True,
        reasoning=True,
        thinking_types=("adaptive",),
        effort_levels=("low", "medium", "high", "xhigh", "max"),
        can_disable=True,
    )
    assert models["claude-sonnet-4-6"].thinking_types == ("adaptive", "enabled")
    assert models["claude-sonnet-4-6"].effort_levels == ("low", "medium", "high", "max")
    # Effort unsupported: no ladder, even though each level is listed.
    assert models["claude-sonnet-4-5"].thinking_types == ("enabled",)
    assert models["claude-sonnet-4-5"].effort_levels == ()
    # Thinking unsupported: no types, and says so.
    assert models["claude-3-5-haiku"].reasoning is False
    assert models["claude-3-5-haiku"].thinking_types == ()
    assert models["claude-3-5-haiku"].image_input is True
    # No capabilities in the payload: nothing known.
    assert models["claude-3-haiku"] == ListedModel("claude-3-haiku", display_name="Claude Haiku 3")


async def test_anthropic_says_whether_thinking_can_be_turned_off():
    """``capabilities.thinking.types.disabled.supported``: False exactly when a
    request with thinking disabled is a 400 from that model."""
    models = by_id(await listed("anthropic", serving(payload("anthropic"))))

    assert models["claude-sonnet-5-5"] == ListedModel(
        id="claude-sonnet-5-5",
        display_name="Claude Sonnet 5.5",
        context_window=1000000,
        max_output=128000,
        image_input=True,
        reasoning=True,
        thinking_types=("adaptive",),
        effort_levels=("low", "medium", "high", "xhigh", "max"),
        can_disable=False,
    )
    assert models["claude-opus-4-7"].can_disable is True
    assert models["claude-sonnet-4-5"].can_disable is True
    # The types without a disabled entry, or no capabilities: not said.
    assert models["claude-haiku-4-5"].can_disable is None
    assert models["claude-3-haiku"].can_disable is None


@pytest.mark.parametrize(
    "disabled",
    [{"supported": "no"}, {"supported": None}, {}, True, "false"],
    ids=["a string flag", "a null flag", "no flag", "a bool, not an object", "a string"],
)
async def test_a_disabled_type_that_is_not_a_flag_says_nothing(disabled):
    types = {"adaptive": {"supported": True}, "disabled": disabled}
    entry = {
        "id": "claude-sonnet-5-5",
        "capabilities": {"thinking": {"supported": True, "types": types}},
    }

    listing = await listed("anthropic", serving({"data": [entry], "has_more": False}))

    assert listing.ok, listing.error
    assert by_id(listing)["claude-sonnet-5-5"].can_disable is None


def test_a_listed_model_does_not_say_whether_thinking_can_be_turned_off_by_default():
    assert ListedModel(id="any").can_disable is None


async def test_a_dated_anthropic_id_is_listed_once_as_its_alias():
    data = payload("anthropic")["data"]
    dated = next(m for m in data if m["id"] == "claude-sonnet-4-5-20250929")
    page = {
        "data": [
            dated,
            {**dated, "id": "claude-sonnet-4-5", "display_name": "The alias"},
            {**dated, "id": "claude-sonnet-4-5@2025-09-29", "display_name": "The Vertex form"},
        ],
        "has_more": False,
    }

    listing = await listed("anthropic", serving(page))

    # The first one listed wins, as in 0.6.
    assert [(m.id, m.display_name) for m in listing.models] == [
        ("claude-sonnet-4-5", "Claude Sonnet 4.5"),
    ]


async def test_openai_lists_ids_only():
    listing = await listed("openai", serving(payload("openai")))

    assert listing.models[:2] == (ListedModel(id="gpt-5.5"), ListedModel(id="gpt-5.5-2026-03-05"))
    assert len(listing.models) == 10
    assert all(m == ListedModel(id=m.id) for m in listing.models)


async def test_openrouter_fields():
    models = by_id(await listed("openrouter", serving(payload("openrouter"))))

    assert models["qwen/qwen3-235b-a22b"] == ListedModel(
        id="qwen/qwen3-235b-a22b",
        display_name="Qwen: Qwen3 235B A22B",
        context_window=131072,
        max_output=8192,
        image_input=False,
        tools=True,
        reasoning=True,
    )
    # The free variant has no tools and no max completion size.
    free = models["qwen/qwen3-235b-a22b:free"]
    assert (free.tools, free.reasoning, free.max_output) == (False, True, None)
    assert models["anthropic/claude-sonnet-4.5"].image_input is True
    assert models["meta-llama/llama-3.3-70b-instruct"].reasoning is False


async def test_openrouter_without_supported_parameters_knows_nothing_of_tools():
    model = dict(payload("openrouter")["data"][0])
    del model["supported_parameters"], model["architecture"], model["top_provider"]

    (listed_model,) = (await listed("openrouter", serving({"data": [model]}))).models

    assert listed_model.tools is listed_model.reasoning is listed_model.image_input is None
    assert listed_model.max_output is None


async def test_an_endpoint_lists_its_served_names():
    listing = await listed("local-vllm", serving(payload("vllm")))

    assert listing.models == (
        ListedModel(id="Qwen/Qwen3-32B", context_window=32768),
        ListedModel(id="qwen3-sql-lora", context_window=32768),
    )


async def test_an_endpoint_without_max_model_len():
    """llama.cpp, Ollama and most other servers send ids only."""
    listing = await listed("local-vllm", serving({"object": "list", "data": [{"id": "llama3.2"}]}))

    assert listing.models == (ListedModel(id="llama3.2"),)


async def test_entries_that_are_not_models_are_skipped():
    page = {"data": [{"id": "gpt-5.5"}, "gpt-5", {"object": "model"}, {"id": ""}, {"id": 5}]}

    listing = await listed("openai", serving(page))

    assert listing.ok
    assert listing.models == (ListedModel(id="gpt-5.5"),)


async def test_numbers_that_are_not_positive_integers_are_unknown():
    model = {**payload("vllm")["data"][0], "max_model_len": "32768"}
    other = {"id": "b", "max_model_len": True}
    third = {"id": "c", "max_model_len": 0}

    listing = await listed("local-vllm", serving({"data": [model, other, third]}))

    assert [m.context_window for m in listing.models] == [None, None, None]


# --- the listing -------------------------------------------------------------


async def test_the_listing_names_its_provider_and_time():
    listing = await list_models(
        endpoint("local-vllm"), transport=serving(payload("vllm")).transport, clock=lambda: 1234.5
    )

    assert (listing.provider, listing.ok, listing.fetched_at, listing.error) == (
        "local-vllm", True, 1234.5, None,
    )


async def test_an_injected_transport_is_left_open():
    """The caller owns a transport it passes in; one listing must not close it
    under the others sharing it."""

    class Recording(httpx.MockTransport):
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    transport = Recording(lambda r: httpx.Response(200, json=payload("openai")))

    await list_models(endpoint("openai"), transport=transport)

    assert transport.closed is False


# --- failures ----------------------------------------------------------------


def _status(code: int, provider: str):
    # A server that echoes the key back in its error body must not get it
    # into the listing's error.
    return lambda r: httpx.Response(code, json={"error": {"message": f"bad key {KEYS[provider]}"}})


def _raise(exc_type):
    def handler(request):
        raise exc_type("boom " + request.headers.get("authorization", ""), request=request)

    return handler


FAILURES = {
    "401": (lambda p: _status(401, p), "HTTP 401"),
    "429": (lambda p: _status(429, p), "HTTP 429"),
    "500": (lambda p: _status(500, p), "HTTP 500"),
    "timeout": (lambda p: _raise(httpx.ReadTimeout), "timed out"),
    "connect": (lambda p: _raise(httpx.ConnectError), "connection failed"),
    "invalid-json": (lambda p: lambda r: httpx.Response(200, content=b"<html>oops</html>"),
                     "invalid JSON"),
    "wrong-shape": (lambda p: lambda r: httpx.Response(200, json={"data": "x", "models": 5}),
                    "unexpected"),
    "not-an-object": (lambda p: lambda r: httpx.Response(200, json=["gpt-5"]), "unexpected"),
}


@pytest.mark.parametrize("failure", list(FAILURES))
@pytest.mark.parametrize("provider", list(URLS))
async def test_a_failure_is_a_failed_listing_never_an_exception(provider, failure):
    make_handler, reason = FAILURES[failure]

    listing = await list_models(
        endpoint(provider), transport=Server(make_handler(provider)).transport, clock=lambda: 99.0
    )

    assert listing.ok is False
    assert listing.models == ()
    assert listing.fetched_at == 99.0
    assert reason in listing.error
    for key in KEYS.values():
        assert key not in listing.error


async def test_a_hung_server_is_cut_off_at_the_timeout():
    """A mock transport is not subject to httpx's socket timeouts, so this
    checks the listing's own deadline, which bounds every page together."""

    async def hang(request):
        await asyncio.sleep(30)

    started = time.monotonic()
    listing = await list_models(endpoint("google"), transport=Server(hang).transport, timeout=0.2)

    assert time.monotonic() - started < 2
    assert listing.ok is False
    assert listing.error == "timed out after 0.2s"

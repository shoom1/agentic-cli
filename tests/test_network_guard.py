"""The offline tests' provider-host guard, and what it covers.

Every test but a live one (marked ``llm``) runs with each provider SDK's host
variable at a local port nothing listens on, no proxy for it, and without the
variables that would send a client elsewhere (``tests/network_guard.py``,
applied by the root conftest's ``_dead_provider_urls``): a built-in
provider's client that a test did not route fails at once, locally. A
configured provider's host is its settings entry's, so routing alone protects
it, and a child process given its own environment (``child_env``) runs
without the guard, as does a module- or session-scoped fixture. Each built-in
provider's client, ADK's and LangChain's, is built here as its adapter builds
it, unrouted and with fake keys, and its host and proxy read: nothing is sent.
No run uploads LangSmith traces: the root conftest clears the tracing
switches once, as the session starts.
"""

from __future__ import annotations

import os
import textwrap
from unittest.mock import patch

import pytest

from tests.network_guard import DEAD_URL

pytest_plugins = ["pytester"]

#: Each SDK's host variable: the dead URL in an offline test.
DEAD = (
    "ANTHROPIC_BASE_URL",
    "OPENAI_BASE_URL",
    "GOOGLE_GEMINI_BASE_URL",
    "GOOGLE_VERTEX_BASE_URL",
    "OPENROUTER_API_BASE",
    "OPENROUTER_BASE_URL",
)
#: No proxy for the dead URL: a request to it stays local.
NO_PROXY = "127.0.0.1,localhost"
#: What an offline test's environment holds.
SET = {**dict.fromkeys(DEAD, DEAD_URL), "NO_PROXY": NO_PROXY, "no_proxy": NO_PROXY}
#: What would send a client elsewhere whatever its host variable says: unset.
UNSET = (
    "ANTHROPIC_API_URL",
    "OPENAI_API_BASE",
    "LANGSMITH_GATEWAY",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_GENAI_USE_ENTERPRISE",
    "ANTHROPIC_PROXY",
    "OPENAI_PROXY",
)
#: LangSmith's tracing switches (``langsmith.utils.tracing_is_enabled``).
TRACING = (
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING",
)


class TestTheGuard:
    def test_an_offline_test_runs_with_every_host_dead(self):
        assert {name: os.environ.get(name) for name in SET} == SET
        assert [name for name in UNSET if name in os.environ] == []

    def test_a_live_test_keeps_the_real_environment(self, pytester, monkeypatch):
        """The marker decides, as the live modules carry it (``pytestmark``):
        an offline test of a session started with real hosts and Vertex AI on
        gets the dead ones, a live test of the same session the real ones."""
        real = "https://provider.invalid"
        for name in SET:
            monkeypatch.setenv(name, real)
        for name in UNSET:
            monkeypatch.setenv(name, "true")
        pytester.makeini(
            "[pytest]\nasyncio_default_fixture_loop_scope = function\n"
            "markers =\n    llm: a live test\n"
        )
        pytester.makepyfile(
            test_offline=f"""
                import os

                def test_offline():
                    assert {{n: os.environ.get(n) for n in {tuple(SET)!r}}} == {SET!r}
                    assert [n for n in {UNSET!r} if n in os.environ] == []
            """,
            test_live=f"""
                import os

                import pytest

                pytestmark = pytest.mark.llm

                def test_live():
                    assert {{n: os.environ.get(n) for n in {tuple(SET)!r}}} == dict.fromkeys(
                        {tuple(SET)!r}, {real!r}
                    )
                    assert {{n: os.environ.get(n) for n in {UNSET!r}}} == dict.fromkeys(
                        {UNSET!r}, "true"
                    )
            """,
        )

        # The suite's root conftest, the guard among its fixtures.
        result = pytester.runpytest("-p", "tests.conftest", "-p", "no:cacheprovider")

        result.assert_outcomes(passed=2)


class TestNoRunTraces:
    def test_no_run_uploads_traces(self, pytester, monkeypatch):
        """A session started with LangSmith tracing on, as a developer's shell
        or the live tests' env file may have it: neither an offline nor a
        live test traces."""
        langsmith_utils = pytest.importorskip("langsmith.utils")
        for name in TRACING:
            monkeypatch.setenv(name, "true")
        pytester.makeini(
            "[pytest]\nasyncio_default_fixture_loop_scope = function\n"
            "markers =\n    llm: a live test\n"
        )
        check = f"""
                    assert [n for n in {TRACING!r} if n in os.environ] == []
                    assert utils.tracing_is_enabled() is False
        """
        pytester.makepyfile(
            test_offline=f"""
                import os

                from langsmith import utils

                def test_offline():{check}""",
            test_live=f"""
                import os

                import pytest
                from langsmith import utils

                pytestmark = pytest.mark.llm

                def test_live():{check}""",
        )

        try:
            result = pytester.runpytest("-p", "tests.conftest", "-p", "no:cacheprovider")
        finally:
            # What the inner session read is cached in this process too.
            langsmith_utils.get_env_var.cache_clear()

        result.assert_outcomes(passed=2)

    def test_the_live_tests_key_file_turns_no_tracing_on(self, tmp_path, monkeypatch):
        """``tests/integration/conftest.py`` loads the live tests' keys from a
        file as it is collected, after the session cleared the switches."""
        keys = tmp_path / "live-keys"
        keys.write_text(
            "NETGUARD_FAKE_KEY=k-test\n" + "".join(f"{name}=true\n" for name in TRACING),
            encoding="utf-8",
        )
        # Before the import, which loads the file it names (the first that
        # exists wins), so no real key file is read here.
        monkeypatch.setenv("AGENTIC_TEST_ENV_FILE", str(keys))

        with patch.dict(os.environ):  # what the loader sets, at import too, is put back
            from tests.integration.conftest import _load_live_test_env

            _load_live_test_env()
            loaded = {name: os.environ.get(name) for name in ("NETGUARD_FAKE_KEY", *TRACING)}

        assert loaded == {"NETGUARD_FAKE_KEY": "k-test", **dict.fromkeys(TRACING)}


# --- every client the framework builds, unrouted ------------------------------------

#: Fake keys: a client is built, never sent.
KEYS = {"google": "g-test-key", "anthropic": "a-test-key", "openai": "o-test-key"}


def resolved(spec: str):
    """``spec`` resolved for an agent, as the manager resolves it, on the
    built-in providers with ``KEYS``."""
    from agentic_cli.models import RoleMap, resolve_model
    from tests.models.fakes import FakeCatalog, providers

    every = providers(**KEYS)
    return resolve_model(
        agent="writer",
        spec=spec,
        generation=None,
        thinking_effort="none",
        roles=RoleMap(default_model=None, model_roles={}),
        catalog=FakeCatalog(),
        providers=every,
    )


def adk(spec: str, settings):
    """The model an ADK agent on ``spec`` gets (``build_adk_model``)."""
    pytest.importorskip("google.adk")
    from agentic_cli.workflow.adk.model_factory import build_adk_model

    return build_adk_model(resolved(spec), settings=settings, app_name="guard-test").model


def chat(spec: str, settings):
    """The chat model a Deep Agents or LangGraph agent on ``spec`` gets
    (``build_chat_model``)."""
    pytest.importorskip("langchain_openai")
    pytest.importorskip("langchain_anthropic")
    pytest.importorskip("langchain_google_genai")
    from agentic_cli.workflow.langchain_models import build_chat_model

    return build_chat_model(resolved(spec), settings=settings, app_name="guard-test")


def host(url) -> str:
    return str(url).rstrip("/")


class TestEveryClientTheFrameworkBuilds:
    """Each one's host is the dead one: what a test leaves unrouted goes nowhere."""

    def test_adk_gemini(self, mock_context):
        gemini = adk("google:gemini-2.5-flash", mock_context.settings)

        assert host(gemini.api_client._api_client._http_options.base_url) == DEAD_URL

    def test_adk_claude_on_the_framework_s_client(self, mock_context):
        claude = adk("anthropic:claude-sonnet-4-6", mock_context.settings)

        assert host(claude.client.base_url) == DEAD_URL

    def test_adk_claude_on_adk_s_own_client(self, tmp_path, monkeypatch):
        """A native ADK YAML app's bare ``claude-*``: ADK builds the model from
        the string and its client from the ``AsyncAnthropic`` its module bound
        at import, keyed by ANTHROPIC_API_KEY."""
        pytest.importorskip("google.adk")
        from agentic_cli.workflow.adk.build import load_native_root

        monkeypatch.setenv("ANTHROPIC_API_KEY", KEYS["anthropic"])
        path = tmp_path / "root_agent.yaml"
        path.write_text(textwrap.dedent("""
            agent_class: LlmAgent
            name: root
            model: claude-sonnet-4-6
            instruction: Help.
        """), encoding="utf-8")

        claude = load_native_root(path).canonical_model

        assert claude.client is None  # ADK's own
        assert host(claude._anthropic_client.base_url) == DEAD_URL

    def test_adk_gpt(self, mock_context):
        gpt = adk("openai:gpt-5.5", mock_context.settings)

        assert host(gpt.client.base_url) == DEAD_URL

    def test_chat_google_generative_ai(self, mock_context):
        gemini = chat("google:gemini-2.5-flash", mock_context.settings)

        assert host(gemini.client._api_client._http_options.base_url) == DEAD_URL

    def test_chat_anthropic(self, mock_context):
        claude = chat("anthropic:claude-sonnet-4-6", mock_context.settings)

        assert host(claude._async_client.base_url) == DEAD_URL

    def test_chat_openai(self, mock_context):
        gpt = chat("openai:gpt-5.5", mock_context.settings)

        assert host(gpt.root_async_client.base_url) == DEAD_URL

    @pytest.mark.parametrize("build", [adk, chat], ids=["adk", "langchain"])
    def test_a_google_client_a_test_puts_on_vertex_ai(self, mock_context, monkeypatch, build):
        """The guard clears the Vertex AI switch; a test that sets it again
        still reaches no host."""
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
        gemini = build("google:gemini-2.5-flash", mock_context.settings)

        client = (gemini.api_client if build is adk else gemini.client)._api_client
        assert client.vertexai is True
        assert host(client._http_options.base_url) == DEAD_URL


class TestNoProxyCarriesAnUnroutedRequest:
    """A proxy in the developer's environment would get the dead URL's request
    in plain text, key header and all: the dead URL is never proxied. Each
    client's proxy for it is read, as it would choose one; nothing is sent."""

    @pytest.fixture(autouse=True)
    def proxies(self, monkeypatch):
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            monkeypatch.setenv(name, "http://proxy.invalid:3128")

    @staticmethod
    def proxied(client) -> bool:
        """Whether an httpx (or httpx2) client sends the dead URL through a proxy."""
        url = type(client.base_url)(DEAD_URL)
        return client._transport_for_url(url) is not client._transport

    def test_adk_claude(self, mock_context):
        claude = adk("anthropic:claude-sonnet-4-6", mock_context.settings)

        assert not self.proxied(claude.client._client)

    def test_adk_gpt(self, mock_context):
        gpt = adk("openai:gpt-5.5", mock_context.settings)

        assert not self.proxied(gpt.client._client)

    def test_chat_anthropic(self, mock_context, monkeypatch):
        from langchain_anthropic import chat_models

        # Its HTTP client is cached per host for the process: build this one.
        monkeypatch.setattr(
            chat_models,
            "_get_default_async_httpx_client",
            chat_models._get_default_async_httpx_client.__wrapped__,
        )
        claude = chat("anthropic:claude-sonnet-4-6", mock_context.settings)

        assert not self.proxied(claude._async_client._client)

    def test_chat_openai(self, mock_context):
        gpt = chat("openai:gpt-5.5", mock_context.settings)

        assert not self.proxied(gpt.root_async_client._client)

    @pytest.mark.parametrize("build", [adk, chat], ids=["adk", "langchain"])
    def test_gemini(self, mock_context, build):
        """google-genai sends through httpx, or through aiohttp (installed
        here) trusting the environment, whose proxy lookup is read."""
        aiohttp_helpers = pytest.importorskip("aiohttp.helpers")
        from yarl import URL

        gemini = build("google:gemini-2.5-flash", mock_context.settings)

        client = (gemini.api_client if build is adk else gemini.client)._api_client
        assert not self.proxied(client._async_httpx_client)
        with pytest.raises(LookupError):
            aiohttp_helpers.get_env_proxy_for_url(URL(DEAD_URL))

"""No offline test reaches a model provider.

Every test but a live one (marked ``llm``) runs with each provider SDK's host
variable at a local port nothing listens on, and without the variables that
would send a client elsewhere (``tests/network_guard.py``, applied by the root
conftest's ``_dead_provider_urls``): a client a test did not route to a fake
fails at once, locally, instead of reaching a provider with whatever key the
environment holds. Each client the framework builds, ADK's and LangChain's, is
built here as its adapter builds it, unrouted and with fake keys, and its host
read: nothing is sent.
"""

from __future__ import annotations

import os
import textwrap

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
#: What would send a client elsewhere whatever its host variable says: unset.
UNSET = (
    "ANTHROPIC_API_URL",
    "OPENAI_API_BASE",
    "LANGSMITH_GATEWAY",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_GENAI_USE_ENTERPRISE",
)


class TestTheGuard:
    def test_an_offline_test_runs_with_every_host_dead(self):
        assert {name: os.environ.get(name) for name in DEAD} == dict.fromkeys(DEAD, DEAD_URL)
        assert [name for name in UNSET if name in os.environ] == []

    def test_a_live_test_keeps_the_real_environment(self, pytester, monkeypatch):
        """The marker decides, as the live modules carry it (``pytestmark``):
        an offline test of a session started with real hosts and Vertex AI on
        gets the dead ones, a live test of the same session the real ones."""
        real = "https://provider.invalid"
        for name in DEAD:
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
                    assert {{n: os.environ.get(n) for n in {DEAD!r}}} == dict.fromkeys(
                        {DEAD!r}, {DEAD_URL!r}
                    )
                    assert [n for n in {UNSET!r} if n in os.environ] == []
            """,
            test_live=f"""
                import os

                import pytest

                pytestmark = pytest.mark.llm

                def test_live():
                    assert {{n: os.environ.get(n) for n in {DEAD!r}}} == dict.fromkeys(
                        {DEAD!r}, {real!r}
                    )
                    assert {{n: os.environ.get(n) for n in {UNSET!r}}} == dict.fromkeys(
                        {UNSET!r}, "true"
                    )
            """,
        )

        # The suite's root conftest, the guard among its fixtures.
        result = pytester.runpytest("-p", "tests.conftest", "-p", "no:cacheprovider")

        result.assert_outcomes(passed=2)


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

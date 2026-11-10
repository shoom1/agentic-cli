"""No offline test reaches a model provider.

Every test but a live one (marked ``llm``) runs with each provider SDK's host
variable at a local port nothing listens on, and without the variables that
would send a client elsewhere (``tests/network_guard.py``, applied by the root
conftest's ``_dead_provider_urls``): a client a test did not route to a fake
fails at once, locally, instead of reaching a provider with whatever key the
environment holds.
"""

from __future__ import annotations

import os

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


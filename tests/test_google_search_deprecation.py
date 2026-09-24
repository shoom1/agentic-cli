"""``agentic_cli.tools.google_search_tool`` is a deprecated re-export.

It re-exports ADK's built-in ``google_search`` singleton, which runs inside
the model and is never seen by the permission engine. It stays importable
through 0.6.x and warns on use; ``web_search`` is the supported alternative.

The name is resolved on every access (nothing is cached), so each access site
warns and these tests can run in-process. Only the claims about a *fresh*
import (silent, loads no ADK) need a separate interpreter.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import types
import warnings

import pytest

import agentic_cli.tools as tools

MESSAGE = re.compile(r"google_search_tool.*0\.7\.0.*web_search", re.S)


def _ours(caught) -> list[warnings.WarningMessage]:
    return [w for w in caught if "google_search_tool" in str(w.message)]


class TestAccess:
    def test_returns_adks_singleton_and_warns(self):
        from google.adk.tools import google_search

        with pytest.warns(DeprecationWarning, match=MESSAGE) as caught:
            obj = tools.google_search_tool
        assert obj is google_search
        assert len(_ours(caught)) == 1

    def test_warning_is_attributed_to_the_accessing_code(self):
        with pytest.warns(DeprecationWarning) as caught:
            from agentic_cli.tools import google_search_tool  # noqa: F401
        assert _ours(caught)[0].filename == __file__

    def test_message_points_at_the_alternative_and_the_permission_gap(self):
        with pytest.warns(DeprecationWarning) as caught:
            tools.google_search_tool
        message = str(_ours(caught)[0].message)
        assert "GoogleSearchTool" in message
        assert "permission engine" in message

    def test_every_access_site_warns(self):
        """Not cached: whichever code touches the name first must not consume
        the only warning another caller would have seen."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tools.google_search_tool
            tools.google_search_tool
        assert len(_ours(caught)) == 2
        assert "google_search_tool" not in vars(tools)

    def test_missing_adk_export_is_an_attribute_error(self, monkeypatch):
        """PEP 562: a failed lookup must raise AttributeError, so hasattr() and
        getattr(..., default) keep working."""
        monkeypatch.setitem(sys.modules, "google.adk.tools", types.ModuleType("google.adk.tools"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert hasattr(tools, "google_search_tool") is False

    def test_unknown_attribute_still_raises(self):
        with pytest.raises(AttributeError, match="no_such_tool"):
            tools.no_such_tool

    def test_not_exported_by_star_import(self):
        """In ``__all__`` a star-import would resolve (and warn about) a name
        the caller never uses."""
        assert "google_search_tool" not in tools.__all__
        assert "web_search" in tools.__all__


class TestConfigReferences:
    """A dotted path in an AgentConfig or agents YAML is resolved by the
    framework, so no frame of the application's own code is on the stack. A
    DeprecationWarning attributed there is hidden by Python's default filters,
    so the resolver re-issues it as a FutureWarning, which is shown."""

    def test_dotted_path_emits_a_visible_future_warning(self):
        from google.adk.tools import google_search
        from agentic_cli.tools.tool_resolver import resolve_tool

        with pytest.warns(FutureWarning, match=MESSAGE) as caught:
            obj = resolve_tool("agentic_cli.tools.google_search_tool")
        assert obj is google_search
        assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]

    def test_future_warning_names_the_config_reference(self):
        from agentic_cli.tools.tool_resolver import resolve_tool

        with pytest.warns(FutureWarning) as caught:
            resolve_tool("agentic_cli.tools.google_search_tool")
        assert "tool reference 'agentic_cli.tools.google_search_tool'" in str(caught[0].message)

    def test_error_filter_on_deprecations_does_not_break_resolution(self):
        from agentic_cli.tools.tool_resolver import resolve_tool

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            assert resolve_tool("agentic_cli.tools.google_search_tool") is not None

    def test_other_dotted_paths_are_unaffected(self):
        from agentic_cli.tools.tool_resolver import resolve_tool
        from agentic_cli.tools.search import web_search

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve_tool("agentic_cli.tools.search.web_search") is web_search


def _fresh(body: str) -> dict:
    """Run ``body`` in a fresh interpreter and return the JSON it prints.

    ``PYTHONWARNINGS`` is cleared so the parent's filters cannot change the
    child's behaviour.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip(), f"child printed nothing; stderr:\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "statement",
    ["import agentic_cli.tools", "from agentic_cli.tools import *"],
)
def test_fresh_import_is_silent_and_loads_no_adk(statement):
    result = _fresh(f"""
        import json, sys, warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            {statement}
        print(json.dumps({{
            "ours": [str(w.message) for w in caught if "google_search_tool" in str(w.message)],
            "adk": sorted(m for m in sys.modules if m.startswith("google.adk")),
        }}))
    """)
    assert result["ours"] == []
    assert result["adk"] == [], "importing agentic_cli.tools must not load ADK"

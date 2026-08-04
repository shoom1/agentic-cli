"""``agentic_cli.tools.google_search_tool`` is a deprecated re-export.

It was never an Agentic CLI integration — just ADK's ``GoogleSearchTool``
singleton re-exported under our namespace, carrying ADK's model/tool
constraints and leaving grounding metadata, citations and the required Search
Suggestions UI entirely to the application. It stays importable through the
0.6.x window and warns on first use; ``web_search`` is the supported
alternative.

Every case runs in its own interpreter: the resolved object is cached in the
module's globals, so the warning fires once *per process*. A sibling test that
had already touched the attribute — or merely a different test order — would
otherwise hide a missing warning entirely.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

_PRELUDE = """
import json, warnings


def summarize(caught):
    return [
        {
            "category": w.category.__name__,
            "message": str(w.message),
            "filename": w.filename,
        }
        for w in caught
    ]


def emit(payload):
    print(json.dumps(payload))
"""


def _run(body: str) -> dict:
    """Run ``body`` in a fresh interpreter, return the JSON payload it emits."""
    script = _PRELUDE + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _deprecations(result: dict) -> list[dict]:
    """The recorded DeprecationWarnings that are about this re-export."""
    return [
        w
        for w in result["warnings"]
        if w["category"] == "DeprecationWarning"
        and "google_search_tool" in w["message"]
    ]


class TestImportingThePackageIsSilent:
    """Nobody pays for a deprecation they did not opt into."""

    def test_importing_tools_emits_no_deprecation_warning(self):
        result = _run(
            """
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                import agentic_cli.tools
            emit({"warnings": summarize(caught)})
            """
        )
        assert _deprecations(result) == []

    def test_name_stays_in_all_without_resolving_it(self):
        """The export survives the window, and reading ``__all__`` is not use."""
        result = _run(
            """
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                import agentic_cli.tools
                names = list(agentic_cli.tools.__all__)
            emit({"warnings": summarize(caught), "all": names})
            """
        )
        assert "google_search_tool" in result["all"]
        assert _deprecations(result) == []


class TestDeprecatedAccess:
    def test_from_import_still_returns_adks_singleton(self):
        result = _run(
            """
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                from agentic_cli.tools import google_search_tool
            from google.adk.tools import google_search
            emit({
                "warnings": summarize(caught),
                "is_adk_singleton": google_search_tool is google_search,
                "type": type(google_search_tool).__name__,
            })
            """
        )
        assert result["is_adk_singleton"] is True, (
            "the shim must hand back ADK's existing object, not a copy or wrapper"
        )
        assert result["type"] == "GoogleSearchTool"
        assert len(_deprecations(result)) == 1

    def test_warning_names_the_removal_target_and_the_alternative(self):
        result = _run(
            """
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                from agentic_cli.tools import google_search_tool
            emit({"warnings": summarize(caught)})
            """
        )
        found = _deprecations(result)
        assert len(found) == 1
        message = found[0]["message"]

        assert "agentic_cli.tools.google_search_tool" in message
        assert "deprecated" in message.lower()
        assert "0.7.0" in message, "the removal release must be stated"
        assert "agentic_cli.tools.web_search" in message, (
            "the supported framework-level alternative must be named"
        )
        assert (
            "from google.adk.tools.google_search_tool import GoogleSearchTool"
            in message
        ), (
            "the escape hatch must name the fully qualified import path of the "
            "configurable GoogleSearchTool class"
        )
        assert "from google.adk.tools import google_search" not in message, (
            "that path imports ADK's pre-built singleton, not the class callers "
            "are told to instantiate and configure"
        )
        lowered = message.lower()
        for responsibility in ("grounding", "citation"):
            assert responsibility in lowered, (
                f"the caller takes on ADK's {responsibility} obligations"
            )
        assert "search suggestions when returned" in lowered, (
            "Search Suggestions are a conditional obligation — they are not "
            "returned on every response"
        )

    def test_warning_is_attributed_to_the_importing_code(self):
        """``stacklevel=2`` — the actionable frame is the caller's import."""
        result = _run(
            """
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                from agentic_cli.tools import google_search_tool
            emit({"warnings": summarize(caught)})
            """
        )
        filename = _deprecations(result)[0]["filename"]
        assert filename == "<string>", (
            f"warning was blamed on {filename!r} instead of the caller"
        )


class TestResolutionIsCached:
    def test_repeated_access_is_the_same_object_and_warns_once(self):
        result = _run(
            """
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                import agentic_cli.tools
                first = agentic_cli.tools.google_search_tool
                second = agentic_cli.tools.google_search_tool
                from agentic_cli.tools import google_search_tool as third
            emit({
                "warnings": summarize(caught),
                "identical": first is second and second is third,
            })
            """
        )
        assert result["identical"] is True
        assert len(_deprecations(result)) == 1, (
            "the resolved object must be cached in module globals, as the "
            "package's other lazy exports are, so one import warns once"
        )


class TestUnknownAttributesStillFail:
    def test_attribute_error_is_unchanged(self):
        result = _run(
            """
            import agentic_cli.tools
            try:
                agentic_cli.tools.no_such_tool
            except AttributeError as exc:
                emit({"warnings": [], "error": str(exc)})
            """
        )
        assert "no_such_tool" in result["error"]


class TestWebSearchIsUntouched:
    def test_web_search_imports_silently_and_is_the_defining_object(self):
        result = _run(
            """
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                import agentic_cli.tools
                from agentic_cli.tools import web_search
            from agentic_cli.tools.search import web_search as defining
            emit({
                "warnings": summarize(caught),
                "is_defining": web_search is defining,
                "in_all": "web_search" in agentic_cli.tools.__all__,
                "callable": callable(web_search),
            })
            """
        )
        assert result["is_defining"] is True
        assert result["in_all"] is True
        assert result["callable"] is True
        assert _deprecations(result) == []

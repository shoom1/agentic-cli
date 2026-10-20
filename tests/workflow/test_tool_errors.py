"""The error results every backend gives a failed tool call.

A tool that raises, and a call to a tool name the agent does not have, cost
that one call: the model reads ``{"success": False, "error": ...}`` and can
recover. ``workflow/tool_errors.py`` builds those results once, so ADK's
``ToolErrorPlugin`` and the Deep Agents middleware say the same thing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_cli.workflow import tool_errors
from agentic_cli.workflow.tool_errors import (
    MAX_TOOL_ERROR,
    tool_error_result,
    unknown_tool_result,
)


class _Recorder:
    """Stands in for the module's structlog logger."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):
        def log(event: str, **fields) -> None:
            self.calls.append((level, event, fields))

        return log


@pytest.fixture
def log(monkeypatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(tool_errors, "logger", recorder)
    return recorder


class TestToolErrorResult:
    def test_it_names_the_tool_and_the_exception(self, log):
        assert tool_error_result("flaky_tool", KeyError("missing")) == {
            "success": False,
            "error": "flaky_tool failed: KeyError: 'missing'",
        }

    def test_the_message_is_cut_to_500_characters(self, log):
        result = tool_error_result("t", RuntimeError("x" * 10_000))

        assert MAX_TOOL_ERROR == 500
        assert result["error"].startswith("t failed: RuntimeError: xxx")
        assert result["error"] == f"t failed: RuntimeError: {'x' * 10_000}"[:500] + "..."

    def test_it_logs_tool_raised(self, log):
        error = ValueError("bad input")

        tool_error_result("parse", error)

        assert log.calls == [
            (
                "warning",
                "tool_raised",
                {
                    "tool": "parse",
                    "error_type": "ValueError",
                    "error": "bad input",
                    "exc_info": error,
                },
            )
        ]

    def test_the_logged_message_is_cut_too(self, log):
        tool_error_result("t", RuntimeError("y" * 10_000))

        (_, _, fields), = log.calls
        assert fields["error"] == "y" * 500 + "..."


class TestUnknownToolResult:
    def test_it_is_adks_text_for_an_unknown_function(self):
        assert unknown_tool_result("nope", ["read_file", "grep", "edit_file"]) == {
            "success": False,
            "error": (
                "nope failed: ValueError: Tool 'nope' not found.\n"
                "Available tools: edit_file, grep, read_file"
            ),
        }

    def test_an_agent_without_tools_lists_none(self):
        assert unknown_tool_result("nope", []) == {
            "success": False,
            "error": "nope failed: ValueError: Tool 'nope' not found.\nAvailable tools: ",
        }

    def test_it_is_cut_to_500_characters(self):
        names = [f"tool_{i:03d}" for i in range(200)]

        error = unknown_tool_result("nope", names)["error"]

        assert len(error) == MAX_TOOL_ERROR + len("...")
        assert error.endswith("...")


class TestAdkPlugin:
    """ADK's plugin answers with the shared result; its own tests are unchanged."""

    async def test_the_plugin_returns_the_shared_result(self, log):
        from agentic_cli.workflow.adk import tool_error_plugin

        error = KeyError("missing")
        result = await tool_error_plugin.ToolErrorPlugin().on_tool_error_callback(
            tool=SimpleNamespace(name="flaky_tool"),
            tool_args={},
            tool_context=None,
            error=error,
        )

        assert result == tool_error_result("flaky_tool", error)
        assert tool_error_plugin.tool_error_result is tool_error_result
